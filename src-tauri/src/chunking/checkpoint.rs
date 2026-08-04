//! Resume state: where to continue, and how much of the transcript to trust.
//!
//! Port of the checkpoint half of `chunking.py`. The durability contract here is
//! the reason a one-hour meeting can survive a crash, a power loss, or Stop:
//!
//! 1. A chunk's text is appended and **fsynced**.
//! 2. Only then is the checkpoint written, atomically, recording the file's new
//!    size in `text_bytes`.
//!
//! A crash between the two leaves a tail past `text_bytes`. [`resume_or_restart`]
//! truncates to `text_bytes` and redoes that chunk, so recovery is idempotent
//! and never yields a duplicated or half-written line.

use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::DEFAULT_CHUNK_SECONDS;

/// Bumped from the Python app's 3. The Rust engines produce different text than
/// openai-whisper/MLX did, so a checkpoint written by the Python app must never
/// be resumed into here — the version alone guarantees that, before the
/// fingerprint even gets a chance to disagree.
pub const CHECKPOINT_VERSION: u32 = 4;

pub const TRANSCRIPT_SUFFIX: &str = "txt";

/// `<name>.transcript.partial.json`, alongside the audio.
pub fn checkpoint_path(source: &Path) -> PathBuf {
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    source.with_file_name(format!("{stem}.transcript.partial.json"))
}

fn tmp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{name}.tmp"))
}

/// Final transcript location: next to the audio it came from.
///
/// Named `<when>-<recording>.txt`, where `when` is the local time the
/// transcription started — first, so a folder of transcripts sorts
/// chronologically. Re-running a recording therefore keeps the earlier
/// transcript instead of overwriting it.
pub fn transcript_path_for(source: &Path, stamp: Option<&str>) -> PathBuf {
    let stamp = stamp
        .map(str::to_string)
        .unwrap_or_else(|| chrono::Local::now().format("%Y%m%d-%H%M%S").to_string());
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    source.with_file_name(format!("{stamp}-{stem}.{TRANSCRIPT_SUFFIX}"))
}

/// The transcript file this run will write.
///
/// A run that resumes has to append to the file the earlier run started, not
/// mint a fresh stamp — otherwise the checkpoint would describe text that is not
/// in the file, and the run would silently restart from silence.
pub fn resolve_transcript_path(source: &Path, engine_key: &str, chunk_sec: f64) -> PathBuf {
    if let Ok(fingerprint) = source_fingerprint(source, engine_key, chunk_sec) {
        if let Some(state) = load_checkpoint(source, &fingerprint) {
            if state.chunks_done > 0 && !state.transcript_name.is_empty() {
                let started = source.with_file_name(&state.transcript_name);
                if started.exists() {
                    return started;
                }
            }
        }
    }
    transcript_path_for(source, None)
}

/// Where to resume, and how much of the transcript file is trustworthy.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    pub fingerprint: String,
    pub next_start_sec: f64,
    pub chunks_done: usize,
    /// The size the transcript had after the last fully written chunk. Anything
    /// past it is the debris of a chunk interrupted mid-write.
    pub text_bytes: u64,
    /// The file that text lives in — stored as a bare name, so moving the
    /// recording and its transcript together doesn't break resuming.
    pub transcript_name: String,
}

/// The JSON actually on disk. Kept separate from [`Checkpoint`] so the wire
/// format is explicit and versioned rather than implied by field order.
#[derive(Serialize, Deserialize)]
struct CheckpointJson {
    version: u32,
    fingerprint: String,
    duration_sec: Option<f64>,
    next_start_sec: f64,
    chunks_done: usize,
    text_bytes: u64,
    #[serde(default)]
    transcript_name: String,
}

/// Identity of "this file transcribed this way".
///
/// Any change to the audio file, the engine/model, the language or the chunk
/// size makes an existing checkpoint meaningless, so it is discarded rather than
/// merged into a transcript produced with different settings.
pub fn source_fingerprint(source: &Path, engine_key: &str, chunk_sec: f64) -> Result<String> {
    let meta = fs::metadata(source)
        .with_context(|| format!("cannot stat {}", source.display()))?;
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|delta| delta.as_nanos())
        .unwrap_or(0);
    Ok(format!(
        "{}:{}:{}:{}",
        meta.len(),
        mtime_ns,
        engine_key,
        format_chunk_sec(chunk_sec)
    ))
}

/// Renders the chunk size the way Python's `f"{chunk_sec:g}"` did, so the
/// fingerprint reads `300` rather than `300.0` for the default.
fn format_chunk_sec(chunk_sec: f64) -> String {
    if chunk_sec.fract() == 0.0 && chunk_sec.abs() < 1e15 {
        format!("{}", chunk_sec.trunc() as i64)
    } else {
        format!("{chunk_sec}")
    }
}

/// Read a resumable checkpoint, or `None` if absent, stale, or corrupt.
///
/// A truncated or garbled checkpoint is worth less than the time it would cost
/// to debug: redo the file rather than emit a corrupt transcript.
pub fn load_checkpoint(source: &Path, fingerprint: &str) -> Option<Checkpoint> {
    let raw = fs::read_to_string(checkpoint_path(source)).ok()?;
    let data: CheckpointJson = serde_json::from_str(&raw).ok()?;

    if data.version != CHECKPOINT_VERSION || data.fingerprint != fingerprint {
        return None;
    }

    Some(Checkpoint {
        fingerprint: fingerprint.to_string(),
        next_start_sec: data.next_start_sec,
        chunks_done: data.chunks_done,
        text_bytes: data.text_bytes,
        transcript_name: data.transcript_name,
    })
}

/// Write the checkpoint atomically so a crash mid-write can't corrupt it.
pub fn save_checkpoint(
    source: &Path,
    checkpoint: &Checkpoint,
    duration_sec: Option<f64>,
) -> Result<()> {
    let path = checkpoint_path(source);
    let tmp = tmp_path(&path);
    let json = serde_json::to_string(&CheckpointJson {
        version: CHECKPOINT_VERSION,
        fingerprint: checkpoint.fingerprint.clone(),
        duration_sec,
        next_start_sec: checkpoint.next_start_sec,
        chunks_done: checkpoint.chunks_done,
        text_bytes: checkpoint.text_bytes,
        transcript_name: checkpoint.transcript_name.clone(),
    })?;

    // The rename is what makes this atomic; fsyncing the temp file first means
    // the rename can never expose a partially written checkpoint.
    {
        let mut handle = File::create(&tmp)
            .with_context(|| format!("cannot write {}", tmp.display()))?;
        handle.write_all(json.as_bytes())?;
        handle.flush()?;
        handle.sync_all()?;
    }
    fs::rename(&tmp, &path)
        .with_context(|| format!("cannot replace {}", path.display()))?;
    Ok(())
}

pub fn clear_checkpoint(source: &Path) {
    let path = checkpoint_path(source);
    // Best-effort: a leftover checkpoint is harmless (the fingerprint or
    // version will reject it), so failing to unlink is not worth reporting.
    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(tmp_path(&path));
}

/// Set up the transcript file for this run and say where to start.
///
/// Resuming trims the transcript back to the last checkpointed chunk. If the
/// transcript is missing or shorter than the checkpoint claims (deleted or
/// edited between runs), the checkpoint is meaningless and the file starts over
/// from silence rather than resuming into a gap.
pub fn resume_or_restart(source: &Path, out_path: &Path, fingerprint: &str) -> Result<Checkpoint> {
    if let Some(mut state) = load_checkpoint(source, fingerprint) {
        if state.chunks_done > 0 {
            if let Ok(meta) = fs::metadata(out_path) {
                if meta.len() >= state.text_bytes {
                    let handle = OpenOptions::new().write(true).open(out_path)?;
                    handle.set_len(state.text_bytes)?;
                    handle.sync_all()?;
                    state.transcript_name = out_path
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    return Ok(state);
                }
            }
        }
    }

    clear_checkpoint(source);
    fs::write(out_path, b"")
        .with_context(|| format!("cannot create {}", out_path.display()))?;
    Ok(Checkpoint {
        fingerprint: fingerprint.to_string(),
        next_start_sec: 0.0,
        chunks_done: 0,
        text_bytes: 0,
        transcript_name: out_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
    })
}

/// Append one chunk's transcript and force it to disk; return the new size.
///
/// `sync_all` is the point of the exercise: without it a crash could lose text
/// the checkpoint has already recorded as written.
pub fn append_text(out_path: &Path, text: &str) -> Result<u64> {
    let mut handle = OpenOptions::new()
        .append(true)
        .create(true)
        .open(out_path)
        .with_context(|| format!("cannot append to {}", out_path.display()))?;
    if !text.is_empty() {
        handle.write_all(text.as_bytes())?;
    }
    handle.flush()?;
    handle.sync_all()?;
    Ok(handle.seek(SeekFrom::End(0))?)
}

/// Seconds of audio already transcribed in a usable checkpoint (0 if none).
///
/// Used only to tell the user "resuming at 25:00" before work restarts.
pub fn resumable_seconds(source: &Path, engine_key: &str, chunk_sec: Option<f64>) -> f64 {
    let chunk_sec = chunk_sec.unwrap_or(DEFAULT_CHUNK_SECONDS);
    let Ok(fingerprint) = source_fingerprint(source, engine_key, chunk_sec) else {
        return 0.0;
    };
    load_checkpoint(source, &fingerprint)
        .map(|state| state.next_start_sec)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample(dir: &Path) -> PathBuf {
        let path = dir.join("standup.m4a");
        fs::write(&path, b"not really audio, but it has a size and an mtime").unwrap();
        path
    }

    fn checkpoint_for(fingerprint: &str, out_name: &str) -> Checkpoint {
        Checkpoint {
            fingerprint: fingerprint.to_string(),
            next_start_sec: 300.0,
            chunks_done: 1,
            text_bytes: 12,
            transcript_name: out_name.to_string(),
        }
    }

    #[test]
    fn transcript_name_puts_the_stamp_first() {
        let path = transcript_path_for(Path::new("/tmp/standup.m4a"), Some("20260728-142530"));
        assert_eq!(path.file_name().unwrap(), "20260728-142530-standup.txt");
    }

    #[test]
    fn checkpoint_sits_next_to_the_audio() {
        let path = checkpoint_path(Path::new("/tmp/standup.m4a"));
        assert_eq!(path.file_name().unwrap(), "standup.transcript.partial.json");
    }

    #[test]
    fn fingerprint_renders_the_default_chunk_size_without_a_decimal() {
        assert_eq!(format_chunk_sec(300.0), "300");
        assert_eq!(format_chunk_sec(12.5), "12.5");
    }

    #[test]
    fn fingerprint_changes_with_the_engine() {
        let dir = tempdir().unwrap();
        let audio = sample(dir.path());
        let a = source_fingerprint(&audio, "whispercpp-large-v3:vi", 300.0).unwrap();
        let b = source_fingerprint(&audio, "whispercpp-large-v3:en", 300.0).unwrap();
        let c = source_fingerprint(&audio, "whispercpp-large-v3:vi", 120.0).unwrap();
        assert_ne!(a, b, "language is part of the engine key");
        assert_ne!(a, c, "chunk size is part of the fingerprint");
    }

    #[test]
    fn a_stale_fingerprint_is_rejected() {
        let dir = tempdir().unwrap();
        let audio = sample(dir.path());
        let fingerprint = source_fingerprint(&audio, "engine-a", 300.0).unwrap();
        save_checkpoint(&audio, &checkpoint_for(&fingerprint, "out.txt"), Some(600.0)).unwrap();

        assert!(load_checkpoint(&audio, &fingerprint).is_some());
        assert!(
            load_checkpoint(&audio, "a-different-engine").is_none(),
            "switching model or language must restart, not resume"
        );
    }

    #[test]
    fn a_corrupt_checkpoint_is_ignored_rather_than_trusted() {
        let dir = tempdir().unwrap();
        let audio = sample(dir.path());
        fs::write(checkpoint_path(&audio), b"{ this is not json").unwrap();
        assert!(load_checkpoint(&audio, "engine-a").is_none());
    }

    #[test]
    fn resume_truncates_the_tail_of_an_interrupted_chunk() {
        let dir = tempdir().unwrap();
        let audio = sample(dir.path());
        let out = dir.path().join("out.txt");

        // 12 bytes were checkpointed; the rest is debris from a chunk that was
        // still being written when the process died.
        fs::write(&out, b"[00:00:00] o\n[00:05:00] half-writ").unwrap();
        let fingerprint = source_fingerprint(&audio, "engine-a", 300.0).unwrap();
        save_checkpoint(&audio, &checkpoint_for(&fingerprint, "out.txt"), Some(600.0)).unwrap();

        let state = resume_or_restart(&audio, &out, &fingerprint).unwrap();
        assert_eq!(state.chunks_done, 1);
        assert_eq!(state.next_start_sec, 300.0);
        assert_eq!(fs::read(&out).unwrap(), b"[00:00:00] o");
    }

    #[test]
    fn resume_restarts_when_the_transcript_is_shorter_than_claimed() {
        let dir = tempdir().unwrap();
        let audio = sample(dir.path());
        let out = dir.path().join("out.txt");

        // The user deleted or edited the transcript between runs.
        fs::write(&out, b"tiny").unwrap();
        let fingerprint = source_fingerprint(&audio, "engine-a", 300.0).unwrap();
        save_checkpoint(&audio, &checkpoint_for(&fingerprint, "out.txt"), Some(600.0)).unwrap();

        let state = resume_or_restart(&audio, &out, &fingerprint).unwrap();
        assert_eq!(state.chunks_done, 0, "must start over rather than resume into a gap");
        assert_eq!(state.next_start_sec, 0.0);
        assert_eq!(fs::read(&out).unwrap(), b"");
        assert!(!checkpoint_path(&audio).exists());
    }

    #[test]
    fn resolve_transcript_path_reuses_the_resumed_file() {
        let dir = tempdir().unwrap();
        let audio = sample(dir.path());
        let started = dir.path().join("20260101-090000-standup.txt");
        fs::write(&started, b"[00:00:00] earlier run\n").unwrap();

        let fingerprint = source_fingerprint(&audio, "engine-a", 300.0).unwrap();
        save_checkpoint(
            &audio,
            &checkpoint_for(&fingerprint, "20260101-090000-standup.txt"),
            Some(600.0),
        )
        .unwrap();

        let resolved = resolve_transcript_path(&audio, "engine-a", 300.0);
        assert_eq!(resolved, started, "a resumed run must append, not mint a new stamp");
    }

    #[test]
    fn append_text_returns_the_new_size() {
        let dir = tempdir().unwrap();
        let out = dir.path().join("out.txt");
        assert_eq!(append_text(&out, "[00:00:00] one\n").unwrap(), 15);
        assert_eq!(append_text(&out, "[00:05:00] two\n").unwrap(), 30);
        // An empty chunk must not disturb the size the checkpoint recorded.
        assert_eq!(append_text(&out, "").unwrap(), 30);
    }

    #[test]
    fn clear_checkpoint_removes_the_temp_file_too() {
        let dir = tempdir().unwrap();
        let audio = sample(dir.path());
        let path = checkpoint_path(&audio);
        fs::write(&path, b"{}").unwrap();
        fs::write(tmp_path(&path), b"{}").unwrap();
        clear_checkpoint(&audio);
        assert!(!path.exists());
        assert!(!tmp_path(&path).exists());
    }
}
