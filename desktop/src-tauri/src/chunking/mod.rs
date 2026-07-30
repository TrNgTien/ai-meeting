//! Chunked, streamed-to-disk, resumable transcription.
//!
//! Port of `chunking.transcribe_chunked`. This is the driver for every engine
//! and every entry point: a file is never transcribed in one unstoppable pass.
//!
//! * **Chunked** — ~5 minutes at a time, cut at the quietest nearby moment.
//! * **Streamed to disk** — each finished chunk is appended and fsynced, so the
//!   transcript is readable in another editor while the run is still going.
//! * **Checkpointed / resumable** — see [`checkpoint`]. A crash costs at most
//!   the chunk in flight.
//! * **Stoppable** — cancellation is checked between chunks.

pub mod checkpoint;
pub mod decode;
pub mod split;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;

use crate::state::CancelFlag;
use crate::transcribe::hallucination::{
    self, drop_segments_in_silence, silent_spans, HALLUCINATION_SILENCE_SEC,
};
use crate::transcribe::{segments_to_text, Engine, TranscriptSegment};
use crate::SAMPLE_RATE;

use checkpoint::{
    append_text, clear_checkpoint, resolve_transcript_path, resume_or_restart, save_checkpoint,
    source_fingerprint,
};
use decode::{audio_duration, decode_range};
use split::find_split_index;

/// Work in 5-minute chunks: long enough that per-chunk model overhead is noise,
/// short enough that a crash costs at most a few minutes of recomputation.
pub const DEFAULT_CHUNK_SECONDS: f64 = 300.0;

/// Extra audio decoded past the nominal chunk end, searched for a quiet spot to
/// cut on so chunk boundaries rarely land in the middle of a word.
const SPLIT_SEARCH_SECONDS: f64 = 20.0;

/// Ignore a trailing sliver of audio rather than feeding a near-empty chunk to
/// the model.
const MIN_TAIL_SECONDS: f64 = 0.2;

#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    /// Cancellation is not a failure: progress is kept in the checkpoint and the
    /// transcript on disk is a complete prefix.
    #[error("transcription cancelled")]
    Cancelled,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Everything `transcribe_chunked` reports as it goes.
///
/// Replaces the Python app's `on_progress` / `on_text` / `on_segment` callback
/// triple. Every method has a default so a caller that only wants text does not
/// have to stub the rest.
pub trait ChunkObserver: Send + Sync {
    fn on_status(&self, _message: &str) {}

    /// Fires once per chunk as it is appended to the file, plus once at startup
    /// with the resumed prefix so a continued file reads as one transcript
    /// rather than starting mid-meeting.
    fn on_text(&self, _text: &str) {}

    /// Fires *within* a chunk, per decode window. These are a preview: the
    /// audio they describe has not been written to disk yet, and a run stopped
    /// mid-chunk must discard them.
    fn on_segment(&self, _segment: &TranscriptSegment) {}

    /// `(chunks_done, transcribed_seconds, total_seconds)`.
    fn on_progress(&self, _chunks_done: usize, _done_sec: f64, _total_sec: Option<f64>) {}
}

/// A no-op observer, for headless runs and tests.
pub struct SilentObserver;
impl ChunkObserver for SilentObserver {}

pub struct ChunkOptions {
    pub chunk_sec: f64,
    /// Where the transcript goes. `None` resolves it from the checkpoint, which
    /// is what makes a resumed run append to the file it started.
    pub output_path: Option<PathBuf>,
    pub cancel: Option<CancelFlag>,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            chunk_sec: DEFAULT_CHUNK_SECONDS,
            output_path: None,
            cancel: None,
        }
    }
}

/// Transcribe `source` chunk by chunk, streaming the text to disk.
///
/// Returns the full transcript text.
pub fn transcribe_chunked(
    source: &Path,
    engine: &dyn Engine,
    observer: &dyn ChunkObserver,
    options: &ChunkOptions,
) -> Result<String, TranscribeError> {
    let engine_key = engine.engine_key();
    let chunk_sec = options.chunk_sec;
    let fingerprint = source_fingerprint(source, &engine_key, chunk_sec)?;
    let duration = audio_duration(source);

    let out_path = match &options.output_path {
        Some(path) => path.clone(),
        None => resolve_transcript_path(source, &engine_key, chunk_sec),
    };

    let mut state = resume_or_restart(source, &out_path, &fingerprint)?;
    let run_started = Instant::now();
    let resumed_from = state.next_start_sec;

    if state.chunks_done > 0 {
        // Hand the caller what was transcribed in earlier runs.
        if let Ok(resumed) = std::fs::read_to_string(&out_path) {
            if !resumed.is_empty() {
                observer.on_text(&resumed);
            }
        }
        observer.on_progress(state.chunks_done, state.next_start_sec, duration);
    }

    let stream_segments = engine.supports_streaming();

    loop {
        if let Some(total) = duration {
            if state.next_start_sec >= total - MIN_TAIL_SECONDS {
                break;
            }
        }
        if options
            .cancel
            .as_ref()
            .is_some_and(CancelFlag::is_cancelled)
        {
            return Err(TranscribeError::Cancelled);
        }

        let start = state.next_start_sec;
        let audio = decode_range(source, start, chunk_sec + SPLIT_SEARCH_SECONDS)?;
        if audio.len() <= (MIN_TAIL_SECONDS * SAMPLE_RATE as f64) as usize {
            break;
        }

        let nominal_end = (chunk_sec * SAMPLE_RATE as f64) as usize;
        let cut = if audio.len() > nominal_end {
            find_split_index(&audio, nominal_end).min(audio.len())
        } else {
            // Short read: this is the tail of the recording.
            audio.len()
        };
        let chunk = &audio[..cut];

        let segments = if stream_segments {
            let sink = |segment: TranscriptSegment| observer.on_segment(&segment);
            engine.transcribe(chunk, start, Some(&sink))?
        } else {
            engine.transcribe(chunk, start, None)?
        };

        // The `hallucination_silence_threshold` behaviour openai-whisper gave us
        // and whisper.cpp does not. Applied here, where the chunk audio is
        // already in hand, so every engine gets it uniformly — the same reason
        // DECODE_OPTIONS was shared between engines in the Python app.
        let spans = silent_spans(chunk, SAMPLE_RATE, HALLUCINATION_SILENCE_SEC);
        let segments = drop_segments_in_silence(segments, &spans, start);
        let text = segments_to_text(&segments);

        // Order matters: the text hits the disk first, and only a chunk the
        // checkpoint has accounted for is treated as done. A crash in between
        // leaves a tail past text_bytes, which the next run truncates and
        // redoes — never a silently missing stretch of the meeting.
        let written = append_text(&out_path, &text)?;
        state.chunks_done += 1;
        state.next_start_sec = start + cut as f64 / SAMPLE_RATE as f64;
        state.text_bytes = written;
        save_checkpoint(source, &state, duration)?;

        if !text.is_empty() {
            observer.on_text(&text);
        }
        observer.on_progress(state.chunks_done, state.next_start_sec, duration);

        if cut >= audio.len()
            && duration.is_none_or(|total| state.next_start_sec >= total - MIN_TAIL_SECONDS)
        {
            break;
        }
    }

    // Only a run that reached the end writes the footer, and only after the last
    // chunk is on disk — so its presence means "this transcript is whole", and a
    // cancelled run leaves a file the next run can still resume into.
    let summary = summary_line(
        duration.unwrap_or(state.next_start_sec),
        run_started.elapsed().as_secs_f64(),
        resumed_from,
    );
    let footer = format!("\n{summary}");
    append_text(&out_path, &footer)?;
    observer.on_text(&footer);

    clear_checkpoint(source);
    Ok(std::fs::read_to_string(&out_path).map_err(anyhow::Error::from)?)
}

/// `HH:MM:SS` — matches the segment timestamps in the transcript body.
fn format_clock(seconds: f64) -> String {
    crate::transcribe::format_timestamp(seconds)
}

/// Wall-clock durations, read at a glance: `45s`, `4m32s`, `3h12m`.
fn format_elapsed(seconds: f64) -> String {
    let total = if seconds.is_finite() && seconds > 0.0 {
        seconds as u64
    } else {
        0
    };
    if total < 60 {
        format!("{total}s")
    } else if total < 3600 {
        format!("{}m{:02}s", total / 60, total % 60)
    } else {
        format!("{}h{:02}m", total / 3600, (total % 3600) / 60)
    }
}

/// The footer closing a finished transcript.
///
/// Records how much audio it covers and how long the machine took, which is the
/// question anyone asks after leaving a long recording to run. Only the time
/// spent in *this* run is knowable — earlier runs' wall time is not carried in
/// the checkpoint — so a resumed file says so rather than reporting a total it
/// cannot vouch for.
pub fn summary_line(audio_sec: f64, run_sec: f64, resumed_from_sec: f64) -> String {
    let mut parts: Vec<String> = Vec::new();
    if audio_sec > 0.0 {
        parts.push(format!("{} of audio", format_clock(audio_sec)));
    }
    if resumed_from_sec > 0.0 {
        parts.push(format!(
            "transcribed in {} this run (resumed from {})",
            format_elapsed(run_sec),
            format_clock(resumed_from_sec)
        ));
    } else {
        parts.push(format!("transcribed in {}", format_elapsed(run_sec)));
    }
    format!("----- Complete: {} -----\n", parts.join(" · "))
}

/// Re-exported so callers don't have to reach into the submodule for the one
/// helper the UI needs before a run starts.
pub use checkpoint::resumable_seconds;

/// Whether a preview segment should be shown to the user.
///
/// The live preview goes straight to the UI without passing through the
/// post-chunk filtering, so it needs the canned-phrase check applied here too —
/// otherwise an invented outro flashes on screen before being dropped from the
/// text that lands on disk.
pub fn preview_is_worth_showing(text: &str) -> bool {
    !hallucination::is_hallucination(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_reads_at_a_glance() {
        assert_eq!(format_elapsed(45.0), "45s");
        assert_eq!(format_elapsed(272.0), "4m32s");
        assert_eq!(format_elapsed(11_520.0), "3h12m");
        assert_eq!(format_elapsed(-1.0), "0s");
    }

    #[test]
    fn summary_of_a_fresh_run_omits_the_resume_note() {
        let line = summary_line(3600.0, 272.0, 0.0);
        assert_eq!(
            line,
            "----- Complete: 01:00:00 of audio · transcribed in 4m32s -----\n"
        );
    }

    #[test]
    fn summary_of_a_resumed_run_says_so() {
        let line = summary_line(3600.0, 272.0, 1500.0);
        assert!(line.contains("resumed from 00:25:00"));
        assert!(
            line.contains("this run"),
            "must not claim a total it cannot vouch for"
        );
    }
}
