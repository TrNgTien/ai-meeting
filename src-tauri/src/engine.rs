//! The job runner behind the Tauri commands — the native replacement for
//! `sidecar.py`.
//!
//! The Python sidecar was a stopgap: a PyInstaller bundle of the root app's
//! modules, spoken to over a JSON-line pipe. Everything it did is now done in
//! this process by [`crate::chunking`], [`crate::transcribe`] and
//! [`crate::transcribe::models`], so the built `.app` ships no Python at all.
//!
//! The **event protocol is unchanged**, deliberately: the frontend still gets
//! one `engine-event` per state change with the same `event` discriminator and
//! the same payload fields (`models`, `mm_progress`, `mm_download_finished`,
//! `model_deleted`, `status`, `file_start`, `chunk_baseline`, `chunk_progress`,
//! `segment_text`, `chunk_text`, `batch_done`, `error`). Only the transport
//! changed, so the panes did not have to.
//!
//! What did *not* survive the port: `vi` mode. It ran PhoWhisper-large from a
//! Hugging Face safetensors checkpoint, and whisper.cpp only loads GGML. `vi+en`
//! covers the same audio — it decodes as Vietnamese on a multilingual
//! checkpoint, which is also what keeps English terms spelled in English.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, Runtime};

use crate::audio::recorder::Recording;
use crate::chunking::{
    checkpoint::resolve_transcript_path, format_elapsed, resumable_seconds, transcribe_chunked,
    ChunkObserver, ChunkOptions, TranscribeError, DEFAULT_CHUNK_SECONDS,
};
use crate::merge::{conversation_path, merge_transcript_files, MIC_LABEL, SYSTEM_LABEL};
use crate::recording::recordings_dir;
use crate::state::{AppState, CancelFlag, LanguageMode, Phase};
use crate::transcribe::models::{
    self, ensure_model_downloaded, is_model_downloaded, DownloadError,
};
use crate::transcribe::whisper_cpp::WhisperCppEngine;
use crate::transcribe::{format_timestamp, Engine, TranscriptSegment, MODEL_OPTIONS};

/// The single Tauri event every state change reaches the frontend on. One
/// channel rather than one event name per message, so the panes can each filter
/// the whole stream for the handful of events they care about.
pub const ENGINE_EVENT: &str = "engine-event";

/// Extensions the import path accepts, copied from `sidecar.py`'s `AUDIO_EXTS`.
/// A directory dropped on the window yields whatever the OS lists; anything not
/// in here is silently skipped rather than handed to ffmpeg to fail on.
const AUDIO_EXTS: &[&str] = &[
    "mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "wma", "mp4",
];

/// Emit one `engine-event`. `payload` must be a JSON object; `event` is folded
/// into it as the discriminator the frontend switches on.
pub fn emit<R: Runtime>(app: &AppHandle<R>, event: &str, payload: Value) {
    let mut value = payload;
    if let Some(map) = value.as_object_mut() {
        map.insert("event".to_string(), json!(event));
    }
    let _ = app.emit(ENGINE_EVENT, value);
}

/// Everything the commands mutate, behind an `Arc` so a worker thread can hold
/// it after the command that spawned it has returned.
#[derive(Default)]
pub struct EngineHost {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    /// `(job id, its cancel flag)`. `Some` means a batch is in flight; a second
    /// `start_transcription` is refused while it is, because the app transcribes
    /// one file at a time by design (a second concurrent whisper.cpp context
    /// would double the memory for no throughput on a machine already at 100%).
    current_job: Mutex<Option<(String, CancelFlag)>>,
    /// One entry per in-flight model download, keyed by model name.
    downloads: Mutex<HashMap<String, CancelFlag>>,
    /// The last loaded engine, keyed by `model|language`. A 1.5 GB checkpoint
    /// takes seconds to load, so a batch of ten files must not pay for it ten
    /// times — this is what `Transcriber` kept in `self.model` on the Python
    /// side.
    loaded: Mutex<Option<(String, Arc<WhisperCppEngine>)>>,
}

impl EngineHost {
    // Every entry point is generic over the Tauri runtime so the same code the
    // app runs is what the tests drive on `tauri::test`'s mock runtime.
    /// The models the UI lists: every selectable checkpoint, each with whether
    /// it is on disk and how big it is. Cheap enough (a `stat` per model) to
    /// answer synchronously on the caller's thread.
    pub fn list_models<R: Runtime>(&self, app: &AppHandle<R>) {
        let models: Vec<Value> = MODEL_OPTIONS
            .iter()
            .map(|name| {
                json!({
                    "name": name,
                    "downloaded": is_model_downloaded(name),
                    "size_bytes": models::model_size_on_disk(name),
                })
            })
            .collect();
        emit(app, "models", json!({ "models": models }));
    }

    /// Fetch a checkpoint on a worker thread, reporting `mm_progress` as it
    /// goes and exactly one `mm_download_finished` however it ends.
    pub fn download_model<R: Runtime>(&self, app: &AppHandle<R>, name: String) {
        let cancel = {
            let mut downloads = self.inner.downloads.lock();
            if downloads.contains_key(&name) {
                // Already running: the second click is a no-op rather than a
                // second connection racing the first into the same file.
                return;
            }
            let cancel = CancelFlag::new();
            downloads.insert(name.clone(), cancel.clone());
            cancel
        };

        let app = app.clone();
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            let progress = |model: &str, done: u64, total: u64| {
                emit(
                    &app,
                    "mm_progress",
                    json!({"model": model, "downloaded": done, "total": total}),
                );
            };
            let result = ensure_model_downloaded(&name, Some(&progress), Some(&cancel));
            let finished = match result {
                Ok(_) => json!({"name": name, "status": "done"}),
                Err(DownloadError::Cancelled) => json!({"name": name, "status": "cancelled"}),
                Err(DownloadError::Other(err)) => {
                    json!({"name": name, "status": "error", "error": err.to_string()})
                }
            };
            // The entry is removed by the thread that owns it, before the
            // finished event fires — so a frontend that reacts by starting a new
            // download of the same model is never refused by a stale entry.
            inner.downloads.lock().remove(&name);
            emit(&app, "mm_download_finished", finished);
        });
    }

    pub fn delete_model<R: Runtime>(&self, app: &AppHandle<R>, name: String) {
        // Dropping the cached engine matters: it holds the loaded weights of
        // whatever ran last, and keeping a deleted model resident would make
        // "Delete" look like it freed nothing until the app restarts. One lock
        // held across the check and the clear — `parking_lot::Mutex` is not
        // reentrant, so taking it twice here would deadlock the UI thread.
        {
            let prefix = format!("{name}|");
            let mut loaded = self.inner.loaded.lock();
            if loaded
                .as_ref()
                .is_some_and(|(key, _)| key.starts_with(&prefix))
            {
                *loaded = None;
            }
        }
        models::delete_model(&name);
        emit(app, "model_deleted", json!({ "name": name }));
    }

    pub fn cancel_download<R: Runtime>(&self, app: &AppHandle<R>, name: String) {
        match self.inner.downloads.lock().get(&name) {
            Some(cancel) => cancel.cancel(),
            None => emit(
                app,
                "error",
                json!({"message": format!("no download in progress for '{name}'")}),
            ),
        }
    }

    /// Start a batch. Returns immediately; everything after this arrives as
    /// events, ending with exactly one `batch_done` carrying `id`.
    pub fn start_transcription<R: Runtime>(
        &self,
        app: &AppHandle<R>,
        id: String,
        paths: Vec<String>,
        lang_mode: &str,
        model: String,
    ) {
        let Some(cancel) = self.claim_job(app, &id) else {
            return;
        };

        let language = LanguageMode::parse(lang_mode);
        let sources: Vec<PathBuf> = paths
            .into_iter()
            .map(PathBuf::from)
            .filter(|path| path.is_file() && is_audio(path))
            .collect();

        let app = app.clone();
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            run_batch(&app, &inner, &id, &sources, language, &model, &cancel);
        });
    }

    /// Transcribe both sides of a finished recording, then weave them into one
    /// conversation. Port of `app._transcribe_recording`.
    ///
    /// Runs as an ordinary job — same slot, same Stop button, same `batch_done`
    /// — so a recording that is still transcribing blocks an import for the same
    /// reason two imports block each other.
    pub fn transcribe_recording<R: Runtime>(
        &self,
        app: &AppHandle<R>,
        recording: Recording,
        lang_mode: &str,
        model: String,
    ) {
        let id = format!("recording-{}", recording.stem);
        let Some(cancel) = self.claim_job(app, &id) else {
            return;
        };

        let language = LanguageMode::parse(lang_mode);
        let app = app.clone();
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            run_recording(&app, &inner, &id, &recording, language, &model, &cancel);
        });
    }

    /// Take the single job slot, or tell the frontend why it could not.
    ///
    /// One job at a time is deliberate (see [`Inner::current_job`]); this is
    /// where both entry points agree on it.
    fn claim_job<R: Runtime>(&self, app: &AppHandle<R>, id: &str) -> Option<CancelFlag> {
        let mut current = self.inner.current_job.lock();
        if let Some((running, _)) = current.as_ref() {
            emit(
                app,
                "error",
                json!({"message": format!("job '{running}' is already running")}),
            );
            return None;
        }
        let cancel = CancelFlag::new();
        *current = Some((id.to_string(), cancel.clone()));
        set_phase(app, Phase::Transcribing);
        Some(cancel)
    }

    pub fn cancel_job<R: Runtime>(&self, app: &AppHandle<R>, id: String) {
        match self.inner.current_job.lock().as_ref() {
            Some((running, cancel)) if *running == id => cancel.cancel(),
            _ => emit(
                app,
                "error",
                json!({"message": format!("no running job '{id}'")}),
            ),
        }
    }
}

/// Release the job slot and return the app to `Idle`.
///
/// Must run on every path out of a batch, including an early break: the slot is
/// what refuses a second job, so leaking it makes every later job impossible
/// with no recovery short of a restart.
fn finish_job<R: Runtime>(app: &AppHandle<R>, inner: &Inner) {
    *inner.current_job.lock() = None;
    set_phase(app, Phase::Idle);
}

/// Record what the app is doing, when there is somewhere to record it.
///
/// `try_state` rather than `state`: the tests drive this code on a mock app that
/// manages only what the test needs, and a missing phase is not worth panicking
/// a worker thread over.
fn set_phase<R: Runtime>(app: &AppHandle<R>, phase: Phase) {
    if let Some(state) = app.try_state::<AppState>() {
        state.set_phase(phase);
    }
}

fn is_audio(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| AUDIO_EXTS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// The batch loop, a port of `sidecar.py`'s `cmd_start_transcription` worker.
///
/// One file failing does not stop the batch — a corrupt file in a folder of
/// twenty should cost that one transcript, not the other nineteen. Cancellation
/// does stop it, since the user asked.
fn run_batch<R: Runtime>(
    app: &AppHandle<R>,
    inner: &Inner,
    job_id: &str,
    sources: &[PathBuf],
    language: LanguageMode,
    model: &str,
    cancel: &CancelFlag,
) {
    let started = Instant::now();
    let total = sources.len();
    let mut saved: Vec<String> = Vec::new();
    let mut cancelled = false;

    for (index, source) in sources.iter().enumerate() {
        let name = source
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| source.display().to_string());
        let counter = if total > 1 {
            format!(" {}/{total}", index + 1)
        } else {
            String::new()
        };
        let label = format!("Transcribing{counter}: {name}");

        emit(app, "status", json!({"message": format!("{label}…")}));
        emit(app, "file_start", json!({ "name": name }));

        match transcribe_one(app, inner, source, &label, language, model, cancel) {
            Ok(out_path) => saved.push(out_path.display().to_string()),
            Err(TranscribeError::Cancelled) => {
                cancelled = true;
                break;
            }
            Err(TranscribeError::Other(err)) => {
                emit(
                    app,
                    "error",
                    json!({"file": name, "message": err.to_string()}),
                );
            }
        }
    }

    // `batch_done` has to fire and the job slot has to clear on every path,
    // including a panic-free early break — otherwise the frontend's running
    // banner never goes away and every later job is refused as "already
    // running", with no recovery short of restarting the app.
    finish_job(app, inner);
    emit(
        app,
        "batch_done",
        json!({
            "id": job_id,
            "count": saved.len(),
            "saved": saved,
            "cancelled": cancelled,
            "elapsed_sec": started.elapsed().as_secs_f64(),
        }),
    );
}

/// Both sides of a recording, then the conversation they make together.
///
/// Port of `app._transcribe_recording`. The two tracks are transcribed
/// separately — that is what makes the merge able to say who spoke, and what
/// stops your own voice, bleeding from the speakers into the microphone, from
/// being transcribed twice.
fn run_recording<R: Runtime>(
    app: &AppHandle<R>,
    inner: &Inner,
    job_id: &str,
    recording: &Recording,
    language: LanguageMode,
    model: &str,
    cancel: &CancelFlag,
) {
    let started = Instant::now();
    let tracks: Vec<(&Path, &str)> = [
        (recording.mic_path.as_deref(), MIC_LABEL),
        (recording.system_path.as_deref(), SYSTEM_LABEL),
    ]
    .into_iter()
    .filter_map(|(path, speaker)| path.map(|path| (path, speaker)))
    .collect();

    let mut saved: Vec<String> = Vec::new();
    let mut cancelled = false;
    // Where each side's transcript landed, so the merge can find them. Held per
    // speaker rather than as a flat list because the merge needs to know which
    // is which.
    let mut mic_transcript: Option<PathBuf> = None;
    let mut system_transcript: Option<PathBuf> = None;

    for (path, speaker) in &tracks {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let label = format!("Transcribing the {speaker} track");

        emit(app, "status", json!({"message": format!("{label}…")}));
        emit(app, "file_start", json!({"name": format!("{speaker} — {name}")}));

        match transcribe_one(app, inner, path, &label, language, model, cancel) {
            Ok(out_path) => {
                saved.push(out_path.display().to_string());
                if *speaker == MIC_LABEL {
                    mic_transcript = Some(out_path);
                } else {
                    system_transcript = Some(out_path);
                }
            }
            Err(TranscribeError::Cancelled) => {
                cancelled = true;
                break;
            }
            Err(TranscribeError::Other(err)) => {
                emit(
                    app,
                    "error",
                    json!({"file": name, "message": err.to_string()}),
                );
            }
        }
    }

    // A cancelled run is not merged. Half a conversation reads worse than two
    // separate transcripts: the missing side looks like silence rather than like
    // something that was never transcribed, and there is no way to tell from the
    // file which it was.
    if !cancelled {
        match merge_recording(recording, mic_transcript.as_deref(), system_transcript.as_deref()) {
            Ok(Some(path)) => {
                saved.push(path.display().to_string());
                // The text rides along with the path: the conversation is the
                // reason the recording was made, so the pane shows it in place
                // of the two per-track transcripts it was building up, the way
                // `app._handle_ui_event`'s `merged_text` branch did.
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                emit(
                    app,
                    "merged_text",
                    json!({"path": path.display().to_string(), "text": text}),
                );
            }
            // Neither side produced a line — a recording of a silent room. The
            // per-track transcripts still exist; an empty conversation file
            // would only be one more thing to delete.
            Ok(None) => {}
            Err(err) => emit(app, "error", json!({"message": err.to_string()})),
        }
    }

    finish_job(app, inner);
    emit(
        app,
        "batch_done",
        json!({
            "id": job_id,
            "count": saved.len(),
            "saved": saved,
            "cancelled": cancelled,
            "elapsed_sec": started.elapsed().as_secs_f64(),
        }),
    );
}

/// Weave the two transcripts into `<stem>-conversation.txt`.
///
/// `Ok(None)` means there was nothing to weave — both sides parsed to zero
/// utterances — and no file was written.
fn merge_recording(
    recording: &Recording,
    mic_transcript: Option<&Path>,
    system_transcript: Option<&Path>,
) -> anyhow::Result<Option<PathBuf>> {
    if mic_transcript.is_none() && system_transcript.is_none() {
        return Ok(None);
    }

    // Next to the tracks, so a meeting is one folder however it was started.
    let dir = recording
        .mic_path
        .as_deref()
        .or(recording.system_path.as_deref())
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(recordings_dir);
    let out_path = conversation_path(&dir, &recording.stem);
    let header = recording_header(recording);

    let count = merge_transcript_files(
        mic_transcript,
        system_transcript,
        &out_path,
        Some(&header),
    )?;
    if count == 0 {
        let _ = std::fs::remove_file(&out_path);
        return Ok(None);
    }
    Ok(Some(out_path))
}

/// `# Meeting recorded 2026-08-04 14:05 (32m10s)` — the line the merged
/// transcript opens with, so a file found later dates itself.
fn recording_header(recording: &Recording) -> String {
    let when = chrono::DateTime::from_timestamp(
        recording.started_at as i64,
        (recording.started_at.fract().max(0.0) * 1e9) as u32,
    )
    .map(|utc| utc.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string())
    .unwrap_or_else(|| "an unknown time".to_string());
    format!(
        "# Meeting recorded {when} ({})",
        format_elapsed(recording.duration_sec)
    )
}

/// One file, start to finish. Returns where the transcript was written.
fn transcribe_one<R: Runtime>(
    app: &AppHandle<R>,
    inner: &Inner,
    source: &Path,
    label: &str,
    language: LanguageMode,
    model: &str,
    cancel: &CancelFlag,
) -> Result<PathBuf, TranscribeError> {
    let engine = resolve_engine(app, inner, model, language, cancel)?;
    let engine_key = engine.engine_key();
    let out_path = resolve_transcript_path(source, &engine_key, DEFAULT_CHUNK_SECONDS);

    let resume_at = resumable_seconds(source, &engine_key, None);
    if resume_at > 0.0 {
        emit(app, "chunk_baseline", json!({ "resume_at_sec": resume_at }));
        emit(
            app,
            "status",
            json!({"message": format!("{label} — resuming at {}…", format_timestamp(resume_at))}),
        );
    }

    let observer = UiObserver {
        app: app.clone(),
        label: label.to_string(),
    };
    let options = ChunkOptions {
        chunk_sec: DEFAULT_CHUNK_SECONDS,
        output_path: Some(out_path.clone()),
        cancel: Some(cancel.clone()),
    };

    transcribe_chunked(source, engine.as_ref(), &observer, &options)?;
    Ok(out_path)
}

/// Get a loaded engine for `model`/`language`, downloading the checkpoint first
/// if it is not on disk.
///
/// Port of `sidecar.py`'s `_resolve_engine` minus the two fallback paths it
/// needed (PhoWhisper and MLX, each of which could fail into the CPU engine).
/// whisper.cpp has one code path — Metal is a compile-time feature that degrades
/// to CPU inside the library, not a separate engine that can fail to load — so
/// there is nothing left to fall back *to*, and a failure here is a real error.
fn resolve_engine<R: Runtime>(
    app: &AppHandle<R>,
    inner: &Inner,
    model: &str,
    language: LanguageMode,
    cancel: &CancelFlag,
) -> Result<Arc<WhisperCppEngine>, TranscribeError> {
    let key = format!("{model}|{}", language.whisper_language().unwrap_or("auto"));
    if let Some((cached_key, engine)) = inner.loaded.lock().as_ref() {
        if *cached_key == key {
            return Ok(engine.clone());
        }
    }

    // A job that has to fetch its checkpoint first reports it on the same two
    // events the Models tab's own Download button uses, so the progress bar
    // there tracks it — including the *terminating* `mm_download_finished`,
    // without which that row would sit at its last byte count forever.
    let downloading = !is_model_downloaded(model);
    if downloading {
        emit(
            app,
            "status",
            json!({"message": format!("Downloading '{model}' (one-time, first use)…")}),
        );
    }
    let progress = |name: &str, done: u64, total: u64| {
        emit(
            app,
            "mm_progress",
            json!({"model": name, "downloaded": done, "total": total}),
        );
    };
    let result = ensure_model_downloaded(model, Some(&progress), Some(cancel));
    if downloading {
        let status = match &result {
            Ok(_) => json!({"name": model, "status": "done"}),
            Err(DownloadError::Cancelled) => json!({"name": model, "status": "cancelled"}),
            Err(DownloadError::Other(err)) => {
                json!({"name": model, "status": "error", "error": err.to_string()})
            }
        };
        emit(app, "mm_download_finished", status);
    }
    let model_path = match result {
        Ok(path) => path,
        Err(DownloadError::Cancelled) => return Err(TranscribeError::Cancelled),
        Err(DownloadError::Other(err)) => return Err(TranscribeError::Other(err)),
    };

    if cancel.is_cancelled() {
        return Err(TranscribeError::Cancelled);
    }

    emit(
        app,
        "status",
        json!({"message": format!("Loading model '{model}'…")}),
    );
    let engine = Arc::new(WhisperCppEngine::load(
        &model_path,
        model,
        language.whisper_language(),
    )?);
    *inner.loaded.lock() = Some((key, engine.clone()));
    Ok(engine)
}

/// Forwards `transcribe_chunked`'s callbacks to the frontend, one event each.
struct UiObserver<R: Runtime> {
    app: AppHandle<R>,
    /// Carried on every progress event so a batch's status line can name the
    /// file being worked on without the frontend tracking it separately.
    label: String,
}

impl<R: Runtime> ChunkObserver for UiObserver<R> {
    fn on_status(&self, message: &str) {
        emit(&self.app, "status", json!({ "message": message }));
    }

    fn on_text(&self, text: &str) {
        emit(&self.app, "chunk_text", json!({ "text": text }));
    }

    fn on_segment(&self, segment: &TranscriptSegment) {
        // The preview bypasses the post-chunk filtering that cleans the saved
        // text, so the canned-phrase check is applied here too — otherwise an
        // invented outro flashes on screen and then vanishes from the file.
        if crate::chunking::preview_is_worth_showing(&segment.text) {
            emit(
                &self.app,
                "segment_text",
                json!({ "text": segment.format_line() }),
            );
        }
    }

    fn on_progress(&self, chunks_done: usize, done_sec: f64, total_sec: Option<f64>) {
        emit(
            &self.app,
            "chunk_progress",
            json!({
                "label": self.label,
                "done_sec": done_sec,
                "total_sec": total_sec,
                "chunks_done": chunks_done,
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{channel, Receiver};
    use std::time::Duration;

    use tauri::test::{mock_builder, mock_context, noop_assets};
    use tauri::Listener;

    use super::*;

    #[test]
    fn only_known_audio_extensions_are_imported() {
        assert!(is_audio(Path::new("/tmp/meeting.m4a")));
        assert!(is_audio(Path::new("/tmp/MEETING.MP3")), "case-insensitive");
        assert!(!is_audio(Path::new("/tmp/meeting.txt")));
        assert!(!is_audio(Path::new("/tmp/meeting")));
    }

    /// A windowless app plus a receiver of every `engine-event` it emits —
    /// enough to drive the whole command layer the way the frontend does.
    fn harness() -> (AppHandle<tauri::test::MockRuntime>, Receiver<Value>) {
        let app = mock_builder()
            .build(mock_context(noop_assets()))
            .expect("mock app");
        let handle = app.handle().clone();
        let (tx, rx) = channel();
        handle.listen(ENGINE_EVENT, move |event| {
            let _ = tx.send(serde_json::from_str(event.payload()).expect("event is JSON"));
        });
        // The `App` owns the event loop; leaking it keeps the handle valid for
        // the rest of the test without running one.
        std::mem::forget(app);
        (handle, rx)
    }

    fn next_event(rx: &Receiver<Value>, name: &str) -> Value {
        let deadline = Duration::from_secs(10);
        loop {
            let event = rx.recv_timeout(deadline).unwrap_or_else(|_| {
                panic!("no '{name}' event arrived");
            });
            if event["event"] == name {
                return event;
            }
        }
    }

    #[test]
    fn list_models_reports_every_selectable_checkpoint() {
        let (app, rx) = harness();
        EngineHost::default().list_models(&app);

        let event = next_event(&rx, "models");
        let models = event["models"].as_array().expect("models is an array");
        assert_eq!(models.len(), MODEL_OPTIONS.len());
        // The shape the Models tab destructures. `downloaded`/`size_bytes`
        // depend on this machine's cache, so only their presence is asserted.
        assert_eq!(models[0]["name"], MODEL_OPTIONS[0]);
        assert!(models[0]["downloaded"].is_boolean());
        assert!(models[0]["size_bytes"].is_u64());
    }

    #[test]
    fn a_batch_with_nothing_transcribable_still_finishes() {
        // The invariant the frontend's running banner depends on: every job
        // ends in exactly one `batch_done` carrying its own id, even when the
        // import contained no audio at all. Without it the UI is stuck running
        // forever and refuses every later job.
        let (app, rx) = harness();
        let host = EngineHost::default();
        host.start_transcription(
            &app,
            "job-1".into(),
            vec!["/nonexistent/notes.txt".into()],
            "vi+en",
            "large-v3".into(),
        );

        let event = next_event(&rx, "batch_done");
        assert_eq!(event["id"], "job-1");
        assert_eq!(event["count"], 0);
        assert_eq!(event["cancelled"], false);
        assert_eq!(
            event["saved"].as_array().map(Vec::len),
            Some(0),
            "a skipped file must not be reported as a written transcript"
        );
    }

    fn recording_of(dir: &Path, stem: &str) -> Recording {
        Recording {
            stem: stem.to_string(),
            mic_path: Some(dir.join(format!("{stem}-me.wav"))),
            system_path: Some(dir.join(format!("{stem}-meeting.wav"))),
            duration_sec: 90.0,
            // 2026-08-04 00:00:00 UTC — asserted on only through the year, so
            // the test does not depend on the machine's timezone.
            started_at: 1_785_801_600.0,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn a_finished_recording_becomes_one_conversation() {
        let dir = tempfile::tempdir().unwrap();
        let mic = dir.path().join("me.txt");
        let system = dir.path().join("meeting.txt");
        std::fs::write(&mic, "[00:00:01] xin chào\n[00:00:09] vâng\n").unwrap();
        std::fs::write(&system, "[00:00:05] hello there\n").unwrap();

        let recording = recording_of(dir.path(), "meeting-20260804-100000");
        let path = merge_recording(&recording, Some(&mic), Some(&system))
            .expect("merge succeeds")
            .expect("a conversation was written");

        assert_eq!(
            path.file_name().unwrap(),
            "meeting-20260804-100000-conversation.txt",
            "the conversation is named after the recording, next to its tracks"
        );
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# Meeting recorded 202"), "{text}");
        assert!(text.contains("(1m30s)"), "the header carries the duration: {text}");
        // Interleaved by timestamp, which is the whole point of recording the
        // two sides against one clock.
        let spoken: Vec<&str> = text
            .lines()
            .filter(|line| line.starts_with('['))
            .collect();
        assert_eq!(
            spoken,
            vec![
                "[00:00:01] Me: xin chào",
                "[00:00:05] Meeting: hello there",
                "[00:00:09] Me: vâng",
            ]
        );
    }

    #[test]
    fn a_recording_nobody_spoke_in_writes_no_conversation() {
        // Both tracks transcribed to nothing. An empty conversation file would
        // only be one more thing to find and delete.
        let dir = tempfile::tempdir().unwrap();
        let mic = dir.path().join("me.txt");
        std::fs::write(&mic, "").unwrap();

        let recording = recording_of(dir.path(), "meeting-20260804-100000");
        let merged = merge_recording(&recording, Some(&mic), None).expect("merge succeeds");

        assert!(merged.is_none());
        assert!(
            !conversation_path(dir.path(), &recording.stem).exists(),
            "no empty file is left behind"
        );
    }

    #[test]
    fn cancelling_something_that_is_not_running_is_an_error_not_a_silence() {
        let (app, rx) = harness();
        let host = EngineHost::default();

        host.cancel_job(&app, "no-such-job".into());
        let event = next_event(&rx, "error");
        assert!(event["message"].as_str().unwrap().contains("no-such-job"));

        host.cancel_download(&app, "large-v3".into());
        let event = next_event(&rx, "error");
        assert!(event["message"].as_str().unwrap().contains("large-v3"));
    }
}
