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
use tauri::{AppHandle, Emitter, Runtime};

use crate::chunking::{
    checkpoint::resolve_transcript_path, resumable_seconds, transcribe_chunked, ChunkObserver,
    ChunkOptions, TranscribeError, DEFAULT_CHUNK_SECONDS,
};
use crate::state::{CancelFlag, LanguageMode};
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
        let cancel = {
            let mut current = self.inner.current_job.lock();
            if let Some((running, _)) = current.as_ref() {
                emit(
                    app,
                    "error",
                    json!({"message": format!("job '{running}' is already running")}),
                );
                return;
            }
            let cancel = CancelFlag::new();
            *current = Some((id.clone(), cancel.clone()));
            cancel
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
    *inner.current_job.lock() = None;
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
