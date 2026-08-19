//! The frontend's entry points into [`crate::engine`].
//!
//! Every command is fire-and-forget: it validates its arguments, hands the work
//! to the engine host, and returns. Results arrive later as `engine-event`s the
//! frontend correlates by `id` (jobs) or `name` (downloads) — the same contract
//! the Python sidecar had, kept so the panes did not have to change when the
//! sidecar went away.

use tauri::{AppHandle, Manager, State};

use crate::audio::devices::{self, InputDevice};
use crate::audio::recorder::RecorderOptions;
use crate::engine::{emit, EngineHost};
use crate::recording::{parse_backend, recordings_dir, LiveLevels, RecordingHost};
use crate::settings::{self, Settings};
use crate::state::{AppState, Phase};

/// Deletes a saved transcript file from disk. Scoped to `.txt` so the Files
/// tab's inline delete button can't be pointed at arbitrary paths.
#[tauri::command]
pub fn delete_transcript(path: String) -> Result<(), String> {
    let path = std::path::Path::new(&path);
    if path.extension().and_then(|e| e.to_str()) != Some("txt") {
        return Err("refusing to delete a non-transcript file".into());
    }
    std::fs::remove_file(path).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_models(app: AppHandle, engine: State<EngineHost>) {
    engine.list_models(&app);
}

#[tauri::command]
pub fn download_model(app: AppHandle, engine: State<EngineHost>, name: String) {
    engine.download_model(&app, name);
}

#[tauri::command]
pub fn delete_model(app: AppHandle, engine: State<EngineHost>, name: String) {
    engine.delete_model(&app, name);
}

/// How big a checkpoint is before committing to fetching it.
///
/// Asked one model at a time, only when something is about to be downloaded: a
/// HEAD per model on every launch would be five needless requests, and the
/// number only matters at the moment someone is deciding whether to wait for it.
/// `None` when the server does not say — the UI then shows an indeterminate
/// download rather than inventing a size.
#[tauri::command]
pub async fn remote_model_size(name: String) -> Option<u64> {
    // On a blocking pool, not the command thread: this is a network round trip
    // to Hugging Face and it is allowed to be slow.
    tauri::async_runtime::spawn_blocking(move || crate::transcribe::models::remote_size(&name))
        .await
        .ok()
        .flatten()
}

/// Cancels an in-flight model download. Kept separate from [`cancel_job`] even
/// though both trip a `CancelFlag`, so the two distinct cancellable things stay
/// distinct at the call site rather than relying on the frontend to pass the
/// right key in an untyped payload.
#[tauri::command]
pub fn cancel_download(app: AppHandle, engine: State<EngineHost>, name: String) {
    engine.cancel_download(&app, name);
}

#[tauri::command]
pub fn start_transcription(
    app: AppHandle,
    engine: State<EngineHost>,
    id: String,
    paths: Vec<String>,
    lang_mode: String,
    model: String,
) {
    engine.start_transcription(&app, id, paths, &lang_mode, model);
}

#[tauri::command]
pub fn cancel_job(app: AppHandle, engine: State<EngineHost>, id: String) {
    engine.cancel_job(&app, id);
}

/// The microphones the device picker offers. Answered synchronously — it is a
/// device enumeration, not I/O on a device.
#[tauri::command]
pub fn list_input_devices() -> Vec<InputDevice> {
    devices::list_input_devices()
}

/// Begin recording a meeting.
///
/// Fire-and-forget like the rest: opening the meeting side can sit behind a
/// Screen Recording permission dialog for as long as the user takes to answer
/// it, so the result arrives as `rec_started` or `rec_failed`.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub fn start_recording(
    app: AppHandle,
    recorder: State<RecordingHost>,
    record_mic: bool,
    record_system: bool,
    mic_device_id: Option<String>,
    backend: Option<String>,
) {
    let state = app.state::<AppState>();
    if state.phase() != Phase::Idle {
        emit(
            &app,
            "rec_failed",
            serde_json::json!({"message": "the app is busy; stop what is running first"}),
        );
        return;
    }
    if !record_mic && !record_system {
        emit(
            &app,
            "rec_failed",
            serde_json::json!({"message": "nothing to record — enable at least one side"}),
        );
        return;
    }

    // Claimed before the backends open, not after: the permission dialog can
    // take seconds, and a second Record click in that window must be refused
    // rather than start a second recorder writing over the first one's files.
    state.set_phase(Phase::Recording);

    let options = RecorderOptions {
        record_mic,
        record_system,
        mic_device_id,
        system_backend: backend.as_deref().map(parse_backend).unwrap_or_default(),
        stem: None,
    };

    let done_app = app.clone();
    recorder.start(
        recordings_dir(),
        options,
        Box::new(move |result| {
            let state = done_app.state::<AppState>();
            match result {
                Ok(started) => {
                    let payload = serde_json::to_value(&started)
                        .unwrap_or_else(|_| serde_json::json!({"stem": started.stem}));
                    emit(&done_app, "rec_started", payload);
                }
                Err(message) => {
                    state.set_phase(Phase::Idle);
                    emit(&done_app, "rec_failed", serde_json::json!({ "message": message }));
                }
            }
        }),
    );
}

/// Stop recording, then transcribe both tracks and merge them.
///
/// The language and model come from the same header controls an import uses, so
/// a recording is transcribed with whatever is selected when it ends.
#[tauri::command]
pub fn stop_recording(
    app: AppHandle,
    recorder: State<RecordingHost>,
    lang_mode: String,
    model: String,
) {
    let done_app = app.clone();
    recorder.stop(Box::new(move |recording| {
        let state = done_app.state::<AppState>();
        let Some(recording) = recording else {
            state.set_phase(Phase::Idle);
            emit(
                &done_app,
                "error",
                serde_json::json!({"message": "no recording was running"}),
            );
            return;
        };

        let payload = serde_json::to_value(&recording).unwrap_or_default();
        emit(&done_app, "rec_stopped", payload);

        // Both sides silent, or neither could be opened: there is nothing to
        // transcribe, and the warnings on `rec_stopped` already said why.
        if recording.paths().is_empty() {
            state.set_phase(Phase::Idle);
            emit(
                &done_app,
                "status",
                serde_json::json!({"message": "Nothing was recorded."}),
            );
            return;
        }

        done_app
            .state::<EngineHost>()
            .transcribe_recording(&done_app, recording, &lang_mode, model);
    }));
}

/// Meter levels and elapsed time, polled while recording.
#[tauri::command]
pub fn recording_levels(recorder: State<RecordingHost>) -> LiveLevels {
    recorder.levels()
}

/// Whether the bundled decoders can be run.
///
/// Asked once at launch. Every format the app accepts goes through ffmpeg, so
/// without it nothing works at all — and finding that out from a failed import
/// three minutes into a meeting recording is the worst possible time.
#[tauri::command]
pub fn ffmpeg_ready() -> bool {
    crate::chunking::decode::ffmpeg_available()
}

/// The choices to restore on launch. Never fails: unreadable settings are the
/// default settings.
#[tauri::command]
pub fn load_settings() -> Settings {
    settings::load()
}

/// Persist the choices. The error is returned rather than swallowed so a
/// read-only config directory is visible instead of silently forgetting every
/// choice the user makes.
#[tauri::command]
pub fn save_settings(settings: Settings) -> Result<(), String> {
    settings::save(&settings).map_err(|error| error.to_string())
}
