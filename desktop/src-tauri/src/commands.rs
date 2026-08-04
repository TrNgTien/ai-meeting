//! The frontend's entry points into [`crate::engine`].
//!
//! Every command is fire-and-forget: it validates its arguments, hands the work
//! to the engine host, and returns. Results arrive later as `engine-event`s the
//! frontend correlates by `id` (jobs) or `name` (downloads) — the same contract
//! the Python sidecar had, kept so the panes did not have to change when the
//! sidecar went away.

use tauri::{AppHandle, State};

use crate::engine::EngineHost;

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
