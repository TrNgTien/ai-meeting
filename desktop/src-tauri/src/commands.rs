//! Tauri commands that drive `sidecar.py`. Each one writes a single JSON
//! command line to the sidecar's stdin (via [`SidecarState::send`]) and
//! returns immediately — the actual result arrives later as a
//! `sidecar-event` the frontend correlates by `id` (jobs) or `name`
//! (downloads). These are fire-and-forget sends, not request/response calls.
//!
//! Every payload shape here is copied from `sidecar.py`'s `cmd_*` methods
//! directly, not from the design doc's illustrative examples, which predate
//! the real implementation (e.g. the design doc's `mm_download_progress`/
//! `done` events don't exist — the real ones are `mm_progress`/
//! `mm_download_finished`/`batch_done`).

use serde_json::json;
use tauri::State;

use crate::sidecar::SidecarState;

#[tauri::command]
pub fn list_models(state: State<SidecarState>) -> Result<(), String> {
    state.send(json!({"cmd": "list_models"}))
}

#[tauri::command]
pub fn download_model(state: State<SidecarState>, name: String) -> Result<(), String> {
    state.send(json!({"cmd": "download_model", "name": name}))
}

#[tauri::command]
pub fn delete_model(state: State<SidecarState>, name: String) -> Result<(), String> {
    state.send(json!({"cmd": "delete_model", "name": name}))
}

/// Cancels an in-flight model download. `sidecar.py`'s `cmd_cancel` also
/// handles cancelling a running transcription job (keyed by `id` instead of
/// `name`) — that's [`cancel_job`], a separate Tauri command, so the two
/// distinct cancellable things stay distinct at the call site rather than
/// relying on the frontend to pass the right key in an untyped payload.
#[tauri::command]
pub fn cancel_download(state: State<SidecarState>, name: String) -> Result<(), String> {
    state.send(json!({"cmd": "cancel", "name": name}))
}

#[tauri::command]
pub fn start_transcription(
    state: State<SidecarState>,
    id: String,
    paths: Vec<String>,
    lang_mode: String,
    model: String,
    mlx: bool,
) -> Result<(), String> {
    state.send(json!({
        "cmd": "start_transcription",
        "id": id,
        "paths": paths,
        "lang_mode": lang_mode,
        "model": model,
        "mlx": mlx,
    }))
}

#[tauri::command]
pub fn cancel_job(state: State<SidecarState>, id: String) -> Result<(), String> {
    state.send(json!({"cmd": "cancel", "id": id}))
}
