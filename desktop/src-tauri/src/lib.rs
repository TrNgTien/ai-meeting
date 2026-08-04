//! Transcriber — local-only, Vietnamese-first meeting transcription.
//!
//! A Rust/Tauri port of the Python/Tkinter app that lives at the repo root. The
//! engines changed (whisper.cpp + Metal in place of openai-whisper and MLX);
//! the behaviour deliberately did not. The four things that make this app worth
//! using are ported rather than reinvented:
//!
//! * `vi+en` routing — decode as Vietnamese on a *multilingual* checkpoint, so
//!   English terms mixed into Vietnamese speech stay spelled in English
//!   (see [`state::LanguageMode`]).
//! * Hallucination and silence control (see [`transcribe::hallucination`]).
//! * Chunked, crash-safe, resumable transcription (see [`chunking`]).
//! * Two-track recording aligned to wall-clock time, merged by timestamp.

pub mod audio;
pub mod chunking;
pub mod commands;
pub mod engine;
pub mod merge;
pub mod state;
pub mod transcribe;

/// Everything upstream of a model runs at 16 kHz mono: it is what Whisper wants,
/// and recording at that rate loses nothing transcription could have used while
/// keeping an hour-long meeting to ~110 MB across both tracks instead of ~700 MB.
pub const SAMPLE_RATE: u32 = 16_000;

use engine::EngineHost;
use state::AppState;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        // Transcription, model management and their worker threads all live in
        // this process now; there is no child process to spawn or supervise.
        .manage(EngineHost::default())
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::download_model,
            commands::delete_model,
            commands::cancel_download,
            commands::start_transcription,
            commands::cancel_job,
            commands::delete_transcript,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the Transcriber");
}
