//! What the app is doing, and the handles a command needs to change it.
//!
//! Port of `app.AppState` plus the parts of `TranscriberApp` that were
//! really shared mutable state rather than UI.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Mirrors `app.AppState`. The UI is driven by which of these we are in:
/// importing files and switching models are blocked outside `Idle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Idle,
    Recording,
    Transcribing,
}

/// The language modes the UI offers, and the engine routing each implies.
///
/// `ViEn` is the default and the reason this app exists: decode as Vietnamese
/// on a *multilingual* checkpoint, so English terms mixed into Vietnamese
/// speech ("deploy cái service này") stay spelled in English instead of being
/// Vietnamised.
///
/// The Python app had a fourth mode, pure `vi`, which swapped in PhoWhisper —
/// more accurate on unmixed Vietnamese, but it phoneticises English and it is a
/// Hugging Face safetensors checkpoint, which whisper.cpp cannot load. It is not
/// offered here; `ViEn` covers the same audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageMode {
    #[serde(rename = "vi+en")]
    ViEn,
    En,
    Auto,
}

impl LanguageMode {
    /// Parse the string the frontend sends. Anything unrecognised — including
    /// the retired `vi` — falls back to the default rather than failing a job:
    /// `vi+en` decodes as Vietnamese too, so an old setting still transcribes
    /// Vietnamese audio correctly.
    pub fn parse(value: &str) -> Self {
        match value {
            "en" => LanguageMode::En,
            "auto" => LanguageMode::Auto,
            _ => LanguageMode::ViEn,
        }
    }

    /// The language code handed to the decoder. `None` means "detect", which
    /// matches how `Transcriber.set_language` stored `auto`.
    pub fn whisper_language(self) -> Option<&'static str> {
        match self {
            LanguageMode::ViEn => Some("vi"),
            LanguageMode::En => Some("en"),
            LanguageMode::Auto => None,
        }
    }
}

/// Cooperative cancellation, checked between chunks.
///
/// Replaces `threading.Event`. Progress is never lost when this trips: the
/// checkpoint for the last completed chunk is already on disk.
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Shared app state, owned by Tauri and reachable from commands.
pub struct AppState {
    pub phase: Mutex<Phase>,
    /// Set while a batch is running so Stop has something to trip.
    pub cancel: Mutex<Option<CancelFlag>>,
    pub language: Mutex<LanguageMode>,
    /// The multilingual checkpoint selected in the header dropdown.
    pub model: Mutex<String>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            phase: Mutex::new(Phase::Idle),
            cancel: Mutex::new(None),
            language: Mutex::new(LanguageMode::ViEn),
            model: Mutex::new(crate::transcribe::DEFAULT_MODEL.to_string()),
        }
    }
}

impl AppState {
    pub fn phase(&self) -> Phase {
        *self.phase.lock()
    }

    pub fn set_phase(&self, phase: Phase) {
        *self.phase.lock() = phase;
    }
}
