//! What the app is doing, and the handles a command needs to change it.
//!
//! Port of `app.AppState` plus the parts of `MeetingTranscriberApp` that were
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
/// Vietnamised. `Vi` switches to PhoWhisper, which is more accurate on pure
/// Vietnamese but phoneticises English.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageMode {
    #[serde(rename = "vi+en")]
    ViEn,
    Vi,
    En,
    Auto,
}

impl LanguageMode {
    /// The language code handed to the decoder. `None` means "detect", which
    /// matches how `Transcriber.set_language` stored `auto`.
    pub fn whisper_language(self) -> Option<&'static str> {
        match self {
            // Both Vietnamese modes decode *as* Vietnamese; they differ in
            // which checkpoint does it.
            LanguageMode::ViEn | LanguageMode::Vi => Some("vi"),
            LanguageMode::En => Some("en"),
            LanguageMode::Auto => None,
        }
    }

    /// Whether this mode routes to PhoWhisper rather than a multilingual
    /// checkpoint. Only pure `vi` does.
    pub fn uses_phowhisper(self) -> bool {
        matches!(self, LanguageMode::Vi)
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
