//! The choices the app remembers between launches.
//!
//! The Python app remembered none of them: language, model, the GPU switch and
//! both recording toggles were rebuilt from constants in `TranscriberApp.__init__`
//! every launch, and `_on_close` wrote nothing. Someone who works in Vietnamese
//! on `large-v3` re-picked both every single time.
//!
//! This is deliberately not the same thing as [`crate::state::AppState`]: that
//! is what the app is doing *now* and dies with the process. This is what the
//! user chose, and outlives it.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Written under the app's config directory — `~/Library/Application Support/`
/// on macOS.
const APP_DIR: &str = "dev.placepad.transcriber";
const FILE_NAME: &str = "settings.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// One of the strings [`crate::state::LanguageMode::parse`] accepts.
    pub language_mode: String,
    pub model: String,
    pub record_mic: bool,
    pub record_system: bool,
    /// A cpal device id, stable across launches — which is why remembering it
    /// is worth anything (the Python app's PortAudio index was not).
    pub mic_device_id: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            language_mode: "vi+en".to_string(),
            model: crate::transcribe::DEFAULT_MODEL.to_string(),
            record_mic: true,
            record_system: true,
            mic_device_id: None,
        }
    }
}

pub fn settings_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_DIR)
        .join(FILE_NAME)
}

/// Read the saved settings, falling back to defaults.
///
/// A corrupt or half-written file is not an error worth showing anyone: the
/// defaults are usable, and the next save overwrites it. `#[serde(default)]`
/// covers the other direction — a file written by an older version is missing
/// whatever was added since, and the missing fields take their defaults rather
/// than discarding the fields that *are* there.
pub fn load() -> Settings {
    let Ok(text) = std::fs::read_to_string(settings_path()) else {
        return Settings::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

pub fn save(settings: &Settings) -> Result<()> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(settings)?;
    std::fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_app_the_python_version_started_as() {
        let settings = Settings::default();
        assert_eq!(settings.language_mode, "vi+en");
        assert!(settings.record_mic && settings.record_system);
        // The one place the Rust app disagreed with itself: the frontend
        // defaulted to `small` while the engine's DEFAULT_MODEL was large-v3.
        assert_eq!(settings.model, crate::transcribe::DEFAULT_MODEL);
    }

    #[test]
    fn a_file_from_an_older_version_keeps_what_it_does_say() {
        let stored: Settings = serde_json::from_str(r#"{"model":"medium"}"#).unwrap();
        assert_eq!(stored.model, "medium");
        assert_eq!(stored.language_mode, "vi+en", "missing fields take defaults");
    }

    #[test]
    fn garbage_does_not_take_the_app_down_with_it() {
        let stored: Settings = serde_json::from_str("not json").unwrap_or_default();
        assert_eq!(stored, Settings::default());
    }

    #[test]
    fn settings_live_beside_the_app_not_beside_the_audio() {
        let path = settings_path();
        assert!(path.ends_with("dev.placepad.transcriber/settings.json"));
    }
}
