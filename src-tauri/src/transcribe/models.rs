//! The GGML model cache: what is on disk, how big it is, and how to get more.
//!
//! Port of the model-cache half of `transcriber.py` (`is_model_downloaded`,
//! `model_size_on_disk`, `delete_model`, `list_downloaded_whisper_models`,
//! `ensure_model_downloaded`) and the backend for the Manage Models dialog.
//!
//! The transfer itself (Range/206 resume, retry, `.part` rename, cancel) lives
//! in [`crate::download`], shared with the LLM GGUF cache. What stays here is
//! whisper-specific: where the files go, what the URL is, and which names are
//! valid.
//!
//! The checkpoints are different files than the Python app's: whisper.cpp wants
//! GGML, openai-whisper wanted `.pt`. They therefore live in a **different
//! directory**, and this module never touches `~/.cache/whisper` — the Python
//! app is still using it, and deleting its models out from under it would be a
//! nasty surprise while both apps are installed.

use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Result};

use crate::download::{self, DownloadError, ProgressCallback};
use crate::state::CancelFlag;

/// Where whisper.cpp GGML checkpoints are published.
const GGML_REPO_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// GGML filename for a model name from [`super::MODEL_OPTIONS`].
pub fn ggml_filename(name: &str) -> String {
    format!("ggml-{name}.bin")
}

/// `$XDG_CACHE_HOME/whisper-cpp`, else `~/.cache/whisper-cpp`.
///
/// Deliberately *not* `~/.cache/whisper`: see the module docs. Also deliberately
/// separate from the LLM cache (`~/.cache/llama-gguf`), so `delete_model` on
/// either side can never touch the other.
pub fn cache_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("whisper-cpp");
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("whisper-cpp")
}

pub fn model_path(name: &str) -> PathBuf {
    cache_dir().join(ggml_filename(name))
}

pub fn is_model_downloaded(name: &str) -> bool {
    model_path(name).is_file()
}

pub fn model_size_on_disk(name: &str) -> u64 {
    fs::metadata(model_path(name)).map(|m| m.len()).unwrap_or(0)
}

/// Delete one model's file, freeing that disk space only.
///
/// Returns whether anything was removed. The model re-downloads automatically
/// the next time it is selected and used.
pub fn delete_model(name: &str) -> bool {
    let path = model_path(name);
    if !path.is_file() {
        return false;
    }
    fs::remove_file(&path).is_ok()
}

/// The models from [`super::MODEL_OPTIONS`] that are actually on disk, in the
/// same fastest-to-most-accurate order the dropdown shows.
///
/// The header dropdown is built from this so that picking a model never stalls
/// on a multi-gigabyte download.
pub fn list_downloaded_models() -> Vec<String> {
    super::MODEL_OPTIONS
        .iter()
        .filter(|name| is_model_downloaded(name))
        .map(|name| name.to_string())
        .collect()
}

/// Size of a model on the server, for showing "1.5 GB" before committing.
///
/// `None` when the server does not say; the UI should then show the download as
/// indeterminate rather than inventing a number.
pub fn remote_size(name: &str) -> Option<u64> {
    download::remote_size_of(&model_url(name))
}

fn model_url(name: &str) -> String {
    format!("{GGML_REPO_BASE}/{}", ggml_filename(name))
}

/// Make sure `name` is on disk, downloading it if not, and return its path.
///
/// Reports progress as it goes and honours `cancel` between chunks, so a user
/// who started a 1.5 GB download by accident is not stuck waiting for it.
pub fn ensure_model_downloaded(
    name: &str,
    progress: Option<&ProgressCallback<'_>>,
    cancel: Option<&CancelFlag>,
) -> Result<PathBuf, DownloadError> {
    let final_path = model_path(name);
    if final_path.is_file() {
        return Ok(final_path);
    }
    if !super::MODEL_OPTIONS.contains(&name) {
        return Err(DownloadError::Other(anyhow!("unknown model '{name}'")));
    }

    download::ensure_downloaded(&model_url(name), &final_path, name, progress, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_names_match_the_published_files() {
        assert_eq!(ggml_filename("large-v3"), "ggml-large-v3.bin");
        assert_eq!(ggml_filename("large-v3-turbo"), "ggml-large-v3-turbo.bin");
        assert_eq!(ggml_filename("small"), "ggml-small.bin");
    }

    #[test]
    fn the_cache_is_not_the_python_apps_cache() {
        // Sharing the directory would let this app's Manage Models dialog
        // delete checkpoints the Python app is still using.
        let dir = cache_dir();
        assert!(dir.ends_with("whisper-cpp"), "{}", dir.display());
        assert!(!dir.ends_with("whisper"));
        assert!(
            !dir.ends_with("llama-gguf"),
            "must not share the LLM cache, or one side's delete could hit the other"
        );
    }

    #[test]
    fn xdg_cache_home_is_honoured() {
        // Serialised with the other env-var-free tests by construction: this is
        // the only test that touches XDG_CACHE_HOME, and it restores it.
        let previous = std::env::var_os("XDG_CACHE_HOME");
        // SAFETY: single-threaded within this test, and restored below.
        unsafe { std::env::set_var("XDG_CACHE_HOME", "/tmp/xdg-example") };
        assert_eq!(cache_dir(), PathBuf::from("/tmp/xdg-example/whisper-cpp"));
        match previous {
            Some(value) => unsafe { std::env::set_var("XDG_CACHE_HOME", value) },
            None => unsafe { std::env::remove_var("XDG_CACHE_HOME") },
        }
    }

    #[test]
    fn unknown_models_are_refused_rather_than_fetched() {
        let error = ensure_model_downloaded("not-a-model", None, None).unwrap_err();
        assert!(matches!(error, DownloadError::Other(_)));
        assert!(error.to_string().contains("unknown model"));
    }
}
