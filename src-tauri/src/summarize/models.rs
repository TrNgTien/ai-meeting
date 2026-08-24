//! The GGUF LLM cache: the catalog the user can pick from, plus the "any HF
//! repo + file" escape hatch, and how to get a model on disk.
//!
//! Mirrors the whisper model cache in [`crate::transcribe::models`] — same
//! shape, same `list`/`delete`/`ensure_downloaded` surface — so the frontend can
//! reuse `components/ModelList.tsx`. The one structural difference is that an
//! LLM is identified by `(repo, file)` rather than a bare name, because a custom
//! model can come from any Hugging Face repo. The transfer itself is delegated
//! to [`crate::download`], which both caches share.
//!
//! The cache directory is deliberately separate from `whisper-cpp/` so
//! `delete_model` on either side can never touch the other's weights.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;

use crate::download::{self, DownloadError, ProgressCallback};
use crate::state::CancelFlag;

/// One selectable LLM. `context` is the conservative context window we run it
/// with — what drives RAM — not what the model *can* do.
#[derive(Debug, Clone, Copy)]
pub struct LlmModel {
    pub id: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub label: &'static str,
    pub context: u32,
}

/// The models the UI offers. Gemma 3n E2B (MatFormer, ~2B effective) is the
/// default: small enough to run locally and decent at Vietnamese. Both are
/// **text-only** builds of the *ungated* GGUF mirrors — `google/gemma-3n-E2B-it`
/// itself is license-gated (401s without a token the downloader does not send)
/// and ships safetensors, which llama.cpp cannot load. See `CLAUDE.md` Gotchas.
pub const LLM_OPTIONS: &[LlmModel] = &[
    LlmModel {
        id: "gemma-3n-E2B-it-Q4_K_M",
        repo: "unsloth/gemma-3n-E2B-it-GGUF",
        file: "gemma-3n-E2B-it-Q4_K_M.gguf",
        label: "Gemma 3n E2B (Q4_K_M)",
        context: 4096,
    },
    LlmModel {
        id: "gemma-3n-E2B-it-Q8_0",
        repo: "ggml-org/gemma-3n-E2B-it-GGUF",
        file: "gemma-3n-E2B-it-Q8_0.gguf",
        label: "Gemma 3n E2B (Q8_0)",
        context: 4096,
    },
];

pub const DEFAULT_LLM: &str = "gemma-3n-E2B-it-Q4_K_M";

/// The "any repo + file" escape hatch, selected when `llm_model == "custom"`.
pub const CUSTOM_LLM: &str = "custom";

/// A resolved model selection: everything [`crate::summarize::llama_cli`] needs.
#[derive(Debug, Clone)]
pub struct LlmSelection {
    pub repo: String,
    pub file: String,
    /// The id used in events and the model's cache key — a catalog id, or the
    /// custom file name (which is unique per file in the cache dir).
    pub model_id: String,
    pub label: String,
    pub context: u32,
}

pub fn llm_by_id(id: &str) -> Option<&'static LlmModel> {
    LLM_OPTIONS.iter().find(|model| model.id == id)
}

/// Validate a custom `{repo, file}` pair before it becomes a path component.
///
/// The file name is joined into the cache dir verbatim, so this is a real
/// traversal boundary, not defensive noise: reject empty parts, `..`, a leading
/// `/`, and a file that does not end in `.gguf`.
pub fn validate_custom(repo: &str, file: &str) -> Result<(), String> {
    let path_ok = |s: &str| !s.trim().is_empty() && !s.contains("..") && !s.starts_with('/');
    if !path_ok(repo) {
        return Err("invalid repository: must be a non-empty path without '..' or a leading '/'".into());
    }
    if !path_ok(file) {
        return Err("invalid file: must be a non-empty name without '..' or a leading '/'".into());
    }
    if !file.ends_with(".gguf") {
        return Err("the model file must end in .gguf".into());
    }
    Ok(())
}

/// Resolve a selection from the saved settings.
///
/// `None` when the selection is unusable — an unknown catalog id, or an invalid
/// custom pair — so callers can refuse to summarise rather than guess.
pub fn resolve(
    llm_model: &str,
    custom_repo: Option<&str>,
    custom_file: Option<&str>,
) -> Option<LlmSelection> {
    if llm_model == CUSTOM_LLM {
        let repo = custom_repo.unwrap_or_default();
        let file = custom_file.unwrap_or_default();
        validate_custom(repo, file).ok()?;
        return Some(LlmSelection {
            repo: repo.trim().to_string(),
            file: file.trim().to_string(),
            model_id: file.trim().to_string(),
            label: format!("{repo}/{file}"),
            context: 4096,
        });
    }
    let model = llm_by_id(llm_model)?;
    Some(LlmSelection {
        repo: model.repo.to_string(),
        file: model.file.to_string(),
        model_id: model.id.to_string(),
        label: model.label.to_string(),
        context: model.context,
    })
}

/// `$XDG_CACHE_HOME/llama-gguf`, else `~/.cache/llama-gguf`.
///
/// Deliberately separate from `whisper-cpp/` (see [`crate::transcribe::models::cache_dir`]),
/// so deleting a model on either side can never touch the other.
pub fn cache_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("llama-gguf");
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("llama-gguf")
}

/// Where a model's GGUF lands. `file` becomes a path component, which is why
/// custom files are validated before reaching here.
pub fn model_path(file: &str) -> PathBuf {
    cache_dir().join(file)
}

pub fn is_downloaded(file: &str) -> bool {
    model_path(file).is_file()
}

pub fn size_on_disk(file: &str) -> u64 {
    fs::metadata(model_path(file)).map(|m| m.len()).unwrap_or(0)
}

/// Delete one model's GGUF, freeing that disk space only.
pub fn delete(file: &str) -> bool {
    let path = model_path(file);
    if !path.is_file() {
        return false;
    }
    fs::remove_file(&path).is_ok()
}

/// The catalog models actually on disk, in catalog order.
pub fn list_downloaded() -> Vec<String> {
    LLM_OPTIONS
        .iter()
        .filter(|model| is_downloaded(model.file))
        .map(|model| model.id.to_string())
        .collect()
}

/// Size of `repo`/`file` on the server, for showing "2.8 GB" before committing.
pub fn remote_size(repo: &str, file: &str) -> Option<u64> {
    download::remote_size_of(&model_url(repo, file))
}

fn model_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

/// Make sure `repo`/`file` is on disk, downloading it if not, and return its path.
///
/// Reports progress as it goes and honours `cancel` between chunks.
pub fn ensure_downloaded(
    repo: &str,
    file: &str,
    progress: Option<&ProgressCallback<'_>>,
    cancel: Option<&CancelFlag>,
) -> Result<PathBuf, DownloadError> {
    let dest = model_path(file);
    download::ensure_downloaded(&model_url(repo, file), &dest, file, progress, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_names_are_well_formed() {
        for model in LLM_OPTIONS {
            assert!(model.id.contains(model.file.trim_end_matches(".gguf")));
            assert!(model.file.ends_with(".gguf"));
            assert!(!model.repo.contains(".."));
        }
        assert_eq!(LLM_OPTIONS[0].id, DEFAULT_LLM, "the default is the first entry");
    }

    #[test]
    fn custom_validation_rejects_traversal_and_bad_files() {
        assert!(validate_custom("unsloth/ok", "model.gguf").is_ok());
        assert!(validate_custom("unsloth/ok", "model.gguf").is_ok());
        assert!(validate_custom("", "model.gguf").is_err(), "empty repo");
        assert!(validate_custom("unsloth/ok", "").is_err(), "empty file");
        assert!(validate_custom("../evil", "model.gguf").is_err(), "repo ..");
        assert!(validate_custom("unsloth/ok", "../model.gguf").is_err(), "file ..");
        assert!(validate_custom("/abs/repo", "model.gguf").is_err(), "absolute repo");
        assert!(validate_custom("unsloth/ok", "/etc/passwd").is_err(), "absolute file");
        assert!(validate_custom("unsloth/ok", "model.bin").is_err(), "non-.gguf");
    }

    #[test]
    fn resolve_handles_catalog_and_custom() {
        let catalog = resolve(DEFAULT_LLM, None, None).expect("default resolves");
        assert_eq!(catalog.file, "gemma-3n-E2B-it-Q4_K_M.gguf");
        assert_eq!(catalog.context, 4096);

        let custom = resolve(CUSTOM_LLM, Some("unsloth/Qwen3-4B-Instruct-2507-GGUF"), Some("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"))
            .expect("valid custom resolves");
        assert!(custom.file.ends_with(".gguf"));
        assert_eq!(custom.model_id, custom.file);

        assert!(resolve("no-such-model", None, None).is_none(), "unknown id");
        assert!(resolve(CUSTOM_LLM, Some("x"), Some("../y.gguf")).is_none(), "invalid custom");
    }

    #[test]
    fn cache_dir_is_distinct_from_whisper() {
        let dir = cache_dir();
        assert!(dir.ends_with("llama-gguf"), "{}", dir.display());
        assert!(!dir.ends_with("whisper-cpp"));
    }

    #[test]
    fn xdg_cache_home_is_honoured() {
        let previous = std::env::var_os("XDG_CACHE_HOME");
        // SAFETY: single-threaded within this test, and restored below.
        unsafe { std::env::set_var("XDG_CACHE_HOME", "/tmp/xdg-llm") };
        assert_eq!(cache_dir(), PathBuf::from("/tmp/xdg-llm/llama-gguf"));
        match previous {
            Some(value) => unsafe { std::env::set_var("XDG_CACHE_HOME", value) },
            None => unsafe { std::env::remove_var("XDG_CACHE_HOME") },
        }
    }
}
