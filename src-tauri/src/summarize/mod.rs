//! Local-LLM transcript summarization — a bundled llama-cli running a GGUF.
//!
//! Same privacy contract as whisper.cpp: nothing leaves the machine, weights are
//! pulled from Hugging Face on first use, and the user can point it at any GGUF
//! they like. [`summarize::models`] owns the GGUF cache, [`summarize::prompt`]
//! owns the map-reduce over a long transcript, and [`summarize::llama_cli`] is
//! the one concrete [`Summarizer`].

pub mod llama_cli;
pub mod models;
pub mod prompt;

use std::path::{Path, PathBuf};

use crate::state::CancelFlag;

/// A destination for streamed completion tokens.
pub type TokenSink<'a> = dyn Fn(&str) + Send + Sync + 'a;

/// Where a summary goes: a sibling of the transcript it summarises.
///
/// `transcript_stem` is the transcript's bare name — for an import
/// `<stamp>-<stem>`, giving `<stamp>-<stem>-summary.md`. A recording's merged
/// conversation is `<stem>-conversation.txt`, and its summary drops the
/// `-conversation` marker so it reads `<stem>-summary.md`, matching how the
/// merge names `<stem>-conversation.txt`.
pub fn summary_path(dir: &Path, transcript_stem: &str) -> PathBuf {
    let stem = transcript_stem
        .strip_suffix("-conversation")
        .unwrap_or(transcript_stem);
    dir.join(format!("{stem}-summary.md"))
}

/// Character budget for one map chunk, derived from the model's context window.
///
/// The model's context is in tokens; a transcript chunk has to leave room for
/// the system prompt, the chat template and the completion. Working on a rough
/// ~3 characters per token (mixed Vietnamese/English) and reserving a slice for
/// overhead keeps the chunk comfortably inside the window without guessing at
/// the exact tokeniser.
pub fn char_budget(context: u32) -> usize {
    let tokens = context.saturating_sub(512);
    (tokens as usize) * 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_and_recording_summaries_land_next_to_their_transcripts() {
        let dir = Path::new("/tmp/meetings");
        assert_eq!(
            summary_path(dir, "20260804-090000-standup"),
            Path::new("/tmp/meetings/20260804-090000-standup-summary.md")
        );
        // A recording's merged conversation drops the `-conversation` marker so
        // `<stem>-summary.md` sits beside `<stem>-conversation.txt`.
        assert_eq!(
            summary_path(dir, "meeting-20260804-090000-conversation"),
            Path::new("/tmp/meetings/meeting-20260804-090000-summary.md")
        );
    }

    #[test]
    fn budget_derives_from_context_tokens() {
        assert_eq!(char_budget(4096), (4096 - 512) * 3);
        assert_eq!(char_budget(0), 0);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SummarizeError {
    /// Cancellation is not a failure: the user stopped, and no summary file is
    /// written. Mirrors [`crate::chunking::TranscribeError::Cancelled`].
    #[error("summarization cancelled")]
    Cancelled,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// One way of turning a transcript into a summary.
///
/// A trait, not a concrete struct, purely so tests get a `StubSummarizer` the
/// way `tests/chunked_run.rs`'s `StubEngine` stubs [`crate::transcribe::Engine`].
/// It deliberately does **not** implement that trait — `Engine` is audio-shaped.
pub trait Summarizer: Send + Sync {
    /// Identity of this summarizer, for events and any cache busting.
    fn key(&self) -> String;

    /// The context window this model is driven with, in tokens. Used to size
    /// map chunks so they fit comfortably alongside the prompts.
    fn context(&self) -> u32;

    /// Run one completion. `system` and `user` are the two chat turns; tokens
    /// stream through `on_token` as they are produced. Honours `cancel` between
    /// reads, killing the child process if it trips.
    fn generate(
        &self,
        system: &str,
        user: &str,
        on_token: Option<&TokenSink<'_>>,
        cancel: &CancelFlag,
    ) -> Result<String, SummarizeError>;
}
