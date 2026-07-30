//! Transcription engines and the text they produce.
//!
//! Every engine implements one contract — [`Engine`] — which is the Rust
//! version of the Python app's `transcribe_audio(audio, offset_sec, on_segment)`
//! callable. `chunking::transcribe_chunked` drives it and never knows which
//! engine it has.

pub mod hallucination;
pub mod models;
pub mod params;
pub mod segment;
pub mod whisper_cpp;

use anyhow::Result;

pub use segment::{format_timestamp, segments_to_text, TranscriptSegment};

/// The multilingual checkpoint used unless the header dropdown says otherwise.
pub const DEFAULT_MODEL: &str = "large-v3";

/// Checkpoints selectable in the UI for the `vi+en` / `en` / `auto` modes
/// (pure `vi` always uses PhoWhisper). Ordered roughly fastest -> most
/// accurate, matching `transcriber.FINAL_MODEL_OPTIONS`.
pub const MODEL_OPTIONS: &[&str] = &["small", "medium", "large-v2", "large-v3", "large-v3-turbo"];

/// Called as the engine finishes each decode window, while the chunk it belongs
/// to is still being transcribed. These are a *preview*: nothing reaching this
/// callback is on disk yet.
pub type SegmentSink<'a> = dyn Fn(TranscriptSegment) + Send + Sync + 'a;

/// One way of turning 16 kHz mono audio into timestamped segments.
pub trait Engine: Send + Sync {
    /// Identity of this engine for the resume checkpoint.
    ///
    /// It has to change whenever the produced text would differ — model,
    /// language, or engine — because a transcript half-written by one engine
    /// must never be continued by another. Splicing two different decodings of
    /// the same meeting into one file is worse than redoing the work.
    fn engine_key(&self) -> String;

    /// Whether this engine can report segments before the chunk is finished.
    ///
    /// Replaces `chunking._accepts_on_segment()`, which had to inspect the
    /// callable's signature; a trait method just answers.
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Transcribe one already-decoded 16 kHz mono chunk.
    ///
    /// `offset_sec` shifts the returned timestamps back onto the original
    /// recording's timeline, so chunks can be merged into one transcript.
    fn transcribe(
        &self,
        audio: &[f32],
        offset_sec: f64,
        on_segment: Option<&SegmentSink<'_>>,
    ) -> Result<Vec<TranscriptSegment>>;
}
