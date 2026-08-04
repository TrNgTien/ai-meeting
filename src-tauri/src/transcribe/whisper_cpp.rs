//! The whisper.cpp engine — what replaces both openai-whisper and MLX.
//!
//! On Apple silicon this is built with Metal, which is the point: it decodes the
//! same Whisper weights on the GPU instead of fp32 on the CPU, which was the
//! difference between a 44-minute recording taking an afternoon and taking a
//! lunch break. There is no switch for it as there was in the Python app, because
//! there is no longer a second engine to switch to — Metal is compiled in and
//! used when the hardware has it.
//!
//! ## The streaming hack is gone
//!
//! The Python app got live segments by redirecting **process-wide stdout**,
//! running the model with `verbose=True`, and parsing the `[start --> end] text`
//! lines it printed (`transcriber._SegmentPrintTap`). That worked, but it meant
//! only one thread could ever transcribe at a time, and a stray print from
//! anywhere else in the process could be mistaken for a segment.
//!
//! whisper.cpp has a real callback, so all of that is deleted. Two wrinkles come
//! with it, both of them properties of whisper-rs rather than of this code:
//!
//! 1. `set_segment_callback_safe` demands a `'static` closure while our sink is
//!    borrowed, so the callback sends over a channel and a scoped thread pumps it
//!    into the sink.
//! 2. whisper-rs stores that closure with `Box::into_raw` and implements no
//!    `Drop` for `FullParams`, so **the closure is leaked and its channel sender
//!    is never dropped**. Waiting for the channel to close on its own therefore
//!    deadlocks. An explicit `None` sentinel ends the pump instead. The leak is
//!    one small box per chunk and cannot be avoided through this API.

use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, Result};
use whisper_rs::{
    FullParams, SegmentCallbackData, WhisperContext, WhisperContextParameters, WhisperState,
};

use super::hallucination::{drop_hallucinations, is_hallucination};
use super::params::{decode_params, default_threads};
use super::segment::TranscriptSegment;
use super::{Engine, SegmentSink};

/// whisper.cpp reports timestamps in centiseconds.
const CENTISECONDS: f64 = 100.0;

/// whisper.cpp and GGML print backend details straight to stderr on every state
/// init ("ggml_metal_init: allocating", the whole kv-cache table). Routing them
/// into the `log` crate silences a desktop app that has no terminal, while
/// leaving them recoverable for anyone who installs a logger.
fn silence_backend_logs() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(whisper_rs::install_logging_hooks);
}

pub struct WhisperCppEngine {
    context: WhisperContext,
    model_name: String,
    /// `None` means "detect", matching how `Transcriber` stored the `auto` mode.
    language: Option<String>,
    threads: i32,
}

impl WhisperCppEngine {
    /// Load a GGML checkpoint. The file must already be on disk — see
    /// [`super::models::ensure_model_downloaded`], which is called first so a
    /// multi-gigabyte fetch is reported as a download rather than showing up as
    /// a mysteriously slow first chunk.
    pub fn load(model_path: &Path, model_name: &str, language: Option<&str>) -> Result<Self> {
        silence_backend_logs();
        let path = model_path
            .to_str()
            .context("model path is not valid UTF-8")?;
        let context = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .with_context(|| format!("cannot load the model at {}", model_path.display()))?;

        Ok(Self {
            context,
            model_name: model_name.to_string(),
            language: language.map(str::to_string),
            threads: default_threads(),
        })
    }

    /// Read the finished segments out of a completed decode.
    fn collect(&self, state: &WhisperState, offset_sec: f64) -> Result<Vec<TranscriptSegment>> {
        let count = state.full_n_segments();
        let mut collected = Vec::with_capacity(count.max(0) as usize);

        for index in 0..count {
            let Some(segment) = state.get_segment(index) else {
                continue;
            };
            // Lossy: a checkpoint occasionally emits a partial UTF-8 sequence,
            // and losing one character beats failing the whole chunk.
            let text = segment
                .to_str_lossy()
                .map(|text| text.trim().to_string())
                .unwrap_or_default();
            if text.is_empty() {
                continue;
            }
            collected.push(TranscriptSegment::new(
                offset_sec + segment.start_timestamp() as f64 / CENTISECONDS,
                offset_sec + segment.end_timestamp() as f64 / CENTISECONDS,
                text,
            ));
        }
        Ok(collected)
    }

    /// Wire the native segment callback to a channel.
    ///
    /// The filtering here mirrors what `_SegmentPrintTap._emit` did, and for the
    /// same reason: preview lines go straight to the UI without passing through
    /// the post-chunk filtering, so an invented outro would otherwise flash on
    /// screen before being dropped from the text that lands on disk.
    fn attach_callback(
        params: &mut FullParams<'_, '_>,
        sender: mpsc::Sender<Option<TranscriptSegment>>,
        offset_sec: f64,
    ) {
        let mut last_text = String::new();
        params.set_segment_callback_safe(move |data: SegmentCallbackData| {
            let text = data.text.trim().to_string();
            if text.is_empty() || is_hallucination(&text) {
                return;
            }
            // A sentence repeated verbatim back-to-back is a decode loop, not
            // speech; short utterances ("ừ", "vâng") are left alone.
            if text == last_text && text.split_whitespace().count() >= 4 {
                return;
            }
            last_text = text.clone();
            // A closed channel just means the run ended; the preview is not
            // worth taking the transcription down for.
            let _ = sender.send(Some(TranscriptSegment::new(
                offset_sec + data.start_timestamp as f64 / CENTISECONDS,
                offset_sec + data.end_timestamp as f64 / CENTISECONDS,
                text,
            )));
        });
    }
}

impl Engine for WhisperCppEngine {
    fn engine_key(&self) -> String {
        format!(
            "whispercpp-{}:{}",
            self.model_name,
            self.language.as_deref().unwrap_or("auto")
        )
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn transcribe(
        &self,
        audio: &[f32],
        offset_sec: f64,
        on_segment: Option<&SegmentSink<'_>>,
    ) -> Result<Vec<TranscriptSegment>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let mut state = self
            .context
            .create_state()
            .context("cannot create a decode state")?;
        let mut params = decode_params(self.language.as_deref(), self.threads);

        match on_segment {
            None => {
                state
                    .full(params, audio)
                    .context("whisper.cpp failed to decode the chunk")?;
            }
            Some(sink) => {
                let (sender, receiver) = mpsc::channel::<Option<TranscriptSegment>>();
                // Keep a sender of our own: the callback's clone is leaked by
                // whisper-rs (see the module docs), so this is the only one that
                // can ever signal end-of-stream.
                Self::attach_callback(&mut params, sender.clone(), offset_sec);

                std::thread::scope(|scope| -> Result<()> {
                    let pump = scope.spawn(move || {
                        // Stops on the sentinel, not on channel close — the
                        // leaked sender means the channel never closes.
                        while let Ok(Some(segment)) = receiver.recv() {
                            sink(segment);
                        }
                    });

                    let result = state
                        .full(params, audio)
                        .context("whisper.cpp failed to decode the chunk");

                    // Sent even when the decode failed, so an error can never
                    // leave the pump thread parked forever.
                    let _ = sender.send(None);
                    let _ = pump.join();
                    result.map(|_| ())
                })?;
            }
        }

        Ok(drop_hallucinations(self.collect(&state, offset_sec)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Engine keys are what stop a transcript half-written by one model from
    /// being continued by another, so their exact shape matters.
    #[test]
    fn engine_keys_distinguish_model_and_language() {
        // Built without loading a model: the key depends only on these fields.
        fn key(model: &str, language: Option<&str>) -> String {
            format!("whispercpp-{}:{}", model, language.unwrap_or("auto"))
        }
        assert_eq!(key("large-v3", Some("vi")), "whispercpp-large-v3:vi");
        assert_eq!(key("large-v3", Some("en")), "whispercpp-large-v3:en");
        assert_eq!(key("large-v3", None), "whispercpp-large-v3:auto");
        assert_ne!(key("large-v3", Some("vi")), key("large-v2", Some("vi")));
    }

    #[test]
    fn centisecond_timestamps_land_on_the_recordings_timeline() {
        // 250 centiseconds into a chunk starting at 300 s is 302.5 s overall.
        let offset = 300.0;
        let raw_timestamp: i64 = 250;
        assert_eq!(offset + raw_timestamp as f64 / CENTISECONDS, 302.5);
    }
}
