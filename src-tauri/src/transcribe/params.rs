//! Decode parameters — the single source of truth for how a model is driven.
//!
//! This is the port of `transcriber.DECODE_OPTIONS`, and the mapping is the most
//! important table in the project, so it is written out explicitly:
//!
//! | openai-whisper / mlx-whisper        | whisper.cpp / whisper-rs                     |
//! |-------------------------------------|----------------------------------------------|
//! | `condition_on_previous_text: False` | [`FullParams::set_no_context(true)`]         |
//! | `logprob_threshold: -1.0`           | `set_logprob_thold(-1.0)`                    |
//! | `compression_ratio_threshold: 2.4`  | `set_entropy_thold(2.4)`                     |
//! | `no_speech_threshold: 0.6`          | `set_no_speech_thold(0.6)`                   |
//! | `word_timestamps: True`             | `set_token_timestamps(true)` + `split_on_word`|
//! | `hallucination_silence_threshold`   | *no equivalent* — see [`super::hallucination`] |
//! | `beam_size=5`                       | `SamplingStrategy::BeamSearch { beam_size: 5 }` |
//! | `fp16=False`, `DEVICE="cpu"`        | n/a — Metal decodes f16 natively             |
//! | `language="vi"`/`"en"`/`None`        | `set_language` / `set_detect_language`       |
//!
//! Why each of these is set is not obvious from the value, so the reasoning from
//! the Python app is carried over with them: Whisper was trained on YouTube
//! captions, so over silence it emits a channel outro rather than nothing, and
//! with previous-text conditioning left on that invented line becomes the prompt
//! for the next window and repeats for the rest of the recording. Dropping the
//! conditioning is the fix for the loop; the thresholds make the model bail out
//! of a window it is not confident about instead of guessing.

use whisper_rs::{FullParams, SamplingStrategy};

/// Below this average token logprob the window is treated as a failed decode.
const LOGPROB_THRESHOLD: f32 = -1.0;

/// whisper.cpp's analogue of openai-whisper's `compression_ratio_threshold`:
/// a window this repetitive is degenerate rather than speech.
const ENTROPY_THRESHOLD: f32 = 2.4;

/// Above this no-speech probability the window is emitted as silence.
const NO_SPEECH_THRESHOLD: f32 = 0.6;

/// Beam search matches the Python app's `beam_size=5`. It is the slow, accurate
/// setting; on Metal the cost is affordable in a way it never was on CPU.
const BEAM_SIZE: i32 = 5;

/// `-1.0` means "use whisper.cpp's default patience" rather than no patience.
const BEAM_PATIENCE: f32 = -1.0;

/// Build the decode parameters for one chunk.
///
/// `language` is `None` for the `auto` mode, which turns on detection; the
/// `vi+en` mode passes `Some("vi")` on a *multilingual* checkpoint, which is
/// what keeps embedded English words spelled in English.
///
/// `streaming` decides whether whisper.cpp emits its own progress prints. It is
/// unrelated to the segment callback — that is wired separately in
/// [`super::whisper_cpp`] — and is kept off so a Tauri app does not spew to a
/// terminal nobody is reading.
pub fn decode_params<'a>(language: Option<&'a str>, threads: i32) -> FullParams<'a, 'a> {
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: BEAM_SIZE,
        patience: BEAM_PATIENCE,
    });

    params.set_n_threads(threads);
    params.set_translate(false);

    match language {
        Some(code) => {
            params.set_language(Some(code));
            params.set_detect_language(false);
        }
        None => {
            // `auto`: let the model decide, as `Transcriber` did with language=None.
            params.set_language(None);
            params.set_detect_language(true);
        }
    }

    // The fix for the repeat loop: never let an invented line become the prompt
    // for the next window.
    params.set_no_context(true);

    params.set_logprob_thold(LOGPROB_THRESHOLD);
    params.set_entropy_thold(ENTROPY_THRESHOLD);
    params.set_no_speech_thold(NO_SPEECH_THRESHOLD);

    // Word-level timings, which the silence post-pass in `hallucination` needs to
    // tell an invented outro from a speaker trailing off.
    params.set_token_timestamps(true);
    params.set_split_on_word(true);

    // An extra guard the Python app had no access to: whisper.cpp can suppress
    // the non-speech tokens ("(music)", "[laughter]") that Whisper emits over
    // noise. New behaviour, not a change to anything that existed.
    params.set_suppress_nst(true);

    // A Tauri app has no terminal to print to, and the segment callback replaces
    // the Python app's reason for wanting verbose output in the first place.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    params
}

/// Threads to decode with: leave a core for the UI and the recorder.
///
/// On Metal most of the work is on the GPU, but the mel spectrogram and the
/// beam-search bookkeeping are still CPU-side.
pub fn default_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    (cores - 1).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_leaves_a_core_free() {
        let threads = default_threads();
        assert!(threads >= 1);
        let cores = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        assert!(threads < cores || cores == 1);
    }

    #[test]
    fn params_build_for_every_language_mode() {
        // Mostly a compile-and-don't-panic check: whisper-rs exposes no getters,
        // so the values themselves are verified by the A/B transcript diff.
        let _vi = decode_params(Some("vi"), 4);
        let _en = decode_params(Some("en"), 4);
        let _auto = decode_params(None, 4);
    }
}
