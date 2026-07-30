//! Hallucination and silence control.
//!
//! Whisper was trained on YouTube captions, so over silence or background noise
//! it does not stay quiet — it emits the most likely thing to follow, which for
//! Vietnamese audio is a channel outro ("Hãy subscribe cho kênh Ghiền Mì Gõ…").
//! With `condition_on_previous_text` left on, that invented line then becomes
//! the prompt for the next window and the model repeats it once per decode
//! window for the rest of the recording.
//!
//! Three layers deal with it, and they are deliberately independent:
//!
//! 1. Decode parameters (see `params.rs`) stop the loop from forming.
//! 2. [`is_hallucination`] drops the known canned phrases.
//! 3. [`drop_hallucinations`] collapses the repeat runs that slip past 1 and 2.
//!
//! Ported from `transcriber.py`. The one thing openai-whisper gave us for free
//! and whisper.cpp does not is `hallucination_silence_threshold`; that is
//! reimplemented here as [`silent_spans`] + [`drop_segments_in_silence`].

use std::sync::LazyLock;

use regex::{Regex, RegexSet};

use super::segment::TranscriptSegment;

/// Stock phrases Whisper falls back on when there is nothing to transcribe.
///
/// Matched case-insensitively against the *whole* segment, after stripping
/// punctuation, so a segment is only dropped when it is entirely boilerplate —
/// the same words spoken inside a real sentence survive.
const HALLUCINATION_PATTERNS: &[&str] = &[
    // Vietnamese YouTube outros
    r"^h[ãa]y subscribe cho k[êe]nh\b.*",
    r".*\bghi[eề]n m[ìi] g[õo]\b.*",
    r"^đăng k[ýy] k[êe]nh\b.*",
    r"^c[ảa]m [ơo]n c[áa]c b[ạa]n đ[ãa] (theo d[õo]i|xem|l[ắa]ng nghe)\b.*",
    r"^h[ẹe]n g[ặa]p l[ạa]i c[áa]c b[ạa]n\b.*",
    r"^c[áa]c b[ạa]n c[óo] th[ểe] nh[ậa]n th[êe]m nhi[ềe]u th[ôo]ng tin\b.*",
    r".*trong ph[ầa]n b[ìi]nh lu[ậa]n\s*$",
    r"^ch[úu]c c[áa]c b[ạa]n (xem )?(video )?vui v[ẻe]\b.*",
    // English equivalents, which show up on mixed-language audio
    r"^thanks? (you )?for watching\b.*",
    r"^(please )?(don't forget to )?subscribe\b.*",
    r"^(subtitles?|amara)\b.*",
];

/// Word count above which a verbatim back-to-back repeat is a decode loop
/// rather than something a person said twice.
const REPEAT_MIN_WORDS: usize = 4;

/// How close two identical sentences must be to be treated as one held line.
/// A duplicate a minute later has silence in between, and claiming the speaker
/// held that line the whole time is worse than leaving the gap.
const REPEAT_ADJACENCY_SEC: f64 = 1.0;

/// Replaces openai-whisper's `hallucination_silence_threshold=2.0`: a span of
/// silence at least this long is long enough that anything "transcribed"
/// inside it was invented.
pub const HALLUCINATION_SILENCE_SEC: f64 = 2.0;

/// Frame length for the silence scan. Same granularity as the chunk-boundary
/// search in `chunking::split`, for the same reason: fine enough to locate a
/// pause, coarse enough to be cheap.
const SILENCE_FRAME_SEC: f64 = 0.1;

/// Mean absolute amplitude below which a frame counts as silence (~-34 dBFS).
/// Above room tone and mic self-noise, well below speech.
const SILENCE_FLOOR: f32 = 0.02;

/// `RegexSet` tests all patterns in one pass. Each is wrapped so it behaves
/// like Python's `fullmatch` — the whole normalized segment must match, which
/// is what keeps these from firing on a real sentence that merely contains one
/// of the phrases.
static HALLUCINATION_SET: LazyLock<RegexSet> = LazyLock::new(|| {
    let anchored: Vec<String> = HALLUCINATION_PATTERNS
        .iter()
        .map(|pattern| format!("(?is)^(?:{pattern})$"))
        .collect();
    RegexSet::new(anchored).expect("hallucination patterns must compile")
});

/// Punctuation and whitespace to ignore when matching the patterns above.
static PUNCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"[.,!?…\-–—"'()\[\]]+"#).expect("punct pattern must compile"));

static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("whitespace pattern must compile"));

/// True if a segment is one of Whisper's canned silence fillers.
///
/// Empty (or punctuation-only) text counts as a hallucination too: there is
/// nothing there to keep.
pub fn is_hallucination(text: &str) -> bool {
    let normalized = normalize(text);
    if normalized.is_empty() {
        return true;
    }
    HALLUCINATION_SET.is_match(&normalized)
}

fn normalize(text: &str) -> String {
    let without_punct = PUNCT.replace_all(text, " ");
    WHITESPACE.replace_all(&without_punct, " ").trim().to_string()
}

/// Whether a verbatim back-to-back repeat of this text is a decode loop.
///
/// Short utterances are left alone — "ừ", "vâng", "okay" really are said twice
/// in a row, and a run of them is speech rather than a loop.
fn is_sentence(text: &str) -> bool {
    text.split_whitespace().count() >= REPEAT_MIN_WORDS
}

/// Strip canned filler and collapse the repeat loops it causes.
///
/// Beyond the known phrases, a *sentence* repeated back-to-back is dropped: a
/// speaker does not say the same eight words verbatim three windows running,
/// but a model that has locked onto its own output does.
pub fn drop_hallucinations(segments: Vec<TranscriptSegment>) -> Vec<TranscriptSegment> {
    let mut kept: Vec<TranscriptSegment> = Vec::with_capacity(segments.len());

    for segment in segments {
        if is_hallucination(&segment.text) {
            continue;
        }
        let text = segment.text.trim();

        if let Some(last) = kept.last_mut() {
            if last.text.trim() == text && is_sentence(text) {
                // Only stretch the surviving copy over a repeat that butts up
                // against it; a duplicate a minute later is silence in
                // between, and claiming the speaker held that line the whole
                // time is worse than leaving the gap.
                if segment.start_sec - last.end_sec <= REPEAT_ADJACENCY_SEC {
                    last.end_sec = last.end_sec.max(segment.end_sec);
                }
                continue;
            }
        }

        kept.push(segment);
    }

    kept
}

/// Spans of near-silence at least `min_len_sec` long, in seconds relative to
/// the start of `audio` (16 kHz mono).
///
/// This is the input to [`drop_segments_in_silence`], and exists because
/// whisper.cpp has no equivalent of openai-whisper's
/// `hallucination_silence_threshold`. Deriving the spans from the chunk audio
/// we already hold in memory keeps the check independent of whether the VAD
/// model happened to load.
pub fn silent_spans(audio: &[f32], sample_rate: u32, min_len_sec: f64) -> Vec<(f64, f64)> {
    let frame = ((SILENCE_FRAME_SEC * sample_rate as f64) as usize).max(1);
    if audio.len() < frame {
        return Vec::new();
    }

    let mut spans: Vec<(f64, f64)> = Vec::new();
    let mut run_start: Option<usize> = None;
    let frame_count = audio.len() / frame;

    for index in 0..frame_count {
        let start = index * frame;
        let window = &audio[start..start + frame];
        let energy = window.iter().map(|s| s.abs()).sum::<f32>() / frame as f32;

        if energy < SILENCE_FLOOR {
            run_start.get_or_insert(index);
        } else if let Some(begin) = run_start.take() {
            push_span(&mut spans, begin, index, frame, sample_rate, min_len_sec);
        }
    }

    // A silent run reaching the end of the buffer — the common case for an
    // invented outro, which is exactly what this is for.
    if let Some(begin) = run_start {
        push_span(
            &mut spans,
            begin,
            frame_count,
            frame,
            sample_rate,
            min_len_sec,
        );
    }

    spans
}

fn push_span(
    spans: &mut Vec<(f64, f64)>,
    begin_frame: usize,
    end_frame: usize,
    frame: usize,
    sample_rate: u32,
    min_len_sec: f64,
) {
    let start = (begin_frame * frame) as f64 / sample_rate as f64;
    let end = (end_frame * frame) as f64 / sample_rate as f64;
    if end - start >= min_len_sec {
        spans.push((start, end));
    }
}

/// Drop segments that lie entirely inside a silent span.
///
/// This is the `hallucination_silence_threshold` behaviour: text the model
/// produced for a stretch where nobody was speaking is not a transcription of
/// anything. Segments that merely *overlap* a pause are kept — a speaker
/// trailing off into silence is real speech.
///
/// `offset_sec` is the chunk's position on the recording's timeline, since
/// `segments` carry absolute timestamps while `spans` are chunk-relative.
pub fn drop_segments_in_silence(
    segments: Vec<TranscriptSegment>,
    spans: &[(f64, f64)],
    offset_sec: f64,
) -> Vec<TranscriptSegment> {
    if spans.is_empty() {
        return segments;
    }
    segments
        .into_iter()
        .filter(|segment| {
            let start = segment.start_sec - offset_sec;
            let end = segment.end_sec - offset_sec;
            !spans
                .iter()
                .any(|&(span_start, span_end)| start >= span_start && end <= span_end)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> TranscriptSegment {
        TranscriptSegment::new(start, end, text)
    }

    #[test]
    fn canned_outros_are_dropped() {
        assert!(is_hallucination("Hãy subscribe cho kênh Ghiền Mì Gõ để không bỏ lỡ!"));
        assert!(is_hallucination("Đăng ký kênh để xem thêm video nhé"));
        assert!(is_hallucination("Thanks for watching!"));
        assert!(is_hallucination("Subtitles by the Amara.org community"));
        assert!(is_hallucination("Hẹn gặp lại các bạn"));
    }

    #[test]
    fn empty_and_punctuation_only_count_as_hallucination() {
        assert!(is_hallucination(""));
        assert!(is_hallucination("   "));
        assert!(is_hallucination("..."));
        assert!(is_hallucination("—"));
    }

    #[test]
    fn real_speech_containing_the_words_survives() {
        // The whole-string match is what protects these.
        assert!(!is_hallucination(
            "Mình nghĩ nên subscribe cho kênh nội bộ của team trước khi deploy"
        ));
        assert!(!is_hallucination("Chào mọi người, hôm nay mình review sprint backlog."));
        assert!(!is_hallucination("Ticket này đang bị block ở phần authentication."));
    }

    #[test]
    fn adjacent_sentence_repeat_collapses_into_one() {
        let segments = vec![
            seg(0.0, 3.0, "Chúng ta cần review lại toàn bộ authentication flow"),
            seg(3.5, 6.0, "Chúng ta cần review lại toàn bộ authentication flow"),
            seg(6.2, 9.0, "Chúng ta cần review lại toàn bộ authentication flow"),
        ];
        let kept = drop_hallucinations(segments);
        assert_eq!(kept.len(), 1);
        // The survivor is stretched over the repeats it absorbed.
        assert_eq!(kept[0].start_sec, 0.0);
        assert_eq!(kept[0].end_sec, 9.0);
    }

    #[test]
    fn distant_repeat_is_dropped_but_not_stretched_over_the_gap() {
        // Consecutive identical sentences are always collapsed — "back-to-back"
        // means adjacent in the segment list, not adjacent in time. What the
        // adjacency rule gates is only whether the survivor's end_sec is
        // stretched: claiming the speaker held one line across a minute of
        // silence is worse than leaving the gap.
        let segments = vec![
            seg(0.0, 3.0, "Chúng ta cần review lại authentication flow"),
            seg(65.0, 68.0, "Chúng ta cần review lại authentication flow"),
        ];
        let kept = drop_hallucinations(segments);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].end_sec, 3.0, "must not stretch across the silence");
    }

    #[test]
    fn short_utterances_may_repeat() {
        let segments = vec![
            seg(0.0, 0.5, "vâng"),
            seg(0.6, 1.0, "vâng"),
            seg(1.1, 1.5, "ừ"),
            seg(1.6, 2.0, "ừ"),
        ];
        let kept = drop_hallucinations(segments);
        assert_eq!(kept.len(), 4, "people really do say these twice in a row");
    }

    #[test]
    fn silent_spans_finds_a_trailing_pause() {
        let sample_rate = 16_000;
        // 1 s of speech-ish noise, then 3 s of silence.
        let mut audio = vec![0.0f32; sample_rate as usize * 4];
        for (index, sample) in audio.iter_mut().take(sample_rate as usize).enumerate() {
            *sample = if index % 2 == 0 { 0.3 } else { -0.3 };
        }
        let spans = silent_spans(&audio, sample_rate, HALLUCINATION_SILENCE_SEC);
        assert_eq!(spans.len(), 1);
        assert!((spans[0].0 - 1.0).abs() < 0.15, "span starts near 1 s: {spans:?}");
        assert!((spans[0].1 - 4.0).abs() < 0.15, "span ends near 4 s: {spans:?}");
    }

    #[test]
    fn short_pauses_are_not_reported() {
        let sample_rate = 16_000;
        // A 0.5 s gap is a breath, not grounds for dropping a segment.
        let mut audio = vec![0.3f32; sample_rate as usize * 3];
        let gap = (sample_rate as usize)..(sample_rate as usize + sample_rate as usize / 2);
        for sample in &mut audio[gap] {
            *sample = 0.0;
        }
        let spans = silent_spans(&audio, sample_rate, HALLUCINATION_SILENCE_SEC);
        assert!(spans.is_empty(), "expected no long spans, got {spans:?}");
    }

    #[test]
    fn segments_inside_silence_are_dropped_but_overlapping_ones_kept() {
        let spans = [(10.0, 15.0)];
        let segments = vec![
            // Invented entirely inside the pause.
            seg(111.0, 114.0, "Hãy đăng ký kênh"),
            // Trailing off into the pause: real speech.
            seg(109.0, 111.0, "Vậy thì mình chốt như thế nhé"),
            // Well outside it.
            seg(120.0, 122.0, "Tiếp theo là phần deploy"),
        ];
        let kept = drop_segments_in_silence(segments, &spans, 100.0);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|s| !s.text.contains("đăng ký")));
    }
}
