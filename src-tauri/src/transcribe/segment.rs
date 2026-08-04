//! Transcript segments and the transcript's on-disk line format.
//!
//! Port of `transcriber.TranscriptSegment`, `format_timestamp` and
//! `segments_to_text`. The line format is load-bearing beyond display:
//! `transcript_merge` parses it back to interleave the two recorded tracks, so
//! `[HH:MM:SS] text` has to stay exactly this shape.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
}

impl TranscriptSegment {
    pub fn new(start_sec: f64, end_sec: f64, text: impl Into<String>) -> Self {
        Self {
            start_sec,
            end_sec,
            text: text.into(),
        }
    }

    pub fn format_line(&self) -> String {
        format!("[{}] {}", format_timestamp(self.start_sec), self.text.trim())
    }
}

/// `HH:MM:SS`, clamped at zero.
///
/// Truncates rather than rounds, matching Python's `int(seconds)`, so a segment
/// starting at 59.9 s reads `00:00:59` in both apps.
pub fn format_timestamp(seconds: f64) -> String {
    let total = if seconds.is_finite() && seconds > 0.0 {
        seconds as u64
    } else {
        0
    };
    format!(
        "{:02}:{:02}:{:02}",
        total / 3600,
        (total % 3600) / 60,
        total % 60
    )
}

/// One chunk's segments as transcript text.
///
/// Mirrors Python's `"\n".join(lines).strip() + ("\n" if lines else "")`: the
/// trailing newline is what makes appending consecutive chunks produce one
/// line per segment, and the empty case must stay genuinely empty so a chunk
/// that decoded to silence adds nothing to the file.
pub fn segments_to_text(segments: &[TranscriptSegment]) -> String {
    if segments.is_empty() {
        return String::new();
    }
    let joined = segments
        .iter()
        .map(TranscriptSegment::format_line)
        .collect::<Vec<_>>()
        .join("\n");
    format!("{}\n", joined.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_truncate_and_clamp() {
        assert_eq!(format_timestamp(0.0), "00:00:00");
        assert_eq!(format_timestamp(59.9), "00:00:59");
        assert_eq!(format_timestamp(60.0), "00:01:00");
        assert_eq!(format_timestamp(3661.4), "01:01:01");
        // Negative and non-finite inputs must not panic or wrap.
        assert_eq!(format_timestamp(-5.0), "00:00:00");
        assert_eq!(format_timestamp(f64::NAN), "00:00:00");
    }

    #[test]
    fn line_format_matches_python() {
        let segment = TranscriptSegment::new(4.2, 6.0, "  Chào mọi người  ");
        assert_eq!(segment.format_line(), "[00:00:04] Chào mọi người");
    }

    #[test]
    fn no_segments_produce_empty_text() {
        // The empty case must stay genuinely empty: a chunk that decoded to
        // silence has to add nothing to the file, or the checkpoint's
        // `text_bytes` would drift from what is actually written.
        assert_eq!(segments_to_text(&[]), "");
    }

    #[test]
    fn a_whitespace_only_segment_still_emits_its_timestamp() {
        // Matches the Python app, which produced "[00:00:00]\n" here. It is
        // unreachable in practice — `is_hallucination` treats empty text as
        // filler and drops the segment before this point — but the two apps
        // agreeing matters more than this edge case being pretty.
        let blank = [TranscriptSegment::new(0.0, 1.0, "   ")];
        assert_eq!(segments_to_text(&blank), "[00:00:00]\n");
    }

    #[test]
    fn text_ends_with_single_newline() {
        let segments = [
            TranscriptSegment::new(0.0, 1.0, "một"),
            TranscriptSegment::new(1.0, 2.0, "hai"),
        ];
        assert_eq!(segments_to_text(&segments), "[00:00:00] một\n[00:00:01] hai\n");
    }
}
