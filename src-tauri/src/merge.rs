//! Weaving the two sides of a recorded meeting into one conversation.
//!
//! Port of `transcript_merge.py`.
//!
//! The microphone track and the system track are transcribed independently,
//! which gives two transcripts of the same span of time. Because both recordings
//! were written against a single `t0` (see `audio::wav_writer`), their timestamps
//! are directly comparable, so interleaving them by time reconstructs the order
//! the meeting actually happened in — and, since each line's origin is known, who
//! said it.
//!
//! Transcribing the tracks separately rather than mixing them is what makes the
//! attribution possible at all, and it also stops the local speaker's voice —
//! bleeding from the speakers back into the microphone — from being transcribed
//! twice.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result};
use regex::Regex;

pub const MIC_LABEL: &str = "Me";
pub const SYSTEM_LABEL: &str = "Meeting";

/// Lines produced by `TranscriptSegment::format_line`: `[HH:MM:SS] spoken text`.
static LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$").expect("transcript line pattern")
});

#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    pub at_sec: f64,
    pub speaker: String,
    pub text: String,
}

impl Utterance {
    pub fn format_line(&self) -> String {
        let total = if self.at_sec.is_finite() && self.at_sec > 0.0 {
            self.at_sec as u64
        } else {
            0
        };
        format!(
            "[{:02}:{:02}:{:02}] {}: {}",
            total / 3600,
            (total % 3600) / 60,
            total % 60,
            self.speaker,
            self.text
        )
    }
}

/// Read a timestamped transcript file into utterances.
///
/// Lines that don't carry a timestamp (blank lines, the `===== file =====`
/// banners, the `----- Complete: … -----` footer) are skipped rather than
/// guessed at: a line with no time has no place in a merge ordered by time.
///
/// An unreadable file yields no utterances rather than an error — one missing
/// side must still produce a labelled transcript for the other.
pub fn parse_transcript(path: &Path, speaker: &str) -> Vec<Utterance> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    parse_transcript_text(&text, speaker)
}

fn parse_transcript_text(text: &str, speaker: &str) -> Vec<Utterance> {
    let mut utterances = Vec::new();
    for line in text.lines() {
        let Some(captures) = LINE.captures(line.trim()) else {
            continue;
        };
        let spoken = captures[4].trim();
        if spoken.is_empty() {
            continue;
        }
        let hours: u64 = captures[1].parse().unwrap_or(0);
        let minutes: u64 = captures[2].parse().unwrap_or(0);
        let seconds: u64 = captures[3].parse().unwrap_or(0);
        utterances.push(Utterance {
            at_sec: (hours * 3600 + minutes * 60 + seconds) as f64,
            speaker: speaker.to_string(),
            text: spoken.to_string(),
        });
    }
    utterances
}

/// Interleave both sides by time.
///
/// Ties go to the microphone: when both sides carry the same second, it is
/// nearly always the local speaker being echoed back through the meeting app a
/// beat later, so putting "Me" first reads the way the exchange happened.
pub fn merge(mic: Vec<Utterance>, system: Vec<Utterance>) -> Vec<Utterance> {
    let mut ordered: Vec<(f64, u8, Utterance)> = mic
        .into_iter()
        .map(|u| (u.at_sec, 0u8, u))
        .chain(system.into_iter().map(|u| (u.at_sec, 1u8, u)))
        .collect();

    // Stable, so utterances from one side keep their original order within a
    // second — the transcripts are already in order and re-sorting them would
    // scramble a fast exchange.
    ordered.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    ordered.into_iter().map(|(_, _, u)| u).collect()
}

/// Merged transcript as text, with a blank line at each change of speaker.
///
/// The blank line is the whole readability win: an unbroken column of
/// alternating labels is much harder to follow than visible turns.
pub fn render(utterances: &[Utterance], header: Option<&str>) -> String {
    let mut lines: Vec<String> = Vec::new();
    if let Some(header) = header {
        lines.push(header.to_string());
        lines.push(String::new());
    }

    let mut previous: Option<&str> = None;
    for utterance in utterances {
        if previous.is_some_and(|prev| prev != utterance.speaker) {
            lines.push(String::new());
        }
        lines.push(utterance.format_line());
        previous = Some(&utterance.speaker);
    }

    if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    }
}

/// Merge the two transcripts on disk into `output_path`.
///
/// Returns how many utterances it holds. Either side may be `None` — a recording
/// with only one usable track still gets a labelled transcript, which keeps the
/// output shape the same either way.
pub fn merge_transcript_files(
    mic_transcript: Option<&Path>,
    system_transcript: Option<&Path>,
    output_path: &Path,
    header: Option<&str>,
) -> Result<usize> {
    let mic = mic_transcript
        .map(|path| parse_transcript(path, MIC_LABEL))
        .unwrap_or_default();
    let system = system_transcript
        .map(|path| parse_transcript(path, SYSTEM_LABEL))
        .unwrap_or_default();

    let merged = merge(mic, system);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    fs::write(output_path, render(&merged, header))
        .with_context(|| format!("cannot write {}", output_path.display()))?;
    Ok(merged.len())
}

/// Where a recording's merged conversation goes: `<stem>-conversation.txt`.
pub fn conversation_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}-conversation.txt"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utterance(at: f64, speaker: &str, text: &str) -> Utterance {
        Utterance {
            at_sec: at,
            speaker: speaker.to_string(),
            text: text.to_string(),
        }
    }

    #[test]
    fn only_timestamped_lines_are_parsed() {
        let text = "\
===== standup.m4a =====
[00:00:04] Chào mọi người
not a transcript line
[00:00:11] Ticket đầu tiên đang bị block

----- Complete: 00:01:30 of audio · transcribed in 15s -----
";
        let parsed = parse_transcript_text(text, MIC_LABEL);
        assert_eq!(parsed.len(), 2, "banners and footers have no place in a merge");
        assert_eq!(parsed[0].at_sec, 4.0);
        assert_eq!(parsed[0].text, "Chào mọi người");
        assert_eq!(parsed[1].at_sec, 11.0);
    }

    #[test]
    fn timestamps_beyond_an_hour_parse() {
        let parsed = parse_transcript_text("[01:02:03] muộn rồi", MIC_LABEL);
        assert_eq!(parsed[0].at_sec, 3723.0);
    }

    #[test]
    fn a_timestamp_with_no_words_is_skipped() {
        assert!(parse_transcript_text("[00:00:04]   ", MIC_LABEL).is_empty());
    }

    #[test]
    fn ties_put_the_microphone_first() {
        // The local speaker echoed back through the meeting app a beat later.
        let mic = vec![utterance(10.0, MIC_LABEL, "mình nói trước")];
        let system = vec![utterance(10.0, SYSTEM_LABEL, "echo")];
        let merged = merge(mic, system);
        assert_eq!(merged[0].speaker, MIC_LABEL);
        assert_eq!(merged[1].speaker, SYSTEM_LABEL);
    }

    #[test]
    fn both_sides_interleave_by_time() {
        let mic = vec![utterance(0.0, MIC_LABEL, "a"), utterance(20.0, MIC_LABEL, "c")];
        let system = vec![
            utterance(10.0, SYSTEM_LABEL, "b"),
            utterance(30.0, SYSTEM_LABEL, "d"),
        ];
        let merged = merge(mic, system);
        let texts: Vec<&str> = merged.iter().map(|u| u.text.as_str()).collect();
        assert_eq!(texts, ["a", "b", "c", "d"]);
    }

    #[test]
    fn same_side_lines_within_one_second_keep_their_order() {
        let mic = vec![
            utterance(5.0, MIC_LABEL, "first"),
            utterance(5.0, MIC_LABEL, "second"),
            utterance(5.0, MIC_LABEL, "third"),
        ];
        let merged = merge(mic, Vec::new());
        let texts: Vec<&str> = merged.iter().map(|u| u.text.as_str()).collect();
        assert_eq!(texts, ["first", "second", "third"]);
    }

    #[test]
    fn render_breaks_a_line_at_each_change_of_speaker() {
        let merged = vec![
            utterance(0.0, MIC_LABEL, "chào"),
            utterance(2.0, MIC_LABEL, "mọi người"),
            utterance(5.0, SYSTEM_LABEL, "hello"),
        ];
        let text = render(&merged, None);
        assert_eq!(
            text,
            "[00:00:00] Me: chào\n[00:00:02] Me: mọi người\n\n[00:00:05] Meeting: hello\n"
        );
    }

    #[test]
    fn render_puts_a_blank_line_under_the_header() {
        let merged = vec![utterance(0.0, MIC_LABEL, "chào")];
        let text = render(&merged, Some("# Meeting recorded 2026-07-30 14:25 (1m30s)"));
        assert!(text.starts_with("# Meeting recorded 2026-07-30 14:25 (1m30s)\n\n[00:00:00] Me:"));
    }

    #[test]
    fn nothing_to_merge_renders_nothing() {
        assert_eq!(render(&[], None), "");
    }

    #[test]
    fn one_missing_side_still_produces_a_labelled_transcript() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mic = dir.path().join("mic.txt");
        fs::write(&mic, "[00:00:01] chỉ có mình thôi\n")?;
        let out = conversation_path(dir.path(), "meeting-20260730-142530");

        let count = merge_transcript_files(Some(&mic), None, &out, None)?;

        assert_eq!(count, 1);
        assert_eq!(fs::read_to_string(&out)?, "[00:00:01] Me: chỉ có mình thôi\n");
        assert!(out.file_name().unwrap().to_string_lossy().ends_with("-conversation.txt"));
        Ok(())
    }

    #[test]
    fn a_missing_file_is_treated_as_an_empty_side() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let out = dir.path().join("merged.txt");
        let count = merge_transcript_files(
            Some(&dir.path().join("does-not-exist.txt")),
            None,
            &out,
            None,
        )?;
        assert_eq!(count, 0);
        Ok(())
    }
}
