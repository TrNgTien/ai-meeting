"""Weaving the two sides of a recorded meeting into one conversation.

The microphone track and the system track are transcribed independently, which
gives two transcripts of the same span of time. Because both recordings were
written against a single t0 (see audio_capture._MonoWavWriter), their timestamps
are directly comparable, so interleaving them by time reconstructs the order the
meeting actually happened in — and, since each line's origin is known, who said
it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Lines produced by TranscriptSegment.format_line: "[HH:MM:SS] spoken text".
_LINE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")

MIC_LABEL = "Me"
SYSTEM_LABEL = "Meeting"


@dataclass(frozen=True)
class Utterance:
    at_sec: float
    speaker: str
    text: str

    def format_line(self) -> str:
        hours, rem = divmod(int(self.at_sec), 3600)
        minutes, seconds = divmod(rem, 60)
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {self.speaker}: {self.text}"


def parse_transcript(path: Path, speaker: str) -> list[Utterance]:
    """Read a timestamped transcript file into utterances.

    Lines that don't carry a timestamp (blank lines, the `===== file =====`
    banners) are skipped rather than guessed at: a line with no time has no
    place in a merge ordered by time.
    """
    utterances: list[Utterance] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return utterances

    for line in text.splitlines():
        match = _LINE.match(line.strip())
        if not match:
            continue
        hours, minutes, seconds, spoken = match.groups()
        spoken = spoken.strip()
        if not spoken:
            continue
        at = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        utterances.append(Utterance(at_sec=float(at), speaker=speaker, text=spoken))
    return utterances


def merge(
    mic: Iterable[Utterance] = (), system: Iterable[Utterance] = ()
) -> list[Utterance]:
    """Interleave both sides by time.

    Ties go to the microphone: when both sides carry the same second, it is
    nearly always the local speaker being echoed back through the meeting app a
    beat later, so putting "Me" first reads the way the exchange happened.
    """
    ordered = [(u.at_sec, 0, u) for u in mic] + [(u.at_sec, 1, u) for u in system]
    ordered.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ordered]


def render(utterances: list[Utterance], *, header: Optional[str] = None) -> str:
    """Merged transcript as text, with a blank line at each change of speaker.

    The blank line is the whole readability win: an unbroken column of
    alternating labels is much harder to follow than visible turns.
    """
    lines: list[str] = []
    if header:
        lines.extend([header, ""])

    previous: Optional[str] = None
    for utterance in utterances:
        if previous is not None and utterance.speaker != previous:
            lines.append("")
        lines.append(utterance.format_line())
        previous = utterance.speaker
    return "\n".join(lines) + ("\n" if lines else "")


def merge_transcript_files(
    *,
    mic_transcript: Optional[Path],
    system_transcript: Optional[Path],
    output_path: Path,
    header: Optional[str] = None,
    mic_label: str = MIC_LABEL,
    system_label: str = SYSTEM_LABEL,
) -> tuple[Path, int]:
    """Merge the two transcripts on disk into `output_path`.

    Returns the path written and how many utterances it holds. Either side may
    be None — a recording with only one usable track still gets a labelled
    transcript, which keeps the output shape the same either way.
    """
    mic = parse_transcript(mic_transcript, mic_label) if mic_transcript else []
    system = parse_transcript(system_transcript, system_label) if system_transcript else []
    merged = merge(mic, system)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(merged, header=header), encoding="utf-8")
    return output_path, len(merged)
