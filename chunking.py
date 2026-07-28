"""Chunked, resumable transcription of long recordings.

A one-hour meeting takes a long time to transcribe on CPU, and the old
one-shot path wrote nothing until the whole file was done — a crash, a quit or
a power loss at minute 55 threw away all of it.

This module splits the audio into chunks and transcribes them one at a time,
appending each finished chunk straight to the transcript file. A small sidecar
checkpoint records how far into the recording that text goes, so re-running the
same file continues from the last line actually written instead of starting
over. The transcript is therefore always readable mid-run, and nothing that has
been transcribed is ever held in memory only.

Chunk boundaries are snapped to the quietest moment near the nominal cut so we
rarely slice a word in half.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable, Optional

import numpy as np

from transcriber import SegmentCallback, TranscriptSegment, segments_to_text

SAMPLE_RATE = 16_000

# Work in 5-minute chunks: long enough that per-chunk model overhead is noise,
# short enough that a crash costs at most a few minutes of recomputation.
DEFAULT_CHUNK_SECONDS = 300.0
# Extra audio decoded past the nominal chunk end, searched for a quiet spot to
# cut on so chunk boundaries rarely land in the middle of a word.
SPLIT_SEARCH_SECONDS = 20.0
# Granularity of the quiet-spot search.
SPLIT_FRAME_SECONDS = 0.1
# Ignore a trailing sliver of audio rather than feeding a near-empty chunk to
# the model.
MIN_TAIL_SECONDS = 0.2

# v3 added transcript_name: a v2 checkpoint points at a transcript named by the
# old scheme, so it is discarded rather than resumed into the wrong file.
CHECKPOINT_VERSION = 3

TRANSCRIPT_SUFFIX = ".txt"

# on_progress(chunk_index, chunks_done, done_seconds, total_seconds|None)
ProgressCallback = Callable[[int, int, float, Optional[float]], None]
# on_text(chunk_transcript) — fired as each chunk is appended to the file
TextCallback = Callable[[str], None]
# transcribe_audio(audio_16k_mono_float32, offset_sec, on_segment=…) -> segments
# The on_segment keyword is optional: engines that cannot produce partial
# results (see WhisperXTranscriber) simply don't take it.
TranscribeAudio = Callable[..., list[TranscriptSegment]]


class TranscriptionCancelled(Exception):
    """Raised when cancel_event is set. Progress is kept in the checkpoint."""


def checkpoint_path(source: Path) -> Path:
    return source.with_name(source.stem + ".transcript.partial.json")


def transcript_path_for(source: Path, stamp: Optional[str] = None) -> Path:
    """Final transcript location: next to the audio it came from.

    Named `<when>-<recording>.txt`: the local time the transcription started,
    first so a folder of transcripts sorts chronologically. Re-running a
    recording therefore keeps the earlier transcript instead of overwriting it.
    A resumed run must reuse the stamp of the file it is continuing — see
    resolve_transcript_path().
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return source.with_name(f"{stamp}-{source.stem}{TRANSCRIPT_SUFFIX}")


def resolve_transcript_path(
    source: Path, engine_key: str, chunk_sec: float = DEFAULT_CHUNK_SECONDS
) -> Path:
    """The transcript file this run will write.

    A run that resumes has to append to the file the earlier run started, not
    mint a fresh stamp — otherwise the checkpoint would describe text that is
    not in the file, and the run would silently restart from silence.
    """
    state = load_checkpoint(source, source_fingerprint(source, engine_key, chunk_sec))
    if state is not None and state.chunks_done and state.transcript_name:
        started = source.with_name(state.transcript_name)
        if started.exists():
            return started
    return transcript_path_for(source)


# --- audio decoding -----------------------------------------------------------


def audio_duration(source: Path) -> Optional[float]:
    """Duration in seconds via ffprobe, or None if it can't be determined.

    None is not fatal: the chunk loop also stops when a decode comes back
    shorter than requested, which is what happens at the end of the file.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(source),
            ],
            capture_output=True,
            check=True,
        )
        value = float(out.stdout.decode().strip())
        return value if value > 0 else None
    except Exception:
        return None


def decode_range(source: Path, start_sec: float, duration_sec: float) -> np.ndarray:
    """Decode [start, start+duration) as mono 16 kHz float32 via ffmpeg.

    Decoding a range at a time keeps memory flat regardless of recording
    length, and avoids loading (and resampling) hours of audio up front.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-ss", f"{max(0.0, start_sec):.3f}",
        "-t", f"{max(0.0, duration_sec):.3f}",
        "-i", str(source),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore").strip().splitlines()
        detail = stderr[-1] if stderr else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed to decode {source.name}: {detail}") from exc
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def find_split_index(audio: np.ndarray, search_from: int) -> int:
    """Index of the quietest short frame at/after search_from.

    Cutting there instead of at a fixed offset keeps chunk boundaries out of
    the middle of words most of the time.
    """
    if search_from >= audio.size:
        return audio.size

    frame = max(1, int(SPLIT_FRAME_SECONDS * SAMPLE_RATE))
    window = audio[search_from:]
    frame_count = window.size // frame
    if frame_count <= 0:
        return audio.size

    frames = window[: frame_count * frame].reshape(frame_count, frame)
    energy = np.abs(frames).mean(axis=1)
    quietest = int(energy.argmin())
    # Cut in the middle of the quiet frame, so neither side clips speech.
    return search_from + quietest * frame + frame // 2


# --- checkpoint ---------------------------------------------------------------


@dataclass
class Checkpoint:
    """Where to resume, and how much of the transcript file is trustworthy.

    text_bytes is the size the transcript had after the last fully written
    chunk. Anything past it is the debris of a chunk that was interrupted
    mid-write, and gets truncated away on resume.

    transcript_name is the file that text lives in — stored as a bare name, so
    moving the recording and its transcript together doesn't break resuming.
    """

    fingerprint: str
    next_start_sec: float
    chunks_done: int
    text_bytes: int
    transcript_name: str = ""

    def to_json(self, duration_sec: Optional[float]) -> str:
        return json.dumps(
            {
                "version": CHECKPOINT_VERSION,
                "fingerprint": self.fingerprint,
                "duration_sec": duration_sec,
                "next_start_sec": self.next_start_sec,
                "chunks_done": self.chunks_done,
                "text_bytes": self.text_bytes,
                "transcript_name": self.transcript_name,
            },
            ensure_ascii=False,
        )


def source_fingerprint(source: Path, engine_key: str, chunk_sec: float) -> str:
    """Identity of "this file transcribed this way".

    Any change to the audio file, the engine/model, the language or the chunk
    size makes an existing checkpoint meaningless, so it is discarded rather
    than merged into a transcript produced with different settings.
    """
    stat = source.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}:{engine_key}:{chunk_sec:g}"


def load_checkpoint(source: Path, fingerprint: str) -> Optional[Checkpoint]:
    """Read a resumable checkpoint, or None if absent/stale/corrupt."""
    path = checkpoint_path(source)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if data.get("version") != CHECKPOINT_VERSION:
        return None
    if data.get("fingerprint") != fingerprint:
        return None

    try:
        return Checkpoint(
            fingerprint=fingerprint,
            next_start_sec=float(data["next_start_sec"]),
            chunks_done=int(data["chunks_done"]),
            text_bytes=int(data["text_bytes"]),
            transcript_name=str(data.get("transcript_name", "")),
        )
    except Exception:
        # A truncated/garbled checkpoint is worth less than the time it would
        # cost to debug: redo the file rather than emit a corrupt transcript.
        return None


def _format_clock(seconds: float) -> str:
    """HH:MM:SS — matches the segment timestamps in the transcript body."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _format_elapsed(seconds: float) -> str:
    """Wall-clock durations, read at a glance: 45s, 4m32s, 3h12m."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


def summary_line(
    audio_sec: Optional[float], run_sec: float, resumed_from_sec: float
) -> str:
    """The footer closing a finished transcript.

    Records how much audio it covers and how long the machine took, which is
    the question anyone asks after leaving a long recording to run. Only the
    time spent in *this* run is knowable — earlier runs' wall time is not
    carried in the checkpoint — so a resumed file says so rather than
    reporting a total it cannot vouch for.
    """
    parts = []
    if audio_sec:
        parts.append(f"{_format_clock(audio_sec)} of audio")
    if resumed_from_sec > 0:
        parts.append(
            f"transcribed in {_format_elapsed(run_sec)} this run "
            f"(resumed from {_format_clock(resumed_from_sec)})"
        )
    else:
        parts.append(f"transcribed in {_format_elapsed(run_sec)}")
    return "----- Complete: " + " · ".join(parts) + " -----\n"


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def save_checkpoint(
    source: Path, checkpoint: Checkpoint, duration_sec: Optional[float]
) -> None:
    """Write the checkpoint atomically so a crash mid-write can't corrupt it."""
    path = checkpoint_path(source)
    tmp = _tmp_path(path)
    tmp.write_text(checkpoint.to_json(duration_sec), encoding="utf-8")
    os.replace(tmp, path)


def clear_checkpoint(source: Path) -> None:
    path = checkpoint_path(source)
    for candidate in (path, _tmp_path(path)):
        try:
            candidate.unlink()
        except OSError:
            pass


# --- orchestration ------------------------------------------------------------


def transcribe_chunked(
    source: Path,
    *,
    transcribe_audio: TranscribeAudio,
    engine_key: str,
    chunk_sec: float = DEFAULT_CHUNK_SECONDS,
    output_path: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
    on_text: Optional[TextCallback] = None,
    on_segment: Optional[SegmentCallback] = None,
    cancel_event: Optional[Event] = None,
) -> str:
    """Transcribe `source` chunk by chunk, streaming the text to disk.

    on_segment, if given and supported by the engine, fires for each segment
    the model finishes *within* the chunk in flight. Those segments are a
    preview only — nothing is written to disk until the chunk completes — but
    they are what makes text appear seconds into a run rather than minutes.

    Every chunk is appended to the transcript file and flushed before the
    checkpoint records it, so the file on disk is always a complete prefix of
    the transcript — readable while the run is still going. Interrupting (crash,
    Stop, power loss) costs at most the chunk in flight; the next run truncates
    any half-written tail and continues from there.

    Returns the full transcript text. Raises TranscriptionCancelled if
    cancel_event is set.
    """
    fingerprint = source_fingerprint(source, engine_key, chunk_sec)
    duration = audio_duration(source)
    stream_segments = on_segment is not None and _accepts_on_segment(transcribe_audio)
    out_path = (
        output_path
        if output_path is not None
        else resolve_transcript_path(source, engine_key, chunk_sec)
    )

    state = _resume_or_restart(source, out_path, fingerprint)
    run_started = time.monotonic()
    resumed_from = state.next_start_sec
    if state.chunks_done:
        # Hand the caller what was transcribed in earlier runs, so a resumed
        # file reads as one transcript rather than starting mid-meeting.
        if on_text is not None:
            resumed = out_path.read_text(encoding="utf-8")
            if resumed:
                on_text(resumed)
        if on_progress is not None:
            on_progress(state.chunks_done, state.chunks_done, state.next_start_sec, duration)

    while duration is None or state.next_start_sec < duration - MIN_TAIL_SECONDS:
        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelled()

        start = state.next_start_sec
        audio = decode_range(source, start, chunk_sec + SPLIT_SEARCH_SECONDS)
        if audio.size <= int(MIN_TAIL_SECONDS * SAMPLE_RATE):
            break

        nominal_end = int(chunk_sec * SAMPLE_RATE)
        if audio.size > nominal_end:
            cut = min(find_split_index(audio, nominal_end), audio.size)
        else:
            # Short read: this is the tail of the recording.
            cut = audio.size

        if stream_segments:
            segments = transcribe_audio(audio[:cut], start, on_segment=on_segment)
        else:
            segments = transcribe_audio(audio[:cut], start)
        text = segments_to_text(segments)

        # Order matters: the text hits the disk first, and only a chunk the
        # checkpoint has accounted for is treated as done. A crash in between
        # leaves a tail past text_bytes, which the next run truncates and
        # redoes — never a silently missing stretch of the meeting.
        written = _append_text(out_path, text)
        state.chunks_done += 1
        state.next_start_sec = start + cut / SAMPLE_RATE
        state.text_bytes = written
        save_checkpoint(source, state, duration)

        if on_text is not None and text:
            on_text(text)
        if on_progress is not None:
            on_progress(state.chunks_done, state.chunks_done, state.next_start_sec, duration)

        if cut >= audio.size and (
            duration is None or state.next_start_sec >= duration - MIN_TAIL_SECONDS
        ):
            break

    # Only a run that reached the end writes the footer, and only after the
    # last chunk is on disk — so its presence means "this transcript is whole",
    # and a cancelled run leaves a file the next run can still resume into.
    summary = summary_line(
        duration if duration is not None else state.next_start_sec,
        time.monotonic() - run_started,
        resumed_from,
    )
    _append_text(out_path, "\n" + summary)
    if on_text is not None:
        on_text("\n" + summary)

    clear_checkpoint(source)
    return out_path.read_text(encoding="utf-8")


def _accepts_on_segment(transcribe_audio: TranscribeAudio) -> bool:
    """Whether this engine can report segments before the chunk is finished.

    Asked once per run rather than caught as a TypeError per chunk, which
    would just as happily swallow a TypeError raised inside the model.
    """
    try:
        params = inspect.signature(transcribe_audio).parameters
    except (TypeError, ValueError):
        return False
    if "on_segment" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _resume_or_restart(source: Path, out_path: Path, fingerprint: str) -> Checkpoint:
    """Set up the transcript file for this run and say where to start.

    Resuming trims the transcript back to the last checkpointed chunk. If the
    transcript is missing or shorter than the checkpoint claims (deleted or
    edited between runs), the checkpoint is meaningless and the file starts
    over from silence rather than resuming into a gap.
    """
    state = load_checkpoint(source, fingerprint)
    if state is not None and state.chunks_done:
        try:
            if out_path.stat().st_size >= state.text_bytes:
                with out_path.open("r+b") as handle:
                    handle.truncate(state.text_bytes)
                state.transcript_name = out_path.name
                return state
        except OSError:
            pass

    clear_checkpoint(source)
    out_path.write_bytes(b"")
    return Checkpoint(
        fingerprint=fingerprint,
        next_start_sec=0.0,
        chunks_done=0,
        text_bytes=0,
        transcript_name=out_path.name,
    )


def _append_text(out_path: Path, text: str) -> int:
    """Append one chunk's transcript and force it to disk; return the new size.

    fsync is the point of the exercise: without it a crash could lose text the
    checkpoint has already recorded as written.
    """
    with out_path.open("ab") as handle:
        if text:
            handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
        return handle.tell()


def resumable_seconds(source: Path, engine_key: str, chunk_sec: float = DEFAULT_CHUNK_SECONDS) -> float:
    """Seconds of audio already transcribed in a usable checkpoint (0 if none).

    Used only to tell the user "resuming at 25:00" before work restarts.
    """
    try:
        fingerprint = source_fingerprint(source, engine_key, chunk_sec)
    except OSError:
        return 0.0
    state = load_checkpoint(source, fingerprint)
    return state.next_start_sec if state else 0.0
