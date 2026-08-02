"""Local Whisper transcription via the official openai-whisper package."""

from __future__ import annotations

import warnings

# Suppress torchcodec / pyannote audio decoder warnings (not needed for Silero VAD / soundfile)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")

import contextlib
import hashlib
import io
import os
import re
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import whisper


class DownloadCancelled(Exception):
    """Raised by ensure_model_downloaded() when cancel_event is set mid-download."""

FINAL_MODEL = "large-v3"
# Multilingual openai-whisper checkpoints selectable in the UI for the
# "vi+en" / "en" / "auto" final pass (pure "vi" always uses PhoWhisper-large
# instead, see WhisperXTranscriber). Ordered roughly fastest -> most accurate.
FINAL_MODEL_OPTIONS = ["small", "medium", "large-v2", "large-v3", "large-v3-turbo"]
DEVICE = "cpu"
# openai-whisper only supports fp16 on CUDA; keep it off for CPU inference.
USE_FP16 = False

# --- hallucination control ----------------------------------------------------

# Whisper was trained on YouTube captions, so over silence or background noise
# it does not stay quiet — it emits the most likely thing to follow, which for
# Vietnamese audio is a channel outro ("Hãy subscribe cho kênh Ghiền Mì Gõ…").
# Worse, with the default condition_on_previous_text=True that invented line
# becomes the prompt for the next window and the model repeats it once per 30 s
# decode window for the rest of the recording.
#
# These options are shared by the openai-whisper and mlx-whisper engines, whose
# transcribe() signatures match. Dropping the previous-text conditioning is the
# fix for the repeat loop; the thresholds make the model bail out of a window
# it is not confident about instead of guessing.
DECODE_OPTIONS = {
    "condition_on_previous_text": False,
    # Below this average token logprob the window is treated as a failed decode.
    "logprob_threshold": -1.0,
    # A window whose gzip ratio exceeds this is degenerate repetition.
    "compression_ratio_threshold": 2.4,
    # Above this no-speech probability the window is emitted as silence.
    "no_speech_threshold": 0.6,
    # Needs word_timestamps: drops words whose timings sit inside a gap of
    # silence this long, which is what an invented outro looks like.
    "word_timestamps": True,
    "hallucination_silence_threshold": 2.0,
}

# Stock phrases Whisper falls back on when there is nothing to transcribe.
# Matched case-insensitively against the whole segment, after stripping
# punctuation, so a segment is only dropped when it is *entirely* boilerplate —
# the same words spoken inside a real sentence survive.
_HALLUCINATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Vietnamese YouTube outros
        r"^h[ãa]y subscribe cho k[êe]nh\b.*",
        r".*\bghi[eề]n m[ìi] g[õo]\b.*",
        r"^đăng k[ýy] k[êe]nh\b.*",
        r"^c[ảa]m [ơo]n c[áa]c b[ạa]n đ[ãa] (theo d[õo]i|xem|l[ắa]ng nghe)\b.*",
        r"^h[ẹe]n g[ặa]p l[ạa]i c[áa]c b[ạa]n\b.*",
        r"^c[áa]c b[ạa]n c[óo] th[ểe] nh[ậa]n th[êe]m nhi[ềe]u th[ôo]ng tin\b.*",
        r".*trong ph[ầa]n b[ìi]nh lu[ậa]n\s*$",
        r"^ch[úu]c c[áa]c b[ạa]n (xem )?(video )?vui v[ẻe]\b.*",
        # English equivalents, which show up on mixed-language audio
        r"^thanks? (you )?for watching\b.*",
        r"^(please )?(don't forget to )?subscribe\b.*",
        r"^(subtitles?|amara)\b.*",
    )
]

# Punctuation and whitespace to ignore when matching the patterns above.
_PUNCT = re.compile(r"[.,!?…\-–—\"'()\[\]]+")


def is_hallucination(text: str) -> bool:
    """True if a segment is one of Whisper's canned silence fillers."""
    normalized = _PUNCT.sub(" ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return True
    return any(pattern.fullmatch(normalized) for pattern in _HALLUCINATION_PATTERNS)


def drop_hallucinations(
    segments: list["TranscriptSegment"],
) -> list["TranscriptSegment"]:
    """Strip canned filler and collapse the repeat loops it causes.

    Beyond the known phrases, a *sentence* repeated back-to-back is dropped: a
    speaker does not say the same eight words verbatim three windows running,
    but a model that has locked onto its own output does. Short utterances are
    left alone — "ừ", "vâng", "okay" really are said twice in a row, and a run
    of them is speech rather than a loop.
    """
    kept: list[TranscriptSegment] = []
    for segment in segments:
        if is_hallucination(segment.text):
            continue
        text = segment.text.strip()
        if kept and kept[-1].text.strip() == text and _is_sentence(text):
            # Only stretch the surviving copy over a repeat that butts up
            # against it; a duplicate a minute later is silence in between,
            # and claiming the speaker held that line the whole time is worse
            # than leaving the gap.
            if segment.start_sec - kept[-1].end_sec <= 1.0:
                kept[-1].end_sec = max(kept[-1].end_sec, segment.end_sec)
            continue
        kept.append(segment)
    return kept


# Word count above which a verbatim back-to-back repeat is a decode loop
# rather than something a person said twice.
_REPEAT_MIN_WORDS = 4


def _is_sentence(text: str) -> bool:
    return len(text.split()) >= _REPEAT_MIN_WORDS

# progress_cb(model_name, downloaded_bytes, total_bytes). total_bytes may be 0
# when the server does not report a Content-Length.
ProgressCallback = Callable[[str, int, int], None]


def _whisper_cache_dir() -> str:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def whisper_model_path(name: str) -> Optional[Path]:
    """Local cache path for an openai-whisper checkpoint, or None if unknown."""
    url = whisper._MODELS.get(name)
    if url is None:
        return None
    return Path(_whisper_cache_dir()) / os.path.basename(url)


def is_model_downloaded(name: str) -> bool:
    path = whisper_model_path(name)
    return path is not None and path.is_file()


def model_size_on_disk(name: str) -> int:
    """Cached file size in bytes, or 0 if the model isn't downloaded."""
    path = whisper_model_path(name)
    if path is not None and path.is_file():
        return path.stat().st_size
    return 0


def delete_model(name: str) -> bool:
    """Remove a cached openai-whisper checkpoint from disk.

    Only deletes the local file; the model re-downloads automatically the
    next time it's selected and used. Returns True if a file was removed.
    """
    path = whisper_model_path(name)
    if path is not None and path.is_file():
        path.unlink()
        return True
    return False


def list_downloaded_whisper_models() -> list[str]:
    """Names of every openai-whisper model with a cached checkpoint on disk.

    Scans the whisper cache dir against whisper's known model registry, so it
    also picks up models downloaded outside of FINAL_MODEL_OPTIONS (e.g. via
    the CLI or an older config).
    """
    cache_dir = Path(_whisper_cache_dir())
    if not cache_dir.is_dir():
        return []
    return [
        name
        for name, url in whisper._MODELS.items()
        if (cache_dir / os.path.basename(url)).is_file()
    ]


def ensure_model_downloaded(
    name: str,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Download the given Whisper model to the cache, reporting progress.

    Mirrors ``whisper._download`` but streams progress through ``progress_cb``
    instead of a terminal tqdm bar. After this returns, ``whisper.load_model``
    finds the cached file and loads without re-downloading.

    If `cancel_event` is set while the download is in progress, it stops
    early, deletes the partial file, and raises `DownloadCancelled`.
    """
    url = whisper._MODELS.get(name)
    if url is None:
        # Not a known model name (e.g. a local checkpoint path); nothing to do.
        return

    root = _whisper_cache_dir()
    os.makedirs(root, exist_ok=True)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    # A file at the final path is written atomically below (or by whisper) only
    # after checksum verification, so treat its presence as "already cached".
    # whisper.load_model re-verifies on load, so corruption is still caught.
    if os.path.isfile(download_target):
        size = os.path.getsize(download_target)
        if progress_cb:
            progress_cb(name, size, size)
        return

    part_target = download_target + ".part"
    with urllib.request.urlopen(url) as source, open(part_target, "wb") as output:
        total = int(source.info().get("Content-Length") or 0)
        downloaded = 0
        if progress_cb:
            progress_cb(name, 0, total)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                output.close()
                os.remove(part_target)
                raise DownloadCancelled(f"Download of '{name}' was cancelled")
            buffer = source.read(1 << 20)  # 1 MiB blocks
            if not buffer:
                break
            output.write(buffer)
            downloaded += len(buffer)
            if progress_cb:
                progress_cb(name, downloaded, total)

    with open(part_target, "rb") as f:
        model_bytes = f.read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        os.remove(part_target)
        raise RuntimeError(
            f"Downloaded {name} but the SHA256 checksum does not match; please retry."
        )
    os.replace(part_target, download_target)


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str

    def format_line(self) -> str:
        return f"[{format_timestamp(self.start_sec)}] {self.text.strip()}"


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# --- live segment streaming ---------------------------------------------------

# on_segment(segment) — fired as the engine finishes each decode window, while
# the chunk it belongs to is still being transcribed.
SegmentCallback = Callable[[TranscriptSegment], None]

# Whisper decodes a chunk in ~30s windows and has a finished segment long
# before the call returns, but neither openai-whisper nor mlx-whisper exposes a
# callback for it. Both do print each segment when verbose=True, in this exact
# shape, so tapping that print is the one way to see inside a decode that runs
# for minutes.
_VERBOSE_SEGMENT_LINE = re.compile(r"^\[([\d:.]+) --> ([\d:.]+)\]\s?(.*)$")


def _parse_clock(value: str) -> float:
    """Seconds from whisper's `[hh:]mm:ss.mmm` verbose timestamps."""
    parts = value.split(":")
    seconds = float(parts[-1])
    if len(parts) > 1:
        seconds += int(parts[-2]) * 60
    if len(parts) > 2:
        seconds += int(parts[-3]) * 3600
    return seconds


class _SegmentPrintTap(io.TextIOBase):
    """A stdout stand-in that turns verbose segment prints into callbacks.

    Anything printed that is *not* a segment line is passed through to the real
    stdout, so warnings from the engine are not swallowed along the way.
    """

    def __init__(
        self,
        on_segment: SegmentCallback,
        offset_sec: float,
        passthrough: Optional[io.TextIOBase],
    ) -> None:
        self._on_segment = on_segment
        self._offset = offset_sec
        self._passthrough = passthrough
        self._buffer = ""
        self._last_text = ""

    def write(self, data: str) -> int:
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(data)

    def flush(self) -> None:
        if self._passthrough is not None:
            try:
                self._passthrough.flush()
            except Exception:
                pass

    def close_out(self) -> None:
        """Emit whatever was written without a trailing newline."""
        if self._buffer:
            line, self._buffer = self._buffer, ""
            self._emit(line)

    def _emit(self, line: str) -> None:
        match = _VERBOSE_SEGMENT_LINE.match(line.strip())
        if match is None:
            if line.strip() and self._passthrough is not None:
                print(line, file=self._passthrough)
            return

        text = match.group(3).strip()
        if not text:
            return
        # The live preview goes straight to the UI without passing through
        # _collect_segments, so it needs the same filtering applied here.
        if is_hallucination(text) or (text == self._last_text and _is_sentence(text)):
            return
        self._last_text = text
        try:
            start = _parse_clock(match.group(1))
            end = _parse_clock(match.group(2))
        except ValueError:
            return

        try:
            self._on_segment(
                TranscriptSegment(
                    start_sec=self._offset + start,
                    end_sec=self._offset + end,
                    text=text,
                )
            )
        except Exception:
            # Streaming is a preview of work already being done; a consumer
            # that trips must never take the transcription down with it.
            pass


@contextlib.contextmanager
def stream_segments(
    on_segment: Optional[SegmentCallback], offset_sec: float
) -> Iterator[bool]:
    """Route verbose segment prints to `on_segment` for the duration.

    Yields whether streaming is on, which the caller passes to the engine as
    `verbose` — the prints only happen when it is. Redirecting stdout is
    process-wide, so this is only safe because transcription runs on one worker
    thread at a time (see app.py); non-segment output is forwarded regardless.
    """
    if on_segment is None:
        yield False
        return

    tap = _SegmentPrintTap(on_segment, offset_sec, sys.stdout)
    try:
        with contextlib.redirect_stdout(tap):
            yield True
    finally:
        tap.close_out()


class Transcriber:
    """Wraps openai-whisper for the final full-file transcription pass."""

    def __init__(
        self,
        *,
        final_model: str = FINAL_MODEL,
        language: str = "vi",
    ) -> None:
        self.final_model_name = final_model
        self.language = language if language != "auto" else None

        self._final_model: Optional[whisper.Whisper] = None

    def set_language(self, language: str) -> None:
        self.language = language if language != "auto" else None

    def set_final_model(self, name: str) -> None:
        """Switch the model used for the full-file final pass.

        Drops the cached loaded model when the name actually changes so the
        next preload/transcribe call downloads (if needed) and loads the
        newly selected checkpoint instead of reusing the old one.
        """
        if name != self.final_model_name:
            self.final_model_name = name
            self._final_model = None

    def _get_final_model(self) -> "whisper.Whisper":
        if self._final_model is None:
            self._final_model = whisper.load_model(self.final_model_name, device=DEVICE)
        return self._final_model

    def preload_final_model(
        self,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        ensure_model_downloaded(self.final_model_name, progress_cb, cancel_event=cancel_event)
        self._get_final_model()

    def transcribe_file(self, wav_path: Path) -> list[TranscriptSegment]:
        model = self._get_final_model()
        result = model.transcribe(
            str(wav_path),
            language=self.language,
            task="transcribe",
            fp16=USE_FP16,
            beam_size=5,
            **DECODE_OPTIONS,
        )
        return self._collect_segments(result, offset_sec=0.0)

    def transcribe_audio(
        self,
        audio: np.ndarray,
        offset_sec: float = 0.0,
        on_segment: Optional[SegmentCallback] = None,
    ) -> list[TranscriptSegment]:
        """Transcribe one already-decoded 16 kHz mono chunk.

        offset_sec shifts the returned timestamps back onto the original
        recording's timeline, so chunks can be merged into one transcript.

        on_segment, if given, is called with each segment the moment the model
        finishes it — minutes before this call returns on a CPU-sized chunk.
        """
        model = self._get_final_model()
        with stream_segments(on_segment, offset_sec) as streaming:
            result = model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                fp16=USE_FP16,
                beam_size=5,
                verbose=True if streaming else None,
                **DECODE_OPTIONS,
            )
        return self._collect_segments(result, offset_sec=offset_sec)

    def transcribe_file_to_text(self, wav_path: Path) -> str:
        segments = self.transcribe_file(wav_path)
        return segments_to_text(segments)

    @staticmethod
    def _collect_segments(
        result: dict,
        *,
        offset_sec: float,
    ) -> list[TranscriptSegment]:
        collected: list[TranscriptSegment] = []
        for segment in result.get("segments", []):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            collected.append(
                TranscriptSegment(
                    start_sec=offset_sec + float(segment["start"]),
                    end_sec=offset_sec + float(segment["end"]),
                    text=text,
                )
            )
        return drop_hallucinations(collected)


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    lines = [segment.format_line() for segment in segments]
    return "\n".join(lines).strip() + ("\n" if lines else "")


# --- PhoWhisper-large via WhisperX (final pass, Vietnamese) --------------------

WHISPERX_BATCH_SIZE = 8
# faster-whisper int8 is the fast, low-memory choice for CPU inference.
WHISPERX_COMPUTE_TYPE = "int8"
# Silero VAD downloads from torch hub without a Hugging Face token, unlike the
# pyannote VAD, keeping this token-free for the accuracy-only use case.
WHISPERX_VAD_METHOD = "silero"


def _load_audio_16k(wav_path: Path) -> np.ndarray:
    """Load a WAV as mono 16 kHz float32 without requiring ffmpeg.

    Falls back to whisperx.load_audio (ffmpeg) for formats soundfile can't read.
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            duration = audio.shape[0] / float(sr)
            target_len = int(round(duration * 16_000))
            if target_len > 0:
                src_idx = np.linspace(0.0, audio.shape[0] - 1, target_len)
                audio = np.interp(
                    src_idx, np.arange(audio.shape[0]), audio
                ).astype(np.float32)
        return np.ascontiguousarray(audio, dtype=np.float32)
    except Exception:
        import whisperx

        return whisperx.load_audio(str(wav_path))


class WhisperXTranscriber:
    """Final-pass transcriber: PhoWhisper-large on the WhisperX pipeline.

    Uses VinAI's Vietnamese-fine-tuned Whisper converted to CTranslate2, run
    through WhisperX (VAD + batched inference) for accurate Vietnamese
    transcripts. Intended for the full-file pass on Stop, not live chunks.
    """

    def __init__(self, *, language: str = "vi") -> None:
        self.language = language
        self._model = None
        self._ct2_dir: Optional[Path] = None

    def is_ready(self) -> bool:
        """True if the CT2 model is loaded or already built on disk."""
        if self._model is not None:
            return True
        from phowhisper import is_ready

        return is_ready()

    def preload(
        self,
        progress_cb: Optional[ProgressCallback] = None,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        from phowhisper import ensure_phowhisper_ct2

        self._ct2_dir = ensure_phowhisper_ct2(
            progress_cb=progress_cb, status_cb=status_cb
        )
        self._get_model()

    def _get_model(self):
        if self._model is None:
            import whisperx

            if self._ct2_dir is None:
                from phowhisper import ensure_phowhisper_ct2

                self._ct2_dir = ensure_phowhisper_ct2()
            self._model = whisperx.load_model(
                str(self._ct2_dir),
                device=DEVICE,
                compute_type=WHISPERX_COMPUTE_TYPE,
                language=self.language,
                vad_method=WHISPERX_VAD_METHOD,
            )
        return self._model

    def transcribe_file(self, wav_path: Path) -> list[TranscriptSegment]:
        audio = _load_audio_16k(Path(wav_path))
        return self.transcribe_audio(audio)

    def transcribe_audio(
        self,
        audio: np.ndarray,
        offset_sec: float = 0.0,
        on_segment: Optional[SegmentCallback] = None,
    ) -> list[TranscriptSegment]:
        """Transcribe one already-decoded 16 kHz mono chunk.

        offset_sec shifts the returned timestamps back onto the original
        recording's timeline, so chunks can be merged into one transcript.

        on_segment is accepted for a uniform engine surface but never fired:
        WhisperX batches the whole chunk through the model and produces
        nothing until it is done, so there is no partial result to stream.
        """
        if audio.size == 0:
            return []
        model = self._get_model()
        result = model.transcribe(
            audio,
            batch_size=WHISPERX_BATCH_SIZE,
            language=self.language,
        )
        collected: list[TranscriptSegment] = []
        for segment in result.get("segments", []):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            collected.append(
                TranscriptSegment(
                    start_sec=offset_sec + float(segment["start"]),
                    end_sec=offset_sec + float(segment["end"]),
                    text=text,
                )
            )
        return drop_hallucinations(collected)

    def transcribe_file_to_text(self, wav_path: Path) -> str:
        return segments_to_text(self.transcribe_file(wav_path))


def _cli(argv: list[str]) -> int:
    """Smoke-test: transcribe a WAV with PhoWhisper-large + WhisperX."""
    if not argv:
        print("usage: python -m transcriber <audio.wav> [language]")
        return 2

    wav_path = Path(argv[0])
    language = argv[1] if len(argv) > 1 else "vi"
    if not wav_path.is_file():
        print(f"file not found: {wav_path}")
        return 1

    def on_progress(name: str, downloaded: int, total: int) -> None:
        if total:
            pct = downloaded / total * 100
            print(f"\rDownloading {name}: {downloaded/1e6:.0f}/{total/1e6:.0f} MB "
                  f"({pct:.0f}%)", end="", flush=True)
        else:
            print(f"\rDownloading {name}: {downloaded/1e6:.0f} MB", end="", flush=True)

    def on_status(msg: str) -> None:
        print(f"\n{msg}", flush=True)

    transcriber = WhisperXTranscriber(language=language)
    print("Preparing PhoWhisper-large (WhisperX)…")
    transcriber.preload(progress_cb=on_progress, status_cb=on_status)
    print("\nTranscribing…")
    text = transcriber.transcribe_file_to_text(wav_path)
    print("\n----- transcript -----")
    print(text)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))
