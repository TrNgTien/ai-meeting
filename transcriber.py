"""Local Whisper transcription via the official openai-whisper package."""

from __future__ import annotations

import warnings

# Suppress torchcodec / pyannote audio decoder warnings (not needed for Silero VAD / soundfile)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import whisper

LIVE_MODEL = "small"
FINAL_MODEL = "large-v3"
DEVICE = "cpu"
# openai-whisper only supports fp16 on CUDA; keep it off for CPU inference.
USE_FP16 = False

# progress_cb(model_name, downloaded_bytes, total_bytes). total_bytes may be 0
# when the server does not report a Content-Length.
ProgressCallback = Callable[[str, int, int], None]


def _whisper_cache_dir() -> str:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def ensure_model_downloaded(
    name: str,
    progress_cb: Optional[ProgressCallback] = None,
) -> None:
    """Download the given Whisper model to the cache, reporting progress.

    Mirrors ``whisper._download`` but streams progress through ``progress_cb``
    instead of a terminal tqdm bar. After this returns, ``whisper.load_model``
    finds the cached file and loads without re-downloading.
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


class Transcriber:
    """Wraps openai-whisper for live chunks and final full-file passes."""

    def __init__(
        self,
        *,
        live_model: str = LIVE_MODEL,
        final_model: str = FINAL_MODEL,
        language: str = "vi",
    ) -> None:
        self.live_model_name = live_model
        self.final_model_name = final_model
        self.language = language if language != "auto" else None

        self._live_model: Optional[whisper.Whisper] = None
        self._final_model: Optional[whisper.Whisper] = None

    def set_language(self, language: str) -> None:
        self.language = language if language != "auto" else None

    def _get_live_model(self) -> "whisper.Whisper":
        if self._live_model is None:
            self._live_model = whisper.load_model(self.live_model_name, device=DEVICE)
        return self._live_model

    def _get_final_model(self) -> "whisper.Whisper":
        if self._final_model is None:
            self._final_model = whisper.load_model(self.final_model_name, device=DEVICE)
        return self._final_model

    def preload_live_model(
        self,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> None:
        ensure_model_downloaded(self.live_model_name, progress_cb)
        self._get_live_model()

    def preload_final_model(
        self,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> None:
        ensure_model_downloaded(self.final_model_name, progress_cb)
        self._get_final_model()

    def transcribe_chunk(
        self,
        audio: np.ndarray,
        *,
        offset_sec: float = 0.0,
    ) -> list[TranscriptSegment]:
        if audio.size == 0:
            return []

        model = self._get_live_model()
        result = model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=self.language,
            task="transcribe",
            fp16=USE_FP16,
            beam_size=1,
            best_of=1,
            # Chunks are transcribed independently, so don't carry text context
            # across calls to avoid repetition/hallucination on chunk seams.
            condition_on_previous_text=False,
        )
        return self._collect_segments(result, offset_sec=offset_sec)

    def transcribe_file(self, wav_path: Path) -> list[TranscriptSegment]:
        model = self._get_final_model()
        result = model.transcribe(
            str(wav_path),
            language=self.language,
            task="transcribe",
            fp16=USE_FP16,
            beam_size=5,
        )
        return self._collect_segments(result, offset_sec=0.0)

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
        return collected


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

    The recorder already writes 16 kHz mono PCM, so this is usually a plain
    read. Falls back to whisperx.load_audio (ffmpeg) for other formats.
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
        model = self._get_model()
        audio = _load_audio_16k(Path(wav_path))
        if audio.size == 0:
            return []
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
                    start_sec=float(segment["start"]),
                    end_sec=float(segment["end"]),
                    text=text,
                )
            )
        return collected

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
