"""Whisper on the Apple GPU via MLX.

The openai-whisper path runs fp32 on the CPU (`transcriber.DEVICE`), because
its MPS backend hits unimplemented sparse ops. On an M2 that transcribes at
roughly a fifth of realtime, so a 44-minute recording costs about three hours
and the machine's GPU sits idle throughout.

MLX runs the same Whisper weights on that GPU, out of the same unified memory,
which is the difference between leaving a meeting to transcribe over lunch and
over the afternoon. The weights are separate converted checkpoints hosted by
the mlx-community org, so a model downloaded for this engine is not the .pt
that `transcriber` uses and vice versa.

Only the decode backend changes: the audio arrives already chunked and decoded
(see chunking.py), and segments come back on the same timeline.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from transcriber import (
    DECODE_OPTIONS,
    SegmentCallback,
    TranscriptSegment,
    drop_hallucinations,
    stream_segments,
)

# openai-whisper checkpoint name -> converted MLX checkpoint on the Hub. Keyed
# by the names in transcriber.FINAL_MODEL_OPTIONS so the Model dropdown means
# the same thing whichever engine is selected.
MLX_REPOS = {
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

StatusCallback = Callable[[str], None]
# progress_cb(model_name, downloaded_bytes, total_bytes) — same shape as the
# openai-whisper downloader, so both engines drive the one progress bar.
ProgressCallback = Callable[[str, int, int], None]


def is_available() -> bool:
    """Whether this machine can run the MLX engine at all.

    MLX is Apple-silicon only, and the package is an optional dependency — the
    app must still work on a machine that has neither.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return importlib.util.find_spec("mlx_whisper") is not None


def repo_for(model_name: str) -> Optional[str]:
    return MLX_REPOS.get(model_name)


def _repo_cache_dir(repo: str) -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + repo.replace("/", "--"))
    )


def is_model_downloaded(model_name: str) -> bool:
    """True when the converted checkpoint is already in the Hub cache.

    Checked against the weights file rather than the directory: an interrupted
    download leaves the directory (and its blobs/refs skeleton) behind, and
    treating that as "downloaded" would fail later at load time instead of
    re-fetching now.
    """
    repo = repo_for(model_name)
    if repo is None:
        return False
    cache = _repo_cache_dir(repo)
    if not cache.is_dir():
        return False
    return any(cache.glob("snapshots/*/weights.npz")) or any(
        cache.glob("snapshots/*/*.safetensors")
    )


def _cached_bytes(repo: str) -> int:
    """Bytes of `repo` materialised on disk.

    Only `blobs/` is measured: `snapshots/` holds symlinks into it, and
    following those would double-count every file.

    Not usable as live download progress — repos served over Xet stage their
    chunks in ~/.cache/huggingface/xet and only land in `blobs/` once the file
    is complete, so this reads 0 for the whole download and then jumps. Byte
    progress comes from the Hub client itself (see `_byte_reporting_tqdm`).
    """
    blobs = _repo_cache_dir(repo) / "blobs"
    if not blobs.is_dir():
        return 0
    total = 0
    for f in blobs.iterdir():
        try:
            total += f.stat().st_size
        except OSError:
            # The file can vanish mid-download as .incomplete is renamed.
            pass
    return total


def _byte_reporting_tqdm(model_name: str, progress_cb: ProgressCallback, total: int):
    """A tqdm subclass that forwards the Hub client's own byte counts.

    huggingface_hub takes no progress callback, but it does take a
    `tqdm_class`, and it drives those bars itself for every storage backend it
    supports. Hooking the bar is therefore the one progress source that stays
    correct whether the repo is served over Xet or classic blob storage.

    Only the per-file byte bars are counted; the Hub also opens a "Fetching N
    files" counter bar, which would otherwise add file counts to a byte total.
    """
    from huggingface_hub.utils import tqdm as hf_tqdm

    lock = threading.Lock()
    per_bar: dict[int, int] = {}

    class _ReportingTqdm(hf_tqdm):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            self._counts_bytes = kwargs.get("unit") == "B"
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            displayed = super().update(n)
            if self._counts_bytes:
                with lock:
                    per_bar[id(self)] = self.n or 0
                    done = sum(per_bar.values())
                progress_cb(model_name, done, total)
            return displayed

    return _ReportingTqdm


def _remote_bytes(repo: str) -> int:
    """Total download size, or 0 if the Hub can't be asked.

    0 is not fatal: the UI falls back to an indeterminate bar, which is still
    better than the silence this replaced.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        return sum(sibling.size or 0 for sibling in (info.siblings or []))
    except Exception:
        return 0


def model_size_on_disk(model_name: str) -> int:
    repo = repo_for(model_name)
    return _cached_bytes(repo) if repo else 0


class MLXTranscriber:
    """Chunk-at-a-time Whisper decoding on the GPU.

    Mirrors the surface `chunking.transcribe_chunked` needs from the
    openai-whisper path: an identity key for the checkpoint fingerprint, and a
    `transcribe_audio(audio, offset)` callable.
    """

    def __init__(self, model_name: str, language: Optional[str]) -> None:
        self.model_name = model_name
        # None means "detect", matching Transcriber's convention for "auto".
        self.language = language
        self.repo = repo_for(model_name)
        if self.repo is None:
            raise ValueError(f"no MLX checkpoint for model '{model_name}'")

    @property
    def engine_key(self) -> str:
        """Identity for the resume checkpoint.

        Includes the engine, so a transcript half-written by openai-whisper is
        never continued with MLX — the two would splice different decodings of
        the same meeting into one file.
        """
        return f"mlx-{self.model_name}:{self.language or 'auto'}"

    def is_ready(self) -> bool:
        return is_model_downloaded(self.model_name)

    def preload(
        self,
        progress_cb: Optional[ProgressCallback] = None,
        status_cb: Optional[StatusCallback] = None,
    ) -> None:
        """Fetch and warm the checkpoint before the first chunk.

        Done up front so the download is reported as a download rather than
        showing up as a mysteriously slow first chunk.
        """
        import mlx_whisper  # imported lazily: optional dependency

        if not self.is_ready():
            if status_cb is not None:
                status_cb(f"Downloading MLX model '{self.model_name}' (one-time)…")
            self._download(progress_cb)

        if status_cb is not None:
            status_cb(f"Loading '{self.model_name}' onto the GPU…")
        # A moment of silence is the cheapest way to force the weights to load
        # and the GPU kernels to compile, so the first real chunk isn't billed
        # for it.
        mlx_whisper.transcribe(
            np.zeros(16_000, dtype=np.float32),
            path_or_hf_repo=self.repo,
            language=self.language,
            verbose=None,
        )

    def _download(self, progress_cb: Optional[ProgressCallback]) -> None:
        """Fetch the checkpoint, reporting bytes as they land.

        `mlx_whisper.transcribe` would fetch it implicitly on first use, but
        silently — a multi-GB download that looks like the app has hung. So the
        download happens here instead, with the Hub client's own progress bars
        redirected into our callback.
        """
        from huggingface_hub import snapshot_download

        if progress_cb is None:
            snapshot_download(self.repo)
            return

        total = _remote_bytes(self.repo)
        snapshot_download(
            self.repo,
            tqdm_class=_byte_reporting_tqdm(self.model_name, progress_cb, total),
        )
        # Land the bar on 100% rather than wherever the last update left it.
        progress_cb(self.model_name, total or _cached_bytes(self.repo), total)

    def transcribe_audio(
        self,
        audio: np.ndarray,
        offset_sec: float = 0.0,
        on_segment: Optional[SegmentCallback] = None,
    ) -> list[TranscriptSegment]:
        """Transcribe one decoded 16 kHz mono chunk, timestamps shifted to `offset_sec`.

        on_segment, if given, receives each segment as the GPU finishes it,
        rather than the caller waiting for the whole chunk.
        """
        import mlx_whisper

        with stream_segments(on_segment, offset_sec) as streaming:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.repo,
                language=self.language,
                task="transcribe",
                verbose=True if streaming else None,
                **DECODE_OPTIONS,
            )
        segments = []
        for raw in result.get("segments", []):
            text = (raw.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start_sec=float(raw.get("start", 0.0)) + offset_sec,
                    end_sec=float(raw.get("end", 0.0)) + offset_sec,
                    text=text,
                )
            )
        return drop_hallucinations(segments)
