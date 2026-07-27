"""Acquire VinAI PhoWhisper-large as a CTranslate2 model for WhisperX.

PhoWhisper-large ships as a transformers/PyTorch checkpoint, while WhisperX runs
on the faster-whisper (CTranslate2) backend. This module downloads the Hugging
Face snapshot once and converts it to a CTranslate2 model directory, caching the
result so subsequent runs are instant.
"""

from __future__ import annotations

import warnings

# Suppress torchcodec / pyannote audio decoder warnings (not needed for Silero VAD / soundfile)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

# Use the Rust-based parallel downloader when available: dramatically faster for
# the ~6 GB PhoWhisper weights. Harmless if hf_transfer isn't installed and the
# env var is ignored by older huggingface_hub versions.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

PHOWHISPER_REPO = "vinai/PhoWhisper-large"

# progress_cb(name, downloaded_bytes, total_bytes); total may be 0 if unknown.
ProgressCallback = Callable[[str, int, int], None]
# status_cb(message) for discrete, non-byte steps (e.g. converting).
StatusCallback = Callable[[str], None]

# Files worth copying next to the converted model so faster-whisper can load the
# tokenizer/feature extractor. Only those present in the snapshot are copied.
_COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "normalizer.json",
    "preprocessor_config.json",
    "generation_config.json",
]


def cache_root() -> Path:
    override = os.getenv("AI_MEETING_CACHE")
    root = Path(override) if override else Path.home() / ".cache" / "ai-meeting"
    return root


def _hf_dir() -> Path:
    return cache_root() / "phowhisper-large-hf"


def _ct2_dir() -> Path:
    return cache_root() / "phowhisper-large-ct2"


def is_ready() -> bool:
    """True if a converted CTranslate2 model is already cached."""
    return (_ct2_dir() / "model.bin").is_file()


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _snapshot_total_bytes() -> int:
    """Best-effort total download size, or 0 if it can't be determined."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(PHOWHISPER_REPO, files_metadata=True)
        return sum(int(s.size) for s in info.siblings if getattr(s, "size", None))
    except Exception:
        return 0


def _download_snapshot(progress_cb: Optional[ProgressCallback]) -> Path:
    from huggingface_hub import snapshot_download

    hf_dir = _hf_dir()
    hf_dir.mkdir(parents=True, exist_ok=True)

    total = _snapshot_total_bytes() if progress_cb else 0
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            if progress_cb:
                progress_cb(PHOWHISPER_REPO, _dir_size(hf_dir), total)
            time.sleep(0.5)

    poller: Optional[threading.Thread] = None
    if progress_cb:
        poller = threading.Thread(target=poll, name="phowhisper-dl", daemon=True)
        poller.start()

    try:
        snapshot_download(
            repo_id=PHOWHISPER_REPO,
            local_dir=str(hf_dir),
            # Skip redundant/large formats we don't need for CT2 conversion.
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.safetensors.index.json"],
        )
    finally:
        stop.set()
        if poller is not None:
            poller.join(timeout=1.0)

    if progress_cb:
        final = _dir_size(hf_dir)
        progress_cb(PHOWHISPER_REPO, final, total or final)
    return hf_dir


def _convert(hf_dir: Path, status_cb: Optional[StatusCallback]) -> Path:
    from ctranslate2.converters import TransformersConverter

    ct2_dir = _ct2_dir()
    tmp_dir = ct2_dir.with_name(ct2_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    copy_files = [name for name in _COPY_FILES if (hf_dir / name).is_file()]

    if status_cb:
        status_cb("Converting PhoWhisper-large to CTranslate2 (one-time)…")

    converter = TransformersConverter(str(hf_dir), copy_files=copy_files)
    converter.convert(str(tmp_dir), quantization="int8", force=True)

    if ct2_dir.exists():
        shutil.rmtree(ct2_dir, ignore_errors=True)
    os.replace(tmp_dir, ct2_dir)
    return ct2_dir


def ensure_phowhisper_ct2(
    *,
    progress_cb: Optional[ProgressCallback] = None,
    status_cb: Optional[StatusCallback] = None,
) -> Path:
    """Return the cached CTranslate2 model dir, building it once if needed.

    Downloads the PhoWhisper-large snapshot from Hugging Face and converts it to
    a CTranslate2 int8 model. Idempotent: returns immediately if already built.
    """
    ct2_dir = _ct2_dir()
    if is_ready():
        return ct2_dir

    cache_root().mkdir(parents=True, exist_ok=True)
    hf_dir = _download_snapshot(progress_cb)
    return _convert(hf_dir, status_cb)
