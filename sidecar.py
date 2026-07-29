"""JSON-line stdin/stdout command loop for the Tauri sidecar.

Re-implements app.py's Tk-glue (_apply_language_selection, _prepare_engine,
_run_final_transcription, the batch loop, the model-manager actions) as a
persistent subprocess protocol instead of tkinter callbacks. transcriber.py,
chunking.py, mlx_engine.py, and phowhisper.py are called unchanged.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import mlx_engine
from chunking import (
    TranscribeAudio,
    TranscriptionCancelled,
    resolve_transcript_path,
    resumable_seconds,
    transcribe_chunked,
)
from transcriber import (
    FINAL_MODEL,
    FINAL_MODEL_OPTIONS,
    DownloadCancelled,
    Transcriber,
    WhisperXTranscriber,
    delete_model,
    ensure_model_downloaded,
    format_timestamp,
    list_downloaded_whisper_models,
    model_size_on_disk,
)

AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4",
}

_stdout_lock = threading.Lock()


def emit(event: str, **payload) -> None:
    """Write one JSON line to stdout — the sidecar -> Rust event channel.

    Guarded by a lock because both the main thread (dispatch errors) and a
    worker thread (progress/status during a job or download) can call this
    concurrently; without the lock two lines could interleave mid-write.
    """
    with _stdout_lock:
        sys.stdout.write(json.dumps({"event": event, **payload}) + "\n")
        sys.stdout.flush()


class Sidecar:
    def __init__(self) -> None:
        self._transcriber = Transcriber(language="vi")
        self._final_transcriber = WhisperXTranscriber(language="vi")
        self._use_phowhisper = False
        self._use_mlx = mlx_engine.is_available()
        self._cancel_event = threading.Event()
        self._current_job_id: Optional[str] = None
        self._download_cancel_events: dict[str, threading.Event] = {}

    def handle(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            emit("error", message=f"unknown cmd '{cmd}'")
            return
        try:
            handler(msg)
        except Exception as exc:
            emit("error", message=str(exc))

    def cmd_list_models(self, msg: dict) -> None:
        extra_downloaded = [
            n for n in list_downloaded_whisper_models() if n not in FINAL_MODEL_OPTIONS
        ]
        names = FINAL_MODEL_OPTIONS + extra_downloaded
        models = [
            {
                "name": name,
                "downloaded": model_size_on_disk(name) > 0,
                "size_bytes": model_size_on_disk(name),
            }
            for name in names
        ]
        emit("models", models=models)


def main() -> None:
    sidecar = Sidecar()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            emit("error", message=f"bad JSON: {exc}")
            continue
        if not isinstance(msg, dict):
            emit("error", message=f"expected object, got {type(msg).__name__}")
            continue
        sidecar.handle(msg)


if __name__ == "__main__":
    main()
