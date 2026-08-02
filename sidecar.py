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
from typing import Optional, Tuple

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

    def cmd_download_model(self, msg: dict) -> None:
        name = msg["name"]
        if name in self._download_cancel_events:
            return
        cancel_event = threading.Event()
        self._download_cancel_events[name] = cancel_event

        def worker() -> None:
            try:
                ensure_model_downloaded(
                    name,
                    lambda m, d, t: emit("mm_progress", model=m, downloaded=d, total=t),
                    cancel_event=cancel_event,
                )
                emit("mm_download_finished", name=name, status="done")
            except DownloadCancelled:
                emit("mm_download_finished", name=name, status="cancelled")
            except Exception as exc:
                emit("mm_download_finished", name=name, status="error", error=str(exc))
            finally:
                # app.py's equivalent cleanup lives in _handle_ui_event on the
                # Tk main thread (a separate poll loop); sidecar.py has no such
                # loop, so the worker that owns this entry cleans it up itself.
                self._download_cancel_events.pop(name, None)

        threading.Thread(target=worker, daemon=True).start()

    def cmd_delete_model(self, msg: dict) -> None:
        name = msg["name"]
        delete_model(name)
        emit("model_deleted", name=name)

    def cmd_cancel(self, msg: dict) -> None:
        if "name" in msg:
            name = msg["name"]
            cancel_event = self._download_cancel_events.get(name)
            if cancel_event is None:
                emit("error", message=f"no download in progress for '{name}'")
                return
            cancel_event.set()
            return

        job_id = msg.get("id")
        if job_id is None or job_id != self._current_job_id:
            emit("error", message=f"no running job '{job_id}'")
            return
        self._cancel_event.set()

    def _resolve_engine(
        self, lang_mode: str, model_name: str, use_mlx: bool
    ) -> Tuple[str, TranscribeAudio]:
        if lang_mode == "vi+en":
            self._transcriber.set_language("vi")
            use_phowhisper = False
        elif lang_mode == "vi":
            self._transcriber.set_language("vi")
            use_phowhisper = True
        else:
            self._transcriber.set_language(lang_mode)
            use_phowhisper = False

        if not use_phowhisper:
            self._transcriber.set_final_model(model_name)

        if use_phowhisper and self._transcriber.language == "vi":
            try:
                if not self._final_transcriber.is_ready():
                    emit("status", message="Installing PhoWhisper-large (one-time download)…")
                self._final_transcriber.preload(
                    progress_cb=lambda m, d, t: emit(
                        "download_progress", model=m, downloaded=d, total=t
                    ),
                    status_cb=lambda msg: emit("status", message=msg),
                )
                emit("hide_progress")
                return "phowhisper-large:vi", self._final_transcriber.transcribe_audio
            except DownloadCancelled:
                emit("hide_progress")
                raise TranscriptionCancelled("model download was cancelled")
            except Exception as exc:
                emit("hide_progress")
                emit(
                    "status",
                    message=f"PhoWhisper failed ({exc}); falling back to "
                    f"{self._transcriber.final_model_name}…",
                )

        if self._cancel_event.is_set():
            raise TranscriptionCancelled("cancelled before model was ready")

        final_model_name = self._transcriber.final_model_name

        if use_mlx and mlx_engine.repo_for(final_model_name) is not None:
            try:
                engine = mlx_engine.MLXTranscriber(final_model_name, self._transcriber.language)
                emit("status", message=f"Loading '{final_model_name}' on the Apple GPU (MLX)…")
                engine.preload(
                    progress_cb=lambda m, d, t: emit(
                        "download_progress", model=m, downloaded=d, total=t
                    ),
                    status_cb=lambda msg: emit("status", message=msg),
                    cancel_event=self._cancel_event,
                )
                emit("hide_progress")
                return engine.engine_key, engine.transcribe_audio
            except DownloadCancelled:
                emit("hide_progress")
                raise TranscriptionCancelled("model download was cancelled")
            except Exception as exc:
                emit("hide_progress")
                emit("status", message=f"MLX unavailable ({exc}); using the CPU engine…")

        if self._cancel_event.is_set():
            raise TranscriptionCancelled("cancelled before model was ready")

        emit("status", message=f"Loading model '{final_model_name}' (downloads on first use)…")
        try:
            self._transcriber.preload_final_model(
                progress_cb=lambda m, d, t: emit(
                    "download_progress", model=m, downloaded=d, total=t
                ),
                cancel_event=self._cancel_event,
            )
        except DownloadCancelled:
            emit("hide_progress")
            raise TranscriptionCancelled("model download was cancelled")
        emit("hide_progress")
        language = self._transcriber.language or "auto"
        return f"whisper-{final_model_name}:{language}", self._transcriber.transcribe_audio

    def cmd_start_transcription(self, msg: dict) -> None:
        if self._current_job_id is not None:
            emit("error", message=f"job '{self._current_job_id}' is already running")
            return

        job_id = msg["id"]
        paths = [Path(p) for p in msg["paths"]]
        lang_mode = msg.get("lang_mode", "vi+en")
        model_name = msg.get("model", FINAL_MODEL)
        use_mlx = msg.get("mlx", self._use_mlx)

        audio_paths = [p for p in paths if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

        self._current_job_id = job_id
        self._cancel_event = threading.Event()

        def worker() -> None:
            saved: list[str] = []
            total = len(audio_paths)
            cancelled = False
            batch_started = time.monotonic()
            try:
                for idx, source in enumerate(audio_paths, start=1):
                    counter = f" {idx}/{total}" if total > 1 else ""
                    label = f"Transcribing{counter}: {source.name}"
                    emit("status", message=f"{label}…")
                    emit("file_start", name=source.name)
                    try:
                        written = self._run_final_transcription(
                            source, label, lang_mode, model_name, use_mlx
                        )
                    except TranscriptionCancelled:
                        cancelled = True
                        break
                    except Exception as exc:
                        emit("error", file=source.name, message=str(exc))
                        continue
                    saved.append(str(written))
            finally:
                # batch_done must fire and _current_job_id must clear no
                # matter what — otherwise the frontend's running state gets
                # stuck forever and every subsequent job is rejected with
                # "already running", with no way to recover short of an app
                # restart.
                emit(
                    "batch_done",
                    id=job_id,
                    count=len(saved),
                    saved=saved,
                    cancelled=cancelled,
                    elapsed_sec=time.monotonic() - batch_started,
                )
                self._current_job_id = None

        threading.Thread(target=worker, daemon=True).start()

    def _run_final_transcription(
        self, wav_path: Path, label: str, lang_mode: str, model_name: str, use_mlx: bool
    ) -> Path:
        engine_key, transcribe_audio = self._resolve_engine(lang_mode, model_name, use_mlx)
        out_path = resolve_transcript_path(wav_path, engine_key)

        resume_at = resumable_seconds(wav_path, engine_key)
        if resume_at > 0:
            emit("chunk_baseline", resume_at_sec=resume_at)
            emit("status", message=f"{label} — resuming at {format_timestamp(resume_at)}…")

        def on_progress(chunk_index, chunks_done, done_sec, total_sec) -> None:
            emit(
                "chunk_progress",
                label=label,
                done_sec=done_sec,
                total_sec=total_sec,
                chunks_done=chunks_done,
            )

        transcribe_chunked(
            wav_path,
            transcribe_audio=transcribe_audio,
            engine_key=engine_key,
            output_path=out_path,
            on_progress=on_progress,
            on_text=lambda text: emit("chunk_text", text=text),
            on_segment=lambda segment: emit("segment_text", text=segment.format_line()),
            cancel_event=self._cancel_event,
        )
        return out_path


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
