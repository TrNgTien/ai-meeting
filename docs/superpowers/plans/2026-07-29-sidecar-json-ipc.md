# sidecar.py JSON IPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sidecar.py`, the new thin JSON-line stdin/stdout command loop that
re-implements `app.py`'s CustomTkinter glue (`_apply_language_selection`,
`_prepare_engine`, `_run_final_transcription`, the batch loop, the model-manager
actions) headlessly, calling `transcriber.py`/`chunking.py`/`mlx_engine.py`/
`phowhisper.py` with the exact same arguments those modules already use — no
Rust, no UI yet. This is **Build phase 1** of
`docs/superpowers/specs/2026-07-29-tauri-rust-rewrite-design.md`.

**Architecture:** One process, two threads at most at any time: the main
thread blocks reading newline-delimited JSON commands from stdin and dispatches
them; a single background worker thread (spawned per `start_transcription` or
per `download_model` call) does the actual model loading/transcription/download
so the main thread can keep accepting commands (in particular `cancel`) while
work is in flight. Every event sidecar emits is one JSON object per stdout
line, written through a single `emit()` helper guarded by a lock (two threads —
main and worker — can both call `emit()`).

**Tech Stack:** Python 3 (same `.venv` as the rest of the repo), stdlib only
(`json`, `threading`, `sys`, `time`, `pathlib`) plus the existing
`transcriber`/`chunking`/`mlx_engine`/`phowhisper` modules.

## Global Constraints

- `transcriber.py`, `chunking.py`, `mlx_engine.py`, `phowhisper.py` are called
  **unchanged** — no edits to any of them in this plan.
- `app.py` is **not touched** — it stays in place, functional, until the Tauri
  app reaches parity (a later, separate decision per the design doc).
- Only one transcription job runs at a time. This isn't just a simplification:
  `transcriber.stream_segments()` redirects `sys.stdout` process-wide to tap
  openai-whisper's/mlx-whisper's verbose print output, which is only safe
  because a single worker thread transcribes at a time today — sidecar.py must
  preserve that invariant, not add concurrency.
- IPC is one JSON object per line: commands arrive on stdin, events are
  written to stdout. No other stdout writes are allowed once the command loop
  starts (this is also why `emit()` must be the only path to `sys.stdout`).
- Event *names* mirror `app.py`'s existing `_ui_queue` tuple names exactly
  (`status`, `download_progress`, `hide_progress`, `file_start`,
  `chunk_baseline`, `chunk_progress`, `chunk_text`, `segment_text`,
  `batch_done`, `mm_progress`, `mm_download_finished`) — **not** the
  illustrative names in the design doc's IPC example (`mm_download_progress`,
  `done`), which don't match the real code the design doc says to mirror
  1:1. Two deliberate, documented departures from that illustrative example:
  - `error` is a real, used event here (`{"event": "error", ...}`) for
    per-file batch failures, instead of `app.py`'s workaround of shoving
    `[error: ...]` into a `chunk_text` line — that workaround only existed
    because Tkinter had one text widget to append to; JSON IPC doesn't have
    that constraint. (`app.py`'s `_handle_ui_event` already has a dead/unused
    `"error"` branch — this finally gives it a producer.)
  - `cancel` covers two distinct cancellable things `app.py` already
    distinguishes: cancelling model **downloads** (keyed by model name, via
    `self._download_cancel_events` in `app.py`) and cancelling the running
    **transcription job** (keyed by job id, which doesn't exist in `app.py`
    since it never runs two things at once — Rust needs the id to correlate).
    `{"cmd": "cancel", "name": "large-v3-turbo"}` cancels a download;
    `{"cmd": "cancel", "id": "job-1"}` cancels the running job.

---

### Task 1: Command loop skeleton + `list_models`

**Files:**
- Create: `sidecar.py`

**Interfaces:**
- Produces: `emit(event: str, **payload) -> None` (module-level, writes one
  JSON line to stdout under `_stdout_lock`); `class Sidecar` with `__init__`
  (attributes: `_transcriber`, `_final_transcriber`, `_use_phowhisper`,
  `_use_mlx`, `_cancel_event`, `_current_job_id`, `_download_cancel_events`)
  and `handle(msg: dict) -> None` (dispatches `msg["cmd"]` to `cmd_<name>`);
  `main()` (stdin read loop).

- [ ] **Step 1: Write `sidecar.py` with the skeleton and `list_models`**

```python
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
        sidecar.handle(msg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify `list_models` responds**

Run:
```bash
printf '{"cmd": "list_models"}\n' | .venv/bin/python sidecar.py
```
Expected: exactly one line of output, valid JSON, shape
`{"event": "models", "models": [{"name": "small", "downloaded": ..., "size_bytes": ...}, ...]}`
covering at least `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`
(plus any extra checkpoints already cached on this machine).

- [ ] **Step 3: Verify unknown-command and bad-JSON handling**

Run:
```bash
printf 'not json\n{"cmd": "nonsense"}\n' | .venv/bin/python sidecar.py
```
Expected: two lines, both `{"event": "error", "message": "..."}` — first
mentioning a JSON decode error, second `"unknown cmd 'nonsense'"`. The process
exits cleanly after stdin closes (no traceback).

- [ ] **Step 4: Commit**

```bash
git add sidecar.py
git commit -m "feat: add sidecar.py JSON command loop skeleton with list_models"
```

---

### Task 2: `download_model`, `delete_model`, `cancel` (download branch)

**Files:**
- Modify: `sidecar.py`

**Interfaces:**
- Consumes: `emit()`, `Sidecar.__init__`'s `_download_cancel_events` (Task 1).
- Produces: `Sidecar.cmd_download_model`, `Sidecar.cmd_delete_model`,
  `Sidecar.cmd_cancel` (handles the `"name"` branch now; the `"id"` branch is
  added in Task 4 once a job can actually be running).

- [ ] **Step 1: Add the three methods to the `Sidecar` class**

```python
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
```

- [ ] **Step 2: Verify a full download/delete round-trip**

Pick the smallest model not already cached on this machine (check with the
`list_models` output from Task 1 — use `"small"` if it isn't already
downloaded; if it is, `delete_model` it first so this step actually exercises
the download path):

```bash
printf '{"cmd": "download_model", "name": "small"}\n' | .venv/bin/python sidecar.py
```
Expected: a stream of `{"event": "mm_progress", "model": "small", "downloaded": ..., "total": ...}`
lines with `downloaded` increasing up to `total`, then one
`{"event": "mm_download_finished", "name": "small", "status": "done"}`.
(The process exits once stdin closes and the daemon worker thread finishes —
if the pipe closes before the download completes, rerun with the fifo pattern
from Task 4's verification instead so stdin stays open.)

```bash
printf '{"cmd": "delete_model", "name": "small"}\n' | .venv/bin/python sidecar.py
```
Expected: `{"event": "model_deleted", "name": "small"}`. Follow with the
`list_models` command from Task 1 to confirm `"small"` now shows
`"downloaded": false`.

- [ ] **Step 3: Verify download cancellation and cancel-with-no-target**

Using the mkfifo pattern (needed here because this test sends two commands to
the *same* running process with a pause between them):

```bash
mkfifo /tmp/sidecar_in
.venv/bin/python sidecar.py < /tmp/sidecar_in &
exec 3>/tmp/sidecar_in
echo '{"cmd": "download_model", "name": "medium"}' >&3
sleep 1
echo '{"cmd": "cancel", "name": "medium"}' >&3
sleep 2
echo '{"cmd": "cancel", "name": "medium"}' >&3
exec 3>&-
```
Expected: `mm_progress` lines, then
`{"event": "mm_download_finished", "name": "medium", "status": "cancelled"}`,
then a final `{"event": "error", "message": "no download in progress for 'medium'"}`
for the second cancel (the entry was already popped) — confirming the
`finally` cleanup in Step 1 actually runs. Clean up: `rm /tmp/sidecar_in`.

- [ ] **Step 4: Commit**

```bash
git add sidecar.py
git commit -m "feat: add sidecar model-manager commands (download/delete/cancel)"
```

---

### Task 3: Engine routing (`_resolve_engine`)

**Files:**
- Modify: `sidecar.py`

**Interfaces:**
- Consumes: `emit()` (Task 1); `Transcriber.set_language`,
  `Transcriber.set_final_model`, `Transcriber.preload_final_model`,
  `Transcriber.transcribe_audio`, `Transcriber.final_model_name`,
  `Transcriber.language`; `WhisperXTranscriber.is_ready`,
  `WhisperXTranscriber.preload`, `WhisperXTranscriber.transcribe_audio`;
  `mlx_engine.repo_for`, `mlx_engine.MLXTranscriber` (all pre-existing,
  unchanged).
- Produces: `Sidecar._resolve_engine(lang_mode: str, model_name: str,
  use_mlx: bool) -> tuple[str, TranscribeAudio]` — used by Task 4.

- [ ] **Step 1: Add `_resolve_engine` to the `Sidecar` class**

Ports `app.py`'s `_apply_language_selection()` + `_prepare_engine()` combined,
taking the three UI values as parameters instead of reading `tk.StringVar`s,
and using `emit()` in place of `self._ui_queue.put()`. The routing decisions
(PhoWhisper only for `"vi"`, MLX only when requested and a checkpoint exists
for the model, CPU openai-whisper otherwise; every engine_key format) are
copied verbatim from `app.py:599-619` and `app.py:915-974`.

```python
    def _resolve_engine(
        self, lang_mode: str, model_name: str, use_mlx: bool
    ) -> tuple[str, TranscribeAudio]:
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
            except Exception as exc:
                emit("hide_progress")
                emit(
                    "status",
                    message=f"PhoWhisper failed ({exc}); falling back to "
                    f"{self._transcriber.final_model_name}…",
                )

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
                )
                emit("hide_progress")
                return engine.engine_key, engine.transcribe_audio
            except Exception as exc:
                emit("hide_progress")
                emit("status", message=f"MLX unavailable ({exc}); using the CPU engine…")

        emit("status", message=f"Loading model '{final_model_name}' (downloads on first use)…")
        self._transcriber.preload_final_model(
            progress_cb=lambda m, d, t: emit("download_progress", model=m, downloaded=d, total=t)
        )
        emit("hide_progress")
        language = self._transcriber.language or "auto"
        return f"whisper-{final_model_name}:{language}", self._transcriber.transcribe_audio
```

- [ ] **Step 2: Verify engine_key routing directly, without going through IPC**

This calls the method directly in a REPL so a full transcription run isn't
needed to check the routing logic — `preload_final_model` still actually
loads/downloads `"small"` (cheap), so this also proves the plumbing works
end-to-end for the CPU path:

```bash
.venv/bin/python -c "
import sidecar
s = sidecar.Sidecar()
key, fn = s._resolve_engine('vi+en', 'small', False)
print(key)
assert key == 'whisper-small:vi'
assert callable(fn)
"
```
Expected: prints `whisper-small:vi`, no assertion error, no traceback (progress
events go to stdout as JSON lines above the printed key — that's expected).

Then check the `"auto"` language path resolves to `None`-language framing:
```bash
.venv/bin/python -c "
import sidecar
s = sidecar.Sidecar()
key, fn = s._resolve_engine('auto', 'small', False)
print(key)
assert key == 'whisper-small:auto'
"
```

- [ ] **Step 3: Commit**

```bash
git add sidecar.py
git commit -m "feat: add sidecar engine routing (_resolve_engine)"
```

---

### Task 4: `start_transcription`

**Files:**
- Modify: `sidecar.py`

**Interfaces:**
- Consumes: `Sidecar._resolve_engine` (Task 3); `resolve_transcript_path`,
  `resumable_seconds`, `transcribe_chunked`, `TranscriptionCancelled` (from
  `chunking`, unchanged); `format_timestamp` (from `transcriber`, unchanged);
  `AUDIO_EXTS`, `emit()` (Task 1).
- Produces: `Sidecar.cmd_start_transcription`, `Sidecar._run_final_transcription`.
  Sets `self._current_job_id` / resets `self._cancel_event` — Task 2's
  `cmd_cancel` id-branch starts having an effect once this lands.

- [ ] **Step 1: Add the batch worker and per-file transcription methods**

Mirrors `app.py`'s `_start_batch_transcription`'s `worker()` (app.py:877-913)
and `_run_final_transcription` (app.py:976-1024) exactly — same per-file
try/except structure (`TranscriptionCancelled` stops the whole batch,
any other exception skips just that file and continues), same
`transcribe_chunked` call shape.

```python
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
```

- [ ] **Step 2: Generate a short synthetic audio fixture for smoke-testing**

Real speech isn't needed to prove the plumbing — a short synthesized tone
exercises `ffmpeg` decode, chunking, checkpointing, and the whisper call path
identically:

```bash
ffmpeg -f lavfi -i "sine=frequency=440:duration=12" -ar 16000 -ac 1 /tmp/sidecar_smoke.wav
```

- [ ] **Step 3: Run a full single-file job end-to-end**

```bash
mkfifo /tmp/sidecar_in
.venv/bin/python sidecar.py < /tmp/sidecar_in &
exec 3>/tmp/sidecar_in
echo '{"cmd": "start_transcription", "id": "job-1", "paths": ["/tmp/sidecar_smoke.wav"], "lang_mode": "vi+en", "model": "small", "mlx": false}' >&3
sleep 30
exec 3>&-
rm /tmp/sidecar_in
```
Expected event sequence on stdout: `status` ("Transcribing: sidecar_smoke.wav…"),
`file_start` (`{"name": "sidecar_smoke.wav"}`), one or more `status`/
`download_progress`/`hide_progress` lines if `"small"` needed loading,
`chunk_progress`, `chunk_text` (may be an empty or near-empty string — it's a
sine tone, not speech, so content doesn't matter here), then `batch_done` with
`"id": "job-1"`, `"count": 1`, `"cancelled": false`. Confirm the transcript
file exists: `ls /tmp/*.txt` (or whatever `resolve_transcript_path` names it)
shows a new file next to `sidecar_smoke.wav`.

- [ ] **Step 4: Verify the concurrent-job guard**

Re-run Step 3's fifo setup but send a second `start_transcription` immediately
after the first, before it finishes:

```bash
mkfifo /tmp/sidecar_in
.venv/bin/python sidecar.py < /tmp/sidecar_in &
exec 3>/tmp/sidecar_in
echo '{"cmd": "start_transcription", "id": "job-1", "paths": ["/tmp/sidecar_smoke.wav"], "lang_mode": "vi+en", "model": "small", "mlx": false}' >&3
echo '{"cmd": "start_transcription", "id": "job-2", "paths": ["/tmp/sidecar_smoke.wav"], "lang_mode": "vi+en", "model": "small", "mlx": false}' >&3
sleep 15
exec 3>&-
rm /tmp/sidecar_in
```
Expected: `{"event": "error", "message": "job 'job-1' is already running"}`
appears (for the rejected `job-2`) before `job-1`'s `batch_done`.

- [ ] **Step 5: Commit**

```bash
git add sidecar.py
git commit -m "feat: add sidecar start_transcription command"
```

---

### Task 5: Verify cancellation and crash-resume through the sidecar

No new production code — this task exists because resumability is the core
design constraint of this whole rewrite (per `CLAUDE.md`'s "Chunked, resumable
transcription" section), and it's worth proving explicitly that wrapping the
existing, unchanged `chunking.py` behavior in a new process boundary hasn't
broken it, rather than assuming it's fine because the chunking code itself
didn't change.

**Files:** none (verification only).

- [ ] **Step 1: Generate a longer fixture spanning multiple chunks**

`transcribe_chunked`'s default chunk size is 300s
(`DEFAULT_CHUNK_SECONDS`), so this needs to clear two chunk boundaries:

```bash
ffmpeg -f lavfi -i "anullsrc=r=16000:cl=mono" -t 700 /tmp/sidecar_long.wav
```

- [ ] **Step 2: Verify mid-run cancel stops before the next chunk, checkpoint intact**

```bash
mkfifo /tmp/sidecar_in
.venv/bin/python sidecar.py < /tmp/sidecar_in &
exec 3>/tmp/sidecar_in
echo '{"cmd": "start_transcription", "id": "job-1", "paths": ["/tmp/sidecar_long.wav"], "lang_mode": "vi+en", "model": "small", "mlx": false}' >&3
sleep 5
echo '{"cmd": "cancel", "id": "job-1"}' >&3
sleep 90
exec 3>&-
rm /tmp/sidecar_in
```
(The `sleep 90` needs to comfortably exceed however long transcribing one
300s chunk takes on this machine with `"small"` — cancellation is only
checked between chunks, so it won't land until the in-flight chunk finishes;
watch for `chunk_progress`/`chunk_text` events to judge timing on a first
run and adjust the sleep if needed.)

Expected: exactly one `chunk_text` (first chunk only), then `batch_done` with
`"cancelled": true`. Confirm the checkpoint survived:
```bash
cat /tmp/sidecar_long.transcript.partial.json
```
Expected: `"chunks_done": 1` and `"next_start_sec"` around `300`.

- [ ] **Step 3: Verify a hard-kill mid-chunk resumes correctly, not from zero**

```bash
mkfifo /tmp/sidecar_in
.venv/bin/python sidecar.py < /tmp/sidecar_in &
SIDECAR_PID=$!
exec 3>/tmp/sidecar_in
echo '{"cmd": "start_transcription", "id": "job-2", "paths": ["/tmp/sidecar_long.wav"], "lang_mode": "vi+en", "model": "small", "mlx": false}' >&3
sleep 5
exec 3>&-
kill -9 $SIDECAR_PID
rm /tmp/sidecar_in
```
This resumes from the Step 2 checkpoint (`chunks_done: 1`) and kills the
process mid-second-chunk, simulating a crash. Then restart clean and re-send
the same command:
```bash
mkfifo /tmp/sidecar_in
.venv/bin/python sidecar.py < /tmp/sidecar_in &
exec 3>/tmp/sidecar_in
echo '{"cmd": "start_transcription", "id": "job-3", "paths": ["/tmp/sidecar_long.wav"], "lang_mode": "vi+en", "model": "small", "mlx": false}' >&3
sleep 120
exec 3>&-
rm /tmp/sidecar_in
```
Expected: a `chunk_baseline` event with `"resume_at_sec"` around `300` (not
`0`) appears before any new `chunk_text` — confirming the sidecar process
boundary doesn't interfere with `chunking.py`'s on-disk checkpoint recovery.
Eventually `batch_done` with `"cancelled": false` and `"count": 1`.

- [ ] **Step 4: Clean up fixtures**

```bash
rm -f /tmp/sidecar_smoke.wav /tmp/sidecar_smoke*.txt /tmp/sidecar_long.wav /tmp/sidecar_long*.txt /tmp/sidecar_long.transcript.partial.json
```

No commit for this task (no file changes).

---

## What's next (not part of this plan)

Once this lands, phases 2–5 of the design doc (Tauri shell, React frontend,
model manager dialog, first-run setup) each get their own plan — they depend
on decisions (exact event payload shapes, error semantics) this plan may
adjust in review, and each is independently substantial. Write the next plan
(`desktop/src-tauri` shell wiring, phase 2) once this one is merged.
