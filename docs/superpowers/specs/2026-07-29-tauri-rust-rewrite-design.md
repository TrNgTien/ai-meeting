# Design: Tauri/Rust rewrite of the meeting transcriber

## Goal

Replace the CustomTkinter desktop shell with a Tauri (Rust) + React desktop app,
following the shape of [Zackriya-Solutions/meetily](https://github.com/Zackriya-Solutions/meetily)
(Tauri + Rust backend + web frontend), while preserving the current transcription
behavior exactly: engine routing (`vi+en`/`en`/`auto`/`vi`), MLX GPU acceleration,
WhisperX + PhoWhisper for pure Vietnamese, the hallucination-control logic
(`transcriber.py:39-136`), and the chunked/resumable/checkpointed transcription
pipeline (`chunking.py`).

## Why hybrid instead of a full native (whisper.cpp) rewrite

meetily uses `whisper.cpp` via Rust bindings for transcription. This repo's
accuracy work is built on PyTorch-ecosystem tooling with no Rust equivalent:

- **WhisperX** — VAD + forced alignment + batched inference, used for the `vi` engine.
- **MLX** — Apple-GPU-accelerated Whisper, used for `vi+en`/`en`/`auto` on Apple silicon.
- **PhoWhisper-large / CTranslate2** — the `vi`-only checkpoint, no Rust binding exists.
- **Hand-tuned hallucination control** (`transcriber.py`) — decode options
  (`condition_on_previous_text=False`, logprob/no-speech/compression-ratio
  thresholds), a Vietnamese/English YouTube-outro regex bank, and back-to-back
  sentence-repeat collapsing — all tuned against openai-whisper's/WhisperX's
  decode behavior specifically.

A full rewrite onto `whisper.cpp` would require re-deriving all of the above
against a different decoder, risking regressions in exactly the behavior this
project exists to get right. Instead: **Rust/Tauri owns the shell (native
packaging, UI, process orchestration); the existing Python engine modules run
unchanged as a subprocess ("sidecar").**

## Architecture

```
Tauri app (Rust)                          Python sidecar (persistent subprocess)
┌─────────────────────────┐   JSON lines  ┌──────────────────────────────────┐
│ src-tauri/               │◄─────────────►│ sidecar.py (NEW, thin)           │
│  - spawns/owns sidecar   │   stdin/stdout│  - command loop                 │
│  - Tauri commands        │               │  - ports app.py's routing glue: │
│    (start/cancel/import/ │               │    _prepare_engine,             │
│     model list/dl/del)   │               │    _run_final_transcription,    │
│  - forwards sidecar      │               │    model manager actions        │
│    events -> frontend    │               │       │                        │
│    via Tauri events      │               │       ▼                        │
├─────────────────────────┤               │  transcriber.py  (UNCHANGED)     │
│ React+Vite frontend      │               │  chunking.py     (UNCHANGED)     │
│  - transcript pane       │               │  mlx_engine.py   (UNCHANGED)     │
│    (+ live preview tag)  │               │  phowhisper.py   (UNCHANGED)     │
│  - model manager dialog  │               └──────────────────────────────────┘
│  - batch import list     │
│  - progress bars         │
└─────────────────────────┘
```

`transcriber.py`, `chunking.py`, `mlx_engine.py`, and `phowhisper.py` have no
Tkinter dependency today — they are already engine-agnostic. Only `app.py`'s
routing/orchestration glue (`_prepare_engine()`, `_run_final_transcription()`,
the batch loop, model-manager actions — all currently wired to `self._ui_queue`)
needs to be re-expressed. `sidecar.py` is a new, thin Python file that
re-implements that same glue as a stdin/stdout JSON command loop instead of Tk
callbacks, calling the exact same engine functions with the exact same
parameters. `app.py` (the CustomTkinter GUI) is left in place, untouched, and
functional until the Tauri app reaches feature parity, at which point it can be
removed.

## Decisions locked in

| Question | Decision |
| --- | --- |
| Transcription engine strategy | Hybrid: Tauri/Rust shell + Python sidecar reusing all four existing engine modules unchanged |
| Python provisioning | Keep the current `setup.sh`/`.venv` flow — Tauri shells out to `.venv/bin/python`, no bundled/frozen Python runtime |
| Frontend stack | React + Vite (Tauri's standard template) — not Next.js, not framework-less |
| Feature scope for v1 | Full parity with the current app: engine routing, batch import, model manager, live preview, resumable chunked transcription, cancel |

## Repo layout

```
ai-meeting/
  app.py, chunking.py, transcriber.py, mlx_engine.py, phowhisper.py   # UNCHANGED
  sidecar.py                        # NEW — JSON-line command loop, replaces app.py's Tk-glue
  desktop/                          # NEW — the Tauri app
    src-tauri/
      src/
        sidecar.rs                 # spawn/own the persistent `python sidecar.py` process
        commands.rs                # #[tauri::command]s: start_transcription, cancel,
                                    #   list_models, download_model, delete_model, import_files
        events.rs                  # parses sidecar stdout JSON lines -> re-emits as Tauri events
    src/                            # React+Vite frontend
      TranscriptPane.tsx            # live preview tag + saved text, mirrors _append_transcript/_drop_preview
      ModelManagerDialog.tsx
      BatchImportBar.tsx
      EngineControls.tsx            # language dropdown, model dropdown, GPU(MLX) switch
```

## IPC protocol (sidecar.py ↔ Rust)

One JSON object per line on stdin/stdout. Event names mirror the current
`_ui_queue` tuple names in `app.py` 1:1, so each event's meaning traces
directly back to existing, working code.

**Rust → sidecar (commands):**

```json
{"cmd": "start_transcription", "id": "job-1", "paths": ["standup.m4a"], "lang_mode": "vi+en", "model": "large-v3", "mlx": true}
{"cmd": "cancel", "id": "job-1"}
{"cmd": "list_models"}
{"cmd": "download_model", "name": "large-v3-turbo"}
{"cmd": "delete_model", "name": "medium"}
```

**sidecar → Rust (events):**

```json
{"event": "status", "message": "Transcribing 2/5: standup.m4a"}
{"event": "chunk_baseline", "resume_at_sec": 1500}
{"event": "chunk_text", "text": "[00:25:04] ..."}
{"event": "segment_text", "text": "...", "tag": "preview"}
{"event": "mm_download_progress", "name": "large-v3-turbo", "downloaded": 12345, "total": 987654}
{"event": "file_start", "name": "standup.m4a"}
{"event": "done", "id": "job-1"}
```

`sidecar.py` is spawned once at app launch and stays alive for the whole
session — this matches the existing single-worker-thread assumption baked
into `transcriber.stream_segments()` (only one transcription runs at a time,
which is what makes its stdout-redirection trick for streaming segments safe
today). Rust reads sidecar stdout line-by-line on a background task and
forwards each parsed event to the frontend via `app_handle.emit()`. The
frontend subscribes to these events and updates React state, replacing
`_poll_ui_queue()`/`_handle_ui_event()`'s role of mutating Tk widgets.

## First-run setup

On launch, Rust checks whether `.venv/.deps-installed` exists. If not, it runs
`setup.sh` as a child process, streaming its stdout into a "Setting up…"
screen in the Tauri window (so the user isn't required to open a terminal),
then spawns the sidecar once setup finishes. This is the same script and
`.venv` layout `make`/`Makefile` already use — Tauri just triggers it instead
of requiring a manual `make setup` first.

## Build phases

1. `sidecar.py` + IPC schema, tested headlessly (send JSON commands on stdin
   by hand, verify events on stdout) — no Rust or UI yet.
2. Tauri shell: spawn the sidecar, wire the commands/events above, minimal
   plain-HTML view just to prove the pipe end-to-end.
3. React frontend: transcript pane with live preview tag, engine controls,
   batch import.
4. Model manager dialog (list/download/delete/cancel, progress).
5. First-run setup screen + polish (drag-drop, resize, status/detail bar,
   reveal-in-Finder/file-manager).

## Out of scope for this design

- **Headless CLI / Docker path** (`make transcribe`, `make run-docker-cli`):
  unaffected by this rewrite — both invoke `transcriber.py` directly and stay
  exactly as they are.
- **Bundling a self-contained Python runtime** inside the Tauri app (so it's a
  single double-click installer with no separate setup step): explicitly
  deferred — see "Python provisioning" decision above. Worth revisiting once
  the hybrid shell is proven out.
- **Windows support**: not currently supported (WSL2 workaround only) and this
  rewrite does not change that, even though Tauri makes native Windows support
  easier in principle.
- **Retiring `app.py`**: stays in place and working throughout the migration;
  removing it is a decision for after the Tauri app reaches parity, not part
  of this design.

## Testing / verification

There is no automated test suite in this repo today (per `CLAUDE.md`); this
rewrite doesn't change that posture. Verification is manual, by exercising the
same paths the existing "Tips & troubleshooting" README section documents:
running each language mode, killing and resuming a long transcription
mid-chunk, downloading/deleting a model, and confirming the CLI/Docker path
(unaffected) still works via `make transcribe`.
