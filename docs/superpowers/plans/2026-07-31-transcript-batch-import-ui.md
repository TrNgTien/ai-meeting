# Transcript Pane, Engine Controls & Batch Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real transcription UI on top of the sidecar bridge from
phase 2 (`docs/superpowers/plans/2026-07-30-tauri-shell-wiring.md`, done):
engine controls (language/model/GPU), a file-picker-driven batch import, and
a transcript pane that streams text as it's produced — live preview lines
included — matching `app.py`'s existing behavior exactly. This is **Build
phase 3** of `docs/superpowers/specs/2026-07-29-tauri-rust-rewrite-design.md`.
Phase 4 (this doc's `ModelManagerDialog.tsx`, already built in phase 2) and
phase 5 (first-run setup, drag-drop, polish) are separate follow-on plans.

**Reference implementation:** `app.py`'s `_build_ui()` (header controls,
`app.py:257-336`), `_start_batch_transcription()` (`app.py:1182-1238`), and
`_handle_ui_event()` (`app.py:1360-1477`, the event → widget-mutation
switch this plan's React state updates replace). Every behavior below cites
the `app.py` lines it mirrors — when in doubt, read those lines rather than
re-deriving the behavior from first principles.

**Tech stack:** React + Vite (existing `desktop/src`), the `sidecar-event`
Tauri event and `list_models`/`start_transcription`/`cancel_job` commands
from phase 2 (`desktop/src-tauri/src/commands.rs`, `sidecar.rs`). No new
Rust work — this plan is frontend-only.

## Global constraints

- Event names and payload shapes come from `sidecar.py`'s actual `emit()`
  calls (`status`, `file_start`, `chunk_baseline`, `segment_text`,
  `chunk_text`, `chunk_progress`, `batch_done`, `error`), not the design
  doc's illustrative examples — same caveat as phase 2.
- `sidecar.py` only implements the **file-import** path today
  (`cmd_start_transcription`) — there is no live-recording sidecar command
  yet (`app.py`'s `rec_started`/`rec_stopped`/`rec_failed`/`merged_text`
  events have no sidecar equivalent; recording stays out of scope for this
  plan, same as the design doc's phased build order implies).
- Only one job runs at a time (`sidecar.py`'s `_current_job_id` guard,
  verified in the phase-1 plan) — the UI must disable "Start" while a job is
  in flight, mirroring `app.py`'s `AppState.TRANSCRIBING` guard in
  `_start_batch_transcription` (`app.py:1183-1184`).
- Preview text (`segment_text`) is never on disk — it is replaced verbatim
  by the real `chunk_text` for the same span once the chunk completes.
  `app.py` implements this as a Tk text-tag whose *range* is deleted
  (`_drop_preview`), not by tracking indices — the React equivalent is
  simpler (preview is separate state, cleared wholesale on the next
  `chunk_text`/`file_start`/`batch_done`), but the invariant it protects
  (preview text must never accumulate as if it were saved text) carries
  over exactly.

---

### Task 1: Engine controls (language, model, GPU switch)

**Files:**
- Create: `desktop/src/EngineControls.tsx`
- Modify: `desktop/src/App.tsx`

**Interfaces:**
- Consumes: `list_models` command + `models` event (already wired, phase 2)
  to populate the model dropdown with `FINAL_MODEL_OPTIONS`-equivalent
  names.
- Produces: `EngineControls` — a language `<select>` (`vi+en`/`vi`/`en`/`auto`,
  default `vi+en`, mirrors `app.py:275-283`), a model `<select>` (mirrors
  `app.py:288-296`), and a GPU/MLX checkbox (mirrors `app.py:310-319` —
  only meaningful on Apple silicon, but unlike `app.py` this plan doesn't
  attempt to detect that client-side; default it on and let a failed/ignored
  `mlx` flag be a later concern, since `sidecar.py`'s `_resolve_engine`
  already falls back to CPU silently when MLX isn't available). Lifts
  `{ langMode, model, mlx }` state up to `App.tsx` via an `onChange` prop —
  `start_transcription` (Task 3) needs all three values.

- [ ] **Step 1: Build the three controls, uncontrolled defaults matching `app.py`'s** (`vi+en`, first entry of `FINAL_MODEL_OPTIONS`, MLX on)

- [ ] **Step 2: Verify manually — change each control, confirm the lifted state in `App.tsx` (temporary console.log) reflects the change**

- [ ] **Step 3: Commit**

```bash
git add desktop/src/EngineControls.tsx desktop/src/App.tsx
git commit -m "feat(desktop): add engine controls (language/model/GPU)"
```

---

### Task 2: Batch import (file picker + validation)

**Files:**
- Create: `desktop/src/BatchImportBar.tsx`
- Modify: `desktop/src/App.tsx`

**Interfaces:**
- Consumes: `@tauri-apps/plugin-dialog`'s `open()` (already a dependency,
  `desktop/package.json`) for a native multi-file picker — `tauri-plugin-dialog`
  is already registered in `lib.rs`, so no Rust change is needed here.
- Produces: `BatchImportBar` — an "Import files…" button that opens the
  picker, filters to the audio extensions `sidecar.py`'s `AUDIO_EXTS` accepts
  (`.mp3 .wav .m4a .aac .flac .ogg .opus .wma .mp4`), and calls an `onImport(paths:
  string[])` prop. Matches `app.py`'s `_start_batch_transcription`'s own
  filter (`app.py:1186-1188`) so a mismatched extension is rejected the same
  way in both apps rather than only server-side in `sidecar.py`.

- [ ] **Step 1: Wire the picker with an extension filter matching `AUDIO_EXTS`**

- [ ] **Step 2: Verify manually — pick a `.mp3` and a `.txt` in the same
  dialog if the picker allows it, confirm only the audio file makes it into
  `onImport`'s argument (or that the dialog's own filter prevents selecting
  the `.txt` at all — either is acceptable, whichever the plugin does by
  default)**

- [ ] **Step 3: Commit**

```bash
git add desktop/src/BatchImportBar.tsx desktop/src/App.tsx
git commit -m "feat(desktop): add batch import file picker"
```

---

### Task 3: Wire Start/Cancel to `start_transcription`/`cancel_job`

**Files:**
- Modify: `desktop/src/App.tsx`

**Interfaces:**
- Consumes: `EngineControls` state (Task 1), `BatchImportBar` (Task 2),
  `start_transcription`/`cancel_job` commands (phase 2,
  `desktop/src-tauri/src/commands.rs`).
- Produces: a `jobId` (client-generated, e.g. `crypto.randomUUID()` — matches
  the design doc's job-id-for-cancellation rationale) held in `App.tsx`
  state; `null` when idle. "Start" is disabled while `jobId` is set
  (mirrors `app.py:1183-1184`'s `AppState.TRANSCRIBING` guard); "Cancel"
  calls `cancel_job` with that id and is disabled while `jobId` is `null`.

- [ ] **Step 1: On import, call `start_transcription` with a fresh `jobId`
  and the current engine-control values; store `jobId` in state**

- [ ] **Step 2: On the `batch_done` sidecar event (Task 4 wires the listener
  that dispatches this), clear `jobId` back to `null` regardless of
  `cancelled`/error, since `sidecar.py` always emits exactly one
  `batch_done` per job whether it finished, errored per-file, or was
  cancelled (`sidecar.py`'s `cmd_start_transcription` worker, and the
  phase-1 plan's Task 5 verification of the cancel path)**

- [ ] **Step 3: Verify manually — start a job, confirm Start is disabled and
  Cancel is enabled; click Cancel mid-job, confirm the UI returns to idle
  once `batch_done` (`cancelled: true`) arrives**

- [ ] **Step 4: Commit**

```bash
git add desktop/src/App.tsx
git commit -m "feat(desktop): wire batch import to start_transcription/cancel_job"
```

---

### Task 4: Transcript pane — status, saved text, live preview

**Files:**
- Create: `desktop/src/TranscriptPane.tsx`
- Modify: `desktop/src/App.tsx` (replace the Task-4-of-phase-2 scratch event
  log with this; remove the temporary `<pre>`/"List models" button as dead
  scaffolding once this pane covers the same ground for real content)

**Interfaces:**
- Consumes: the `sidecar-event` listener — this task is where a real
  `switch (payload.event)` dispatcher replaces the phase-2 scratch listener
  that just stringified everything. Mirrors `app.py`'s
  `_handle_ui_event` (`app.py:1360-1477`) case-by-case:
  - `status` → status line text (`app.py:1361-1362`)
  - `file_start` → append a `===== {name} =====` separator, mirroring
    `app.py:1373-1379` (including the blank-line-between-files rule when the
    pane isn't empty)
  - `chunk_baseline` → informational only for now (`app.py` uses it to seed
    an ETA timer via `_start_live`; this plan doesn't build the live
    ETA/timer UI — track it as a follow-up, not silently dropped)
  - `segment_text` → append to a **separate** `preview` state (dimmed
    styling), never mixed into saved text (`app.py:1390-1396`)
  - `chunk_text` → clear `preview`, append to saved text (`app.py:1397-1402`)
  - `chunk_progress` → informational only for now, same follow-up note as
    `chunk_baseline`
  - `batch_done` → append the saved-count/elapsed summary; the actual
    "cleared to idle" state transition is Task 3's job, this task only
    renders the outcome text
  - `error` → append `[error: {message}]` inline, matching `app.py`'s
    workaround at the time `_handle_ui_event` was ported (`app.py:1220`
    pushes `chunk_text` for per-file errors during the batch loop) — but
    prefer sidecar.py's real `error` event (`{"event": "error", "file":
    ..., "message": ...}`) since, per the phase-1 plan's documented
    departure, that event now has an actual producer and doesn't need the
    `chunk_text`-stuffing workaround `app.py` was stuck with.
  - `_sidecar_exited` (phase-2's synthetic bridge event) → show a
    "backend disconnected, restart the app" banner instead of leaving the
    UI hung waiting for a `batch_done` that will never come.

- [ ] **Step 1: Build `TranscriptPane` — a status line, a preview line
  (dimmed, replaced wholesale on the next `chunk_text`), and a scrolling
  saved-text area**

- [ ] **Step 2: Route every sidecar event above into the right piece of
  state; delete the phase-2 scratch `<pre>` event log and its button**

- [ ] **Step 3: Verify manually — run a real short audio file through
  Start; confirm the status line updates, preview text appears and is
  dimmed, then gets replaced by the real chunk text, and the final
  saved-count summary appears after `batch_done`**

- [ ] **Step 4: Commit**

```bash
git add desktop/src/TranscriptPane.tsx desktop/src/App.tsx
git commit -m "feat(desktop): add transcript pane with live preview"
```

---

## What's next (not part of this plan)

- **Live ETA/progress bar** from `chunk_progress`/`chunk_baseline`
  (`app.py`'s `_start_live`/`_update_chunk_progress`) — deliberately
  deferred in Task 4 above rather than half-built.
- **Live recording** (`rec_started`/`rec_stopped`/`merged_text`,
  `audio_capture.py`'s two-track capture) — `sidecar.py` has no recording
  command yet; needs its own sidecar-side plan before any UI work.
- **First-run setup screen, drag-drop import, reveal-in-Finder, window
  polish** — phase 5 of the design doc, still unplanned.
