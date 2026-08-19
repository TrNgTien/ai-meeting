# Tauri Shell Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `desktop/src-tauri` to spawn `sidecar.py` as a child process, bridge
its JSON-line stdout events to the frontend as Tauri events, expose Tauri
commands for every `sidecar.py` command, and build a minimal React UI —
including the model-manager dialog — on top of that bridge. This is **Build
phase 2** of `docs/superpowers/specs/2026-07-29-tauri-rust-rewrite-design.md`,
following phase 1 (`sidecar.py`, done — see
`docs/superpowers/plans/2026-07-29-sidecar-json-ipc.md`).

**Architecture:** `desktop/src-tauri` spawns `.venv/bin/python sidecar.py`
(resolved relative to the repo root, one directory up from `desktop/`) once at
app launch and keeps it alive for the whole session, matching the sidecar's
own single-worker-thread assumption. A background thread/task reads the
child's stdout line-by-line, parses each line as JSON, and re-emits it as a
Tauri event (`app_handle.emit("sidecar-event", payload)`) for the frontend to
subscribe to. Tauri `#[tauri::command]`s write JSON-encoded commands to the
child's stdin, one per line — this is a fire-and-forget send, not a
request/response call; the actual result comes back asynchronously as a
sidecar event the frontend correlates by `id` (for jobs) or `name` (for
downloads).

**Tech Stack:** Rust (existing `desktop/src-tauri` crate), no new heavy
dependency needed — `std::process::Command` with piped stdin/stdout plus a
`std::thread::spawn` reader loop covers this; `tauri-plugin-shell` is not
required since sidecar.py always ships alongside this repo (not a bundled
sidecar binary per Tauri's sidecar convention) and `std::process` is enough.
React + Vite (already scaffolded in `desktop/src`).

## Global constraints

- `sidecar.py`'s event and command shapes are exactly what's implemented
  today, **not** the illustrative examples in the design doc (which predate
  the actual implementation and review). Concretely: use `mm_progress` /
  `mm_download_finished` (not `mm_download_progress` / `done`), and note that
  `batch_done` (not a per-job `done`) is the terminal event for
  `start_transcription`. Cross-check every payload shape against
  `sidecar.py`'s `emit()` call sites before wiring a Rust struct to it —
  don't trust the design doc's JSON examples verbatim.
- Exactly one `sidecar.py` process per app session. Do not spawn a second one
  per command — every Tauri command in this plan writes to the *same* child's
  stdin.
- `AppState` in `desktop/src-tauri/src/state.rs` today models the (unused,
  pure-Rust) engine path — `Phase`, `CancelFlag`, `language`, `model` fields
  assume Rust-side transcription. This plan does not delete that file (the
  pure-Rust `transcribe/` modules stay, per the design doc's "hybrid" framing
  being about the *shell*, not a promise to delete the Rust engine code) but
  adds a **separate** `SidecarState` for the child process handle rather than
  overloading the existing struct.
- No test suite convention exists for `desktop/src-tauri` beyond `cargo test`
  (102 tests today, all for the untouched pure-Rust modules). This plan adds
  no automated tests for the sidecar bridge itself — process/IPC plumbing is
  verified manually (per the design doc's "Testing / verification" section)
  the same way `sidecar.py` was: real JSON round-trips, not mocks.

---

### Task 1: Spawn and own the sidecar process

**Files:**
- Create: `desktop/src-tauri/src/sidecar.rs`
- Modify: `desktop/src-tauri/src/lib.rs`

**Interfaces:**
- Produces: `pub struct SidecarState` (holds `Mutex<std::process::ChildStdin>`
  and an `Arc` for the reader thread to signal shutdown); `pub fn spawn(app: &tauri::AppHandle) -> anyhow::Result<SidecarState>` —
  resolves the repo root (`desktop/`'s parent directory) via
  `std::env::current_exe()` in release or `CARGO_MANIFEST_DIR` in dev,
  spawns `<repo_root>/.venv/bin/python <repo_root>/sidecar.py` with piped
  stdin/stdout/stderr, and starts the stdout-reader thread (Task 2 wires what
  that thread does with each line).

- [ ] **Step 1: Resolve the repo root and `.venv` python path reliably in both `cargo tauri dev` and a built app**

`cargo tauri dev` runs with CWD `desktop/src-tauri`; a built `.app` bundle's
CWD is unrelated to the source tree entirely. Do not hardcode a relative
path — in dev, walk up from `CARGO_MANIFEST_DIR` (`desktop/src-tauri` ->
`desktop` -> repo root); document that this only works for `cargo tauri dev`
today and that a bundled `.app` (out of scope until phase 5 per the design
doc) will need `sidecar.py`/`.venv` bundled as Tauri resources instead of a
path lookup.

- [ ] **Step 2: Spawn the child with piped stdio**

```rust
Command::new(venv_python)
    .arg(sidecar_py)
    .current_dir(repo_root)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
```
Capture stderr too (a separate reader thread that just logs each line via
`eprintln!`/`log::warn!`) — `sidecar.py` never writes to stderr in normal
operation, so any line there is a real crash/traceback worth surfacing during
development.

- [ ] **Step 3: Verify manually — spawn, send `list_models`, see it echoed to the Rust side**

Temporarily call `spawn()` from `run()` in `lib.rs`, write `{"cmd":
"list_models"}\n` to its stdin right after spawn, and `println!` whatever the
reader thread (stub: just print raw lines for now) receives. Run `make
desktop-dev` and confirm the `models` JSON line appears in the terminal
running `cargo tauri dev`. Remove the temporary send/print once confirmed —
Task 2 replaces the stub with the real event bridge.

- [ ] **Step 4: Commit**

```bash
git add desktop/src-tauri/src/sidecar.rs desktop/src-tauri/src/lib.rs
git commit -m "feat(desktop): spawn and own the sidecar.py process"
```

---

### Task 2: Bridge sidecar stdout to Tauri events

**Files:**
- Modify: `desktop/src-tauri/src/sidecar.rs`

**Interfaces:**
- Consumes: `SidecarState` (Task 1).
- Produces: the reader thread parses each stdout line as
  `serde_json::Value`, reads its `"event"` field, and calls
  `app_handle.emit("sidecar-event", line_as_json)` — forward the *whole*
  parsed object, not a re-typed Rust struct, since the frontend (Task 5) can
  destructure by `event` name directly and this avoids the bridge having to
  track every payload shape as sidecar.py's protocol evolves.

- [ ] **Step 1: Reader thread — one JSON line in, one Tauri event out**

```rust
std::thread::spawn(move || {
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() { continue; }
        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(value) => { let _ = app_handle.emit("sidecar-event", value); }
            Err(err) => eprintln!("sidecar: bad JSON line ({err}): {line}"),
        }
    }
    // stdout closed: the sidecar process exited. Emit a distinct event so
    // the frontend can show "backend disconnected" rather than hanging.
    let _ = app_handle.emit("sidecar-event", json!({"event": "_sidecar_exited"}));
});
```
The synthetic `_sidecar_exited` event (underscore prefix signals "not a real
sidecar.py event, bridge-internal") matters because every `start_transcription`
result depends on `batch_done` arriving — if the process dies mid-job, the
frontend needs to stop waiting rather than spin forever.

- [ ] **Step 2: Verify with the same manual `list_models` send from Task 1 Step 3, but now listening on the frontend side**

In `desktop/src/main.tsx` or a scratch effect, `listen("sidecar-event", console.log)`
and confirm the `models` event lands in the browser devtools console when
sent from Rust. This proves the full pipe: Python stdout -> Rust reader ->
Tauri event -> JS listener.

- [ ] **Step 3: Commit**

```bash
git add desktop/src-tauri/src/sidecar.rs
git commit -m "feat(desktop): bridge sidecar stdout lines to Tauri events"
```

---

### Task 3: Tauri commands that write to sidecar stdin

**Files:**
- Create: `desktop/src-tauri/src/commands.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (register commands, `.manage(SidecarState)`)

**Interfaces:**
- Consumes: `SidecarState` (Task 1).
- Produces: `#[tauri::command] list_models`, `download_model(name: String)`,
  `delete_model(name: String)`, `cancel_download(name: String)`,
  `start_transcription(id: String, paths: Vec<String>, lang_mode: String,
  model: String, mlx: bool)`, `cancel_job(id: String)` — each one serializes
  its args into the exact `{"cmd": ...}` shape `sidecar.py` expects (verify
  every field name against `sidecar.py`'s `cmd_*` methods, not the design
  doc) and writes it + `"\n"` to the locked `ChildStdin`, then flushes.

- [ ] **Step 1: One helper, `fn send(state: &SidecarState, msg: serde_json::Value) -> Result<(), String>`**

Locks the stdin mutex, writes `msg.to_string() + "\n"`, flushes, maps any IO
error to a `String` (Tauri commands need `Result<T, E: Serialize>` — a plain
`String` error is enough here, matching this repo's "no premature
abstraction" convention). Every command below is a one-line call into this
helper with its own JSON literal.

- [ ] **Step 2: Implement all six commands**

Match `sidecar.py`'s actual dispatch: `cmd_list_models` takes no args;
`cmd_download_model`/`cmd_delete_model` take `name`; `cmd_cancel` takes either
`name` (download) or `id` (job) — so `cancel_download` and `cancel_job` are
two distinct Tauri commands even though they both map to `sidecar.py`'s one
`cmd_cancel`, because Rust's type system should make the two cancellable
things distinct at the call site rather than relying on the caller to pass
the right key in an untyped payload.

- [ ] **Step 3: Verify each command from the frontend with `invoke()`**

`await invoke("list_models")`, confirm (via the Task 2 listener) the `models`
event arrives. Repeat for `download_model` with a small model and watch
`mm_progress`/`mm_download_finished` land.

- [ ] **Step 4: Commit**

```bash
git add desktop/src-tauri/src/commands.rs desktop/src-tauri/src/lib.rs
git commit -m "feat(desktop): add Tauri commands that drive the sidecar"
```

---

### Task 4: Minimal plain view proving the pipe end-to-end

**Files:**
- Modify: `desktop/src/App.tsx`

**Interfaces:** none new — this task is deliberately UI-poor, per the design
doc's phase 2 scope ("minimal plain-HTML view just to prove the pipe
end-to-end"). Real UI is Task 5.

- [ ] **Step 1: Replace the placeholder with a `<pre>` event log and one button**

A button that calls `invoke("list_models")`, and a `<pre>` that appends every
`sidecar-event` payload as JSON. This is scaffolding, not product — it
exists so a human can watch the whole round trip in the actual running app
(not just devtools console) before investing in real components.

- [ ] **Step 2: Verify — run `make desktop-dev`, click the button, see the model list JSON appear in the window**

- [ ] **Step 3: Commit**

```bash
git add desktop/src/App.tsx
git commit -m "feat(desktop): minimal event-log view proving the sidecar pipe works"
```

---

### Task 5: Model manager UI

**Files:**
- Create: `desktop/src/ModelManagerDialog.tsx`
- Modify: `desktop/src/App.tsx` (mount point / open trigger)

**Interfaces:**
- Consumes: `list_models`, `download_model`, `delete_model`,
  `cancel_download` commands (Task 3); `models`, `mm_progress`,
  `mm_download_finished`, `model_deleted` events (Task 2).
- Produces: `ModelManagerDialog` — a list of models from `FINAL_MODEL_OPTIONS`
  (mirrors `sidecar.py`'s `cmd_list_models`: name, downloaded, size_bytes),
  a Download/Delete/Cancel action per row, and a progress bar driven by
  `mm_progress` while a download for that row is in flight.

- [ ] **Step 1: Call `list_models` on dialog open, render the table**

- [ ] **Step 2: Wire Download — call the command, subscribe to `mm_progress`
  filtered by `model === row.name`, update a per-row progress bar, and on
  `mm_download_finished` (status `done`/`cancelled`/`error`) clear it and
  re-fetch `list_models` (simplest way to get the authoritative
  `downloaded`/`size_bytes` state back, rather than hand-computing it
  client-side)**

- [ ] **Step 3: Wire Delete and Cancel similarly**

- [ ] **Step 4: Verify manually — download a small model, confirm the
  progress bar moves and the row flips to "downloaded"; delete it, confirm
  the row flips back; start a download and cancel it mid-flight, confirm
  `mm_download_finished` with `status: "cancelled"` arrives and the partial
  file is gone (`ls ~/.cache/whisper-cpp` — wait, this UI drives the
  *Python* sidecar, so the cache is `~/.cache/whisper`, not
  `whisper-cpp` — that path belongs to the unrelated pure-Rust
  `transcribe/models.rs`; don't confuse the two caches when verifying this
  step)**

- [ ] **Step 5: Commit**

```bash
git add desktop/src/ModelManagerDialog.tsx desktop/src/App.tsx
git commit -m "feat(desktop): add model manager dialog"
```

---

## What's next (not part of this plan)

Per the design doc's build phases: phase 3 (transcript pane with live preview
tag, engine controls, batch import) and phase 4/5 (first-run setup screen,
drag-drop, polish) each still need their own plan once this one lands — they
depend on UI/UX decisions (how live preview text is visually distinguished,
what the batch import list looks like) this plan doesn't make.
