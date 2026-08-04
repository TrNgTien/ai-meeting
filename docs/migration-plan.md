# Python → Rust: full migration plan

**Status: all phases landed, including Phase 5.** The Python tree is deleted, the
Makefile and docs now describe a single Rust app, and `desktop/` was moved to the
repo root. Chunk parity was verified (`desktop-parity` on a real meeting file)
immediately before the delete.

| Phase | State |
| --- | --- |
| 0 — Live recording wired | Done. `src-tauri/src/recording.rs`, `src/RecordBar.tsx`. |
| 1 — Two-track merge wired | Done. `engine.rs:run_recording`. |
| 2 — ffmpeg/ffprobe bundled | Done. `scripts/build-ffmpeg.sh`, LGPL, ~3 MB each. |
| 3 — Settings persisted | Done. `src-tauri/src/settings.rs`, `src/lib/settings.ts`. |
| 4 — Parity gaps closed | Done. ETA UI, pre-download size, `src/bin/transcribe.rs`. |
| 5 — Delete Python, rewrite docs | **Done.** See "What changed while executing this plan". |

## Context

Two implementations live side by side: the Python/CustomTkinter reference app at the repo
root, and the Rust/Tauri 2 + React port in `desktop/`. The goal is one implementation —
delete Python entirely, ship a self-contained `.app`.

The port is further along than the docs suggest. Transcription (whisper.cpp + Metal),
chunking, crash-safe checkpoints, hallucination + silence filtering, model download and
management, and the whole import/transcribe UI are **done**. Recording (`src/audio/*`,
7 files, ~1 500 lines) and transcript merge (`src/merge.rs`) are **fully written and
unit-tested, but reachable from no Tauri command** — `Phase::Recording` is never set.
There is already zero Python at Rust runtime; the only `Command::new` calls are `ffmpeg`
and `ffprobe`.

So the migration is mostly **wiring, not writing**. Four decisions taken up front:

- **`vi`/PhoWhisper mode is dropped.** `vi+en` on a multilingual checkpoint covers the
  same audio. Already reflected in `state.rs:29-53` — `"vi"` parses to `ViEn`.
- **MLX is dropped.** Metal is a compile-time `whisper-rs` feature; no runtime toggle.
- **ffmpeg/ffprobe get bundled** as Tauri sidecars — no `brew install` prerequisite.
- **Settings get persisted** (Python persisted nothing; every choice reset on quit).

---

## Phase 0 — Wire live recording ✅

Backend exists. Missing: commands, phase transitions, UI.

**Rust** — new commands in `desktop/src-tauri/src/commands.rs`, registered in
`src/lib.rs:39-47`, backed by handlers on `EngineHost` (`src/engine.rs`):

| Command | Uses |
| --- | --- |
| `list_input_devices()` | `audio::devices::list_input_devices()` (`devices.rs:28`) |
| `start_recording(record_mic, record_system, mic_device_id, backend)` | `MeetingRecorder::start(out_dir, RecorderOptions)` (`recorder.rs:98`) |
| `stop_recording()` | `MeetingRecorder::stop() -> Recording` (`recorder.rs:238`) |
| `recording_levels()` | `MeetingRecorder::levels()` + `elapsed()` (`recorder.rs:217-226`) |

- Hold `Option<MeetingRecorder>` in `EngineHost::Inner` next to the job slot. Reject
  `start_recording` unless `Phase::Idle`; set `Phase::Recording` on success, back to
  `Idle` on failure — mirrors `app.py:986-1011` and `_set_state` (`app.py:933-958`).
- New events on the existing `engine-event` channel, same shapes as Python's `_ui_queue`:
  `rec_started {system_description, warnings}`, `rec_failed {message}`,
  `rec_stopped {Recording}` (already `Serialize`, `recorder.rs:23`).
- Levels: poll from the frontend on a 100 ms timer (matches Python's `METER_TICK_MS`);
  a command returning `{mic, system, elapsed_sec}` is simpler than a push event.

**Output directory** — Python wrote to a repo-relative `recordings/` (`app.py:73`), which
is wrong inside a bundled `.app`. Assumption: `~/Documents/Transcriber/recordings/`,
created on demand, revealed via the existing `opener` plugin. Same `<stem>-me.wav` /
`<stem>-meeting.wav` naming (`recorder.rs:114`).

**Frontend** — record row in `desktop/src/App.tsx` (or a new `RecordBar.tsx`): record
button, mic/system checkboxes, mic device dropdown (reuse `components/Dropdown.tsx`),
two level meters, elapsed timer. Disable Import + language/model controls while recording,
per the Python transition table.

**Permissions** — `src-tauri/Info.plist` has only `NSMicrophoneUsageDescription`. Add
`NSScreenCaptureUsageDescription`, or ScreenCaptureKit fails silently on first use.
Reference `Info.plist` explicitly from `tauri.conf.json` rather than relying on the
bundler's implicit merge.

## Phase 1 — Wire the two-track merge ✅

`merge_transcript_files` (`src/merge.rs:150`) and `conversation_path` (`:174`) are written
and tested; nothing calls them.

In `engine.rs`, after `rec_stopped`, feed `Recording::paths()` into the existing
`run_batch` (`engine.rs:241`) with `Me` / `Meeting` labels, then merge into
`<stem>-conversation.txt` with header `# Meeting recorded YYYY-MM-DD HH:MM (dur)`
(`app.py:1113`). Emit `merged_text {path}`; the frontend clears the transcript pane and
loads the merged file, as `app.py:1434-1443` did.

**A cancelled run must skip the merge** — half a conversation reads worse than two
transcripts (`app.py:1100-1112`). Preserve that.

## Phase 2 — Bundle ffmpeg + ffprobe ✅

Removes the last external runtime dependency.

- Add static **LGPL** macOS arm64 builds (decode-only; no GPL encoders) as
  `src-tauri/binaries/ffmpeg-aarch64-apple-darwin` and `ffprobe-…`, declared under
  `bundle.externalBin` in `tauri.conf.json`.
- `src/chunking/decode.rs` currently hardcodes `Command::new("ffmpeg")` (`:25`) and
  `"ffprobe"` (`:88`). Introduce one resolver — bundled sidecar path first, `PATH`
  fallback — and route both call sites plus `ffmpeg_available()` (`:110`) through it.
  Keep the `Command`/stdout-pipe shape unchanged so the `-f s16le … /32768.0` decode
  stays bit-identical (`decode.rs:76-81`).
- Call `ffmpeg_available()` at startup and surface a clear error instead of failing at
  the first chunk — it is written but never called today.
- Fetch the binaries in `make desktop-setup`; add a licence note to the README.
- Sanity-check `.dmg` size after (currently 255.9 MB; expect ~+80 MB).

## Phase 3 — Persist settings ✅

New `src/settings.rs`: `{language_mode, model, record_mic, record_system, mic_device_id}`
as JSON in Tauri's app-config dir. Load in `run()` to seed `AppState` (`state.rs:97-106`),
save on every change. Commands `load_settings` / `save_settings`.

Also fix the default mismatch while here: `DEFAULT_MODEL = "large-v3"`
(`transcribe/mod.rs:19`) vs the React default `"small"` (`App.tsx:21`). Settings become
the single source; `large-v3` wins on first run.

## Phase 4 — Close the remaining parity gaps ✅

Small, independent items:

- **Progress / ETA.** `chunk_baseline` and `chunk_progress` are emitted and explicitly
  ignored by the frontend (`TranscriptPane.tsx:56-59`). Port Python's live ticker
  (`app.py:1597-1641`): elapsed, `~N left` from the audio-sec/wall-sec rate, and the
  "progress saved" hint.
- **Pre-download size.** `models::remote_size()` (`models.rs:139`) is written, never
  called — show the size before a download starts, as the Manage Models dialog did.
- **Headless CLI.** Promote `examples/transcribe.rs` to a real `[[bin]]` so
  `make transcribe FILE=… LANG_MODE=…` keeps working after `transcriber.py` dies.
- **Warnings surfacing.** Render `Recording::warnings` as `[note] …` lines, per
  `app.py:1414-1429` — a meeting is not repeatable, so a missing side must be visible.

## Phase 5 — Cutover ✅

**Delete from the repo root:** `app.py`, `transcriber.py`,
`chunking.py`, `audio_capture.py`, `transcript_merge.py`, `mlx_engine.py`,
`phowhisper.py`, `setup.sh`, `requirements.txt`, `Dockerfile`, `.venv/`, the
stray `path/to/venv/`, and `desktop/scripts/{ab_python,chunk_parity}.py` —
all done.

**Makefile:** the Python targets (`start`, `setup`, `run-local`, `build`,
`run-docker-cli`, `clean`) and the `$(STAMP)` venv machinery are gone; the
default target is `dev`; `transcribe` runs the Rust bin; the `desktop-*`
targets became `setup` / `dev` / `build` / `test` / `release` / `install`.

**Docs:** `CLAUDE.md` and `README.md` were rewritten around the single Rust
app. The README notes that `~/.cache/whisper`, `~/.cache/ai-meeting` and the
MLX HF snapshots are dead weight and can be deleted by hand — caches are never
deleted programmatically.

**`desktop/**` was moved to the repo root** as a separate step of the same
commit, per the "final, separate commit" note below.

---

## Verification

Run at each phase, not only at the end. There is no test suite for the Python side, so the
reference is real audio.

1. `make desktop-test` — 106 existing `#[test]`s plus `tests/chunked_run.rs`; the
   durability/resume invariants must stay green through every phase.
2. **Decode parity, before deleting Python.** `make desktop-parity` diffs Rust chunk
   boundaries and sample checksums against `scripts/chunk_parity.py`. Run it after the
   Phase 2 ffmpeg-sidecar change — that is the one edit that could shift decoded samples.
3. **Transcript A/B, before deleting Python.** `.venv/bin/python desktop/scripts/ab_python.py`
   on 2–3 real meeting files, `vi+en` and `en`. Expect wording differences between engines;
   what must match is timestamp alignment and the absence of hallucinated outros.
4. **Recording, end to end.** Record ~2 min with something playing, both sides on; confirm
   two WAVs land in the recordings dir, `-conversation.txt` interleaves `Me` / `Meeting`
   correctly, and drift stays under 100 ms on a 30-minute run. Then repeat with the mic
   disabled, and with Screen Recording permission denied — both must warn, not crash.
5. **Crash safety.** Kill the app mid-chunk on an hour-long file; relaunch; it must resume
   at the checkpoint and produce a transcript identical to an uninterrupted run.
6. **Self-containment.** `make desktop-install`, then `PATH=/usr/bin:/bin open -a Transcriber`
   with Homebrew ffmpeg temporarily renamed — transcription must still work. This is the
   test that proves the migration actually removed the runtime dependencies.
7. `pnpm build` (`tsc --noEmit && vite build`) clean at each frontend change.

## Sequencing

Phases 0–4 are independent enough to land as separate commits. Phase 5 is a single
irreversible commit and must come last. Nothing in Phases 0–4 touches the Python tree, so
the reference implementation stays runnable for comparison the entire time.

---

## What changed while executing this plan

- **Phase 5 landed as one commit: delete, move, rename.** The parity gate was
  run (`make desktop-parity` on a real Vietnamese meeting file — chunk
  boundaries and sample checksums matched) and then the Python tree was deleted,
  `desktop/**` moved to the repo root with plain `mv` (git's rename detection
  keeps history), the Makefile targets were renamed, and both docs were
  rewritten. The two Python harness scripts died with the tree, as planned.
- **ffmpeg is built from source, not downloaded.** The plan assumed a static LGPL macOS
  build could be fetched. None exists — evermeet, osxexperts and Homebrew are all GPL
  because they link x264 to encode video. `scripts/build-ffmpeg.sh` configures
  `--disable-gpl --disable-nonfree --disable-version3 --disable-everything` plus the
  audio decoders `AUDIO_EXTS` implies, which is both correctly licensed and much smaller
  than expected: **3.1 MB per binary, not ~80 MB**.
- **Recording output moved** to `~/Documents/Transcriber/recordings/`. The Python app's
  repo-relative `recordings/` cannot work inside a bundled `.app`.
- **The recorder owns a thread.** `cpal::Stream` and the ScreenCaptureKit stream are not
  `Send`, so `MeetingRecorder` never crosses a thread boundary; commands talk to it over
  a channel. This also keeps a Screen Recording permission dialog from freezing the
  window while it is being answered.
- **`Info.plist` needed no config change.** tauri-bundler merges `src-tauri/Info.plist`
  into the generated one implicitly; `bundle.macOS.files` would have *replaced* it and
  lost `CFBundleIdentifier`. Verified both usage descriptions survive in the built `.app`.

Two traps worth not rediscovering:

- ffmpeg's `configure` component name for the raw PCM muxer is `pcm_s16le`, **not**
  `s16le` — that is only what `-f` calls it. The wrong flag builds, links and runs
  happily, then fails at output with `Requested output format 's16le' is not known`.
  The build script now self-tests on the exact command line `decode.rs` runs.
- Bundled ffmpeg 7.1.1 and system 8.1.2 decode bit-identically for mp3, m4a, flac, wav
  and wma. Opus differs by at most 1 LSB out of 32768 (mean 0.00) — rounding between
  versions, not a codec mismatch. Do not chase it in a transcript A/B diff.
