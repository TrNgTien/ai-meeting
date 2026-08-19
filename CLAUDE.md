# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local-only macOS desktop app that records a meeting and turns audio into timestamped, Vietnamese-first transcripts. No audio leaves the machine: whisper.cpp runs on the Metal GPU, models download on first use, and ffmpeg is bundled.

**One implementation: Rust + Tauri 2 + React.** The Python/CustomTkinter reference app that used to live here was deleted after the port proved itself (see `docs/migration-plan.md` for what was dropped and why). There is no Python, no Homebrew ffmpeg, and no external dependency at runtime. The app, the headless CLI, and every test are the same code.

## Commands

```bash
make                              # setup if needed, then launch the GUI (the one command)
make setup                        # install missing toolchain/deps, pnpm install, build ffmpeg
make dev                          # launch with hot reload
make build                        # produce the .app/.dmg
make test                         # cargo test (116 tests)
make transcribe FILE=meeting.m4a [MODEL=large-v3] [LANG_MODE=vi+en]   # headless CLI
make release / make install       # .dmg to dist-release/ / install to /Applications
```

- `LANG_MODE`, never `LANG` — make inherits `LANG` from the shell locale.
- `make` re-runs `setup` whenever `package.json` or `pnpm-lock.yaml` is newer than `node_modules`. `setup` is idempotent; `make ffmpeg` skips itself once the binaries exist.
- Verification is `make test` plus running the app or the CLI on a real audio file.

## Architecture

All Rust lives in `src-tauri/src/`, all UI in `src/`.

### Engine abstraction

`transcribe::Engine` (`transcribe/mod.rs:32`) is the contract — one way of turning 16 kHz mono audio into timestamped segments:

```rust
fn transcribe_chunk(&self, audio: &[f32], offset_sec: f64, observer: &dyn ChunkObserver) -> Result<Vec<TranscriptSegment>>;
```

One implementation today: `WhisperCppEngine` (`transcribe/whisper_cpp.rs`), whisper.cpp compiled with Metal. Routing is language-only, in `state::LanguageMode` (`state.rs:35`): `vi+en` (default — decode as Vietnamese on a *multilingual* checkpoint so English terms mixed into Vietnamese speech stay spelled in English), `en`, `auto`. The old `vi`/PhoWhisper mode was dropped; anything unrecognised falls back to `vi+en`. Every decode path runs the hallucination filter (`transcribe/hallucination.rs`) before returning.

### Chunked, resumable transcription

`chunking::transcribe_chunked()` (`chunking/mod.rs`) is the driver for every entry point — GUI, recording pipeline, and the `transcribe` binary. Per ~5-minute chunk (`DEFAULT_CHUNK_SECONDS = 300.0`):

1. `decode_range()` shells out to the bundled ffmpeg `-ss`/`-t` (`chunking/decode.rs`) — keeps memory flat on hour-long files.
2. `find_split_index()` snaps the cut to the quietest point in the following 20 s so boundaries land in a pause.
3. **Durability order, do not reorder:** append text → `flush` + `sync_all` → atomically write the checkpoint (`checkpoint.rs`). A crash in between leaves a tail past the checkpointed size; `resume_or_restart()` truncates to it and redoes that chunk, so recovery is idempotent.

`source_fingerprint()` = audio size + mtime + `engine_key` + chunk size. Any mismatch discards the checkpoint and restarts the file, which is why `engine_key` must change whenever the produced text would differ (model, language). `<name>.transcript.partial.json` sits next to the audio; it is deleted when the file completes.

Cancellation is a `CancelFlag` (`state.rs:70`) checked between chunks, surfacing as `TranscribeError::Cancelled` — never a failure, since the checkpoint already holds a complete prefix.

### Streaming preview

whisper-rs exposes a real segment callback, so unlike the Python app there is no stdout tap: `WhisperCppEngine` forwards segments as they decode, and `transcribe_chunked` fans them out through `ChunkObserver::on_segment` (`chunking/mod.rs:63`). The trait's `can_stream()` method answers whether previews exist (replacing Python's signature inspection). **Exactly one worker thread transcribes at a time** — do not parallelise transcription. Preview segments are *not* on disk; the frontend keeps them separate from the appended chunk text.

### Threading / UI

`commands.rs` is the thin layer; `engine::EngineHost` (`engine.rs`) owns the worker-thread state behind an `Arc<Inner>`. Tauri owns an `AppState` (`state.rs`) with a `Phase` state machine (`Idle`/`Recording`/`Transcribing`) — import, model switching, and new recordings are rejected unless `Idle`. All progress reaches the frontend as one `engine-event` (JSON with an `event` discriminator); the React panes each filter the whole stream (`src/lib/` has the typed helpers). Never call `emit` from a worker without the `AppHandle` being held by the job (the recorder keeps its own, see below).

### Live recording

`audio/` captures the two sides of a meeting to two time-aligned mono 16 kHz WAVs in `~/Documents/Transcriber/recordings/`: microphone (`mic.rs`, cpal) and system playback (`system.rs` — ScreenCaptureKit on macOS 13+, loopback fallback). Either side may be missing; the recording still runs and the reason lands in `Recording::warnings`, because a meeting is not repeatable. `RecordingHost` (`recording.rs`) owns the recorder: cpal streams and ScreenCaptureKit are not `Send`, so commands talk to it over a channel and it runs the job on its own thread.

The two streams have independent clocks, so `wav_writer.rs` pins each track to wall-clock time and pads/trims when drift exceeds 100 ms — that alignment is what makes the merge meaningful. After `stop_recording`, `engine::run_recording` transcribes both tracks *separately* (so your own voice bleeding from the speakers into the mic is not transcribed twice), then `merge::merge_transcript_files()` interleaves by timestamp into `<stem>-conversation.txt` with `Me` / `Meeting` labels. **A cancelled run skips the merge** — half a conversation reads worse than two transcripts.

### Settings

`settings.rs` persists `{language_mode, model, record_mic, record_system, mic_device_id}` as JSON in `~/.config/dev.placepad.transcriber/settings.json`, loaded at startup to seed `AppState` and saved on every change. It is the single source of truth for the defaults — `DEFAULT_MODEL = "large-v3"` wins on first run, whatever the UI placeholder says.

## Gotchas

- **`whisper-rs` leaks its segment callback.** It stores the closure with `Box::into_raw` and implements no `Drop` for `FullParams`, so a channel sender captured by that closure is *never dropped*. Waiting for the channel to close deadlocks; `whisper_cpp.rs` sends an explicit `None` sentinel instead.
- **`.cargo/config.toml` sets `http.multiplexing = false`.** Without it, cargo's LibreSSL intermittently fails the TLS handshake against some crates.io CDN nodes and reports a bogus "failed to download".
- **`screencapturekit` is pinned to 2.1 on purpose.** 3.0+ depend on `apple-metal`, whose Swift bridge needs a newer Metal SDK than macOS 15 Command Line Tools ships.
- **The bundled ffmpeg (7.1, built by `scripts/build-ffmpeg.sh`) is the one true decoder**, resolved in `decode.rs` sidecar-first with a `PATH` fallback and a `TRANSCRIBER_FFMPEG_DIR` override. It and a system ffmpeg (8.1) agree bit-for-bit on mp3, m4a, flac, wav and wma; opus differs by at most 1 LSB out of 32768 — rounding between versions, do not chase it in a diff.
- **Do not reorder the chunk durability steps** in `checkpoint.rs` (text → fsync → checkpoint with new size). The truncate-and-redo recovery in `resume_or_restart()` depends on that exact order.
- **Changing what a decode produces (model, language, engine) must change `engine_key()`**, or a checkpoint written by the old config will be resumed by the new one and the two halves spliced.
- Data lives in: `~/Documents/Transcriber/recordings/` (recordings + transcripts), `~/.cache/whisper-cpp/` (models), `~/.config/dev.placepad.transcriber/settings.json`. `~/.cache/whisper`, `~/.cache/ai-meeting` and the MLX HF snapshots are dead weight from the Python era — deletable by hand, never programmatically.
- `.gitignore` excludes `*.txt`, `*.mp3`, `*.m4a`, `*.mp4` and the checkpoint files — sample audio and transcripts stay local by design.
- `target/debug` plus a 1.6 GB model is ~5 GB; `rm -rf src-tauri/target/debug` is the quick disk win.
- `README.md` is written for non-technical end users and is the source of truth for UI behaviour; it overlaps this file in the "What this is" section.
