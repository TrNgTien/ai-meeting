# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local-only desktop app that records a meeting and turns audio into timestamped, Vietnamese-first transcripts. No audio leaves the machine.

**Two implementations live side by side:**

| | Where | Status |
| --- | --- | --- |
| Python + CustomTkinter/Tk | repo root (`app.py` + 6 modules) | **Working.** The reference implementation; `make` runs it. |
| Rust + Tauri 2 + React | `desktop/` | **Usable.** Import + transcribe + model management, all native — no Python at runtime. No live recording UI, no `vi`/PhoWhisper mode. See `desktop/README.md`. |

The Rust port is deliberately additive — every root-level `make` target still works unchanged, so the two can be compared on real audio while the port catches up. When changing shared *behaviour*, change both or say which one you changed.

## Commands

```bash
make                              # setup if needed, then launch the GUI (the one command)
make setup                        # force a full setup pass (system deps, venv, pip install)
make transcribe FILE=meeting.m4a LANG_MODE=vi   # headless CLI
make build && make run-docker-cli FILE=x.mp3    # containerised CLI

.venv/bin/python app.py           # launch directly, skipping the setup stamp check
.venv/bin/python transcriber.py <file> [lang]   # same as `make transcribe`
```

- `LANG_MODE`, never `LANG` — make inherits `LANG` from the shell locale.
- `make` re-runs `setup.sh` whenever `requirements.txt` or `setup.sh` is newer than `.venv/.deps-installed`. `setup.sh` is idempotent.
- Python must be 3.10/3.11/3.12 (`SUPPORTED_VERSIONS` in `setup.sh`); override with `PYTHON_BIN=/path/to/python3.11 make setup`.
- **There is no test suite, linter, or formatter configured.** Verification is running the app or the CLI on a real audio file.

## Architecture

### Engine abstraction

Everything hinges on one contract. Each engine exposes:

```python
transcribe_audio(audio: np.ndarray, offset_sec: float, on_segment=None) -> list[TranscriptSegment]
```

taking 16 kHz mono float32; `offset_sec` shifts the chunk's timestamps back onto the recording's timeline. Three implementations:

| Engine | Class | Used for |
| --- | --- | --- |
| openai-whisper on CPU | `transcriber.Transcriber` | `vi+en`, `en`, `auto` fallback path |
| mlx-whisper on Apple GPU | `mlx_engine.MLXTranscriber` | same modes, when the GPU switch is on and a checkpoint exists |
| PhoWhisper-large via WhisperX/CTranslate2 | `transcriber.WhisperXTranscriber` | `vi` only |

Routing is `TranscriberApp._apply_language_selection()` (UI choice → language + `_use_phowhisper`) then `_prepare_engine()` (loads/downloads, returns `(engine_key, transcribe_audio)`). Both PhoWhisper and MLX failures fall back to the CPU whisper engine with a status message rather than failing the run — preserve that when touching `_prepare_engine`.

### Chunked, resumable transcription

`chunking.transcribe_chunked()` is the driver for every engine and every entry point. Per ~5-minute chunk (`DEFAULT_CHUNK_SECONDS`):

1. `decode_range()` shells out to ffmpeg `-ss`/`-t` — keeps memory flat on hour-long files and avoids the float64 blow-up in whisper's own loader.
2. `find_split_index()` snaps the cut to the quietest point in the following 20 s so boundaries land in a pause.
3. **Durability order, do not reorder:** append text → `flush` + `os.fsync` → atomically write the checkpoint with `text_bytes` = new file size. A crash in between leaves a tail past `text_bytes`; `_resume_or_restart()` truncates to it and redoes that chunk, so recovery is idempotent.

`source_fingerprint()` = audio size + mtime + `engine_key` + chunk size. Any mismatch discards the checkpoint and restarts the file, which is why `engine_key` must change whenever the produced text would differ (model, language, CPU vs MLX vs PhoWhisper). `<name>.transcript.partial.json` sits next to the audio; it is deleted when the file completes.

Cancellation is a `threading.Event` checked between chunks, raising `TranscriptionCancelled`.

### Streaming preview (the fragile part)

Neither openai-whisper nor mlx-whisper offers a segment callback. `transcriber.stream_segments()` redirects stdout for the duration of the call, runs the engine with `verbose=True`, and parses the `[start --> end] text` lines both print (`_SegmentPrintTap`). **This is only safe because exactly one worker thread transcribes at a time** — do not parallelise transcription without replacing this mechanism. `chunking._accepts_on_segment()` inspects the signature to skip engines that can't stream (WhisperX/PhoWhisper batches the chunk).

Preview lines carry the `preview` Tk tag and are *not* on disk; `_drop_preview()` deletes the tag's range before the saved chunk text is appended. The tag range *is* the bookkeeping — no index tracking, so nothing can drift.

### Threading / UI

`app.py` is a single `TranscriberApp(ctk.CTk)` with an `AppState` enum (`IDLE`/`RECORDING`/`TRANSCRIBING`). All work happens on worker threads that push `(event, payload)` tuples onto `self._ui_queue`; `_poll_ui_queue()` drains it on the Tk thread. Never touch widgets from a worker — add a queue event instead. Existing events include `status`, `chunk_text`, `segment_text`, `file_start`, `chunk_baseline`, `batch_done`, `rec_started`/`rec_stopped`/`rec_failed`, `merged_text`, `mm_download_*`, `hide_progress`.

### Live recording

`audio_capture.py` records the two sides of a meeting to two time-aligned mono 16 kHz WAVs in `recordings/`: microphone (`MicCapture`, sounddevice) and system playback (`SystemCapture` — `ScreenCaptureKitCapture` on macOS 13+ via pyobjc, else `LoopbackDeviceCapture`; chosen by `open_system_capture()`). Either side may be missing; the recording still runs and the reason lands in `Recording.warnings`, because a meeting is not repeatable.

The two streams have independent clocks, so `_MonoWavWriter` pins each track to wall-clock time and pads/trims when drift exceeds 100 ms — that alignment is what makes the merge meaningful. The tracks are transcribed *separately* (`_transcribe_recording`), then `transcript_merge.merge_transcript_files()` interleaves them by timestamp into `<stem>-conversation.txt` with `Me` / `Meeting` speaker labels. Separate tracks are also what stops your own voice, bleeding from the speakers into the mic, from being transcribed twice. A cancelled run skips the merge (half a conversation reads worse than two transcripts).

### Hallucination control

Whisper invents YouTube-outro boilerplate over silence, and with `condition_on_previous_text=True` that invention prompts the next window into a repeat loop. Two layers, both in `transcriber.py`:

- `DECODE_OPTIONS` — shared verbatim by the openai-whisper and mlx engines (matching `transcribe()` signatures). Conditioning off, plus logprob/compression/no-speech thresholds and `hallucination_silence_threshold`.
- `is_hallucination()` / `drop_hallucinations()` — pattern match against the *whole* segment after stripping punctuation, so the same words inside a real sentence survive. Every engine's terminal path calls `drop_hallucinations()` before returning.

## Gotchas

- `requirements.txt` does not declare `sounddevice` or the pyobjc frameworks that `audio_capture` imports lazily, though both are present in the current `.venv`. A fresh `make setup` produces an app that transcribes but cannot record.
- Model caches live in three places: `~/.cache/whisper` (openai-whisper `.pt`, managed by the Manage Models dialog), the HF hub cache (MLX conversions, `mlx_engine`), and `phowhisper.cache_root()` (HF snapshot + CTranslate2 conversion).
- Changing the active model must go through `Transcriber.set_final_model()`, which unloads the loaded model so the next run picks up the new one.
- `.gitignore` excludes `*.txt`, `*.mp3`, `*.m4a`, `*.mp4` and `recordings/` — sample audio and transcripts stay local by design.
- `path/to/venv/` at the repo root is a stray untracked virtualenv from a mistyped command, not part of the project. The real one is `.venv/`.
- `README.md` is written for non-technical end users and is the source of truth for UI behaviour; its "For developers" section overlaps this file.
