# Meeting Transcriber

Simple macOS and cross-platform desktop application that produces high-accuracy Vietnamese-first speech-to-text transcripts from imported audio files. Runs fully local and offline using `openai-whisper` and VinAI's `PhoWhisper-large` via `WhisperX`.

## Features

- **File Import & Drag-and-Drop:** Transcribe pre-recorded audio files (`.mp3`, `.wav`, `.m4a`, etc.) — via the **Import Files…** button or by dropping files onto the transcript area.
- **Accurate Transcription:** High-accuracy transcript with timestamps (`PhoWhisper-large` + `WhisperX` for `vi`, or a selectable `openai-whisper` model for `vi+en`/`en`/`auto`).
- **Language Options:** Mixed Vietnamese & English (`vi+en`, default), pure Vietnamese (`vi`), English (`en`), or Auto-detection.
- **Selectable Model:** For `vi+en` / `en` / `auto`, pick which `openai-whisper` checkpoint runs the transcription (speed vs. accuracy trade-off); downloads on first use with a progress bar.
- **Model Manager:** View every model cached locally, download one ahead of time, or delete ones you don't need.
- **Privacy First:** 100% local processing; transcripts are saved next to each imported audio file as `<name>.transcript.txt`.

## Prerequisites

- **Python:** Version 3.10 is recommended (compatible with `faster-whisper` and `ctranslate2` wheels).
- **macOS:** [Homebrew](https://brew.sh) installed (used to install system libraries automatically).
- **Linux (Ubuntu/Debian):** `sudo` access (used to install system libraries automatically).

## Quick Start (New Machine)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-meeting
   ```

2. **Run one-command setup** (installs `ffmpeg`, Tk, creates a virtualenv, and installs Python dependencies):
   ```bash
   make setup
   # OR directly:
   ./setup.sh
   ```

3. **Run the desktop app:**
   ```bash
   make run-local
   # OR manually:
   source .venv/bin/activate
   python app.py
   ```

No manual research into system libraries is needed — `setup.sh` detects your OS (macOS/Linux) and installs everything required.

Whichever model you select (`small` through `large-v3`/`large-v3-turbo`, or `PhoWhisper-large` for `vi`) downloads and caches automatically the first time you use it (~0.5–1.5 GB depending on size), with progress shown in the app. Internet is required only for that first download per model; transcription runs completely offline afterward.

## Docker Usage

For headless transcription without local environment setup:

```bash
# Build Docker image
make build

# Transcribe a file headlessly via Docker
make run-docker-cli FILE=viet-voice.mp3
```

## Language & Model Selection

The app header has two dropdowns that together decide which model transcribes imported audio files.

| Language dropdown | Transcription engine | Model dropdown |
| --- | --- | --- |
| `vi` (pure Vietnamese) | `PhoWhisper-large` via `WhisperX` (`WhisperXTranscriber` in `transcriber.py`) | Disabled — not applicable |
| `vi+en` (mixed Vietnamese/English) | `openai-whisper`, decoded with `language="vi"` so code-switched English words are kept rather than Vietnamized | Enabled |
| `en` | `openai-whisper`, decoded with `language="en"` | Enabled |
| `auto` | `openai-whisper`, language auto-detected | Enabled |

When the Model dropdown is enabled, it selects which `openai-whisper` checkpoint (`FINAL_MODEL_OPTIONS` in `transcriber.py`) is used for those three modes:

```
small < medium < large-v2 < large-v3 (default) < large-v3-turbo
```

Smaller models transcribe faster with lower accuracy; larger models are slower but more accurate. `large-v3-turbo` targets large-v3-level accuracy with faster inference.

The Model dropdown marks already-downloaded checkpoints with a **✓** (e.g. `large-v3 ✓`). To download a model ahead of time, or to see/remove models you no longer need, use **Manage Models…** — see below.

**Behavior notes (for humans and agents modifying this code):**
- Switching the Model dropdown calls `Transcriber.set_final_model()`, which drops any already-loaded model so the next transcription loads the newly selected one.
- The selected model downloads automatically on first use (cached under `~/.cache/whisper`, or `$XDG_CACHE_HOME/whisper` if set) with progress shown in the app's status bar and progress bar — no separate download step is required.
- Routing logic lives in `MeetingTranscriberApp._apply_language_selection()` and `_run_final_transcription()` in `app.py`; the ✓ labels are rebuilt by `_refresh_model_menu_labels()`.

### Managing Local Models (Download / Disk Space)

Click **Manage Models…** next to the Model dropdown to see every `openai-whisper` model cached on this machine, proactively download one, or free up disk space. It stays usable even while a transcription/download is already running in the background.

- Lists all `FINAL_MODEL_OPTIONS` plus any other openai-whisper checkpoint found in the cache (e.g. downloaded previously via CLI), each with its size on disk or `not downloaded`.
- **Use** makes a model the active selection (mirrored in the header's Model dropdown) — disabled while a transcription is running, to avoid racing with the worker thread reading the active model. **✓ Selected** marks the current one.
- **Download** fetches a not-yet-downloaded model in the background (progress shown in the dialog) without needing to transcribe anything first. While it's running, the button becomes **Cancel** — clicking it stops the download and deletes the partial file, leaving the model as `not downloaded`.
- **Delete** removes only the local cache file for that model — it re-downloads automatically the next time it's selected and used. Deleting does not affect any other installed model.
- PhoWhisper-large (used automatically for the `vi` language mode) isn't listed here — it's managed separately by `phowhisper.py` and isn't part of the Model dropdown's selectable set.
- Implemented by `MeetingTranscriberApp._open_model_manager()` in `app.py`, backed by `is_model_downloaded()` / `model_size_on_disk()` / `delete_model()` / `list_downloaded_whisper_models()` / `ensure_model_downloaded()` (which accepts a `cancel_event` and raises `DownloadCancelled`) in `transcriber.py`.

## Project Structure & Output

Transcribing an imported file saves the transcript next to it:

```
my-meeting.mp3
my-meeting.transcript.txt
```

Example transcript format:

```
[00:01:23] Xin chào mọi người, hôm nay chúng ta review sprint backlog.
```

