# Meeting Transcriber

Desktop app that turns recorded meeting audio into timestamped, Vietnamese-first transcripts. Everything runs locally — no audio ever leaves your machine.

```bash
make setup      # once, on a new machine
make run-local  # start the app
```

---

## Using the app

### 1. Pick a language mode

The **Language** dropdown decides which engine transcribes your audio:

| Choose | When your recording is | Engine used |
| --- | --- | --- |
| `vi+en` *(default)* | Vietnamese with English words mixed in ("deploy cái service này…") | openai-whisper, decoded as Vietnamese so English terms stay in English |
| `vi` | Pure Vietnamese, no code-switching | PhoWhisper-large + WhisperX (most accurate for Vietnamese) |
| `en` | English only | openai-whisper, forced to English |
| `auto` | You're not sure / mixed sources | openai-whisper, language auto-detected |

Most meeting recordings in a Vietnamese team should stay on **`vi+en`**. Switch to **`vi`** only when there is genuinely no English — PhoWhisper tends to Vietnamize English words.

### 2. Pick a model (optional)

The **Model** dropdown applies to `vi+en`, `en`, and `auto` (it's disabled for `vi`, which always uses PhoWhisper-large).

```
small  <  medium  <  large-v2  <  large-v3 (default)  <  large-v3-turbo
```

- Smaller = faster, less accurate. Larger = slower, more accurate.
- `large-v3-turbo` is the good middle ground: near `large-v3` accuracy, noticeably faster.
- **The dropdown lists only models you've already downloaded**, so picking one never stalls on a multi-GB download. To use a different model, download it from **Manage Models…** first (~0.5–1.5 GB each) — it appears in the dropdown as soon as the download finishes.
- If you delete the model that's currently selected, the app switches the selection to another downloaded one.
- On a fresh machine with nothing downloaded, the dropdown is disabled and reads `No models downloaded` — open **Manage Models…** to fetch one. (Starting a transcription anyway still works: it downloads the default `large-v3` on demand.)

Changing the model mid-session is fine — the next transcription picks it up.

### 3. Import audio

Two ways, both accept **multiple files at once**:

- Click **Import Files…** and select one or more files.
- Drag files straight onto the transcript area.

Supported: `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`, `.wma`, `.mp4`

Files are processed one after another. The status bar shows `Transcribing 2/5: standup.m4a…`, and each finished transcript is appended to the window under a header:

```
===== standup.m4a =====
[00:00:04] Chào mọi người, hôm nay mình review sprint backlog.
[00:00:11] Ticket đầu tiên đang bị block ở phần authentication.
```

The app is busy until the whole batch finishes — importing more files or switching the active model is blocked until then.

### 4. Collect the transcripts

Each transcript is saved as a `.txt` file **next to the original audio file**:

```
~/recordings/standup.m4a
~/recordings/standup.transcript.txt   ← written for you
```

The bar under the transcript lists every file that was saved. Nothing is uploaded, and nothing is written anywhere else.

---

## Managing downloaded models

Click **Manage Models…** to see disk usage and control downloads. It stays usable while a transcription is running.

- Every model is listed with its size on disk, or `not downloaded` — this dialog is where you get at models the header dropdown doesn't show yet.
- **Download** — fetch a model ahead of time (e.g. before a flight) instead of waiting mid-meeting. While downloading, the button becomes **Cancel**; cancelling deletes the partial file.
- **Use** — make a downloaded model the active one (mirrors the header dropdown). Only shown for models on disk, and disabled during a transcription.
- **Delete** — frees the disk space for that model only. It re-downloads automatically next time you use it.
- Models downloaded previously via the whisper CLI show up here too.
- PhoWhisper-large (the `vi` engine) isn't listed — it's managed separately and installs automatically the first time you transcribe in `vi` mode.

---

## Tips & troubleshooting

- **First run of any model is slow.** It's downloading. Subsequent runs are fully offline.
- **Transcription seems stuck.** Long recordings simply take a while — a 1-hour file on `large-v3` can take several minutes on Apple Silicon, longer on CPU-only machines. Drop to `medium` or `large-v3-turbo` if you need speed.
- **English words come out as Vietnamese phonetics.** You're in `vi` mode — switch to `vi+en`.
- **Drag-and-drop doesn't work.** It's optional (needs `tkinterdnd2`); the **Import Files…** button always works.
- **A file failed.** The batch continues, and that file shows `[error: …]` in the transcript pane instead of stopping everything.
- **`PhoWhisper failed …; falling back to …`** in the status bar means the `vi` engine couldn't load, so the selected openai-whisper model handled the file instead — the transcript is still produced.

---

## Headless / batch use without the GUI

Transcribe from the command line (PhoWhisper + WhisperX):

```bash
python transcriber.py "The Qafé.m4a"        # defaults to vi
python transcriber.py meeting.wav vi        # explicit language
```

Or via Docker, with no local Python setup:

```bash
make build
make run-docker-cli FILE=viet-voice.mp3
```

---

## Setup

**Requirements:** Python 3.10 recommended. macOS needs [Homebrew](https://brew.sh); Linux (Ubuntu/Debian) needs `sudo` — both only so the setup script can install `ffmpeg` and Tk for you.

```bash
git clone <repository-url>
cd ai-meeting
make setup      # installs system libs, creates .venv, installs Python deps
make run-local  # or: source .venv/bin/activate && python app.py
```

`setup.sh` detects your OS and installs everything needed — no manual dependency hunting. Internet is required only for `make setup` and the first download of each model.

## For developers

| File | Responsibility |
| --- | --- |
| `app.py` | CustomTkinter UI, batch import, model manager dialog, worker threads |
| `transcriber.py` | openai-whisper + WhisperX transcribers, model cache helpers, CLI entrypoint |
| `phowhisper.py` | PhoWhisper-large download and CTranslate2 conversion |

Notes for anyone modifying this:

- Engine routing lives in `MeetingTranscriberApp._apply_language_selection()` and `_run_final_transcription()` in `app.py`.
- Changing the model calls `Transcriber.set_final_model()`, which unloads the current model so the next run loads the new one.
- `_refresh_model_menu_labels()` rebuilds the header dropdown from the models actually on disk, reassigns `_selected_model_name` if the selected one was deleted, and disables the menu with `NO_MODELS_LABEL` when the cache is empty.
- Model cache lives in `~/.cache/whisper` (or `$XDG_CACHE_HOME/whisper`); helpers are `is_model_downloaded()`, `model_size_on_disk()`, `delete_model()`, `list_downloaded_whisper_models()`, and `ensure_model_downloaded()` (accepts a `cancel_event`, raises `DownloadCancelled`).
- Transcription runs on a worker thread and reports back through `self._ui_queue`, polled by `_poll_ui_queue()`.
