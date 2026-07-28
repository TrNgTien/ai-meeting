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

Files are processed one after another, each in ~5-minute chunks, so the status bar tells you how far in you are:

```
Transcribing 2/5: standup.m4a — 00:25:00 / 01:02:00 transcribed (saved)
```

Each finished transcript is appended to the window under a header:

```
===== standup.m4a =====
[00:00:04] Chào mọi người, hôm nay mình review sprint backlog.
[00:00:11] Ticket đầu tiên đang bị block ở phần authentication.
```

The app is busy until the whole batch finishes — importing more files or switching the active model is blocked until then. **Stop** ends the run after the chunk in flight, keeping everything done so far (see below).

### 4. Collect the transcripts

Each transcript is saved as a `.txt` file **next to the original audio file**:

```
~/recordings/standup.m4a
~/recordings/standup.transcript.txt   ← written for you
```

The bar under the transcript lists every file that was saved. Nothing is uploaded, and nothing is written anywhere else.

---

## Long meetings (1 hour and up)

Meeting recordings are usually long, and on CPU a 1-hour file takes a while. So a file is never transcribed in one unstoppable pass:

- **Chunked.** Audio is transcribed ~5 minutes at a time. Boundaries are snapped to the quietest moment nearby, so cuts land in a pause rather than mid-word.
- **Streamed to disk.** Each finished chunk is appended to `<name>.transcript.txt` and flushed (`fsync`) immediately. You can open the transcript in another editor and read it while the run is still going — it's never held in memory only.
- **Checkpointed.** A small `<name>.transcript.partial.json` next to the audio records how far into the recording the text goes.
- **Resumable.** Import the same file again and it continues from the last written line (`resuming at 00:25:00…`). A crash, a power loss, or quitting costs at most the chunk in flight — everything before it is already in the `.txt`.
- **Stoppable.** **Stop** ends the run after the current chunk. Files later in the batch aren't started. Nothing is lost.
- **Self-healing.** If the app died halfway through writing a chunk, the leftover tail past the checkpoint is truncated on the next run and that chunk is redone — so you never get a duplicated or half-written line.

The transcript pane fills in the same way, chunk by chunk, and shows the earlier text again when you resume a file.

A checkpoint is only reused if the audio file, the language mode, and the model all still match. Change the model (or edit the audio) and the stale checkpoint is discarded and the file restarts, so a transcript never mixes output from two different models.

When a file finishes, its `.partial.json` is deleted; the `.txt` is the complete transcript. A leftover `.partial.json` just means that file was interrupted. Deleting it by hand means the next run starts that file from scratch.

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
- **Transcription seems stuck.** Watch the `mm:ss / mm:ss transcribed` counter — it advances once per ~5-minute chunk. Everything runs on CPU (`DEVICE = "cpu"`), so `large-v3` with beam search is the slowest option by a wide margin; `large-v3-turbo`, or `vi` mode (PhoWhisper is int8 + VAD + batched), are much faster on long recordings.
- **English words come out as Vietnamese phonetics.** You're in `vi` mode — switch to `vi+en`.
- **Drag-and-drop doesn't work.** It's optional (needs `tkinterdnd2`); the **Import Files…** button always works.
- **A file failed.** The batch continues, and that file shows `[error: …]` in the transcript pane instead of stopping everything. Its checkpoint is kept, so re-importing retries from where it got to.
- **You need the machine back.** Hit **Stop** (or just quit) — re-import later to resume.
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
| `chunking.py` | ffmpeg range decoding, chunk splitting, checkpoint read/write, resumable chunk loop |
| `transcriber.py` | openai-whisper + WhisperX transcribers, model cache helpers, CLI entrypoint |
| `phowhisper.py` | PhoWhisper-large download and CTranslate2 conversion |

Notes for anyone modifying this:

- Engine routing lives in `MeetingTranscriberApp._apply_language_selection()` and `_prepare_engine()` in `app.py`; `_prepare_engine()` returns `(engine_key, transcribe_audio)` and `chunking.transcribe_chunked()` drives the loop.
- `engine_key` goes into the checkpoint fingerprint alongside the audio file's size + mtime and the chunk size — mismatches invalidate the checkpoint instead of resuming into it.
- Both engines expose `transcribe_audio(audio, offset_sec)` taking a 16 kHz mono float32 array; `offset_sec` puts the chunk's timestamps back on the recording's timeline.
- Chunk decoding goes through `chunking.decode_range()` (ffmpeg `-ss`/`-t`), which keeps memory flat on long files and bypasses the float64 blow-up in `_load_audio_16k()`'s resampler.
- Durability order per chunk: append text → `flush` + `os.fsync` → write the checkpoint (`text_bytes` = new file size) atomically. A crash between the two leaves a tail past `text_bytes`; `_resume_or_restart()` truncates to `text_bytes` and redoes that chunk, so recovery is idempotent.
- Cancellation is a `threading.Event` checked between chunks; the loop raises `TranscriptionCancelled` and the per-chunk checkpoint is already on disk.
- `on_text` fires per chunk (plus once with the resumed prefix at startup) and drives the live transcript pane via the `chunk_text` UI event.
- Changing the model calls `Transcriber.set_final_model()`, which unloads the current model so the next run loads the new one.
- `_refresh_model_menu_labels()` rebuilds the header dropdown from the models actually on disk, reassigns `_selected_model_name` if the selected one was deleted, and disables the menu with `NO_MODELS_LABEL` when the cache is empty.
- Model cache lives in `~/.cache/whisper` (or `$XDG_CACHE_HOME/whisper`); helpers are `is_model_downloaded()`, `model_size_on_disk()`, `delete_model()`, `list_downloaded_whisper_models()`, and `ensure_model_downloaded()` (accepts a `cancel_event`, raises `DownloadCancelled`).
- Transcription runs on a worker thread and reports back through `self._ui_queue`, polled by `_poll_ui_queue()`.
