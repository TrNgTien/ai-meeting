# Meeting Transcriber

Desktop app that turns recorded meeting audio into timestamped, Vietnamese-first transcripts. Everything runs locally — no audio ever leaves your machine.

## Quick start (macOS)

Open **Terminal** (`Cmd + Space`, type `Terminal`). First, install Apple's command line tools — if a dialog appears, click **Install** and wait for it to finish before continuing:

```bash
xcode-select --install
```

Then paste this whole block at once:

```bash
command -v brew >/dev/null || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
git clone https://github.com/TrNgTien/ai-meeting.git && cd ai-meeting && make
```

That's the whole setup — `make` installs everything else (ffmpeg, Python, all packages) and opens the app. **The first run takes several minutes** downloading packages, so let it run even if it looks stuck. Afterwards, `cd ai-meeting && make` opens it in seconds.

<details>
<summary>Not on a Mac?</summary>

**Ubuntu / Debian** — asks for your password once, when it installs `ffmpeg` and Tk:

```bash
sudo apt-get update && sudo apt-get install -y git make
git clone https://github.com/TrNgTien/ai-meeting.git && cd ai-meeting && make
```

**Windows** — not supported directly. Install [WSL2](https://learn.microsoft.com/windows/wsl/install) (`wsl --install` in PowerShell as Administrator, then reboot), open the Ubuntu terminal it gives you, and run the commands above.

</details>

Then, in the window:

1. **Import Files…** (or drag audio onto the transcript area) — `.mp3`, `.wav`, `.m4a`, `.mp4`, and more.
2. Watch the transcript appear line by line. Nothing else to configure — the defaults (`vi+en`, `large-v3`) suit a Vietnamese team meeting with English terms mixed in.
3. The finished `.txt` is saved next to your audio file, named `YYYYMMDD-HHMMSS-<recording>.txt`.

Long recordings are transcribed in ~5-minute chunks, saved as they go, and **resume where they left off** if you stop or quit.

**Internet:** needed for that first `make`, and once more the first time you transcribe (the speech model, ~1.5 GB, is downloaded and cached). Everything after that works offline.

Only if you like: `make` accepts other targets — `make transcribe FILE=meeting.m4a` for headless CLI use, `make setup` to reinstall dependencies. Read on for what the dropdowns do.

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
- **On the CPU engine the dropdown lists only models you've already downloaded**, so picking one never stalls on a multi-GB download. To use a different model, download it from **Manage Models…** first (~0.5–1.5 GB each) — it appears in the dropdown as soon as the download finishes.
- If you delete the model that's currently selected, the app switches the selection to another downloaded one.
- On a fresh machine with nothing downloaded, the dropdown is disabled and reads `No models downloaded` — open **Manage Models…** to fetch one. (Starting a transcription anyway still works: it downloads the default `large-v3` on demand.)

Changing the model mid-session is fine — the next transcription picks it up.

### 2b. GPU (MLX) — Apple silicon only

On an M-series Mac the header shows a **GPU (MLX)** switch, **on by default**. Leave it on: it decodes the same Whisper models on the Apple GPU instead of the CPU, which is the difference between a 44-minute recording taking an afternoon and taking a lunch break. The switch simply doesn't appear on Intel Macs, Linux, or Windows — nothing to configure there.

- It applies to `vi+en`, `en`, and `auto` (the `vi` engine has its own path).
- The two engines use different checkpoints (MLX conversions vs openai-whisper `.pt`), so the dropdown changes with the switch. MLX offers every size and fetches the one you pick on first use — the hint under the dropdown says when a download is pending. **Manage Models…** manages the CPU engine's `.pt` files only.
- Because the engine is part of a checkpoint's fingerprint, flipping the switch mid-file restarts that file rather than resuming into output from the other engine.

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
~/recordings/20260728-142530-standup.txt   ← written for you
```

The name is `<when>-<recording>.txt`, where `when` is the local time the transcription started (`YYYYMMDD-HHMMSS`) — so a folder of transcripts sorts oldest-first, and transcribing the same recording again (after switching model or language, say) leaves the earlier transcript alone instead of overwriting it.

The bar under the transcript lists every file that was saved. Nothing is uploaded, and nothing is written anywhere else.

---

## Long meetings (1 hour and up)

Meeting recordings are usually long, and on CPU a 1-hour file takes a while. So a file is never transcribed in one unstoppable pass:

- **Chunked.** Audio is transcribed ~5 minutes at a time. Boundaries are snapped to the quietest moment nearby, so cuts land in a pause rather than mid-word.
- **Streamed to disk.** Each finished chunk is appended to `<timestamp>-<name>.txt` and flushed (`fsync`) immediately. You can open the transcript in another editor and read it while the run is still going — it's never held in memory only.
- **Checkpointed.** A small `<name>.transcript.partial.json` next to the audio records how far into the recording the text goes.
- **Resumable.** Import the same file again and it continues from the last written line (`resuming at 00:25:00…`), appending to the transcript the interrupted run started rather than opening a new one. A crash, a power loss, or quitting costs at most the chunk in flight — everything before it is already in the `.txt`.
- **Stoppable.** **Stop** ends the run after the current chunk. Files later in the batch aren't started. Nothing is lost.
- **Self-healing.** If the app died halfway through writing a chunk, the leftover tail past the checkpoint is truncated on the next run and that chunk is redone — so you never get a duplicated or half-written line.

The transcript pane fills in faster than the file does: lines appear **while the chunk is still being transcribed**, dimmed, as the model finishes each ~30-second window — you never wait for a whole chunk (let alone the whole recording) to see text. When the chunk is saved, those dimmed lines are replaced by the text that actually went to disk, so what you read always ends at the last saved line. Stopping mid-chunk drops the dimmed preview, because that audio was never written. Resuming a file shows the earlier text again first.

(Preview lines need an engine that reports as it decodes: `vi+en`/`en`/`auto` on either whisper engine do; `vi` mode's PhoWhisper batches the chunk and fills in chunk by chunk as before.)

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

**Getting it installed (macOS)**

- **`brew: command not found`, right after installing Homebrew.** It isn't on your PATH yet. Run `eval "$(/opt/homebrew/bin/brew shellenv)"`, or just quit Terminal and open it again.
- **`make: command not found`.** Step 1 didn't complete. Run `xcode-select --install` and let the dialog finish.
- **The first `make` sits there for minutes with no output.** That's normal — it's downloading a few GB of Python packages. Leave it alone.
- **`Repository not found` on `git clone`.** Ask the repo owner for access; the clone URL is correct but the repo may be private.

**Using it**

- **First run of any model is slow.** It's downloading. Subsequent runs are fully offline.
- **Transcription seems stuck.** Watch the transcript pane — dimmed lines land every ~30 seconds of audio. The `mm:ss / mm:ss transcribed` counter behind them advances once per ~5-minute chunk. Without the GPU (MLX) switch everything runs on CPU (`DEVICE = "cpu"`), where `large-v3` with beam search is the slowest option by a wide margin; turn **GPU (MLX)** on if you're on Apple silicon, or pick `large-v3-turbo`, or use `vi` mode (PhoWhisper is int8 + VAD + batched).
- **`MLX unavailable …; using the CPU engine…`** in the status bar means the GPU path couldn't load, so the CPU engine took the file. The transcript is still produced, just slower.
- **English words come out as Vietnamese phonetics.** You're in `vi` mode — switch to `vi+en`.
- **Drag-and-drop doesn't work.** It's optional (needs `tkinterdnd2`); the **Import Files…** button always works.
- **A file failed.** The batch continues, and that file shows `[error: …]` in the transcript pane instead of stopping everything. Its checkpoint is kept, so re-importing retries from where it got to.
- **You need the machine back.** Hit **Stop** (or just quit) — re-import later to resume.
- **`PhoWhisper failed …; falling back to …`** in the status bar means the `vi` engine couldn't load, so the selected openai-whisper model handled the file instead — the transcript is still produced.

---

## Headless / batch use without the GUI

Transcribe from the command line (PhoWhisper + WhisperX). Same one-command story — `make transcribe` sets up whatever is missing first:

```bash
make transcribe FILE="The Qafé.m4a"             # defaults to vi
make transcribe FILE=meeting.wav LANG_MODE=vi   # explicit language
```

Or via Docker, with no local Python setup:

```bash
make build
make run-docker-cli FILE=viet-voice.mp3
```

---

## Setup, in detail

You don't need this section — `make` does all of it. It's here so you know what the machine is being asked to do.

**Requirements:** macOS (with [Homebrew](https://brew.sh)) or Ubuntu/Debian (with `sudo`), so the setup script can install `ffmpeg` and Tk for you. Python 3.10, 3.11, or 3.12 — on macOS one is installed for you if you have none.

| Command | What it does |
| --- | --- |
| `make` | Sets up if needed, then launches the app. **This is the one command.** |
| `make setup` | Forces a full setup pass without launching. |
| `make run-local` | Launches the app (setting up first if it has never been set up). |
| `make transcribe FILE=… [LANG_MODE=…]` | Headless CLI transcription. |
| `make build` / `make run-docker-cli FILE=…` | Docker image and containerised CLI. |

`setup.sh` is idempotent — it only installs what's actually missing, so re-running it is cheap. `make` re-runs it automatically whenever `requirements.txt` or `setup.sh` changes, so pulling new dependencies never means remembering a second command. Internet is required for that first setup and for the first download of each model.

If your Python lives somewhere unusual: `PYTHON_BIN=/path/to/python3.10 make setup`.

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
- Both engines expose `transcribe_audio(audio, offset_sec, on_segment=None)` taking a 16 kHz mono float32 array; `offset_sec` puts the chunk's timestamps back on the recording's timeline.
- Chunk decoding goes through `chunking.decode_range()` (ffmpeg `-ss`/`-t`), which keeps memory flat on long files and bypasses the float64 blow-up in `_load_audio_16k()`'s resampler.
- Durability order per chunk: append text → `flush` + `os.fsync` → write the checkpoint (`text_bytes` = new file size) atomically. A crash between the two leaves a tail past `text_bytes`; `_resume_or_restart()` truncates to `text_bytes` and redoes that chunk, so recovery is idempotent.
- Cancellation is a `threading.Event` checked between chunks; the loop raises `TranscriptionCancelled` and the per-chunk checkpoint is already on disk.
- `on_text` fires per chunk (plus once with the resumed prefix at startup) and drives the transcript pane via the `chunk_text` UI event.
- `on_segment` fires *within* a chunk, per decode window, via the `segment_text` UI event. Neither openai-whisper nor mlx-whisper offers a callback, so `transcriber.stream_segments()` redirects stdout for the duration of the call, runs the engine with `verbose=True`, and parses the `[start --> end] text` lines both print (non-segment output is passed through). It is safe only because one worker thread transcribes at a time; `chunking._accepts_on_segment()` skips engines that can't stream (WhisperX/PhoWhisper batches the chunk).
- Those preview lines carry the `preview` tag in the transcript box and are **not** on disk. `_drop_preview()` deletes the tag's range before the saved chunk text is appended, and on stop/error — the tag range *is* the bookkeeping, so no index tracking can drift out of sync.
- Changing the model calls `Transcriber.set_final_model()`, which unloads the current model so the next run loads the new one.
- `_refresh_model_menu_labels()` rebuilds the header dropdown from the models actually on disk, reassigns `_selected_model_name` if the selected one was deleted, and disables the menu with `NO_MODELS_LABEL` when the cache is empty.
- Model cache lives in `~/.cache/whisper` (or `$XDG_CACHE_HOME/whisper`); helpers are `is_model_downloaded()`, `model_size_on_disk()`, `delete_model()`, `list_downloaded_whisper_models()`, and `ensure_model_downloaded()` (accepts a `cancel_event`, raises `DownloadCancelled`).
- Transcription runs on a worker thread and reports back through `self._ui_queue`, polled by `_poll_ui_queue()`.
