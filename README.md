# Meeting Transcriber

Simple macOS and cross-platform desktop application to record meetings and produce high-accuracy Vietnamese-first speech-to-text transcripts. Runs fully local and offline using `openai-whisper` and VinAI's `PhoWhisper-large` via `WhisperX`.

## Features

- **Microphone Recording:** Live capture with device selector, live VU volume meter, 1.5s "Test Mic" diagnostic, and start/stop timer controls.
- **Live Rough Transcripts:** Fast live feedback in ~25s chunks during meetings (`whisper-small`).
- **Accurate Final Pass:** High-accuracy final transcript with timestamps on Stop (`PhoWhisper-large` + `WhisperX` or `large-v3`).
- **File Import & Drag-and-Drop:** Transcribe pre-recorded audio files (`.mp3`, `.wav`, `.m4a`, etc.).
- **Language Options:** Optimized for Vietnamese (`vi`), mixed Vietnamese & English (`vi+en`), English (`en`), or Auto-detection.
- **Privacy First:** 100% local processing; session WAV audio and text transcripts saved under `recordings/`.

## Prerequisites

- **Python:** Version 3.10 is recommended (compatible with `faster-whisper` and `ctranslate2` wheels).
- **System Libraries:**
  - **macOS:**
    ```bash
    brew install ffmpeg python-tk@3.10
    ```
  - **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt-get update && sudo apt-get install -y ffmpeg python3-tk libportaudio2
    ```

## Quick Start for Developers

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-meeting
   ```

2. **Set up virtual environment & install dependencies:**
   ```bash
   make setup
   # OR manually:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the desktop app:**
   ```bash
   make run-local
   # OR manually:
   source .venv/bin/activate
   python app.py
   ```

On first run, models are downloaded and cached automatically (~0.5 GB for `small`, ~1.5 GB for `large-v3` / `PhoWhisper-large`). Internet is required only for initial model download; subsequent runs operate completely offline.

## Docker Usage

For headless transcription without local environment setup:

```bash
# Build Docker image
make build

# Transcribe a file headlessly via Docker
make run-docker-cli FILE=viet-voice.mp3
```

## macOS Microphone Permissions

When running locally on macOS, grant microphone access to the terminal application launching the app (Terminal, iTerm2, or Cursor).
If recording does not capture audio, check:
**System Settings → Privacy & Security → Microphone** and enable your terminal app.

## Project Structure & Output

Each recording session saves its output to:

```
recordings/YYYY-MM-DD_HH-MM-SS/
  ├── audio.wav
  └── transcript.txt
```

Example transcript format:

```
[00:01:23] Xin chào mọi người, hôm nay chúng ta review sprint backlog.
```

