# PhoWhisper-large + WhisperX for the final transcript

Date: 2026-07-24
Status: Approved (design)

## Goal

Improve Vietnamese transcription accuracy of the final transcript by running
**VinAI `PhoWhisper-large`** (a Vietnamese-fine-tuned Whisper model) through the
**WhisperX** pipeline (VAD preprocessing + batched faster-whisper inference).

Scope is **accuracy only**: no speaker diarization, no word-level alignment. The
existing per-segment output format `[HH:MM:SS] text` is preserved.

## Current state (before this change)

- macOS desktop app: `app.py` (CustomTkinter) + `recorder.py` + `transcriber.py`.
- `transcriber.py` wraps `openai-whisper` directly, CPU, default language `vi`.
  - Live chunks: `small` model (`transcribe_chunk`).
  - Final pass on Stop: `large-v3` (`transcribe_file` / `transcribe_file_to_text`).
- Output saved to `recordings/<timestamp>/transcript.txt`.

## Target behavior

| Path | Model | Notes |
| --- | --- | --- |
| Live chunks | `openai-whisper` `small` | Unchanged. Fast, incremental. |
| Final pass, language `vi` | PhoWhisper-large via WhisperX | New. Better VN accuracy. |
| Final pass, language `en`/`auto` | `openai-whisper` `large-v3` | Existing path, kept. |
| Final pass, WhisperX unavailable / error | `openai-whisper` `large-v3` | Graceful fallback + status note. |

PhoWhisper is Vietnamese-only, so it is used only when the selected language is
`vi`. Any import/conversion/runtime failure falls back to `large-v3` so the app
never hard-breaks.

## Components

### 1. `phowhisper.py` (new) — model acquisition

`PhoWhisper-large` ships as a transformers/PyTorch model; WhisperX needs a
CTranslate2 model directory. Conversion happens once and is cached.

- `PHOWHISPER_REPO = "vinai/PhoWhisper-large"`
- Cache root: `~/.cache/ai-meeting/` (override via env `AI_MEETING_CACHE`).
  - HF snapshot: `<cache>/phowhisper-large-hf/`
  - Converted CT2 model: `<cache>/phowhisper-large-ct2/`
- `ensure_phowhisper_ct2(progress_cb) -> Path`:
  1. If the CT2 dir already contains a converted model, return it (idempotent).
  2. Download the HF snapshot (`huggingface_hub.snapshot_download`), reporting
     byte progress through `progress_cb(name, downloaded, total)` (same
     signature already used by the app's download progress UI).
  3. Convert to CT2 with `ctranslate2.converters.TransformersConverter(
     hf_dir).convert(ct2_dir, quantization="int8")`. Reported as a discrete
     "Converting…" status step (indeterminate progress).
  4. Return the CT2 dir path.

Conversion is written to a `.part`/temp dir and renamed on success so a crashed
conversion is not mistaken for a completed one.

### 2. `transcriber.py` (extended) — `WhisperXTranscriber`

New class alongside the existing `Transcriber` (which is untouched):

- `preload(progress_cb)`: calls `ensure_phowhisper_ct2`, then lazy-loads
  `whisperx.load_model(ct2_dir, device="cpu", compute_type="int8",
  language="vi", vad_method=<default>)`.
- `transcribe_file(wav_path) -> list[TranscriptSegment]`:
  - Load audio at 16 kHz float32 (`whisperx.load_audio`).
  - `model.transcribe(audio, batch_size=8, language="vi")`.
  - Map each WhisperX segment (`start`, `end`, `text`) to the existing
    `TranscriptSegment`; reuse `_collect_segments`-style filtering of empties.
- `transcribe_file_to_text(wav_path) -> str`: reuse `segments_to_text`.

The shared `TranscriptSegment`, `format_timestamp`, and `segments_to_text`
helpers stay in `transcriber.py` and are reused (no duplication).

### 3. `app.py` (light wiring)

- Instantiate a `WhisperXTranscriber` for the final pass in addition to the
  existing `Transcriber` for live chunks.
- Final worker (`_stop_recording` → `stop_and_transcribe`) picks the final
  transcriber by language:
  - `vi` → `WhisperXTranscriber.transcribe_file_to_text`, on failure fall back
    to `Transcriber.transcribe_file_to_text` (large-v3) and set a status note.
  - `en`/`auto` → existing `large-v3` path.
- Model preloading (`_initialize_models`): only warm the small live model so
  the app is usable quickly. PhoWhisper-large installs **on demand** (first
  Vietnamese final pass or import), not at startup, so the one-time ~6 GB
  download never blocks the UI. The download/conversion runs in the existing
  worker thread with progress shown in the progress bar; the Tk main loop stays
  responsive. Large-v3 download stays lazy for the fallback path.

## Import audio file

An **Import File…** button (enabled once the live model is ready) opens a file
dialog for common audio formats (mp3/m4a/wav/flac/…). The selected file is run
through the same final-pass pipeline as recordings (PhoWhisper+WhisperX for
`vi`, else large-v3), decoding compressed formats via ffmpeg. The transcript is
saved next to the source as `<name>.transcript.txt`.

## Data flow (final pass, vi)

```
wav file
  -> whisperx.load_audio (16 kHz float32)
  -> whisperx.load_model(PhoWhisper CT2, cpu, int8).transcribe(batch_size=8)
  -> segments [{start,end,text}]
  -> [TranscriptSegment(start,end,text)]
  -> segments_to_text -> "[HH:MM:SS] text\n..."
  -> recordings/<ts>/transcript.txt
```

## Dependencies

Add to the project's dependency file (and install into `.venv`, Python 3.10):

- `whisperx`
- `ctranslate2`
- `faster-whisper`
- `huggingface_hub`

`torch` and `transformers` are already present. Pin versions that resolve
together on Python 3.10 / macOS / CPU; verify the venv installs cleanly and
`import whisperx` succeeds before wiring the GUI.

## Error handling

- Missing/failed WhisperX import → final pass falls back to `large-v3`.
- HF download or CT2 conversion failure → surface message, fall back to
  `large-v3` for that run; partial files/dirs cleaned up so a retry is clean.
- Empty/short audio → returns empty transcript (existing behavior).

## Testing

- CLI smoke test: `python -m transcriber <wav>` transcribes one existing
  `recordings/*/audio.wav` via the PhoWhisper+WhisperX path and prints the
  timestamped transcript, for eyeballing VN quality vs. the old `large-v3`
  output — before touching the GUI flow.
- Manual GUI test: record a short Vietnamese clip, Stop, confirm the final
  transcript is produced and saved, and that `en`/`auto` still use `large-v3`.

## Out of scope (YAGNI)

- Speaker diarization (pyannote, HF token).
- Word-level forced alignment (Vietnamese wav2vec2).
- Using PhoWhisper for live chunks.
- GPU / CUDA paths.
