"""Microphone capture, WAV writing, and live-chunk buffering."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"
BLOCK_DURATION_SEC = 0.5


@dataclass
class LiveChunk:
    """Audio chunk ready for live transcription."""

    audio: np.ndarray
    start_sec: float
    end_sec: float


@dataclass
class RecorderState:
    recording: bool = False
    session_dir: Optional[Path] = None
    wav_path: Optional[Path] = None
    total_samples: int = 0
    max_amplitude: float = 0.0
    live_buffer: list[np.ndarray] = field(default_factory=list)
    live_buffer_samples: int = 0
    live_offset_sec: float = 0.0


def get_input_devices() -> list[tuple[Optional[int], str]]:
    """Return list of available audio input devices as (device_id, display_name)."""
    result: list[tuple[Optional[int], str]] = [(None, "Default Microphone")]
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                name = dev["name"]
                if idx == default_in:
                    name += " (Default)"
                result.append((idx, name))
    except Exception:
        pass
    return result


def test_microphone(device: Optional[int] = None, duration_sec: float = 1.5) -> float:
    """Record brief test snippet and return peak amplitude (0.0 to 1.0)."""
    try:
        rec = sd.rec(
            int(duration_sec * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device,
        )
        sd.wait()
        if rec.size == 0:
            return 0.0
        return float(np.max(np.abs(rec)))
    except Exception:
        return 0.0


class AudioRecorder:
    """Captures mic audio to WAV and feeds rolling chunks for live STT."""

    def __init__(
        self,
        *,
        chunk_duration_sec: float = 25.0,
        on_chunk: Optional[Callable[[LiveChunk], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.chunk_duration_sec = chunk_duration_sec
        self.on_chunk = on_chunk
        self.on_error = on_error
        self.on_level = on_level

        self._state = RecorderState()
        self._lock = threading.Lock()
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._wav_file: Optional[sf.SoundFile] = None

    @property
    def max_amplitude(self) -> float:
        with self._lock:
            return self._state.max_amplitude

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._state.recording

    @property
    def elapsed_sec(self) -> float:
        with self._lock:
            return self._state.total_samples / SAMPLE_RATE

    @property
    def session_dir(self) -> Optional[Path]:
        with self._lock:
            return self._state.session_dir

    @property
    def wav_path(self) -> Optional[Path]:
        with self._lock:
            return self._state.wav_path

    def start(self, session_dir: Path, device: Optional[int] = None) -> None:
        if self.is_recording:
            raise RuntimeError("Already recording")

        session_dir.mkdir(parents=True, exist_ok=True)
        wav_path = session_dir / "audio.wav"

        with self._lock:
            self._state = RecorderState(
                recording=True,
                session_dir=session_dir,
                wav_path=wav_path,
                max_amplitude=0.0,
            )

        self._stop_event.clear()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._wav_file = sf.SoundFile(
            str(wav_path),
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            subtype="PCM_16",
        )

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="audio-writer",
            daemon=True,
        )
        self._writer_thread.start()

        blocksize = int(SAMPLE_RATE * BLOCK_DURATION_SEC)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=blocksize,
            device=device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> Optional[Path]:
        if not self.is_recording:
            return None

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._stop_event.set()

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None

        with self._lock:
            self._state.recording = False
            wav_path = self._state.wav_path

        self._flush_live_buffer(force=True)
        return wav_path

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            print(f"Audio status: {status}")
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
        with self._lock:
            if peak > self._state.max_amplitude:
                self._state.max_amplitude = peak

        if self.on_level is not None:
            self.on_level(peak)

        self._frame_queue.put(mono)

    def _writer_loop(self) -> None:
        try:
            while not self._stop_event.is_set() or not self._frame_queue.empty():
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._process_frame(frame)
        except Exception as exc:
            if self.on_error:
                self.on_error(exc)
            else:
                raise

    def _process_frame(self, frame: np.ndarray) -> None:
        if self._wav_file is not None:
            self._wav_file.write(frame)

        with self._lock:
            self._state.total_samples += len(frame)
            self._state.live_buffer.append(frame)
            self._state.live_buffer_samples += len(frame)

        chunk_samples = int(self.chunk_duration_sec * SAMPLE_RATE)
        with self._lock:
            ready = self._state.live_buffer_samples >= chunk_samples

        if ready:
            self._flush_live_buffer(force=False)

    def _flush_live_buffer(self, *, force: bool) -> None:
        with self._lock:
            if not self._state.live_buffer:
                return

            chunk_samples = int(self.chunk_duration_sec * SAMPLE_RATE)
            if not force and self._state.live_buffer_samples < chunk_samples:
                return

            if force:
                frames = list(self._state.live_buffer)
                self._state.live_buffer.clear()
                self._state.live_buffer_samples = 0
            else:
                needed = chunk_samples
                collected: list[np.ndarray] = []
                collected_samples = 0
                while self._state.live_buffer and collected_samples < needed:
                    frame = self._state.live_buffer.pop(0)
                    collected.append(frame)
                    collected_samples += len(frame)
                    self._state.live_buffer_samples -= len(frame)

                if collected_samples > needed:
                    extra = collected_samples - needed
                    tail = collected[-1]
                    split_at = len(tail) - extra
                    head = tail[:split_at]
                    remainder = tail[split_at:]
                    collected[-1] = head
                    self._state.live_buffer.insert(0, remainder)
                    self._state.live_buffer_samples += len(remainder)
                    collected_samples = needed

                frames = collected

            start_sec = self._state.live_offset_sec
            audio = np.concatenate(frames) if frames else np.array([], dtype=np.float32)
            end_sec = start_sec + (len(audio) / SAMPLE_RATE)
            self._state.live_offset_sec = end_sec

        if len(audio) == 0 or self.on_chunk is None:
            return

        self.on_chunk(
            LiveChunk(
                audio=audio,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
