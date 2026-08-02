"""Live recording of a meeting: your microphone and the machine's own output.

A meeting has two sides and the machine hears them on two different paths. Your
voice arrives through the microphone. Everyone else arrives as *playback* —
Zoom, Teams, Meet and friends decode the far end and send it to the speakers,
where an ordinary input device can't reach it.

So the two sides are captured separately and kept separately:

    microphone  -> mic WAV     -> transcript labelled "Me"
    system mix  -> system WAV  -> transcript labelled "Meeting"

Keeping them apart is what makes the merged transcript able to say who spoke,
and it also avoids the microphone's echo of the speakers being transcribed
twice. The cost is that the two streams are driven by independent clocks and
independent callbacks, which is what `_MonoWavWriter` exists to deal with.

Capturing the system mix is the platform-specific part; see `open_system_capture`
for how a backend is chosen.
"""

from __future__ import annotations

import math
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

# The transcription pipeline decodes everything to mono 16 kHz before it
# reaches a model (see chunking.decode_range), so recording at that rate loses
# nothing that transcription could have used and keeps an hour-long meeting
# down to ~110 MB across both tracks instead of ~700 MB.
SAMPLE_RATE = 16000

# How far a track may drift from wall-clock time before the writer corrects it.
# Small enough that the two transcripts stay aligned to well within a spoken
# word, large enough that ordinary callback jitter isn't constantly "corrected".
_RESYNC_TOLERANCE_SEC = 0.10

# Input history kept across callbacks when resampling, so the anti-alias filter
# starts each block warm instead of ringing at every buffer boundary. Comfortably
# longer than resample_poly's default filter for the ratios we hit in practice.
_RESAMPLE_HISTORY = 256


class CaptureError(RuntimeError):
    """Raised when a capture backend can't be started."""


# --- writing ------------------------------------------------------------------


class _MonoWavWriter:
    """Appends mono 16 kHz audio to a WAV, pinned to wall-clock time.

    The naive thing — write every buffer end to end — quietly desynchronises the
    two tracks. The microphone runs on its device's clock, the system mix on
    CoreAudio's, neither is exactly 16000 Hz, and either side can drop a buffer
    when the machine is busy transcribing. Each of those shifts everything after
    it, and the drift shows up as the "Me" and "Meeting" lines sliding apart
    over a long meeting — exactly the thing the merged transcript depends on.

    So position, not arrival order, decides where audio lands: each block is
    written where the wall clock says it belongs, with silence padding a gap and
    frames dropped from a block that would run ahead. Both tracks share one t0,
    so both stay locked to the same timeline and therefore to each other.
    """

    def __init__(self, path: Path) -> None:
        self._file = sf.SoundFile(
            str(path), mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16"
        )
        self._t0: Optional[float] = None
        self._frames = 0
        self._peak = 0.0
        self._drift_frames = 0
        self._lock = threading.Lock()

    def set_origin(self, t0: float) -> None:
        """Declare the instant that becomes 00:00:00 in this track.

        Set once both sides are actually running, never at construction: a
        first-run permission prompt can hold a backend open for as long as the
        user takes to click it, and a t0 chosen before that would bake the whole
        wait into the file as silence.
        """
        with self._lock:
            self._t0 = t0

    def append(self, mono: np.ndarray, started_at: float) -> None:
        """Write one block. `started_at` is when its first sample was captured."""
        if mono.size:
            self._peak = max(self._peak, float(np.abs(mono).max()))

        with self._lock:
            if self._file.closed or self._t0 is None:
                # Pre-roll: audio from a backend that started before the other
                # side was ready. It predates the shared timeline, so it has no
                # position on it.
                return
            expected = int(max(0.0, started_at - self._t0) * SAMPLE_RATE)
            gap = expected - self._frames
            tolerance = int(_RESYNC_TOLERANCE_SEC * SAMPLE_RATE)

            if gap > tolerance:
                # Late, or a buffer went missing: hold the timeline open with
                # silence so what follows keeps its true offset.
                self._file.write(np.zeros(gap, dtype=np.float32))
                self._frames += gap
                self._drift_frames += gap
            elif gap < -tolerance:
                # Running ahead of the clock; drop the overlap rather than push
                # every later word further out of place.
                trim = min(-gap, mono.size)
                mono = mono[trim:]
                self._drift_frames -= trim

            if mono.size:
                self._file.write(mono)
                self._frames += mono.size

    def close(self) -> int:
        with self._lock:
            if not self._file.closed:
                self._file.close()
            return self._frames

    @property
    def seconds(self) -> float:
        return self._frames / SAMPLE_RATE

    @property
    def drift_seconds(self) -> float:
        """Signed silence/trim applied so far, as a health signal for the UI."""
        return self._drift_frames / SAMPLE_RATE

    def take_peak(self) -> float:
        """Loudest sample since the last call — drives the level meter."""
        peak, self._peak = self._peak, 0.0
        return peak


class _Resampler:
    """Rate conversion that keeps its filter warm across callbacks.

    Resampling each block in isolation leaves a discontinuity at every boundary
    — a faint buzz at the block rate, which is exactly the kind of artefact
    Whisper turns into invented words. Carrying a little input history and
    discarding the output it produces gives a continuous result.
    """

    def __init__(self, src_rate: int) -> None:
        from scipy.signal import resample_poly  # noqa: PLC0415 — optional at import time

        self._resample = resample_poly
        divisor = math.gcd(int(src_rate), SAMPLE_RATE)
        self._up = SAMPLE_RATE // divisor
        self._down = int(src_rate) // divisor
        self._history = np.zeros(_RESAMPLE_HISTORY, dtype=np.float32)

    def __call__(self, block: np.ndarray) -> np.ndarray:
        if self._up == 1 and self._down == 1:
            return block
        padded = np.concatenate([self._history, block])
        out = self._resample(padded, self._up, self._down)
        lead = int(round(self._history.size * self._up / self._down))
        self._history = padded[-_RESAMPLE_HISTORY:].astype(np.float32, copy=False)
        return np.asarray(out[lead:], dtype=np.float32)


def _to_mono(block: np.ndarray) -> np.ndarray:
    if block.ndim > 1:
        block = block.mean(axis=1)
    return np.ascontiguousarray(block, dtype=np.float32)


# --- microphone ---------------------------------------------------------------


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    samplerate: int

    @property
    def label(self) -> str:
        return self.name


def list_input_devices() -> list[InputDevice]:
    """Input-capable devices, for the microphone picker."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    devices = []
    try:
        for index, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                devices.append(
                    InputDevice(
                        index=index,
                        name=str(dev.get("name", f"Device {index}")),
                        channels=int(dev["max_input_channels"]),
                        samplerate=int(dev.get("default_samplerate") or 48000),
                    )
                )
    except Exception:
        return []
    return devices


def default_input_device() -> Optional[InputDevice]:
    try:
        import sounddevice as sd

        index = sd.default.device[0]
    except Exception:
        return None
    if index is None or index < 0:
        devices = list_input_devices()
        return devices[0] if devices else None
    for dev in list_input_devices():
        if dev.index == index:
            return dev
    return None


class MicCapture:
    """Microphone -> mono 16 kHz, via PortAudio."""

    name = "microphone"

    def __init__(self, writer: _MonoWavWriter, device: Optional[int] = None) -> None:
        self._writer = writer
        self._device = device
        self._stream = None
        self._rate = SAMPLE_RATE
        self._resampler: Optional[_Resampler] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - install-time problem
            raise CaptureError(f"sounddevice is not available: {exc}") from exc

        # Ask CoreAudio for 16 kHz directly — its own rate conversion is better
        # than ours and costs nothing. Only fall back to converting ourselves on
        # devices that refuse the rate.
        rate, resampler = SAMPLE_RATE, None
        try:
            sd.check_input_settings(device=self._device, samplerate=SAMPLE_RATE, channels=1)
        except Exception:
            info = sd.query_devices(self._device, "input") if self._device is not None else sd.query_devices(kind="input")
            rate = int(info["default_samplerate"])
            resampler = _Resampler(rate)
        self._rate = rate
        self._resampler = resampler

        try:
            self._stream = sd.InputStream(
                device=self._device,
                samplerate=rate,
                channels=1,
                dtype="float32",
                callback=self._on_block,
            )
            self._stream.start()
        except Exception as exc:
            raise CaptureError(f"could not open the microphone: {exc}") from exc

    def _on_block(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            # Overflows mean the machine couldn't keep up; the writer's
            # resync turns the lost audio into silence of the right length
            # rather than a shift in everything that follows.
            self.error = str(status)
        # The callback runs after the block was captured, so its first sample
        # belongs one block-length back on the timeline.
        started_at = time.monotonic() - frames / self._rate
        block = _to_mono(np.asarray(indata))
        if self._resampler is not None:
            block = self._resampler(block)
        self._writer.append(block, started_at)

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


# --- system audio -------------------------------------------------------------


class SystemCapture:
    """Interface for the machine's own output. See the two implementations."""

    name = "system audio"
    #: Shown in the UI when the backend is chosen, so it is obvious what is
    #: being recorded and where it came from.
    description = ""

    def start(self) -> None: ...
    def stop(self) -> None: ...


class ScreenCaptureKitCapture(SystemCapture):
    """System audio through ScreenCaptureKit (macOS 13+).

    This is the route that needs nothing installed and changes nothing about
    how the Mac plays audio: no virtual driver, no Multi-Output Device, no
    losing the volume keys mid-meeting. macOS hands over a copy of the system
    mix once Screen Recording permission is granted.

    A display filter is used because SCK's audio comes from a *stream*, and a
    stream needs something to capture. The video side is configured as small
    and as slow as it will go and its frames are never collected — audio is the
    only output we register for.
    """

    description = "macOS system audio (ScreenCaptureKit)"

    def __init__(self, writer: _MonoWavWriter, exclude_own_audio: bool = True) -> None:
        self._writer = writer
        self._exclude_own_audio = exclude_own_audio
        self._stream = None
        self._delegate = None
        self._queue = None
        self.error: Optional[str] = None

    @staticmethod
    def available() -> bool:
        if platform.system() != "Darwin":
            return False
        try:
            release = int(platform.mac_ver()[0].split(".")[0])
        except (ValueError, IndexError):
            return False
        if release < 13:
            return False
        try:
            import ScreenCaptureKit  # noqa: F401
            import CoreMedia  # noqa: F401
            import libdispatch  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def permission_granted(timeout: float = 8.0) -> Optional[bool]:
        """True/False if known, None if the check itself failed.

        Asking for shareable content is also what triggers the system's
        permission prompt the first time, so this doubles as the request.
        """
        try:
            import ScreenCaptureKit as SCK
        except Exception:
            return None

        done = threading.Event()
        box: dict = {}

        def handler(content, error) -> None:
            box["content"] = content
            box["error"] = error
            done.set()

        try:
            SCK.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
        except Exception:
            return None
        if not done.wait(timeout):
            return None
        content = box.get("content")
        return bool(content is not None and len(content.displays()) > 0)

    def start(self) -> None:
        import CoreMedia as CM
        import libdispatch
        import ScreenCaptureKit as SCK
        from Foundation import NSObject

        content = self._shareable_content()
        displays = content.displays()
        if not displays:
            raise CaptureError("no display available to attach the audio stream to")

        writer = self._writer
        outer = self

        class _AudioDelegate(NSObject):
            def stream_didOutputSampleBuffer_ofType_(self, stream, sbuf, otype):  # noqa: N802
                if otype != SCK.SCStreamOutputTypeAudio:
                    return
                try:
                    block = _samples_from_sample_buffer(sbuf)
                except Exception as exc:  # pragma: no cover - defensive
                    outer.error = f"audio buffer decode failed: {exc}"
                    return
                if block is None or not block.size:
                    return
                writer.append(block, time.monotonic() - block.size / SAMPLE_RATE)

            def stream_didStopWithError_(self, stream, error):  # noqa: N802
                outer.error = str(error) if error else "system audio stream stopped"

        config = SCK.SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setSampleRate_(SAMPLE_RATE)
        config.setChannelCount_(1)
        if self._exclude_own_audio:
            # Otherwise anything this app plays would be recorded back into the
            # meeting track.
            config.setExcludesCurrentProcessAudio_(True)
        # The video half is mandatory but unwanted: keep it at the smallest,
        # slowest setting available and never register for its frames.
        config.setWidth_(2)
        config.setHeight_(2)
        config.setMinimumFrameInterval_(CM.CMTimeMake(1, 1))
        config.setQueueDepth_(6)

        content_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            displays[0], []
        )
        delegate = _AudioDelegate.alloc().init()
        stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, delegate
        )
        queue = libdispatch.dispatch_queue_create(b"transcriber.audio", None)

        ok, err = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            delegate, SCK.SCStreamOutputTypeAudio, queue, None
        )
        if not ok:
            raise CaptureError(f"could not attach the audio output: {err}")

        started = threading.Event()
        start_error: list = []

        def on_start(error) -> None:
            if error:
                start_error.append(error)
            started.set()

        stream.startCaptureWithCompletionHandler_(on_start)
        if not started.wait(10):
            raise CaptureError("system audio capture did not start in time")
        if start_error:
            raise CaptureError(f"system audio capture failed to start: {start_error[0]}")

        # Held so Python doesn't collect the delegate or the queue out from
        # under a stream that is still calling into them.
        self._stream = stream
        self._delegate = delegate
        self._queue = queue

    def _shareable_content(self):
        import ScreenCaptureKit as SCK

        done = threading.Event()
        box: dict = {}

        def handler(content, error) -> None:
            box["content"] = content
            box["error"] = error
            done.set()

        SCK.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
        if not done.wait(15):
            raise CaptureError("timed out asking macOS what can be captured")
        content = box.get("content")
        if content is None:
            raise CaptureError(
                "Screen Recording permission is required to record system audio. "
                "Grant it in System Settings > Privacy & Security > Screen & System "
                "Audio Recording, then restart this app."
            )
        return content

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        stopped = threading.Event()
        try:
            stream.stopCaptureWithCompletionHandler_(lambda error: stopped.set())
            stopped.wait(5)
        except Exception:
            pass
        self._delegate = None
        self._queue = None


def _samples_from_sample_buffer(sbuf) -> Optional[np.ndarray]:
    """Mono float32 samples out of one ScreenCaptureKit audio sample buffer.

    The buffer's PCM is copied out of its block buffer rather than read through
    an AudioBufferList: copying is the one path pyobjc exposes that needs no
    manually built C structs, and at 16 kHz mono the copy is a few kilobytes.
    """
    import CoreMedia as CM

    block_buffer = CM.CMSampleBufferGetDataBuffer(sbuf)
    if block_buffer is None:
        return None
    length = CM.CMBlockBufferGetDataLength(block_buffer)
    if not length:
        return None
    status, raw = CM.CMBlockBufferCopyDataBytes(block_buffer, 0, length, None)
    if status != 0 or raw is None:
        return None

    samples = np.frombuffer(bytes(raw), dtype=np.float32)

    fmt = CM.CMSampleBufferGetFormatDescription(sbuf)
    asbd = CM.CMAudioFormatDescriptionGetStreamBasicDescription(fmt) if fmt else None
    channels = int(getattr(asbd, "mChannelsPerFrame", 1) or 1) if asbd else 1
    if channels > 1:
        # Non-interleaved is what SCK delivers: channel after channel, not
        # sample after sample. Either way the mean across channels is the mix
        # we want, so reshape by whichever layout the format flags describe.
        non_interleaved = bool(int(getattr(asbd, "mFormatFlags", 0)) & 0x20)
        usable = samples.size - (samples.size % channels)
        if usable <= 0:
            return None
        block = samples[:usable]
        if non_interleaved:
            samples = block.reshape(channels, -1).mean(axis=0)
        else:
            samples = block.reshape(-1, channels).mean(axis=1)

    return np.ascontiguousarray(samples, dtype=np.float32)


class LoopbackDeviceCapture(SystemCapture):
    """System audio through a virtual loopback input device.

    The fallback for machines where ScreenCaptureKit isn't an option — an older
    macOS, Linux, Windows, or a denied permission. It needs the user to have
    installed something like BlackHole and to be sending playback through it,
    at which point the system mix simply arrives as an ordinary input device.
    """

    #: Substrings of device names that are loopback devices rather than real
    #: inputs. Matched case-insensitively.
    KNOWN = ("blackhole", "soundflower", "loopback", "vb-audio", "cable output", "stereo mix", "monitor of")

    def __init__(self, writer: _MonoWavWriter, device: InputDevice) -> None:
        self._device = device
        self.description = f"loopback device — {device.name}"
        self._mic = MicCapture(writer, device=device.index)
        self.error: Optional[str] = None

    @classmethod
    def find(cls) -> Optional[InputDevice]:
        for dev in list_input_devices():
            lowered = dev.name.lower()
            if any(token in lowered for token in cls.KNOWN):
                return dev
        return None

    def start(self) -> None:
        self._mic.start()

    def stop(self) -> None:
        self._mic.stop()
        self.error = self._mic.error


def open_system_capture(
    writer: _MonoWavWriter, prefer: str = "auto"
) -> tuple[Optional[SystemCapture], Optional[str]]:
    """Pick a backend for the system mix.

    Returns (capture, problem). A None capture with a problem message means the
    meeting side can't be recorded on this machine right now — the microphone
    still can, which is why this reports rather than raises.
    """
    want_sck = prefer in ("auto", "screencapturekit")
    want_loopback = prefer in ("auto", "loopback")

    if want_sck and ScreenCaptureKitCapture.available():
        granted = ScreenCaptureKitCapture.permission_granted()
        if granted:
            return ScreenCaptureKitCapture(writer), None
        sck_problem = (
            "Screen Recording permission is not granted, so the meeting side "
            "can't be recorded. Grant it in System Settings > Privacy & Security "
            "> Screen & System Audio Recording, then restart this app."
        )
    elif want_sck and platform.system() == "Darwin":
        sck_problem = (
            "System audio capture needs macOS 13+ and the pyobjc ScreenCaptureKit "
            "packages (pip install -r requirements.txt)."
        )
    else:
        sck_problem = "System audio capture is not available on this platform."

    if want_loopback:
        device = LoopbackDeviceCapture.find()
        if device is not None:
            return LoopbackDeviceCapture(writer, device), None

    return None, sck_problem


# --- the recording itself -----------------------------------------------------


@dataclass
class Recording:
    """What a finished recording left on disk."""

    stem: str
    mic_path: Optional[Path]
    system_path: Optional[Path]
    duration_sec: float
    started_at: float
    warnings: list[str]

    @property
    def paths(self) -> list[Path]:
        return [p for p in (self.mic_path, self.system_path) if p is not None]


@dataclass
class Levels:
    """Meter reading for the UI, 0.0-1.0 per side."""

    mic: float = 0.0
    system: float = 0.0


class MeetingRecorder:
    """Records both sides of a meeting to two time-aligned WAV files.

    Either side may be absent — no microphone selected, or no way to reach the
    system mix on this machine — and the recording still runs with what it has.
    Whatever is missing is reported in `warnings` rather than failing the whole
    take, because a meeting is not repeatable and half a recording beats none.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        record_mic: bool = True,
        record_system: bool = True,
        mic_device: Optional[int] = None,
        system_backend: str = "auto",
        stem: Optional[str] = None,
    ) -> None:
        self._out_dir = out_dir
        self._record_mic = record_mic
        self._record_system = record_system
        self._mic_device = mic_device
        self._system_backend = system_backend
        self._stem = stem or time.strftime("meeting-%Y%m%d-%H%M%S")

        self._t0 = 0.0
        self._mic: Optional[MicCapture] = None
        self._system: Optional[SystemCapture] = None
        self._mic_writer: Optional[_MonoWavWriter] = None
        self._system_writer: Optional[_MonoWavWriter] = None
        self._mic_path: Optional[Path] = None
        self._system_path: Optional[Path] = None
        self._warnings: list[str] = []
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def system_description(self) -> str:
        return getattr(self._system, "description", "") if self._system else ""

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self._t0) if self._running else 0.0

    def levels(self) -> Levels:
        return Levels(
            mic=self._mic_writer.take_peak() if self._mic_writer else 0.0,
            system=self._system_writer.take_peak() if self._system_writer else 0.0,
        )

    def start(self) -> None:
        if self._running:
            return
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._warnings = []

        if self._record_mic:
            self._mic_path = self._out_dir / f"{self._stem}-me.wav"
            self._mic_writer = _MonoWavWriter(self._mic_path)
            self._mic = MicCapture(self._mic_writer, device=self._mic_device)
            try:
                self._mic.start()
            except CaptureError as exc:
                self._warnings.append(str(exc))
                self._discard_mic()

        if self._record_system:
            self._system_path = self._out_dir / f"{self._stem}-meeting.wav"
            self._system_writer = _MonoWavWriter(self._system_path)
            capture, problem = open_system_capture(self._system_writer, self._system_backend)
            if capture is None:
                self._warnings.append(problem or "system audio is unavailable")
                self._discard_system()
            else:
                self._system = capture
                try:
                    capture.start()
                except CaptureError as exc:
                    self._warnings.append(str(exc))
                    self._system = None
                    self._discard_system()

        if self._mic is None and self._system is None:
            message = "Nothing could be recorded."
            if self._warnings:
                message = f"{message} {' '.join(self._warnings)}"
            raise CaptureError(message)

        # Both sides are live: from here the two tracks share one origin, which
        # is what makes their transcripts mergeable on timestamp.
        self._t0 = time.monotonic()
        for writer in (self._mic_writer, self._system_writer):
            if writer is not None:
                writer.set_origin(self._t0)
        self._running = True

    def _discard_mic(self) -> None:
        if self._mic_writer is not None:
            self._mic_writer.close()
        self._mic_writer = None
        self._mic = None
        if self._mic_path is not None:
            self._mic_path.unlink(missing_ok=True)
        self._mic_path = None

    def _discard_system(self) -> None:
        if self._system_writer is not None:
            self._system_writer.close()
        self._system_writer = None
        if self._system_path is not None:
            self._system_path.unlink(missing_ok=True)
        self._system_path = None

    def stop(self) -> Recording:
        if not self._running:
            raise CaptureError("not recording")
        self._running = False
        duration = max(0.0, time.monotonic() - self._t0)

        for capture in (self._mic, self._system):
            if capture is not None:
                try:
                    capture.stop()
                except Exception:
                    pass

        for capture, side in ((self._mic, "microphone"), (self._system, "system audio")):
            problem = getattr(capture, "error", None) if capture else None
            if problem:
                self._warnings.append(f"{side}: {problem}")

        mic_frames = self._mic_writer.close() if self._mic_writer else 0
        system_frames = self._system_writer.close() if self._system_writer else 0

        # A track that stayed silent usually means the wrong device, or nothing
        # was actually playing — worth saying, since it looks like a bug in the
        # transcript rather than in the setup.
        if self._mic_path is not None and mic_frames == 0:
            self._warnings.append("the microphone recorded nothing")
            self._mic_path.unlink(missing_ok=True)
            self._mic_path = None
        if self._system_path is not None and system_frames == 0:
            self._warnings.append("no system audio was captured — was anything playing?")
            self._system_path.unlink(missing_ok=True)
            self._system_path = None

        recording = Recording(
            stem=self._stem,
            mic_path=self._mic_path,
            system_path=self._system_path,
            duration_sec=duration,
            started_at=time.time() - duration,
            warnings=list(self._warnings),
        )
        self._mic = self._system = None
        self._mic_writer = self._system_writer = None
        return recording
