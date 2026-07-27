"""Meeting Transcriber desktop app."""

from __future__ import annotations

import warnings

# Suppress torchcodec / pyannote audio decoder warnings (not needed for Silero VAD / soundfile)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")

import queue
import threading
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import customtkinter as ctk

from recorder import AudioRecorder, LiveChunk, get_input_devices, test_microphone
from transcriber import Transcriber, WhisperXTranscriber, segments_to_text

APP_TITLE = "Meeting Transcriber"
RECORDINGS_DIR = Path("recordings")
LIVE_CHUNK_SEC = 25.0
POLL_MS = 200
IMPORT_FILETYPES = [
    ("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.mp4"),
    ("All files", "*.*"),
]
AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4",
}


class AppState(Enum):
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


def _format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


class MeetingTranscriberApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.title(APP_TITLE)
        self.geometry("860x640")
        self.minsize(720, 520)

        self._state = AppState.IDLE
        self._models_ready = False
        # Whether the final pass uses PhoWhisper (pure Vietnamese). Disabled for
        # the "vi+en" mixed mode, which needs multilingual large-v3 to keep
        # English words instead of Vietnamizing them.
        self._use_phowhisper = True
        self._session_dir: Optional[Path] = None
        self._elapsed_sec = 0.0
        self._timer_job: Optional[str] = None

        # Skeleton/loading overlay shown over the transcript area while a final
        # or import transcription is running.
        self._skeleton_job: Optional[str] = None
        self._skeleton_bars: list[ctk.CTkFrame] = []
        self._skeleton_highlight = 0

        self._ui_queue: queue.Queue[tuple] = queue.Queue()
        self._live_worker_busy = threading.Event()
        self._final_worker_busy = threading.Event()
        self._progress_indeterminate = False

        self._input_devices = get_input_devices()
        self._device_map = {display_name: dev_id for dev_id, display_name in self._input_devices}
        self._selected_device_id: Optional[int] = self._input_devices[0][0] if self._input_devices else None

        self._recorder = AudioRecorder(
            chunk_duration_sec=LIVE_CHUNK_SEC,
            on_chunk=self._on_live_chunk,
            on_error=self._on_recorder_error,
            on_level=self._on_mic_level,
        )
        self._transcriber = Transcriber(language="vi")
        # PhoWhisper-large via WhisperX handles the accurate Vietnamese final
        # pass; the openai-whisper large-v3 path stays as the fallback.
        self._final_transcriber = WhisperXTranscriber(language="vi")

        self._build_ui()
        self.after(POLL_MS, self._poll_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start background model loading/downloading
        threading.Thread(target=self._initialize_models, daemon=True).start()

    def _initialize_models(self) -> None:
        # Only warm the small live model so the app is usable quickly. The large
        # PhoWhisper-large model installs on demand at the first final/import
        # transcription so it never blocks startup.
        self._ui_queue.put(("status", "Preparing live model…"))
        try:
            self._transcriber.preload_live_model(progress_cb=self._on_download_progress)
            self._ui_queue.put(("models_ready", None))
        except Exception as exc:
            self._ui_queue.put(("error", f"Failed to load models: {exc}"))

    def _on_download_progress(self, model_name: str, downloaded: int, total: int) -> None:
        self._ui_queue.put(
            (
                "download_progress",
                {"model": model_name, "downloaded": downloaded, "total": total},
            )
        )

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(3, weight=1)

        title = ctk.CTkLabel(
            header,
            text=APP_TITLE,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))

        lang_label = ctk.CTkLabel(header, text="Language:")
        lang_label.grid(row=1, column=0, sticky="w", padx=(0, 8))

        self.language_var = tk.StringVar(value="vi")
        self.language_menu = ctk.CTkOptionMenu(
            header,
            values=["vi", "vi+en", "en", "auto"],
            variable=self.language_var,
            command=self._on_language_change,
            width=90,
        )
        self.language_menu.grid(row=1, column=1, sticky="w", padx=(0, 16))

        mic_label = ctk.CTkLabel(header, text="Microphone:")
        mic_label.grid(row=1, column=2, sticky="w", padx=(0, 8))

        mic_names = list(self._device_map.keys()) if self._device_map else ["Default Microphone"]
        self.mic_var = tk.StringVar(value=mic_names[0])
        self.mic_menu = ctk.CTkOptionMenu(
            header,
            values=mic_names,
            variable=self.mic_var,
            command=self._on_device_change,
            width=230,
        )
        self.mic_menu.grid(row=1, column=3, sticky="w", padx=(0, 12))

        self.test_mic_button = ctk.CTkButton(
            header,
            text="Test Mic",
            command=self._test_mic,
            width=90,
            fg_color="gray30",
            hover_color="gray40",
        )
        self.test_mic_button.grid(row=1, column=4, sticky="w")

        controls = ctk.CTkFrame(self)
        controls.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        controls.grid_columnconfigure(4, weight=1)

        self.start_button = ctk.CTkButton(
            controls,
            text="Start",
            command=self._start_recording,
            width=110,
            state="disabled",
        )
        self.start_button.grid(row=0, column=0, padx=(0, 8))

        self.stop_button = ctk.CTkButton(
            controls,
            text="Stop",
            command=self._stop_recording,
            width=110,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 8))

        self.import_button = ctk.CTkButton(
            controls,
            text="Import Files…",
            command=self._import_file,
            width=110,
            state="disabled",
        )
        self.import_button.grid(row=0, column=2, padx=(0, 16))

        self.timer_label = ctk.CTkLabel(
            controls,
            text="00:00:00",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.timer_label.grid(row=0, column=3, sticky="w", padx=(0, 16))

        # Audio level indicator (VU meter)
        level_frame = ctk.CTkFrame(controls, fg_color="transparent")
        level_frame.grid(row=0, column=4, sticky="e", padx=(0, 8))

        self.mic_level_label = ctk.CTkLabel(
            level_frame,
            text="Mic: 0%",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.mic_level_label.pack(side="left", padx=(0, 6))

        self.mic_level_bar = ctk.CTkProgressBar(
            level_frame,
            width=100,
            height=12,
            progress_color="#2ecc71",
        )
        self.mic_level_bar.set(0.0)
        self.mic_level_bar.pack(side="left")

        self.status_label = ctk.CTkLabel(
            self,
            text="Checking models...",
            anchor="w",
        )
        self.status_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 4))

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.progress_bar.grid_remove()

        transcript_frame = ctk.CTkFrame(self)
        transcript_frame.grid(row=4, column=0, sticky="nsew", padx=16, pady=(0, 8))
        transcript_frame.grid_columnconfigure(0, weight=1)
        transcript_frame.grid_rowconfigure(1, weight=1)

        transcript_title = ctk.CTkLabel(
            transcript_frame,
            text="Transcript",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        transcript_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))

        self.drop_hint = ctk.CTkLabel(
            transcript_frame,
            text="Drag & drop audio files here to transcribe",
            text_color="gray",
        )
        self.drop_hint.grid(row=0, column=1, sticky="e", padx=12, pady=(12, 4))

        self.transcript_box = ctk.CTkTextbox(
            transcript_frame,
            wrap="word",
            font=ctk.CTkFont(size=14),
        )
        self.transcript_box.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12)
        )

        self._build_skeleton(transcript_frame)

        self._setup_dnd()

        self.saved_label = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            text_color="gray",
        )
        self.saved_label.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 16))

    def _build_skeleton(self, parent: ctk.CTkFrame) -> None:
        """Create a shimmer skeleton overlay that covers the transcript box."""
        self.skeleton_frame = ctk.CTkFrame(
            parent,
            fg_color=self.transcript_box.cget("fg_color"),
            corner_radius=6,
        )
        self.skeleton_frame.grid_columnconfigure(0, weight=1)

        self.skeleton_label = ctk.CTkLabel(
            self.skeleton_frame,
            text="Transcribing…",
            text_color="gray",
            anchor="w",
        )
        self.skeleton_label.grid(row=0, column=0, sticky="w", padx=16, pady=(20, 12))

        # A stack of gray "text line" bars of varying widths that shimmer.
        bar_widths = [560, 700, 480, 640, 420, 600, 360, 520]
        self._skeleton_bars = []
        for i, width in enumerate(bar_widths):
            bar = ctk.CTkFrame(
                self.skeleton_frame,
                height=14,
                width=width,
                corner_radius=7,
                fg_color=("gray80", "gray28"),
            )
            bar.grid(row=i + 1, column=0, sticky="w", padx=16, pady=6)
            bar.grid_propagate(False)
            self._skeleton_bars.append(bar)

        self.skeleton_frame.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12)
        )
        self.skeleton_frame.grid_remove()

    def _show_skeleton(self) -> None:
        if not self._skeleton_bars:
            return
        self.skeleton_frame.grid()
        self.skeleton_frame.tkraise()
        self._skeleton_highlight = 0
        self._animate_skeleton()

    def _hide_skeleton(self) -> None:
        if self._skeleton_job is not None:
            self.after_cancel(self._skeleton_job)
            self._skeleton_job = None
        if getattr(self, "skeleton_frame", None) is not None:
            self.skeleton_frame.grid_remove()

    def _animate_skeleton(self) -> None:
        base = ("gray80", "gray28")
        highlight = ("gray92", "gray45")
        for idx, bar in enumerate(self._skeleton_bars):
            bar.configure(fg_color=highlight if idx == self._skeleton_highlight else base)
        self._skeleton_highlight = (self._skeleton_highlight + 1) % len(self._skeleton_bars)
        self._skeleton_job = self.after(140, self._animate_skeleton)

    def _on_device_change(self, choice: str) -> None:
        self._selected_device_id = self._device_map.get(choice)

    def _test_mic(self) -> None:
        if self._state != AppState.IDLE:
            return
        self.test_mic_button.configure(state="disabled", text="Testing…")
        self._set_status("Testing microphone for 1.5 seconds… speak now!")

        dev_id = self._selected_device_id

        def worker() -> None:
            peak = test_microphone(device=dev_id, duration_sec=1.5)
            self._ui_queue.put(("test_mic_result", peak))

        threading.Thread(target=worker, daemon=True).start()

    def _on_mic_level(self, peak: float) -> None:
        self._ui_queue.put(("mic_level", peak))

    def _on_language_change(self, value: str) -> None:
        self._apply_language_selection()

    def _apply_language_selection(self) -> None:
        """Translate the UI language choice into transcriber + model routing.

        "vi" uses PhoWhisper-large (best pure Vietnamese). "vi+en" decodes in
        Vietnamese but on multilingual large-v3, which keeps embedded English
        words instead of forcing them into Vietnamese spelling. "en"/"auto" also
        use large-v3.
        """
        value = self.language_var.get()
        if value == "vi+en":
            self._transcriber.set_language("vi")
            self._use_phowhisper = False
        elif value == "vi":
            self._transcriber.set_language("vi")
            self._use_phowhisper = True
        else:
            self._transcriber.set_language(value)
            self._use_phowhisper = False

    def _set_state(self, state: AppState) -> None:
        self._state = state
        if state == AppState.IDLE:
            ready = "normal" if self._models_ready else "disabled"
            self.start_button.configure(state=ready)
            self.import_button.configure(state=ready)
            self.stop_button.configure(state="disabled")
            self.language_menu.configure(state="normal")
            self.mic_menu.configure(state="normal")
            self.test_mic_button.configure(state="normal")
            self.mic_level_bar.set(0.0)
            self.mic_level_label.configure(text="Mic: 0%")
            self._hide_skeleton()
        elif state == AppState.RECORDING:
            self.start_button.configure(state="disabled")
            self.import_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.language_menu.configure(state="disabled")
            self.mic_menu.configure(state="disabled")
            self.test_mic_button.configure(state="disabled")
            self._hide_skeleton()
        elif state == AppState.TRANSCRIBING:
            self.start_button.configure(state="disabled")
            self.import_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.language_menu.configure(state="disabled")
            self.mic_menu.configure(state="disabled")
            self.test_mic_button.configure(state="disabled")
            self.mic_level_bar.set(0.0)
            self.mic_level_label.configure(text="Mic: 0%")
            self._show_skeleton()

    def _start_recording(self) -> None:
        if self._state != AppState.IDLE:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_dir = RECORDINGS_DIR / timestamp
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._elapsed_sec = 0.0
        self._update_timer_label()
        self.transcript_box.delete("1.0", "end")
        self.saved_label.configure(text="", text_color="gray")
        self._set_status("Loading live model…")

        self._apply_language_selection()
        self._set_state(AppState.RECORDING)

        dev_id = self._selected_device_id

        def preload_and_start() -> None:
            try:
                self._recorder.start(self._session_dir, device=dev_id)
                self._ui_queue.put(("recording_started", None))
            except Exception as exc:
                self._ui_queue.put(("error", str(exc)))

        threading.Thread(target=preload_and_start, daemon=True).start()

    def _stop_recording(self) -> None:
        if self._state != AppState.RECORDING:
            return

        self._set_state(AppState.TRANSCRIBING)
        self._set_status("Stopping recording…")
        self._stop_timer()

        def stop_and_transcribe() -> None:
            try:
                wav_path = self._recorder.stop()
                max_amp = self._recorder.max_amplitude
                if wav_path is None:
                    self._ui_queue.put(("error", "No recording found."))
                    return

                if max_amp < 0.001:
                    self._ui_queue.put(
                        (
                            "silence_warning",
                            "⚠️ Recorded audio was completely silent (0% volume).\n"
                            "This usually means macOS has BLOCKED Microphone permissions for your Terminal / IDE.\n"
                            "Fix: Go to System Settings -> Privacy & Security -> Microphone and enable your terminal app (Cursor / Terminal / iTerm2), then restart.",
                        )
                    )

                self._ui_queue.put(("status", "Waiting for live transcription to finish…"))
                while self._live_worker_busy.is_set():
                    self._live_worker_busy.wait(timeout=0.2)

                self._final_worker_busy.set()
                try:
                    final_text = self._run_final_transcription(wav_path)
                    transcript_path = wav_path.parent / "transcript.txt"
                    transcript_path.write_text(final_text, encoding="utf-8")
                    self._ui_queue.put(
                        (
                            "final_done",
                            {
                                "text": final_text,
                                "transcript_path": str(transcript_path),
                                "wav_path": str(wav_path),
                            },
                        )
                    )
                finally:
                    self._final_worker_busy.clear()
            except Exception as exc:
                self._ui_queue.put(("error", str(exc)))

        threading.Thread(target=stop_and_transcribe, daemon=True).start()

    def _setup_dnd(self) -> None:
        """Enable dropping audio files onto the transcript area (best-effort)."""
        self._dnd_enabled = False
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES

            self.TkdndVersion = TkinterDnD._require(self)
            target = getattr(self.transcript_box, "_textbox", self.transcript_box)
            target.drop_target_register(DND_FILES)
            target.dnd_bind("<<Drop>>", self._on_drop)
            self._dnd_enabled = True
        except Exception:
            # Drag-and-drop is optional; the Import Files button still works.
            self.drop_hint.configure(text="")

    def _on_drop(self, event) -> None:
        if self._state != AppState.IDLE or not self._models_ready:
            self._set_status("Busy — finish the current task before importing.")
            return
        try:
            raw = self.tk.splitlist(event.data)
        except Exception:
            raw = event.data.split()
        self._start_batch_transcription([Path(p) for p in raw])

    def _import_file(self) -> None:
        if self._state != AppState.IDLE or not self._models_ready:
            return

        selected = filedialog.askopenfilenames(
            title="Select audio file(s) to transcribe",
            filetypes=IMPORT_FILETYPES,
        )
        if not selected:
            return
        self._start_batch_transcription([Path(p) for p in selected])

    def _start_batch_transcription(self, paths: list[Path]) -> None:
        if self._state != AppState.IDLE or not self._models_ready:
            return

        audio_paths = [
            p for p in paths if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
        if not audio_paths:
            self._set_status("No supported audio files to import.")
            return

        self._apply_language_selection()
        self._set_state(AppState.TRANSCRIBING)
        self.transcript_box.delete("1.0", "end")
        self.saved_label.configure(text="")

        def worker() -> None:
            self._final_worker_busy.set()
            saved: list[str] = []
            total = len(audio_paths)
            try:
                for idx, source in enumerate(audio_paths, start=1):
                    self._ui_queue.put(
                        ("status", f"Transcribing {idx}/{total}: {source.name}…")
                    )
                    try:
                        text = self._run_final_transcription(source)
                    except Exception as exc:
                        self._ui_queue.put(
                            ("batch_item", {"name": source.name, "text": f"[error: {exc}]"})
                        )
                        continue
                    transcript_path = source.with_name(source.stem + ".transcript.txt")
                    transcript_path.write_text(text, encoding="utf-8")
                    saved.append(str(transcript_path))
                    self._ui_queue.put(
                        ("batch_item", {"name": source.name, "text": text})
                    )
                self._ui_queue.put(("batch_done", {"count": len(saved), "saved": saved}))
            finally:
                self._final_worker_busy.clear()

        threading.Thread(target=worker, daemon=True).start()

    def _run_final_transcription(self, wav_path: Path) -> str:
        """Final pass: PhoWhisper+WhisperX for pure Vietnamese, else large-v3.

        Runs in a worker thread. PhoWhisper-large is used only for the "vi" mode
        (pure Vietnamese); "vi+en"/"en"/"auto" use multilingual large-v3 so
        code-switched English words survive. For "vi", PhoWhisper-large is
        installed on demand (downloaded + converted) the first time, with
        progress reported to the UI; the UI stays responsive because this runs
        off the main thread.
        """
        if self._use_phowhisper and self._transcriber.language == "vi":
            try:
                if not self._final_transcriber.is_ready():
                    self._ui_queue.put(
                        ("status", "Installing PhoWhisper-large (one-time download)…")
                    )
                self._final_transcriber.preload(
                    progress_cb=self._on_download_progress,
                    status_cb=lambda msg: self._ui_queue.put(("status", msg)),
                )
                self._ui_queue.put(("hide_progress", None))
                self._ui_queue.put(
                    ("status", "Running final transcription (PhoWhisper + WhisperX)…")
                )
                return self._final_transcriber.transcribe_file_to_text(wav_path)
            except Exception as exc:
                self._ui_queue.put(
                    ("hide_progress", None)
                )
                self._ui_queue.put(
                    ("status", f"PhoWhisper failed ({exc}); falling back to large-v3…")
                )

        self._ui_queue.put(("status", "Running final transcription (large-v3)…"))
        return self._transcriber.transcribe_file_to_text(wav_path)

    def _on_live_chunk(self, chunk: LiveChunk) -> None:
        if self._live_worker_busy.is_set():
            return

        self._live_worker_busy.set()

        def worker() -> None:
            try:
                segments = self._transcriber.transcribe_chunk(
                    chunk.audio,
                    offset_sec=chunk.start_sec,
                )
                if segments:
                    text = segments_to_text(segments)
                    self._ui_queue.put(("live_text", text))
            except Exception as exc:
                self._ui_queue.put(("error", str(exc)))
            finally:
                self._live_worker_busy.clear()

        threading.Thread(target=worker, daemon=True).start()

    def _on_recorder_error(self, exc: Exception) -> None:
        self._ui_queue.put(("error", str(exc)))

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                event, payload = self._ui_queue.get_nowait()
                self._handle_ui_event(event, payload)
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll_ui_queue)

    def _handle_ui_event(self, event: str, payload) -> None:
        if event == "recording_started":
            self._set_status("Recording… live transcript will appear shortly.")
            self._start_timer()
        elif event == "mic_level":
            peak = float(payload)
            self.mic_level_bar.set(min(1.0, peak * 3.0))
            self.mic_level_label.configure(text=f"Mic: {int(peak * 100)}%")
        elif event == "test_mic_result":
            peak = float(payload)
            self.test_mic_button.configure(state="normal", text="Test Mic")
            if peak >= 0.001:
                self._set_status(f"✅ Microphone working! Peak volume: {int(peak * 100)}%")
                self.saved_label.configure(text="", text_color="gray")
            else:
                self._set_status("⚠️ Mic captured 0 sound! Check macOS Privacy & Security -> Microphone permissions.")
                self.saved_label.configure(
                    text="⚠️ SILENT MIC: macOS is blocking microphone access for your Terminal/IDE.\n"
                         "Open System Settings -> Privacy & Security -> Microphone -> Enable permission for Terminal/Cursor.",
                    text_color="red",
                )
        elif event == "silence_warning":
            self.saved_label.configure(text=payload, text_color="red")
        elif event == "live_text":
            self._append_transcript(payload)
            if self._state == AppState.RECORDING:
                self._set_status("Recording… updating live transcript.")
        elif event == "status":
            self._set_status(payload)
        elif event == "download_progress":
            self._update_download_progress(payload)
        elif event == "hide_progress":
            self._hide_progress_bar()
        elif event == "models_ready":
            self._models_ready = True
            self.progress_bar.grid_remove()
            self._set_status("Ready. Press Start to record or Import File to transcribe.")
            self.start_button.configure(state="normal")
            self.import_button.configure(state="normal")
        elif event == "final_done":
            self._hide_progress_bar()
            self.transcript_box.delete("1.0", "end")
            self._append_transcript(payload["text"])
            self.saved_label.configure(
                text=f"Saved: {payload['transcript_path']} | {payload['wav_path']}"
            )
            self._set_status("Done. Final transcript saved.")
            self._set_state(AppState.IDLE)
        elif event == "batch_item":
            self._hide_skeleton()
            self._append_transcript(f"===== {payload['name']} =====")
            self._append_transcript(payload["text"])
            self._append_transcript("")
        elif event == "batch_done":
            self._hide_progress_bar()
            count = payload["count"]
            if payload["saved"]:
                self.saved_label.configure(
                    text=f"Saved {count} transcript(s): " + " | ".join(payload["saved"])
                )
            self._set_status(f"Done. Transcribed {count} file(s).")
            self._set_state(AppState.IDLE)
        elif event == "error":
            self._set_status(f"Error: {payload}")
            self._stop_timer()
            self._set_state(AppState.IDLE)

    def _append_transcript(self, text: str) -> None:
        if not text:
            return
        self.transcript_box.insert("end", text if text.endswith("\n") else text + "\n")
        self.transcript_box.see("end")

    def _update_download_progress(self, payload: dict) -> None:
        model = payload.get("model", "")
        downloaded = int(payload.get("downloaded", 0))
        total = int(payload.get("total", 0))

        if not self.progress_bar.winfo_ismapped():
            self.progress_bar.grid()

        if total > 0:
            if self._progress_indeterminate:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self._progress_indeterminate = False
            fraction = min(1.0, downloaded / total)
            self.progress_bar.set(fraction)
            self._set_status(
                f"Downloading model '{model}': "
                f"{_format_bytes(downloaded)} / {_format_bytes(total)} "
                f"({fraction * 100:.0f}%)"
            )
        else:
            if not self._progress_indeterminate:
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start()
                self._progress_indeterminate = True
            self._set_status(f"Downloading model '{model}': {_format_bytes(downloaded)}")

    def _hide_progress_bar(self) -> None:
        if self._progress_indeterminate:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self._progress_indeterminate = False
        self.progress_bar.set(0.0)
        if self.progress_bar.winfo_ismapped():
            self.progress_bar.grid_remove()

    def _set_status(self, message: str) -> None:
        self.status_label.configure(text=message)

    def _start_timer(self) -> None:
        self._stop_timer()
        self._tick_timer()

    def _stop_timer(self) -> None:
        if self._timer_job is not None:
            self.after_cancel(self._timer_job)
            self._timer_job = None

    def _tick_timer(self) -> None:
        if self._state == AppState.RECORDING:
            self._elapsed_sec = self._recorder.elapsed_sec
            self._update_timer_label()
            self._timer_job = self.after(500, self._tick_timer)

    def _update_timer_label(self) -> None:
        total = max(0, int(self._elapsed_sec))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        self.timer_label.configure(text=f"{hours:02d}:{minutes:02d}:{secs:02d}")

    def _on_close(self) -> None:
        if self._state == AppState.RECORDING:
            self._recorder.stop()
        self.destroy()


def main() -> None:
    RECORDINGS_DIR.mkdir(exist_ok=True)
    app = MeetingTranscriberApp()
    app.mainloop()


if __name__ == "__main__":
    main()
