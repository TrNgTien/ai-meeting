"""Meeting Transcriber desktop app."""

from __future__ import annotations

import warnings

# Suppress torchcodec / pyannote audio decoder warnings (not needed for Silero VAD / soundfile)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import customtkinter as ctk

from chunking import (
    TranscribeAudio,
    TranscriptionCancelled,
    resumable_seconds,
    transcribe_chunked,
    transcript_path_for,
)
from transcriber import (
    FINAL_MODEL,
    FINAL_MODEL_OPTIONS,
    DownloadCancelled,
    Transcriber,
    WhisperXTranscriber,
    delete_model,
    ensure_model_downloaded,
    format_timestamp,
    is_model_downloaded,
    list_downloaded_whisper_models,
    model_size_on_disk,
    segments_to_text,
)

APP_TITLE = "Meeting Transcriber"
POLL_MS = 200
IMPORT_FILETYPES = [
    ("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.mp4"),
    ("All files", "*.*"),
]
AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4",
}

# Placeholder shown in the Model dropdown when no checkpoint is cached on disk.
NO_MODELS_LABEL = "No models downloaded"


class AppState(Enum):
    IDLE = auto()
    TRANSCRIBING = auto()


def _center_geometry(win: tk.Misc, width: int, height: int, over: Optional[tk.Misc] = None) -> None:
    """Set win's size and position so it's centered on `over`, or the screen if omitted."""
    if over is not None:
        over.update_idletasks()
        cx = over.winfo_rootx() + over.winfo_width() // 2
        cy = over.winfo_rooty() + over.winfo_height() // 2
    else:
        win.update_idletasks()
        cx = win.winfo_screenwidth() // 2
        cy = win.winfo_screenheight() // 2
    x = max(0, cx - width // 2)
    y = max(0, cy - height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")


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
        _center_geometry(self, 860, 640)
        self.minsize(720, 520)

        self._state = AppState.IDLE
        # Whether the final pass uses PhoWhisper (pure Vietnamese). Disabled for
        # the "vi+en" mixed mode (the default), which needs a multilingual
        # openai-whisper model to keep English words instead of Vietnamizing
        # them.
        self._use_phowhisper = False
        # Raw openai-whisper checkpoint name selected in the Model dropdown,
        # which lists downloaded models only (see _refresh_model_menu_labels).
        self._selected_model_name = FINAL_MODEL
        self._model_label_to_name: dict[str, str] = {}
        self._no_models_downloaded = False
        # Set by Stop; the chunk loop checks it between chunks and saves the
        # checkpoint before bailing out.
        self._cancel_event = threading.Event()

        # Skeleton/loading overlay shown over the transcript area while a
        # transcription is running.
        self._skeleton_job: Optional[str] = None
        self._skeleton_bars: list[ctk.CTkFrame] = []
        self._skeleton_highlight = 0

        self._ui_queue: queue.Queue[tuple] = queue.Queue()
        self._progress_indeterminate = False

        # Set while the Manage Models dialog is open, so background downloads
        # started from the main transcription flow or from that dialog can
        # refresh its status text/list via the UI queue.
        self._model_manager_refresh = None
        self._mm_status_var: Optional[tk.StringVar] = None
        # name -> cancel signal for downloads started from Manage Models.
        self._download_cancel_events: dict[str, threading.Event] = {}

        self._transcriber = Transcriber(language="vi")
        # PhoWhisper-large via WhisperX handles the accurate Vietnamese final
        # pass; the openai-whisper path stays as the fallback/mixed-language engine.
        self._final_transcriber = WhisperXTranscriber(language="vi")

        self._build_ui()
        self._update_model_menu_state()
        self.after(POLL_MS, self._poll_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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

        self.language_var = tk.StringVar(value="vi+en")
        self.language_menu = ctk.CTkOptionMenu(
            header,
            values=["vi+en", "vi", "en", "auto"],
            variable=self.language_var,
            command=self._on_language_change,
            width=90,
        )
        self.language_menu.grid(row=1, column=1, sticky="w", padx=(0, 16))

        model_label = ctk.CTkLabel(header, text="Model:")
        model_label.grid(row=1, column=2, sticky="w", padx=(0, 8))

        self.model_var = tk.StringVar(value=FINAL_MODEL)
        self.model_menu = ctk.CTkOptionMenu(
            header,
            values=FINAL_MODEL_OPTIONS,
            variable=self.model_var,
            command=self._on_model_change,
            width=160,
        )
        self.model_menu.grid(row=1, column=3, sticky="w", padx=(0, 16))

        self.manage_models_button = ctk.CTkButton(
            header,
            text="Manage Models…",
            command=self._open_model_manager,
            width=130,
            fg_color="gray30",
            hover_color="gray40",
        )
        self.manage_models_button.grid(row=1, column=4, sticky="w")

        model_hint = ctk.CTkLabel(
            header,
            text="(used for vi+en / en / auto · only downloaded models are listed · "
                 "need another model? Use Manage Models…)",
            text_color="gray",
            font=ctk.CTkFont(size=11),
        )
        model_hint.grid(row=2, column=0, columnspan=5, sticky="w", pady=(2, 0))

        self._refresh_model_menu_labels()

        controls = ctk.CTkFrame(self)
        controls.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        controls.grid_columnconfigure(2, weight=1)

        self.import_button = ctk.CTkButton(
            controls,
            text="Import Files…",
            command=self._import_file,
            width=130,
        )
        self.import_button.grid(row=0, column=0, padx=(0, 8))

        # Stopping is safe at any point: the finished chunks are already on
        # disk, so re-importing the file resumes instead of restarting.
        self.stop_button = ctk.CTkButton(
            controls,
            text="Stop",
            command=self._request_stop,
            width=90,
            fg_color="#8b2e2e",
            hover_color="#a13a3a",
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 16))

        self.status_label = ctk.CTkLabel(
            self,
            text="Ready. Import audio file(s) to transcribe.",
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

    def _on_language_change(self, value: str) -> None:
        self._apply_language_selection()
        self._update_model_menu_state()

    def _on_model_change(self, label: str) -> None:
        self._selected_model_name = self._model_label_to_name.get(label, self._selected_model_name)
        self._apply_language_selection()

    def _refresh_model_menu_labels(self) -> None:
        """Rebuild the Model dropdown so it lists only checkpoints already
        cached on disk — anything else has to be fetched from Manage Models…
        first, so offering it here would just stall the next transcription on a
        multi-GB download.

        If the selected model is no longer on disk (deleted from Manage
        Models…), the selection moves to a downloaded one so the dropdown never
        lists something that isn't there. When nothing is downloaded at all the
        dropdown is disabled with a placeholder; the default model is still used
        (and downloaded on demand) if a transcription is started anyway.
        """
        names = [name for name in FINAL_MODEL_OPTIONS if is_model_downloaded(name)]
        self._model_label_to_name = {name: name for name in names}
        self._no_models_downloaded = not names

        if not names:
            self.model_menu.configure(values=[NO_MODELS_LABEL])
            self.model_var.set(NO_MODELS_LABEL)
            self._update_model_menu_state()
            return

        if self._selected_model_name not in names:
            self._selected_model_name = FINAL_MODEL if FINAL_MODEL in names else names[-1]
            if not self._use_phowhisper:
                self._transcriber.set_final_model(self._selected_model_name)

        self.model_menu.configure(values=names)
        self.model_var.set(self._selected_model_name)
        self._update_model_menu_state()

    def _update_model_menu_state(self) -> None:
        """Model choice only applies to the openai-whisper path (not PhoWhisper)."""
        busy = self._state != AppState.IDLE
        if busy or self._no_models_downloaded:
            # Nothing downloaded yet means there is nothing to choose between;
            # Manage Models… is the way out.
            self.model_menu.configure(state="disabled")
        else:
            is_pure_vi = self.language_var.get() == "vi"
            self.model_menu.configure(state="disabled" if is_pure_vi else "normal")
        # Manage Models stays open even mid-transcription/download, so users
        # can free disk space or queue up another model without waiting.
        self.manage_models_button.configure(state="normal")

    def _apply_language_selection(self) -> None:
        """Translate the UI language + model choice into transcriber routing.

        "vi" uses PhoWhisper-large (best pure Vietnamese). "vi+en" decodes in
        Vietnamese but on the user-selected multilingual openai-whisper model
        (default large-v3), which keeps embedded English words instead of
        forcing them into Vietnamese spelling. "en"/"auto" also use that model.
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

        if not self._use_phowhisper:
            self._transcriber.set_final_model(self._selected_model_name)

    def _open_model_manager(self) -> None:
        """Show every openai-whisper model cached on disk: download ones you
        need, or delete ones you don't. Available even while a transcription/
        download is running in the background.
        """
        win = ctk.CTkToplevel(self)
        win.title("Manage Local Models")
        _center_geometry(win, 640, 440, over=self)
        win.transient(self)
        win.grab_set()

        ctk.CTkLabel(
            win,
            text="Locally cached models",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 4))

        ctk.CTkLabel(
            win,
            text="Click Use to make a model the active selection (mirrored in the\n"
                 "Model dropdown). Download one here to have it ready before you\n"
                 "transcribe, or delete one you no longer need — deleted models\n"
                 "re-download automatically the next time you select and use them.",
            text_color="gray",
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        status_var = tk.StringVar(value="")
        ctk.CTkLabel(win, textvariable=status_var, text_color="gray").pack(
            anchor="w", padx=16, pady=(0, 8)
        )

        rows_frame = ctk.CTkScrollableFrame(win)
        rows_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        rows_frame.grid_columnconfigure(0, weight=1)

        def refresh() -> None:
            for child in rows_frame.winfo_children():
                child.destroy()

            extra_downloaded = [
                n for n in list_downloaded_whisper_models() if n not in FINAL_MODEL_OPTIONS
            ]
            names = FINAL_MODEL_OPTIONS + extra_downloaded

            for idx, name in enumerate(names):
                size = model_size_on_disk(name)
                downloaded = size > 0
                is_current = name == self._selected_model_name

                ctk.CTkLabel(
                    rows_frame,
                    text=name,
                    anchor="w",
                    font=ctk.CTkFont(weight="bold") if is_current else None,
                ).grid(row=idx, column=0, sticky="w", pady=4, padx=(4, 8))
                ctk.CTkLabel(
                    rows_frame,
                    text=_format_bytes(size) if downloaded else "not downloaded",
                    anchor="w",
                    text_color="gray" if not downloaded else None,
                ).grid(row=idx, column=1, sticky="w", pady=4, padx=(0, 8))

                if is_current:
                    ctk.CTkLabel(
                        rows_frame, text="✓ Selected", text_color="#2ecc71"
                    ).grid(row=idx, column=2, sticky="e", padx=(0, 8), pady=4)
                elif downloaded:
                    # Switching the active model while a transcription is
                    # running would race with the worker thread reading
                    # it, so only allow it when idle. Not-downloaded models
                    # have no Use button at all: the Model dropdown lists
                    # downloaded checkpoints only, so they must be
                    # downloaded here first.
                    ctk.CTkButton(
                        rows_frame,
                        text="Use",
                        width=70,
                        fg_color="gray30",
                        hover_color="gray40",
                        state="normal" if self._state == AppState.IDLE else "disabled",
                        command=lambda name=name: do_select(name),
                    ).grid(row=idx, column=2, sticky="e", padx=(0, 8), pady=4)

                downloading = name in self._download_cancel_events
                if downloading:
                    ctk.CTkButton(
                        rows_frame,
                        text="Cancel",
                        width=80,
                        fg_color="#8b2e2e",
                        hover_color="#a13a3a",
                        command=lambda name=name: do_cancel(name),
                    ).grid(row=idx, column=3, sticky="e", pady=4)
                elif downloaded:
                    ctk.CTkButton(
                        rows_frame,
                        text="Delete",
                        width=80,
                        fg_color="#8b2e2e",
                        hover_color="#a13a3a",
                        command=lambda name=name: do_delete(name),
                    ).grid(row=idx, column=3, sticky="e", pady=4)
                else:
                    ctk.CTkButton(
                        rows_frame,
                        text="Download",
                        width=80,
                        command=lambda name=name: do_download(name),
                    ).grid(row=idx, column=3, sticky="e", pady=4)

        def do_select(name: str) -> None:
            self._selected_model_name = name
            if not self._use_phowhisper:
                self._transcriber.set_final_model(name)
            self._refresh_model_menu_labels()
            refresh()

        def do_delete(name: str) -> None:
            if not messagebox.askyesno(
                "Delete model", f"Delete cached model '{name}'?", parent=win
            ):
                return
            delete_model(name)
            refresh()
            self._refresh_model_menu_labels()

        def do_download(name: str) -> None:
            if name in self._download_cancel_events:
                return
            cancel_event = threading.Event()
            self._download_cancel_events[name] = cancel_event
            status_var.set(f"Downloading {name}…")
            refresh()

            def worker() -> None:
                try:
                    ensure_model_downloaded(
                        name,
                        lambda m, d, t: self._ui_queue.put(
                            ("mm_progress", {"model": m, "downloaded": d, "total": t})
                        ),
                        cancel_event=cancel_event,
                    )
                    self._ui_queue.put(("mm_download_finished", {"name": name, "status": "done"}))
                except DownloadCancelled:
                    self._ui_queue.put(("mm_download_finished", {"name": name, "status": "cancelled"}))
                except Exception as exc:
                    self._ui_queue.put(
                        ("mm_download_finished", {"name": name, "status": "error", "error": str(exc)})
                    )

            threading.Thread(target=worker, daemon=True).start()

        def do_cancel(name: str) -> None:
            cancel_event = self._download_cancel_events.get(name)
            if cancel_event is not None:
                status_var.set(f"Cancelling {name}…")
                cancel_event.set()

        refresh()
        self._model_manager_refresh = refresh
        self._mm_status_var = status_var

        def on_close() -> None:
            self._model_manager_refresh = None
            self._mm_status_var = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)
        ctk.CTkButton(win, text="Close", command=on_close, width=100).pack(pady=(0, 16))

    def _set_state(self, state: AppState) -> None:
        self._state = state
        if state == AppState.IDLE:
            self.import_button.configure(state="normal")
            self.language_menu.configure(state="normal")
            self.stop_button.configure(state="disabled", text="Stop")
            self._hide_skeleton()
        elif state == AppState.TRANSCRIBING:
            self.import_button.configure(state="disabled")
            self.language_menu.configure(state="disabled")
            self.stop_button.configure(state="normal", text="Stop")
            self._show_skeleton()
        self._update_model_menu_state()

    def _request_stop(self) -> None:
        """Ask the worker to stop after the chunk it is currently transcribing.

        Nothing is lost: chunks finished before this point are already in the
        checkpoint file, so re-importing resumes from there.
        """
        if self._state != AppState.TRANSCRIBING:
            return
        self._cancel_event.set()
        self.stop_button.configure(state="disabled", text="Stopping…")
        self._set_status("Stopping after the current chunk — progress is saved.")

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
        if self._state != AppState.IDLE:
            self._set_status("Busy — finish the current task before importing.")
            return
        try:
            raw = self.tk.splitlist(event.data)
        except Exception:
            raw = event.data.split()
        self._start_batch_transcription([Path(p) for p in raw])

    def _import_file(self) -> None:
        if self._state != AppState.IDLE:
            return

        selected = filedialog.askopenfilenames(
            title="Select audio file(s) to transcribe",
            filetypes=IMPORT_FILETYPES,
        )
        if not selected:
            return
        self._start_batch_transcription([Path(p) for p in selected])

    def _start_batch_transcription(self, paths: list[Path]) -> None:
        if self._state != AppState.IDLE:
            return

        audio_paths = [
            p for p in paths if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
        if not audio_paths:
            self._set_status("No supported audio files to import.")
            return

        self._apply_language_selection()
        self._cancel_event = threading.Event()
        self._set_state(AppState.TRANSCRIBING)
        self.transcript_box.delete("1.0", "end")
        self.saved_label.configure(text="")

        def worker() -> None:
            saved: list[str] = []
            total = len(audio_paths)
            cancelled = False
            for idx, source in enumerate(audio_paths, start=1):
                label = f"Transcribing {idx}/{total}: {source.name}"
                self._ui_queue.put(("status", f"{label}…"))
                self._ui_queue.put(("file_start", source.name))
                try:
                    self._run_final_transcription(source, label)
                except TranscriptionCancelled:
                    # Chunks finished before this point are already written to
                    # the transcript file; the remaining files never started.
                    cancelled = True
                    break
                except Exception as exc:
                    self._ui_queue.put(("chunk_text", f"[error: {exc}]"))
                    continue
                # The transcript file is written chunk by chunk by
                # transcribe_chunked(), so there is nothing to save here.
                saved.append(str(transcript_path_for(source)))
            self._ui_queue.put(
                ("batch_done", {"count": len(saved), "saved": saved, "cancelled": cancelled})
            )

        threading.Thread(target=worker, daemon=True).start()

    def _prepare_engine(self) -> tuple[str, TranscribeAudio]:
        """Load the engine for the current language/model choice.

        Returns an identity key (part of the checkpoint fingerprint, so
        switching model or language never resumes into a transcript produced
        with different settings) and the per-chunk transcribe callable.

        PhoWhisper-large is used only for the "vi" mode (pure Vietnamese);
        "vi+en"/"en"/"auto" use the selected multilingual openai-whisper model
        so code-switched English words survive. Models install on demand
        (downloaded, and converted for PhoWhisper) the first time, with
        progress reported to the UI. Runs in the worker thread, so the UI stays
        responsive.
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
                return "phowhisper-large:vi", self._final_transcriber.transcribe_audio
            except Exception as exc:
                self._ui_queue.put(("hide_progress", None))
                self._ui_queue.put(
                    ("status", f"PhoWhisper failed ({exc}); falling back to {self._transcriber.final_model_name}…")
                )

        model_name = self._transcriber.final_model_name
        self._ui_queue.put(("status", f"Loading model '{model_name}' (downloads on first use)…"))
        self._transcriber.preload_final_model(progress_cb=self._on_download_progress)
        self._ui_queue.put(("hide_progress", None))
        language = self._transcriber.language or "auto"
        return f"whisper-{model_name}:{language}", self._transcriber.transcribe_audio

    def _run_final_transcription(self, wav_path: Path, label: str) -> str:
        """Transcribe one file in chunks, resuming from a checkpoint if present.

        Long meetings are the normal case here, so the file is processed a few
        minutes at a time and every finished chunk is persisted (see
        chunking.py). A crash, a quit, or Stop therefore costs at most the
        chunk in flight rather than the whole recording.
        """
        engine_key, transcribe_audio = self._prepare_engine()

        resume_at = resumable_seconds(wav_path, engine_key)
        if resume_at > 0:
            self._ui_queue.put(
                ("status", f"{label} — resuming at {format_timestamp(resume_at)}…")
            )

        def on_progress(
            chunk_index: int, chunks_done: int, done_sec: float, total_sec: Optional[float]
        ) -> None:
            self._ui_queue.put(
                (
                    "chunk_progress",
                    {
                        "label": label,
                        "done_sec": done_sec,
                        "total_sec": total_sec,
                        "chunks_done": chunks_done,
                    },
                )
            )

        return transcribe_chunked(
            wav_path,
            transcribe_audio=transcribe_audio,
            engine_key=engine_key,
            output_path=transcript_path_for(wav_path),
            on_progress=on_progress,
            on_text=lambda text: self._ui_queue.put(("chunk_text", text)),
            cancel_event=self._cancel_event,
        )

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                event, payload = self._ui_queue.get_nowait()
                self._handle_ui_event(event, payload)
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll_ui_queue)

    def _handle_ui_event(self, event: str, payload) -> None:
        if event == "status":
            self._set_status(payload)
        elif event == "download_progress":
            self._update_download_progress(payload)
        elif event == "hide_progress":
            self._hide_progress_bar()
            self._refresh_model_menu_labels()
        elif event == "file_start":
            # Keep the skeleton up until real text arrives — loading a model
            # can take a while before the first chunk lands.
            if self.transcript_box.index("end-1c") != "1.0":
                self._append_transcript("\n")  # blank line between files
            self._append_transcript(f"===== {payload} =====")
        elif event == "chunk_text":
            # Each chunk lands here as it is appended to the transcript file,
            # so a long meeting fills in as it goes instead of at the end.
            self._hide_skeleton()
            self._append_transcript(payload)
        elif event == "chunk_progress":
            self._update_chunk_progress(payload)
        elif event == "batch_done":
            self._hide_progress_bar()
            count = payload["count"]
            if payload["saved"]:
                self.saved_label.configure(
                    text=f"Saved {count} transcript(s): " + " | ".join(payload["saved"])
                )
            if payload.get("cancelled"):
                self._set_status(
                    f"Stopped. {count} file(s) finished; partial progress saved — "
                    "import the same file again to resume."
                )
            else:
                self._set_status(f"Done. Transcribed {count} file(s).")
            self._set_state(AppState.IDLE)
        elif event == "error":
            self._set_status(f"Error: {payload}")
            self._set_state(AppState.IDLE)
        elif event == "mm_progress":
            if self._mm_status_var is not None:
                total = int(payload.get("total", 0))
                downloaded = int(payload.get("downloaded", 0))
                model = payload.get("model", "")
                if total > 0:
                    pct = min(100, downloaded / total * 100)
                    self._mm_status_var.set(
                        f"Downloading {model}: {_format_bytes(downloaded)} / "
                        f"{_format_bytes(total)} ({pct:.0f}%)"
                    )
                else:
                    self._mm_status_var.set(f"Downloading {model}: {_format_bytes(downloaded)}")
        elif event == "mm_download_finished":
            name = payload["name"]
            status = payload["status"]
            self._download_cancel_events.pop(name, None)
            if self._mm_status_var is not None:
                if status == "done":
                    self._mm_status_var.set(f"{name} downloaded.")
                elif status == "cancelled":
                    self._mm_status_var.set(f"Download of {name} cancelled.")
                else:
                    self._mm_status_var.set(f"Error downloading {name}: {payload.get('error')}")
            if status == "done":
                self._refresh_model_menu_labels()
            if self._model_manager_refresh is not None:
                self._model_manager_refresh()

    def _append_transcript(self, text: str) -> None:
        if not text:
            return
        self.transcript_box.insert("end", text if text.endswith("\n") else text + "\n")
        self.transcript_box.see("end")

    def _update_chunk_progress(self, payload: dict) -> None:
        """Show how far into the recording the transcription has got.

        Unlike the old single-shot pass, chunking gives a real "X of Y minutes
        transcribed" figure, which matters when a meeting takes a while.
        """
        label = payload.get("label", "Transcribing")
        done_sec = float(payload.get("done_sec", 0.0))
        total_sec = payload.get("total_sec")

        if not self.progress_bar.winfo_ismapped():
            self.progress_bar.grid()
        if self._progress_indeterminate:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self._progress_indeterminate = False

        if total_sec:
            self.progress_bar.set(min(1.0, done_sec / float(total_sec)))
            self._set_status(
                f"{label} — {format_timestamp(done_sec)} / "
                f"{format_timestamp(float(total_sec))} transcribed (saved)"
            )
        else:
            self._set_status(
                f"{label} — {format_timestamp(done_sec)} transcribed (saved)"
            )

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

    def _on_close(self) -> None:
        # The worker is a daemon thread and dies with the process; signalling it
        # first just gives the in-flight chunk a chance to checkpoint cleanly.
        self._cancel_event.set()
        self.destroy()


def main() -> None:
    app = MeetingTranscriberApp()
    app.mainloop()


if __name__ == "__main__":
    main()
