"""Transcribe a file the way the *Python* app does, for A/B against the Rust port.

Mirrors `app.TranscriberApp._prepare_engine()`'s CPU whisper path exactly:
the same `Transcriber`, the same language routing, the same `transcribe_chunked`
driver and chunk size. Only the engine differs from the Rust build, which is the
point — anything else differing would mean the port is wrong.

    .venv/bin/python desktop/scripts/ab_python.py <audio> [model] [lang] [chunk_sec]

`lang` is one of vi+en (default), en, auto. Pure `vi` uses PhoWhisper and is
compared separately.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chunking import DEFAULT_CHUNK_SECONDS, transcribe_chunked  # noqa: E402
from transcriber import FINAL_MODEL, Transcriber  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    source = Path(argv[0])
    model = argv[1] if len(argv) > 1 else FINAL_MODEL
    lang_mode = argv[2] if len(argv) > 2 else "vi+en"
    chunk_sec = float(argv[3]) if len(argv) > 3 else DEFAULT_CHUNK_SECONDS

    if lang_mode == "vi":
        print("`vi` mode uses PhoWhisper; compare that path separately.", file=sys.stderr)
        return 2

    # Exactly app.py's routing: vi+en decodes as Vietnamese on the *multilingual*
    # checkpoint, so English terms keep their English spelling.
    language = "vi" if lang_mode == "vi+en" else lang_mode
    transcriber = Transcriber(final_model=model, language=language)

    print(f"Loading '{model}' (language {transcriber.language})…", file=sys.stderr)
    transcriber.preload_final_model()

    engine_key = f"whisper-{model}:{transcriber.language or 'auto'}"
    print(f"engine_key = {engine_key}", file=sys.stderr)

    previews = 0

    def on_segment(_segment) -> None:
        nonlocal previews
        previews += 1

    def on_text(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_progress(_index, chunks_done, done_sec, total_sec) -> None:
        print(f"  [chunk {chunks_done}] {done_sec:.0f}s / {total_sec or 0:.0f}s", file=sys.stderr)

    started = time.monotonic()
    transcribe_chunked(
        source,
        transcribe_audio=transcriber.transcribe_audio,
        engine_key=engine_key,
        chunk_sec=chunk_sec,
        on_text=on_text,
        on_segment=on_segment,
        on_progress=on_progress,
    )
    print(
        f"done in {time.monotonic() - started:.1f}s, {previews} preview segments",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
