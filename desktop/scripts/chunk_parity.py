"""Print the chunk boundaries the *Python* app would pick, for diffing.

Counterpart to `src-tauri/examples/chunk_parity.rs`. Run both on the same file
and diff: identical output means the Rust decode + split port is faithful.

    .venv/bin/python desktop/scripts/chunk_parity.py data/some.mp3 4
"""
import sys
from pathlib import Path

# The Python app lives at the repo root, two levels up from this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from chunking import (  # noqa: E402
    DEFAULT_CHUNK_SECONDS,
    MIN_TAIL_SECONDS,
    SAMPLE_RATE,
    SPLIT_SEARCH_SECONDS,
    audio_duration,
    decode_range,
    find_split_index,
)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: chunk_parity.py <audio> [max_chunks]")
        return 2
    source = Path(argv[0])
    max_chunks = int(argv[1]) if len(argv) > 1 else 4

    duration = audio_duration(source)
    print(f"duration_sec={duration:.3f}" if duration else "duration_sec=none")

    start = 0.0
    for index in range(max_chunks):
        if duration is not None and start >= duration - MIN_TAIL_SECONDS:
            break
        audio = decode_range(source, start, DEFAULT_CHUNK_SECONDS + SPLIT_SEARCH_SECONDS)
        if audio.size <= int(MIN_TAIL_SECONDS * SAMPLE_RATE):
            break

        nominal_end = int(DEFAULT_CHUNK_SECONDS * SAMPLE_RATE)
        if audio.size > nominal_end:
            cut = min(find_split_index(audio, nominal_end), audio.size)
        else:
            cut = audio.size
        nxt = start + cut / SAMPLE_RATE

        checksum = float(np.abs(audio[:cut]).astype(np.float64).sum())
        print(
            f"chunk={index} start={start:.3f} decoded={audio.size} "
            f"cut={cut} next={nxt:.3f} sum={checksum:.6f}"
        )
        start = nxt
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
