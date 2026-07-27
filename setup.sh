#!/usr/bin/env bash
# One-shot setup: installs system dependencies + Python virtualenv + packages
# so a new machine can run `make run-local` (or `python app.py`) immediately.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

echo "==> Meeting Transcriber setup"

install_macos_deps() {
    if ! command -v brew >/dev/null 2>&1; then
        echo "Error: Homebrew not found. Install it from https://brew.sh then re-run this script." >&2
        exit 1
    fi
    echo "==> Installing system packages via Homebrew (ffmpeg, python-tk@3.10)"
    brew install ffmpeg python-tk@3.10
}

install_linux_deps() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "==> Installing system packages via apt-get (ffmpeg, python3-tk)"
        sudo apt-get update
        sudo apt-get install -y ffmpeg python3-tk python3-venv
    else
        echo "Warning: apt-get not found. Please manually install: ffmpeg, python3-tk, python3-venv" >&2
    fi
}

case "$(uname -s)" in
    Darwin)
        install_macos_deps
        ;;
    Linux)
        install_linux_deps
        ;;
    *)
        echo "Warning: unsupported OS $(uname -s). Skipping system package install; ensure ffmpeg and tk are available." >&2
        ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Warning: $PYTHON_BIN not found, falling back to python3" >&2
    PYTHON_BIN="python3"
fi

echo "==> Creating virtual environment (.venv) with $PYTHON_BIN"
"$PYTHON_BIN" -m venv .venv

echo "==> Installing Python dependencies"
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

cat <<'EOF'

==> Setup complete!

Run the app with:
    make run-local
or:
    source .venv/bin/activate && python app.py

On first run, models are downloaded and cached automatically (internet required once).
EOF
