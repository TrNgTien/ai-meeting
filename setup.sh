#!/usr/bin/env bash
# One-shot setup: installs system dependencies + Python virtualenv + packages
# so a new machine can run the app immediately. Idempotent — re-running it only
# does the work that is actually missing.
set -euo pipefail

# Python versions this project is known to work on, best first.
SUPPORTED_VERSIONS=(3.10 3.11 3.12)

echo "==> Meeting Transcriber setup"

have() { command -v "$1" >/dev/null 2>&1; }

# Echoes the first supported python3.X on PATH, or nothing.
find_python() {
    local v
    for v in "${SUPPORTED_VERSIONS[@]}"; do
        if have "python$v"; then
            echo "python$v"
            return
        fi
    done
    # A bare python3 counts only if it is one of the supported versions.
    if have python3; then
        local bare
        bare="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        for v in "${SUPPORTED_VERSIONS[@]}"; do
            [ "$bare" = "$v" ] && { echo python3; return; }
        done
    fi
}

install_macos_deps() {
    if ! have brew; then
        cat >&2 <<'EOF'
Error: Homebrew is required but was not found.

Install it by pasting this into your terminal:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

On Apple Silicon, then add it to your PATH:

    eval "$(/opt/homebrew/bin/brew shellenv)"

Afterwards, run `make` again.
EOF
        exit 1
    fi

    if ! have ffmpeg; then
        echo "==> Installing ffmpeg (Homebrew)"
        brew install ffmpeg
    fi

    if [ -z "$(find_python)" ]; then
        echo "==> Installing Python ${SUPPORTED_VERSIONS[0]} (Homebrew)"
        brew install "python@${SUPPORTED_VERSIONS[0]}"
    fi

    # Tk ships separately from Homebrew's Python, and the app is a Tk GUI.
    local pybin ver
    pybin="$(find_python)"
    if [ -n "$pybin" ] && ! "$pybin" -c 'import tkinter' >/dev/null 2>&1; then
        ver="$("$pybin" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        echo "==> Installing Tk for Python $ver (Homebrew)"
        brew install "python-tk@$ver"
    fi
}

install_linux_deps() {
    if have apt-get; then
        local missing=()
        have ffmpeg || missing+=(ffmpeg)
        python3 -c 'import tkinter' >/dev/null 2>&1 || missing+=(python3-tk)
        python3 -c 'import venv' >/dev/null 2>&1 || missing+=(python3-venv)
        if [ ${#missing[@]} -gt 0 ]; then
            echo "==> Installing system packages via apt-get (${missing[*]})"
            sudo apt-get update
            sudo apt-get install -y "${missing[@]}"
        fi
    else
        echo "Warning: apt-get not found. Please manually install: ffmpeg, python3-tk, python3-venv" >&2
    fi
}

case "$(uname -s)" in
    Darwin) install_macos_deps ;;
    Linux)  install_linux_deps ;;
    *)
        cat >&2 <<EOF
Error: unsupported operating system ($(uname -s)).

This app runs on macOS and Ubuntu/Debian. On Windows, install WSL2:

    wsl --install      (in PowerShell, as Administrator, then reboot)

then open the Ubuntu terminal and run the setup there.
EOF
        exit 1
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$(find_python)}"
if [ -z "$PYTHON_BIN" ]; then
    echo "Error: no supported Python found (need one of: ${SUPPORTED_VERSIONS[*]})." >&2
    echo "       Install one, or point PYTHON_BIN at it:  PYTHON_BIN=/path/to/python3.10 ./setup.sh" >&2
    exit 1
fi
echo "==> Using $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"

if [ ! -x .venv/bin/python ]; then
    echo "==> Creating virtual environment (.venv)"
    "$PYTHON_BIN" -m venv .venv
fi

echo "==> Installing Python dependencies (a few minutes the first time)"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

cat <<'EOF'

==> Setup complete!

Start the app with:
    make

On first transcription the speech model is downloaded and cached (internet
required once); everything after that runs fully offline.
EOF
