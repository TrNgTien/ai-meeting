.PHONY: setup ffmpeg dev build test transcribe release install

.DEFAULT_GOAL := dev

# The only command a new machine needs: `make`. Sets up whatever is missing
# (Rust, cmake, pnpm deps, bundled ffmpeg), then launches the app.
# Later runs skip straight to launching.

# rustup installs to ~/.cargo/bin, which a shell only picks up after sourcing
# ~/.cargo/env (or restarting). Prepending it here means every target below
# sees `cargo` immediately after setup installs it, even though each recipe
# runs in its own subshell.
export PATH := $(HOME)/.cargo/bin:$(PATH)

# Installs whatever is missing: the Rust toolchain, cmake (whisper-rs-sys
# needs it), the frontend dependencies, and the bundled ffmpeg binaries.
setup:
	@if ! command -v cargo >/dev/null 2>&1; then \
		echo "==> Rust (cargo) not found, installing via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		. "$$HOME/.cargo/env"; \
	fi
	@if ! command -v cmake >/dev/null 2>&1; then \
		echo "==> cmake not found, installing via Homebrew..."; \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "Error: Homebrew is required to install cmake but was not found."; \
			echo "Install it from https://brew.sh, then re-run make setup."; \
			exit 1; \
		fi; \
		brew install cmake; \
	fi
	pnpm install
	$(MAKE) ffmpeg
	@echo ""
	@echo "==> setup done. If this was the first install of Rust in this"
	@echo "    shell, restart your terminal (or run: . \"\$$HOME/.cargo/env\")"
	@echo "    before running make dev."

# The ffmpeg and ffprobe the .app bundles, built from source because every
# ready-made static macOS build is GPL (they link x264 to encode video, which
# this app never does). Audio decoders only: LGPL v2.1, ~3 MB each, and about
# ten minutes the first time. Skips itself once built.
ffmpeg:
	./scripts/build-ffmpeg.sh

dev: setup
	pnpm tauri dev

build: setup
	pnpm tauri build

test:
	cargo test --manifest-path src-tauri/Cargo.toml

# Headless transcription, no window. There is no Python left in this repo —
# this is the same engine the app uses.
# Usage: make transcribe FILE=meeting.m4a [MODEL=large-v3] [LANG_MODE=vi+en]
# (LANG_MODE, not LANG — make inherits LANG from the shell's locale settings.)
transcribe:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make transcribe FILE=your_audio_file.mp3 [MODEL=large-v3] [LANG_MODE=vi+en]"; \
		exit 1; \
	fi
	cargo run --release --manifest-path src-tauri/Cargo.toml --bin transcribe -- \
		"$(CURDIR)/$(FILE)" "$(or $(MODEL),large-v3)" "$(or $(LANG_MODE),vi+en)"

# Builds the .dmg and copies it to dist-release/Transcriber-<version>.dmg
# so it can be handed to someone else directly.
release: setup
	./scripts/build-release.sh

# Builds and installs straight to /Applications on this Mac, replacing
# whatever version is already there. Handy for trying a build without
# leaving the terminal.
install: release
	./scripts/install.sh "$$(ls -t dist-release/*.dmg | head -n1)"
