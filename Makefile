.PHONY: start setup build run-local run-docker-cli transcribe clean

.DEFAULT_GOAL := start

IMAGE_NAME = transcriber
VENV = .venv
PY = $(VENV)/bin/python
STAMP = $(VENV)/.deps-installed

# The only command a new machine needs: `make`.
# Installs whatever is missing (system libs, virtualenv, Python deps) the first
# time, then launches the app. Later runs skip straight to launching.
start: $(STAMP)
	$(PY) app.py

# Re-runs setup.sh whenever it or requirements.txt is newer than the stamp,
# so changed dependencies are picked up without anyone remembering to re-setup.
$(STAMP): requirements.txt setup.sh
	./setup.sh
	@touch $(STAMP)

# Force a full setup pass (system deps, virtualenv, Python deps) without launching.
setup:
	./setup.sh
	@touch $(STAMP)

# Launch the app, assuming setup already ran.
run-local: $(STAMP)
	$(PY) app.py

# Headless transcription without the GUI.
# Usage: make transcribe FILE=meeting.m4a [LANG_MODE=vi]
# (LANG_MODE, not LANG — make inherits LANG from the shell's locale settings.)
transcribe: $(STAMP)
	@if [ -z "$(FILE)" ]; then \
		echo "Error: You must provide a FILE to transcribe."; \
		echo "Usage: make transcribe FILE=your_audio_file.mp3 [LANG_MODE=vi]"; \
		exit 1; \
	fi
	$(PY) transcriber.py "$(FILE)" $(LANG_MODE)

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the CLI transcriber headlessly via Docker on a specific file
# Usage: make run-docker-cli FILE=viet-voice.mp3
run-docker-cli:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: You must provide a FILE to transcribe."; \
		echo "Usage: make run-docker-cli FILE=your_audio_file.mp3"; \
		exit 1; \
	fi
	docker run --rm -it \
		-v "$$(pwd):/app/data" \
		$(IMAGE_NAME) python transcriber.py "/app/data/$(FILE)"

clean:
	docker rmi $(IMAGE_NAME) || true

# --- Rust + Tauri port (desktop/) ---------------------------------------------
# Additive: every target above keeps working unchanged, so the Python app stays
# runnable while the port catches up and the two can be compared on real audio.

DESKTOP = desktop

# rustup installs to ~/.cargo/bin, which a shell only picks up after sourcing
# ~/.cargo/env (or restarting). Prepending it here means desktop-dev/-build/
# -test/-parity see `cargo` immediately after desktop-setup installs it, even
# though each target's recipe runs in its own subshell.
export PATH := $(HOME)/.cargo/bin:$(PATH)

.PHONY: desktop-setup desktop-dev desktop-build desktop-test desktop-parity desktop-release desktop-install

# Frontend dependencies (React + Vite + the Tauri CLI), plus the Rust toolchain
# and cmake that tauri/whisper-rs-sys need to build. Installs whatever is
# missing so a fresh machine can go straight to desktop-dev. Rust crate deps
# come down on the first cargo build.
desktop-setup:
	@if ! command -v cargo >/dev/null 2>&1; then \
		echo "==> Rust (cargo) not found, installing via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		. "$$HOME/.cargo/env"; \
	fi
	@if ! command -v cmake >/dev/null 2>&1; then \
		echo "==> cmake not found, installing via Homebrew..."; \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "Error: Homebrew is required to install cmake but was not found."; \
			echo "Install it from https://brew.sh, then re-run make desktop-setup."; \
			exit 1; \
		fi; \
		brew install cmake; \
	fi
	cd $(DESKTOP) && pnpm install
	@echo ""
	@echo "==> desktop-setup done. If this was the first install of Rust in this"
	@echo "    shell, restart your terminal (or run: . \"\$$HOME/.cargo/env\") before"
	@echo "    running make desktop-dev."

desktop-dev: export MACOSX_DEPLOYMENT_TARGET ?= 14.0
desktop-dev: desktop-setup
	cd $(DESKTOP) && pnpm tauri dev

desktop-build: desktop-setup
	cd $(DESKTOP) && pnpm tauri build

desktop-test:
	cd $(DESKTOP)/src-tauri && cargo test

# Builds the .dmg and copies it to desktop/dist-release/MeetingTranscriber-<version>.dmg
# so it can be handed to someone else directly.
desktop-release: desktop-setup
	cd $(DESKTOP) && ./scripts/build-release.sh

# Builds and installs straight to /Applications on this Mac, replacing
# whatever version is already there. Handy for trying a build without
# leaving the terminal.
desktop-install: desktop-release
	cd $(DESKTOP) && ./scripts/install.sh "$$(ls -t dist-release/*.dmg | head -n1)"

# Prove the Rust chunk decode/split is faithful to the Python app's: same
# boundaries and same sample checksums on the same file, or the A/B transcript
# comparison would be measuring the decoder instead of the model.
# Usage: make desktop-parity FILE=data/meeting.mp3
desktop-parity: $(STAMP)
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make desktop-parity FILE=your_audio_file.mp3"; \
		exit 1; \
	fi
	@cd $(DESKTOP)/src-tauri && cargo run --quiet --example chunk_parity -- "$(CURDIR)/$(FILE)" 4 > "$(CURDIR)/.parity-rust.txt"
	@$(PY) $(DESKTOP)/scripts/chunk_parity.py "$(FILE)" 4 > "$(CURDIR)/.parity-python.txt"
	@diff -u .parity-python.txt .parity-rust.txt \
		&& echo "==> chunk boundaries and sample checksums match"
	@rm -f .parity-rust.txt .parity-python.txt
