.PHONY: start setup build run-local run-docker-cli transcribe clean

.DEFAULT_GOAL := start

IMAGE_NAME = meeting-transcriber
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

.PHONY: desktop-setup desktop-dev desktop-build desktop-test desktop-parity

# Frontend dependencies (React + Vite + the Tauri CLI). Rust deps come down on
# the first cargo build.
desktop-setup:
	cd $(DESKTOP) && pnpm install

desktop-dev: desktop-setup
	cd $(DESKTOP) && pnpm tauri dev

desktop-build: desktop-setup
	cd $(DESKTOP) && pnpm tauri build

desktop-test:
	cd $(DESKTOP)/src-tauri && cargo test

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
