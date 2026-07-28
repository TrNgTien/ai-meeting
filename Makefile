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
