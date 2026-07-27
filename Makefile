.PHONY: setup build run-local run-docker-cli clean

IMAGE_NAME = meeting-transcriber

# Full setup: installs system deps (ffmpeg/tk), creates virtualenv,
# and installs Python dependencies. Safe to run on a fresh macOS or Linux machine.
setup:
	./setup.sh

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the GUI application locally (uses virtualenv if present)
run-local:
	@if [ -d ".venv" ]; then \
		.venv/bin/python app.py; \
	else \
		python app.py; \
	fi

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
