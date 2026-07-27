FROM python:3.10-slim

# Install system dependencies
# - ffmpeg: required for whisperx and handling mp3 files
# - python3-tk: required by tkinterdnd2 and customtkinter
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (ignoring files specified in .dockerignore)
COPY . .

# Default command to run the GUI app
# Note: Running GUI from Docker on macOS requires XQuartz and DISPLAY configuration.
# To run headless transcription instead, override the command:
# docker run -v $(pwd):/app/data meeting-transcriber python transcriber.py /app/data/file.mp3
CMD ["python", "app.py"]
