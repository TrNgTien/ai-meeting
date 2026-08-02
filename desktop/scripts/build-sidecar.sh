#!/usr/bin/env bash
# Builds sidecar.py (the persistent JSON-protocol transcription worker the
# Tauri app talks to) into a self-contained onedir executable via PyInstaller,
# so the bundled .app never depends on the repo's .venv or a system Python.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DESKTOP="$REPO_ROOT/desktop"
DIST_DIR="$DESKTOP/src-tauri/sidecar-dist"
PY="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "error: $PY not found — run 'make setup' at the repo root first" >&2
  exit 1
fi

# torch and transformers already ship curated PyInstaller hooks
# (_pyinstaller_hooks_contrib/stdhooks/hook-{torch,transformers}.py) that pull
# in exactly the compiled libs/data those packages need at runtime — a blanket
# --collect-all on them instead drags in ~hundreds of MB of unrelated test
# suites, C++ headers (torch/include), and the ONNX/distributed/inductor
# subsystems we never touch, roughly doubling bundle size for nothing.
#
# hook-torch.py's own hiddenimports is `collect_submodules("torch")` — every
# torch submodule, unconditionally — so torch.fx/_dynamo/_inductor pull in
# sympy (78MB) and networkx (18MB) even though openai-whisper's own code
# (checked: no `torch.jit`/`torch.compile`/`torch.onnx`/`torch.distributed`/
# `torch.ao` references anywhere in the installed `whisper` package) never
# touches compile/export/distributed/quantization. The --exclude-module
# torch.* entries below claw those back; PyInstaller resolves excludes after
# hidden-import merging, so they win over the hook's blanket collection.
#
# `vi`-mode (PhoWhisper via WhisperX/CTranslate2, plus its own use of
# transformers/tokenizers/safetensors — nothing else in this bundle needs
# them) is deliberately left out: sidecar.py's `_resolve_engine` wraps that
# path in a broad `except Exception` and falls back to the plain CPU whisper
# engine on any failure (see sidecar.py:170-177), so excluding it here just
# means `vi` mode always takes that fallback in the packaged app instead of
# failing to import — same degrade-gracefully behavior the code already has
# for a missing/broken PhoWhisper install. This drops ~400MB+ of transitive
# deps (transformers, tokenizers, pandas, torchvision, torchcodec,
# pytorch-lightning, torchmetrics, nltk, ...) that only PhoWhisper ever
# touches; `phowhisper.py` itself still gets bundled (transcriber.py imports
# it with a static `from phowhisper import ...`, which PyInstaller's analysis
# does see), it just can't succeed past its own `import ctranslate2`.
"$PY" -m PyInstaller \
  --name sidecar \
  --onedir \
  --noconfirm \
  --distpath "$DIST_DIR" \
  --workpath "$DESKTOP/src-tauri/sidecar-build" \
  --specpath "$DESKTOP/src-tauri/sidecar-build" \
  --paths "$REPO_ROOT" \
  --collect-all whisper \
  --collect-all mlx_whisper \
  --collect-all mlx \
  --collect-all tiktoken \
  --exclude-module matplotlib \
  --exclude-module IPython \
  --exclude-module notebook \
  --exclude-module tensorboard \
  --exclude-module onnx \
  --exclude-module onnxruntime \
  --exclude-module gradio \
  --exclude-module streamlit \
  --exclude-module whisperx \
  --exclude-module faster_whisper \
  --exclude-module ctranslate2 \
  --exclude-module pyannote \
  --exclude-module pytorch_lightning \
  --exclude-module lightning \
  --exclude-module torchmetrics \
  --exclude-module torchvision \
  --exclude-module torchcodec \
  --exclude-module pandas \
  --exclude-module nltk \
  --exclude-module omegaconf \
  --exclude-module asteroid_filterbanks \
  --exclude-module speechbrain \
  --exclude-module transformers \
  --exclude-module tokenizers \
  --exclude-module safetensors \
  --exclude-module sklearn \
  --exclude-module grpc \
  --exclude-module sqlalchemy \
  --exclude-module alembic \
  --exclude-module optuna \
  --exclude-module opentelemetry \
  --exclude-module torch.onnx \
  --exclude-module torch.distributed \
  --exclude-module torch._dynamo \
  --exclude-module torch._inductor \
  --exclude-module torch.ao \
  --exclude-module torch.testing \
  --exclude-module torch.compiler \
  --exclude-module torch.func \
  --exclude-module torch.masked \
  --exclude-module torch.sparse \
  --exclude-module torch.distributions \
  --hidden-import mlx_engine \
  --hidden-import chunking \
  --hidden-import transcriber \
  "$REPO_ROOT/sidecar.py"

echo "sidecar bundle: $DIST_DIR/sidecar/sidecar"
