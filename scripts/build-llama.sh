#!/usr/bin/env bash
# Build the llama-cli the app ships with, for local-LLM transcript
# summarization (see src-tauri/src/summarize/).
#
# The summarizer shells out to a bundled llama.cpp `llama-cli`, the same
# sidecar pattern the ffmpeg/ffprobe decoders use (see build-ffmpeg.sh). Like
# whisper.cpp it runs on the Metal GPU and never sends anything off the
# machine, and because it is invoked as a separate process the app binary stays
# small and free of a second cmake-built ggml.
#
# Licensing is the one thing that is *easier* than ffmpeg: llama.cpp is MIT, so
# there is no GPL config dance and no need to audit linked components. The
# resulting binary is bundled unmodified under its own MIT license.
#
# Run by `make setup` (via `make llama`); skips itself when the binary is
# already built. The output is deliberately untracked (see .gitignore).

set -euo pipefail

LLAMA_VERSION="v0.2.0"
LLAMA_SHA256="72e6c3e70c584f84e61697e449ee388f43458d662ef8f3bd3f6b4a054c947958"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
out_dir="$repo_dir/src-tauri/binaries"
work_dir="${LLAMA_BUILD_DIR:-$repo_dir/.llama-build}"

# Tauri resolves an `externalBin` by appending the target triple, so the file
# has to be named for the platform it was built on.
triple="$(rustc -vV | awk '/^host:/ { print $2 }')"
if [ -z "$triple" ]; then
  echo "error: cannot determine the Rust host triple — is rustc on PATH?" >&2
  exit 1
fi

llama_out="$out_dir/llama-cli-$triple"

if [ "${LLAMA_FORCE_REBUILD:-0}" != "1" ] && [ -x "$llama_out" ]; then
  echo "llama-cli $LLAMA_VERSION already built for $triple — skipping."
  exit 0
fi

for tool in curl cmake make clang shasum tar; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: '$tool' is required to build llama.cpp" >&2
    exit 1
  }
done

mkdir -p "$work_dir" "$out_dir"
tarball="$work_dir/llama.cpp-$LLAMA_VERSION.tar.gz"
# GitHub archive tarballs strip the leading `v` from the tag when naming the
# extracted directory (`refs/tags/v0.2.0` -> `llama.cpp-0.2.0`).
source_dir="$work_dir/llama.cpp-${LLAMA_VERSION#v}"

if [ ! -f "$tarball" ]; then
  echo "Downloading llama.cpp $LLAMA_VERSION..."
  curl -fL --retry 3 -o "$tarball.part" \
    "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LLAMA_VERSION.tar.gz"
  mv "$tarball.part" "$tarball"
fi

echo "$LLAMA_SHA256  $tarball" | shasum -a 256 -c - >/dev/null || {
  echo "error: llama.cpp tarball failed its checksum — refusing to build it" >&2
  rm -f "$tarball"
  exit 1
}

rm -rf "$source_dir"
tar -xzf "$tarball" -C "$work_dir"

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"

echo "Configuring llama.cpp $LLAMA_VERSION (Metal, llama-cli only)…"
cmake -S "$source_dir" -B "$source_dir/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  >"$work_dir/configure.log" 2>&1 || {
  echo "error: llama.cpp configure failed — see $work_dir/configure.log" >&2
  tail -20 "$work_dir/configure.log" >&2
  exit 1
}

# LLAMA_BUILD_SERVER is ON not because we want the server, but because since
# v0.2.0 `llama-cli` is gated behind it in tools/CMakeLists.txt; we only build
# the llama-cli target below, never the llama-server binary.
echo "Building llama-cli (this takes a few minutes, once)…"
cmake --build "$source_dir/build" --target llama-cli -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" \
  >"$work_dir/build.log" 2>&1 || {
  echo "error: llama.cpp build failed — see $work_dir/build.log" >&2
  tail -20 "$work_dir/build.log" >&2
  exit 1
}

install -m 755 "$source_dir/build/bin/llama-cli" "$llama_out"

# Prove the binary is actually usable for its one job, on the exact command
# line the app runs. GGML_METAL_EMBED_LIBRARY=ON is what makes this pass inside
# a .app bundle; without it the binary silently falls back to CPU at runtime.
# With the flag set, the whole `.metal` shader source is embedded as a string,
# so `#include <metal_stdlib>` in the binary's strings is the marker for it.
# (`llama-cli --version` does not print the backend, and plain `ggml-metal.metal`
# also appears in compiler command lines, so neither is a reliable check.)
#
# Counted with `grep -c` rather than `grep -q`: `-q` closes the pipe the moment
# it matches, `strings` dies of SIGPIPE, and under `set -o pipefail` that makes
# the check fail even when the shaders are present.
metal_embed=$(strings "$llama_out" 2>/dev/null | grep -c '#include <metal_stdlib>' || true)
if [ "$metal_embed" -eq 0 ]; then
  echo "error: the built llama-cli has no embedded Metal shaders — the build is unusable" >&2
  exit 1
fi

echo "Built:"
ls -lh "$llama_out" | sed 's/^/  /'
echo "License: MIT (see $source_dir/LICENSE)"
