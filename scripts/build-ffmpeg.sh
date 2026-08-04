#!/usr/bin/env bash
# Build the ffmpeg and ffprobe the app ships with.
#
# The app decodes every audio format it accepts by shelling out to ffmpeg (see
# src-tauri/src/chunking/decode.rs). Requiring the user to `brew install ffmpeg`
# first is the last thing standing between "download the app" and "use the app",
# so the binaries are bundled — which makes *which* ffmpeg a licensing question.
#
# Every ready-made static macOS build (evermeet, osxexperts, Homebrew) is GPL,
# because they all link x264 to encode video. This app decodes audio and encodes
# nothing but raw PCM, so none of that is needed: configured below with
# --disable-gpl --disable-nonfree --disable-version3 and an explicit list of the
# decoders the app's own AUDIO_EXTS imply, the result is LGPL v2.1 and about a
# tenth the size. --disable-autodetect keeps it that way by refusing to link
# whatever happens to be installed on the build machine.
#
# Run by `make setup` (via `make ffmpeg`); skips itself when the binaries are
# already built. The output is deliberately untracked (see .gitignore) — build
# artefacts of someone else's source do not belong in this repo's history.

set -euo pipefail

FFMPEG_VERSION="7.1.1"
FFMPEG_SHA256="733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
out_dir="$repo_dir/src-tauri/binaries"
work_dir="${FFMPEG_BUILD_DIR:-$repo_dir/.ffmpeg-build}"

# Tauri resolves an `externalBin` by appending the target triple, so the files
# have to be named for the platform they were built on.
triple="$(rustc -vV | awk '/^host:/ { print $2 }')"
if [ -z "$triple" ]; then
  echo "error: cannot determine the Rust host triple — is rustc on PATH?" >&2
  exit 1
fi

ffmpeg_out="$out_dir/ffmpeg-$triple"
ffprobe_out="$out_dir/ffprobe-$triple"

if [ "${FFMPEG_FORCE_REBUILD:-0}" != "1" ] && [ -x "$ffmpeg_out" ] && [ -x "$ffprobe_out" ]; then
  echo "ffmpeg $FFMPEG_VERSION already built for $triple — skipping."
  exit 0
fi

for tool in curl make clang shasum tar; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: '$tool' is required to build ffmpeg" >&2
    exit 1
  }
done

mkdir -p "$work_dir" "$out_dir"
tarball="$work_dir/ffmpeg-$FFMPEG_VERSION.tar.xz"
source_dir="$work_dir/ffmpeg-$FFMPEG_VERSION"

if [ ! -f "$tarball" ]; then
  echo "Downloading ffmpeg $FFMPEG_VERSION…"
  curl -fL --retry 3 -o "$tarball.part" \
    "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz"
  mv "$tarball.part" "$tarball"
fi

echo "$FFMPEG_SHA256  $tarball" | shasum -a 256 -c - >/dev/null || {
  echo "error: ffmpeg tarball failed its checksum — refusing to build it" >&2
  rm -f "$tarball"
  exit 1
}

rm -rf "$source_dir"
tar -xf "$tarball" -C "$work_dir"

# The app's floor, matching bundle.macOS.minimumSystemVersion.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"

cd "$source_dir"

# --disable-everything, then re-enable exactly what the app's AUDIO_EXTS need:
#   mp3          mp3 demuxer + mp3float decoder
#   m4a/mp4/aac  mov + aac demuxers, aac and alac decoders
#   flac         flac demuxer + decoder
#   ogg/opus     ogg demuxer, vorbis and opus decoders
#   wav          wav demuxer + the pcm decoders
#   wma          asf demuxer + wmav1/wmav2 decoders
# Output is always `-f s16le` on a pipe, so one encoder and one muxer cover it,
# plus the two filters ffmpeg auto-inserts to hit `-ar 16000 -ac 1`.
echo "Configuring ffmpeg $FFMPEG_VERSION (LGPL, audio decode only)…"
./configure \
  --disable-gpl \
  --disable-nonfree \
  --disable-version3 \
  --disable-everything \
  --disable-autodetect \
  --disable-network \
  --disable-doc \
  --disable-htmlpages \
  --disable-manpages \
  --disable-podpages \
  --disable-txtpages \
  --disable-avdevice \
  --disable-postproc \
  --disable-shared \
  --enable-static \
  --enable-swresample \
  --enable-avfilter \
  --disable-programs \
  --enable-ffmpeg \
  --enable-ffprobe \
  --enable-decoder=aac,aac_fixed,aac_latm,alac,flac,mp1,mp1float,mp2,mp2float,mp3,mp3float,opus,vorbis,wmav1,wmav2,wmalossless,wmapro,pcm_alaw,pcm_f32be,pcm_f32le,pcm_f64le,pcm_mulaw,pcm_s16be,pcm_s16le,pcm_s24be,pcm_s24le,pcm_s32le,pcm_s8,pcm_u8 \
  --enable-demuxer=aac,asf,flac,matroska,mov,mp3,ogg,pcm_s16le,w64,wav \
  --enable-parser=aac,aac_latm,flac,mpegaudio,opus,vorbis \
  --enable-protocol=file,pipe \
  --enable-encoder=pcm_s16le \
  --enable-muxer=pcm_s16le,wav \
  --enable-filter=aformat,anull,aresample,atrim \
  --prefix="$work_dir/install" >"$work_dir/configure.log" 2>&1 || {
  echo "error: ffmpeg configure failed — see $work_dir/configure.log" >&2
  tail -20 "$work_dir/configure.log" >&2
  exit 1
}

echo "Building ffmpeg (this takes a few minutes, once)…"
make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" ffmpeg ffprobe \
  >"$work_dir/build.log" 2>&1 || {
  echo "error: ffmpeg build failed — see $work_dir/build.log" >&2
  tail -20 "$work_dir/build.log" >&2
  exit 1
}

install -m 755 ffmpeg "$ffmpeg_out"
install -m 755 ffprobe "$ffprobe_out"

# Prove the build can do the one job it has, on the exact command line the app
# runs. A build missing a decoder or a muxer configures and links perfectly
# happily and only fails on the user's first real file, which is far too late.
#
# The probe is a hand-built 0.1 s 16 kHz mono WAV: 3200 bytes of silence behind
# a 44-byte header. Decoding it must yield those 3200 bytes back.
probe_wav="$work_dir/selftest.wav"
printf 'RIFF\xa4\x0c\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x80\x0c\x00\x00' >"$probe_wav"
head -c 3200 /dev/zero >>"$probe_wav"

decoded=$("$ffmpeg_out" -hide_banner -loglevel error -nostdin -i "$probe_wav" \
  -f s16le -ac 1 -acodec pcm_s16le -ar 16000 - | wc -c | tr -d ' ')
if [ "$decoded" != "3200" ]; then
  echo "error: the built ffmpeg decoded $decoded bytes, expected 3200 — the build is unusable" >&2
  exit 1
fi

duration=$("$ffprobe_out" -v error -show_entries format=duration -of default=nw=1:nk=1 "$probe_wav")
case "$duration" in
0.1*) ;;
*)
  echo "error: the built ffprobe reported duration '$duration', expected 0.1 — the build is unusable" >&2
  exit 1
  ;;
esac

echo "Built:"
ls -lh "$ffmpeg_out" "$ffprobe_out" | sed 's/^/  /'
echo "License: LGPL v2.1 (no GPL or nonfree components; see $source_dir/COPYING.LGPLv2.1)"
