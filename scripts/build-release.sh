#!/usr/bin/env bash
# Builds the signed-or-not .dmg bundle and drops a version-stamped copy in
# dist-release/, so a build can be handed to someone else without them digging
# through src-tauri/target.
#
# Usage: scripts/build-release.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

pnpm tauri build

BUNDLE_DMG_DIR="src-tauri/target/release/bundle/dmg"
DMG=$(find "$BUNDLE_DMG_DIR" -maxdepth 1 -name '*.dmg' | head -n1)
if [ -z "$DMG" ]; then
  echo "error: no .dmg produced under $BUNDLE_DMG_DIR" >&2
  exit 1
fi

VERSION=$(node -p "require('./src-tauri/tauri.conf.json').version")
OUT_DIR="dist-release"
OUT="$OUT_DIR/Transcriber-$VERSION.dmg"
mkdir -p "$OUT_DIR"
cp "$DMG" "$OUT"

echo "==> Built $OUT"
echo "==> Hand this file (or run scripts/install.sh) to install/update on a Mac."
