#!/usr/bin/env bash
# One-shot installer for Transcriber.app: quits any running copy,
# removes whatever is already in /Applications, mounts the .dmg, copies the
# new .app in, clears the quarantine flag (the build isn't notarized, so
# Gatekeeper would otherwise call it "damaged"), and launches it.
#
# Usage:
#   ./install.sh [/path/to/Transcriber-X.Y.Z.dmg]
#
# With no argument, it looks for a .dmg next to this script, then in
# ~/Downloads (newest first).
set -euo pipefail

APP_NAME="Transcriber"
DEST="/Applications/$APP_NAME.app"

DMG="${1:-}"

if [ -z "$DMG" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  DMG=$(find "$SCRIPT_DIR" -maxdepth 1 -name '*.dmg' | head -n1)
fi

if [ -z "$DMG" ]; then
  DMG=$(find "$HOME/Downloads" -maxdepth 1 -iname 'Transcriber*.dmg' -print0 \
    | xargs -0 ls -t 2>/dev/null | head -n1 || true)
fi

if [ -z "$DMG" ] || [ ! -f "$DMG" ]; then
  echo "error: no .dmg found. Usage: $0 /path/to/Transcriber-X.Y.Z.dmg" >&2
  exit 1
fi

echo "==> Installing from $DMG"

echo "==> Quitting any running copy of $APP_NAME"
osascript -e "tell application \"$APP_NAME\" to quit" >/dev/null 2>&1 || true
sleep 1
pkill -f "/Applications/$APP_NAME.app" >/dev/null 2>&1 || true

if [ -d "$DEST" ]; then
  echo "==> Removing existing install at $DEST"
  rm -rf "$DEST"
fi

MOUNT_POINT=$(mktemp -d /tmp/transcriber-install.XXXXXX)
cleanup() { hdiutil detach "$MOUNT_POINT" -quiet >/dev/null 2>&1 || true; rmdir "$MOUNT_POINT" 2>/dev/null || true; }
trap cleanup EXIT

echo "==> Mounting installer image"
hdiutil attach "$DMG" -nobrowse -mountpoint "$MOUNT_POINT" -quiet

SRC_APP=$(find "$MOUNT_POINT" -maxdepth 1 -name '*.app' | head -n1)
if [ -z "$SRC_APP" ]; then
  echo "error: no .app found inside $DMG" >&2
  exit 1
fi

echo "==> Copying to /Applications"
ditto "$SRC_APP" "$DEST"

echo "==> Clearing quarantine flag (unsigned build)"
xattr -cr "$DEST" 2>/dev/null || true

echo "==> Installed $APP_NAME. Launching..."
open "$DEST"
