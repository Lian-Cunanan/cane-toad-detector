#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE="${1:-debug}"
APP_DIR="/mnt/c/Users/Carlo/Desktop/AI Components/mobile_app"
WORK_DIR="/tmp/cane_toad_mobile_build_$(date +%s)"

export PIP_BREAK_SYSTEM_PACKAGES=1

mkdir -p "$WORK_DIR/mobile_app"

(
  cd "$APP_DIR"
  tar --exclude='.venv' --exclude='.buildozer' --exclude='bin' --exclude='build' -cf - .
) | (
  cd "$WORK_DIR/mobile_app"
  tar -xf -
)

cd "$WORK_DIR/mobile_app"
yes | buildozer android "$BUILD_TYPE"

mkdir -p "$APP_DIR/bin"
cp -f "$WORK_DIR/mobile_app/bin/"*.apk "$APP_DIR/bin/"

echo "APK copied to $APP_DIR/bin"
