#!/usr/bin/env bash
# Build Oscilloom installer locally (for development/testing).
# Prerequisites: Python 3.11, Node.js 20+, Rust, Tauri CLI
#
# Usage: ./scripts/build-installer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Building Oscilloom installer..."

# Detect target triple
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  TARGET="aarch64-apple-darwin" ;;
  Darwin-x86_64) TARGET="x86_64-apple-darwin" ;;
  Linux-x86_64)  TARGET="x86_64-unknown-linux-gnu" ;;
  *)             echo "Unsupported platform"; exit 1 ;;
esac

echo "    Target: $TARGET"

# Step 1: Bundle Python backend
echo "==> Step 1/3: Building Python backend with PyInstaller..."
cd "$ROOT_DIR"
pip install pyinstaller -q
pyinstaller oscilloom.spec --noconfirm

# Copy sidecar with target triple name (Tauri convention)
cp "dist/oscilloom-server" "src-tauri/oscilloom-server-$TARGET"
chmod +x "src-tauri/oscilloom-server-$TARGET"

# Step 2: Build React frontend
echo "==> Step 2/3: Building React frontend..."
cd "$ROOT_DIR/frontend"
npm ci --silent
npm run build

# Step 3: Build Tauri app
echo "==> Step 3/3: Building Tauri desktop app..."
cd "$ROOT_DIR"
cargo tauri build --target "$TARGET"

echo ""
echo "==> Done! Installer is in src-tauri/target/$TARGET/release/bundle/"
echo "    Look for .dmg (macOS), .AppImage (Linux), or .msi (Windows)"
