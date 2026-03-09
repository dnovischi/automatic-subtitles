#!/usr/bin/env bash
# setup.sh - Install all dependencies for the Video Subtitle Pipeline
set -euo pipefail

VENV_DIR=".venv"

echo "========================================"
echo "  Video Subtitle Pipeline - Setup"
echo "========================================"

# System packages
echo ""
echo "[1/3] Checking system dependencies ..."

if ! command -v ffmpeg &>/dev/null; then
    echo "  Installing ffmpeg ..."
    sudo apt-get update -qq
    sudo apt-get install -y ffmpeg
else
    echo "  ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
fi

# Virtual environment
echo ""
echo "[2/3] Setting up virtual environment ($VENV_DIR) ..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created $VENV_DIR"
else
    echo "  $VENV_DIR already exists - reusing"
fi

# Python packages
echo ""
echo "[3/3] Installing Python packages into $VENV_DIR ..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Usage:"
echo "  python automatic-subtitles.py <video> <text_file> <language>"
echo ""
echo "Examples:"
echo "  python automatic-subtitles.py lecture.mp4 lecture.txt ro"
echo "  python automatic-subtitles.py lecture.mp4 lecture.txt es --model large"
echo "  python automatic-subtitles.py lecture.mp4 lecture.txt fr --device cuda --keep-srt"
echo ""
echo "Run  python automatic-subtitles.py --help  for all options."
