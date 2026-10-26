#!/usr/bin/env bash
# Fetches ComfyUI (GPL-3.0) at a pinned revision into BUILD_DIR/ComfyUI; the worker (run.py) starts it as a
# separate headless process and talks to its HTTP API. Its Python dependencies are in pyproject.toml.
#
#   tools/pixal3d/post_install.sh PYTHON BUILD_DIR
set -euo pipefail

COMFYUI_REV=63be2f9b82ad0dac88a7ceaa28a9cf3154125a18   # v0.42.1; keep pyproject.toml in sync with its requirements.txt
dest=$2/ComfyUI

if [[ $(cat "$dest/.rev" 2>/dev/null) != "$COMFYUI_REV" ]]; then
  rm -rf "$dest" "$dest.tmp"
  git init -q "$dest.tmp"
  git -C "$dest.tmp" fetch -q --depth 1 https://github.com/Comfy-Org/ComfyUI "$COMFYUI_REV"
  git -C "$dest.tmp" checkout -q FETCH_HEAD
  rm -rf "$dest.tmp/.git"
  echo "$COMFYUI_REV" > "$dest.tmp/.rev"
  mv "$dest.tmp" "$dest"
fi
echo "ComfyUI ${COMFYUI_REV:0:12} in $dest"
