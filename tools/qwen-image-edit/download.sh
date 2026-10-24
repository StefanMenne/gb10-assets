#!/usr/bin/env bash
# Qwen-Image-Edit-2511 (Qwen, Apache-2.0, not gated) -> $MODELS_DIR/qwen-image-edit/Qwen-Image-Edit-2511 (~58 GB)
# https://huggingface.co/Qwen/Qwen-Image-Edit-2511
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch Qwen/Qwen-Image-Edit-2511 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
  "$MODELS_DIR/qwen-image-edit/Qwen-Image-Edit-2511"
