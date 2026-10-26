#!/usr/bin/env bash
# Qwen-Image-Edit-2511 (Qwen, Apache-2.0, not gated) -> $WEIGHTS_DIR/Qwen-Image-Edit-2511 (~58 GB)
# https://huggingface.co/Qwen/Qwen-Image-Edit-2511
# plus its 4-step Lightning LoRA (lightx2v, Apache-2.0) -> $WEIGHTS_DIR/Qwen-Image-Edit-2511-Lightning
# https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning (only the bf16 LoRA, 0.85 GB; the repo holds
# >100 GB of fp8 checkpoints for ComfyUI)
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch Qwen/Qwen-Image-Edit-2511 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
  "$WEIGHTS_DIR/Qwen-Image-Edit-2511"
hf_fetch lightx2v/Qwen-Image-Edit-2511-Lightning 107d238d68b8387e32843d0b4a2e24ceb0d3ad86 \
  "$WEIGHTS_DIR/Qwen-Image-Edit-2511-Lightning" \
  --include "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
