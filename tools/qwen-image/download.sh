#!/usr/bin/env bash
# Qwen-Image-2512 (Qwen, Apache-2.0, not gated) -> $WEIGHTS_DIR/Qwen-Image-2512 (~58 GB)
# https://huggingface.co/Qwen/Qwen-Image-2512
# plus its 4-step Lightning LoRA (lightx2v, Apache-2.0) -> $WEIGHTS_DIR/Qwen-Image-2512-Lightning
# https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning (only the bf16 LoRA, 0.85 GB; the repo holds
# >100 GB of fp8/int8 checkpoints for ComfyUI)
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch Qwen/Qwen-Image-2512 25468b98e3276ca6700de15c6628e51b7de54a26 \
  "$WEIGHTS_DIR/Qwen-Image-2512"
hf_fetch lightx2v/Qwen-Image-2512-Lightning 4609161e308ce82018a42c2a0c9b107e865bb751 \
  "$WEIGHTS_DIR/Qwen-Image-2512-Lightning" \
  --include "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
