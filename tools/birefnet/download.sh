#!/usr/bin/env bash
# BiRefNet-matting (MIT, not gated): BiRefNet trained for alpha matting, soft edges at hair, fur and motion blur,
# 0.9 GB (fp32). Same architecture and tensor names as the general BiRefNet that Comfy-Org repackaged for
# ComfyUI's native background removal nodes, so ComfyUI loads it as it is.
#   https://huggingface.co/ZhengPeng7/BiRefNet-matting -> $WEIGHTS_DIR/background_removal/model.safetensors
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch ZhengPeng7/BiRefNet-matting eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9 "$WEIGHTS_DIR/background_removal" \
  --include model.safetensors
