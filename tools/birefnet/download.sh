#!/usr/bin/env bash
# BiRefNet (MIT, not gated), repackaged by Comfy-Org for ComfyUI's native background removal nodes, 0.4 GB in
# ComfyUI's model folder layout -> $WEIGHTS_DIR/background_removal (the same file as tools/pixal3d uses)
#   https://huggingface.co/Comfy-Org/BiRefNet
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch Comfy-Org/BiRefNet 25511f8787e51912e1480706b4e47b8f467fbf72 "$WEIGHTS_DIR" \
  --include background_removal/birefnet.safetensors
