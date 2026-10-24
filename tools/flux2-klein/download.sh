#!/usr/bin/env bash
# FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0, not gated) -> $MODELS_DIR/flux2-klein/FLUX.2-klein-4B (~16 GB)
# https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

# diffusers layout only: the single-file checkpoint in the repo root and the sample images are not needed
hf_fetch black-forest-labs/FLUX.2-klein-4B e7b7dc27f91deacad38e78976d1f2b499d76a294 \
  "$MODELS_DIR/flux2-klein/FLUX.2-klein-4B" --exclude "flux-2-klein-4b.safetensors" --exclude "*.jpg"
