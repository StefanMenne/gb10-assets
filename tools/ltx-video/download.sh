#!/usr/bin/env bash
# LTX-2.5 (Lightricks, LTX-2.x Community License, gated) -> $MODELS_DIR/ltx-video/LTX-2.5 (~50 GB)
# https://huggingface.co/Lightricks/LTX-2.5
# Only the files the NVFP4 distilled pipeline needs: the NVFP4 transformer, the Gemma 4 text encoder, both video VAEs,
# the audio VAE, the duration head and the two latent upscalers (the repo holds ~220 GB of bf16/int8 variants).
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch Lightricks/LTX-2.5 49b093a39c19169a8a81721dcc541e07e945232f "$MODELS_DIR/ltx-video/LTX-2.5" \
  --gated https://huggingface.co/Lightricks/LTX-2.5 \
  --include "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors" \
  --include "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --include "vae/ltx-2.5-video-vae-bf16.safetensors" \
  --include "vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --include "vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --include "model_patches/ltx-2.5-duration-head-bf16.safetensors" \
  --include "latent_upscale_models/*.safetensors"
