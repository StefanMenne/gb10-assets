#!/usr/bin/env bash
# LTX-2.5 (Lightricks, LTX-2.x Community License, gated) -> $WEIGHTS_DIR/LTX-2.5 (~51 GB)
# https://huggingface.co/Lightricks/LTX-2.5
# Only the files the NVFP4 distilled pipeline needs: the NVFP4 transformer, the Gemma 4 text encoder, both video VAEs,
# the audio VAE, the duration head and the two latent upscalers (the repo holds ~220 GB of bf16/int8 variants).
# Plus the Union Control IC-LoRA (depth/canny/pose control videos, ~1 GB): trained on LTX-2.3, Lightricks' own
# LTX-2.5 workflow uses it as it is.
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch Lightricks/LTX-2.5 49b093a39c19169a8a81721dcc541e07e945232f "$WEIGHTS_DIR/LTX-2.5" \
  --gated https://huggingface.co/Lightricks/LTX-2.5 \
  --include "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors" \
  --include "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --include "vae/ltx-2.5-video-vae-bf16.safetensors" \
  --include "vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --include "vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --include "model_patches/ltx-2.5-duration-head-bf16.safetensors" \
  --include "latent_upscale_models/*.safetensors"

hf_fetch Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control 20a580c8606c06a0988d40375a61309e0fe5893d "$WEIGHTS_DIR/LTX-2.5/loras" \
  --include "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
