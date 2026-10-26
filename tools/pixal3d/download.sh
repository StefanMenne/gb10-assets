#!/usr/bin/env bash
# Pixal3D and the models its ComfyUI workflow uses, repackaged by Comfy-Org (all MIT, not gated), ~9 GB in
# ComfyUI's model folder layout -> $WEIGHTS_DIR/{clip_vision,diffusion_models,vae,background_removal,geometry_estimation}
#   https://huggingface.co/Comfy-Org/Pixal3D   the int8 Pixal3D DiT (5.6 GB), DINOv3 ViT-L image encoder (1.2 GB;
#                                              Meta's DINOv3 License applies to it), TRELLIS.2 shape/texture VAEs
#   https://huggingface.co/Comfy-Org/BiRefNet  background removal (0.4 GB)
#   https://huggingface.co/Comfy-Org/MoGe      MoGe-2, estimates the camera field of view of the image (0.6 GB)
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch Comfy-Org/Pixal3D f37641be376725d324c56626007cc477f1249a69 "$WEIGHTS_DIR" \
  --include clip_vision/dino_v3_L_naf_fp32.safetensors --include diffusion_models/pixal3d_int8_convrot.safetensors \
  --include vae/trellis_2_shape_vae_bf16.safetensors --include vae/trellis_2_texture_vae_bf16.safetensors
hf_fetch Comfy-Org/BiRefNet 25511f8787e51912e1480706b4e47b8f467fbf72 "$WEIGHTS_DIR" \
  --include background_removal/birefnet.safetensors
hf_fetch Comfy-Org/MoGe 1484985258ba7ef45c314144fcf0e90924ffbf44 "$WEIGHTS_DIR" \
  --include geometry_estimation/moge_2_vitl_normal_fp16.safetensors
