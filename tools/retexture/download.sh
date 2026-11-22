#!/usr/bin/env bash
# The models of the retexturing workflows (all Apache-2.0, not gated), ~69 GB, in ComfyUI's model folder layout
# -> $WEIGHTS_DIR/{diffusion_models,text_encoders,vae,loras,controlnet}
#   https://huggingface.co/Comfy-Org/Qwen-Image       Qwen-Image-2512 repackaged for ComfyUI: the bf16 DiT (41 GB),
#                                                     the Qwen2.5-VL 7B text encoder (17 GB) and the VAE
#   https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning   its 4-step Lightning LoRA (0.85 GB)
#   https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union   ControlNet for depth, canny, ... and
#                                                     gray (version 2602, 3.5 GB)
#   https://huggingface.co/Comfy-Org/SeedVR2          SeedVR2 3B upscaler and its VAE (7.3 GB)
# The DiT and text encoder hold the same tensors as the diffusers shards of tools/qwen-image (names, shapes, dtypes
# and bytes, at its pinned revision): when qwen-image is installed they are joined from its files instead of
# downloading 58 GB again (merge_shards.py).
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

QWEN=$INSTALL_DIR/qwen-image/weights/Qwen-Image-2512
PY=$(dirname "$WEIGHTS_DIR")/env/bin/python
mkdir -p "$WEIGHTS_DIR/diffusion_models" "$WEIGHTS_DIR/text_encoders" "$WEIGHTS_DIR/vae"
parts=(--include split_files/vae/qwen_image_vae.safetensors)
for f in diffusion_models/qwen_image_2512_bf16:transformer text_encoders/qwen_2.5_vl_7b:text_encoder; do
  file=$WEIGHTS_DIR/${f%%:*}.safetensors
  shards=("$QWEN/${f##*:}"/*.safetensors)
  if [[ -f $file || -L $file ]]; then continue
  elif [[ -f $INSTALL_DIR/qwen-image/.installed && -f ${shards[0]} ]]; then
    "$PY" "$(dirname "$0")/merge_shards.py" "$file" "${shards[@]}"
  else
    parts+=(--include "split_files/${f%%:*}.safetensors")
  fi
done
hf_fetch Comfy-Org/Qwen-Image 1f12b17be14c89b026c51a91d67c32f84bb047bc "$WEIGHTS_DIR/repo/Qwen-Image" "${parts[@]}"
hf_fetch lightx2v/Qwen-Image-2512-Lightning 4609161e308ce82018a42c2a0c9b107e865bb751 "$WEIGHTS_DIR/loras" \
  --include Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors
hf_fetch alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union c21dcc372f9f52a55a6d4984cb56a0f0ac0807bc \
  "$WEIGHTS_DIR/controlnet" --include Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors
hf_fetch Comfy-Org/SeedVR2 df48879708206a403d2a61acd55578c2e80fd233 "$WEIGHTS_DIR" \
  --include diffusion_models/seedvr2_3b_fp16.safetensors --include vae/seedvr2_ema_vae_fp16.safetensors

# the repo keeps its files under split_files/<folder>/: link the downloaded ones into ComfyUI's folders
for f in diffusion_models/qwen_image_2512_bf16 text_encoders/qwen_2.5_vl_7b vae/qwen_image_vae; do
  if [[ -f $WEIGHTS_DIR/repo/Qwen-Image/split_files/$f.safetensors && ! -f $WEIGHTS_DIR/$f.safetensors ]]; then
    ln -sfn "../repo/Qwen-Image/split_files/$f.safetensors" "$WEIGHTS_DIR/$f.safetensors"
  fi
done
