#!/usr/bin/env bash
# Confucius4-TTS (NetEase Youdao, Apache-2.0, not gated) and the models it builds on, ~5.9 GB in total:
#   https://huggingface.co/netease-youdao/Confucius4-TTS -> $MODELS_DIR/confucius-tts/Confucius4-TTS (3.1 GB)
#   https://huggingface.co/facebook/w2v-bert-2.0 (MIT)    -> .../w2v-bert-2.0 (only the safetensors, 2.3 GB)
#   https://huggingface.co/funasr/campplus (Apache-2.0)   -> .../campplus (speaker encoder, 28 MB)
#   https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x (MIT) -> .../bigvgan_v2_22khz_80band_256x
#     (only the generator, 0.45 GB; the repo also holds 2.9 GB of discriminator/optimizer states)
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

dest=$MODELS_DIR/confucius-tts
hf_fetch netease-youdao/Confucius4-TTS ac9ca93f2db168f241bba1b57003401553c41f20 "$dest/Confucius4-TTS"
hf_fetch facebook/w2v-bert-2.0 da985ba0987f70aaeb84a80f2851cfac8c697a7b "$dest/w2v-bert-2.0" \
  --include config.json --include preprocessor_config.json --include model.safetensors
hf_fetch funasr/campplus e4b6ede7ce16997aff4ae69fbca1f0175e2afede "$dest/campplus" \
  --include campplus_cn_common.bin
hf_fetch nvidia/bigvgan_v2_22khz_80band_256x 633ff708ed5b74903e86ff1298cf4a98e921c513 \
  "$dest/bigvgan_v2_22khz_80band_256x" --include config.json --include bigvgan_generator.pt
