#!/usr/bin/env bash
# Qwen3-TTS 1.7B (Qwen, Apache-2.0, not gated), both with their 12 Hz speech tokenizer, ~4.5 GB each:
#   VoiceDesign (voice from a description) -> $MODELS_DIR/qwen-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign
#   https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
#   Base (voice cloning)                   -> $MODELS_DIR/qwen-tts/Qwen3-TTS-12Hz-1.7B-Base
#   https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign 5ecdb67327fd37bb2e042aab12ff7391903235d3 \
  "$MODELS_DIR/qwen-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
hf_fetch Qwen/Qwen3-TTS-12Hz-1.7B-Base fd4b254389122332181a7c3db7f27e918eec64e3 \
  "$MODELS_DIR/qwen-tts/Qwen3-TTS-12Hz-1.7B-Base"
