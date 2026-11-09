#!/usr/bin/env bash
# BootsTAPIR (Google DeepMind TAPNet, Apache-2.0 like the code, not gated): the offline TAPIR point tracker
# fine-tuned with BootsTAP, PyTorch weights, 0.1 GB. Trained at 256x256 and 512x512.
#   https://huggingface.co/google/tapnet -> $WEIGHTS_DIR/bootstapir_checkpoint_v2.pt
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib/hf.sh"

hf_fetch google/tapnet 5d3fb48e76c5422841e38501514121e251beabb7 "$WEIGHTS_DIR" \
  --include bootstapir_checkpoint_v2.pt
