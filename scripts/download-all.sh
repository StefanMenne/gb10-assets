#!/usr/bin/env bash
# Download model weights for all tools, or only the ones named:  scripts/download-all.sh [tool ...]
# Weights go to $MODELS_DIR/<tool>/: environment, else MODELS_DIR in <repo>/.env, else <repo>/models (where the
# server looks too). Gated models need HF_TOKEN. Needs uv on the host, or use `scripts/docker.sh download`.
set -euo pipefail
source "$(dirname "$0")/lib/config.sh"

if [[ $# -gt 0 ]]; then tools=("$@"); else
  tools=(); for f in "$ROOT"/tools/*/download.sh; do tools+=("$(basename "$(dirname "$f")")"); done
fi
failed=()
for t in "${tools[@]}"; do
  script="$ROOT/tools/$t/download.sh"
  [[ -x $script ]] || { echo "ERROR: unknown tool $t (no tools/$t/download.sh)" >&2; failed+=("$t"); continue; }
  echo "=== $t"
  "$script" || failed+=("$t")
done
if [[ ${#failed[@]} -gt 0 ]]; then echo "FAILED: ${failed[*]}" >&2; exit 1; fi
echo "all downloads complete: $MODELS_DIR"
