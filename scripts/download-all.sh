#!/usr/bin/env bash
# Download model weights for all tools, or only the ones named:  scripts/download-all.sh [tool ...]
# Weights go to $MODELS_DIR/<tool>/ (default <repo>/models, where the server looks too). Gated models need
# HF_TOKEN. Needs uv on the host.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MODELS_DIR="${MODELS_DIR:-$ROOT/models}"

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
