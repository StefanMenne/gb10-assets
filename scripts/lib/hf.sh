# Shared helpers for tools/*/download.sh (source this file, do not run it).
#
#   hf_fetch <repo> <revision> <dest> [--gated <license-url>] [hf download options ...]
#
# Downloads a pinned Hugging Face revision into <dest>. Idempotent: `hf download --local-dir` keeps
# metadata in <dest>/.cache and skips files that are already complete. Gated models need HF_TOKEN in the
# environment (never write it into the repo); the license link is printed so it can be accepted first.
set -euo pipefail

: "${MODELS_DIR:=./models}"
HF_HUB_VERSION=2.3.0

hf_fetch() {
  local repo=$1 rev=$2 dest=$3; shift 3
  if [[ ${1:-} == --gated ]]; then
    local license=$2; shift 2
    if [[ -z ${HF_TOKEN:-} ]]; then
      echo "ERROR: $repo is gated. Accept the license at $license" >&2
      echo "       then create a read token at https://huggingface.co/settings/tokens and run: HF_TOKEN=... $0" >&2
      return 1
    fi
    echo "[download] $repo is gated, license: $license"
  fi
  mkdir -p "$dest"
  echo "[download] $repo@${rev:0:7} -> $dest"
  uvx --from "huggingface_hub==$HF_HUB_VERSION" hf download "$repo" --revision "$rev" --local-dir "$dest" "$@" --quiet > /dev/null
  echo "[download] ok: $dest ($(du -sh "$dest" | cut -f1))"
}
