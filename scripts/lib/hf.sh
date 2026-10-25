# Shared helpers for tools/*/download.sh (source this file, do not run it).
#
#   hf_fetch <repo> <revision> <dest> [--gated <license-url>] [hf download options ...]
#
# Downloads a pinned Hugging Face revision into <dest>. Idempotent: `hf download --local-dir` keeps
# metadata in <dest>/.cache and skips files that are already complete. Gated models need HF_TOKEN in the
# environment or in <repo>/.env (never commit it); the license link is printed so it can be accepted first.
# Needs uv on the host (https://docs.astral.sh/uv/): `hf` runs via uvx in a throwaway environment.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"   # MODELS_DIR
HF_HUB_VERSION=2.3.0

hf_fetch() {
  local repo=$1 rev=$2 dest=$3; shift 3
  if [[ ${1:-} == --gated ]]; then
    local license=$2; shift 2
    # on the host the token may also live in .env (in the container compose passes it as environment)
    [[ -n ${HF_TOKEN:-} ]] || HF_TOKEN=$(env_value HF_TOKEN)
    [[ -z $HF_TOKEN ]] || export HF_TOKEN
    if [[ -z ${HF_TOKEN:-} ]]; then
      echo "ERROR: $repo is gated. Accept the license at $license" >&2
      echo "       then create a read token at https://huggingface.co/settings/tokens and run: HF_TOKEN=... $0" >&2
      return 1
    fi
    echo "[download] $repo is gated, license: $license"
  fi
  if ! command -v uvx > /dev/null; then
    echo "ERROR: uvx not found. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    return 1
  fi
  mkdir -p "$dest"
  echo "[download] $repo@${rev:0:7} -> $dest"
  # --format human: progress bars on stderr even when run by an agent (auto mode hides them then)
  uvx --from "huggingface_hub==$HF_HUB_VERSION" hf download "$repo" --revision "$rev" --local-dir "$dest" "$@" \
    --format human > /dev/null
  echo "[download] ok: $dest ($(du -sh "$dest" | cut -f1))"
}
