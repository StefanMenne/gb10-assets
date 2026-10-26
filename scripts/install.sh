#!/usr/bin/env bash
# Install tools: their uv environment and model weights, into $INSTALL_DIR/<tool>/ (see scripts/lib/config.sh).
#
#   scripts/install.sh TOOL ...       install these tools, or bring them up to date (idempotent)
#   scripts/install.sh --all          all tools
#   scripts/install.sh --outdated     reinstall the installed tools whose definition changed (after an update)
#   scripts/install.sh --list         tools and their state
#
# Per tool: `uv sync` of tools/<tool> into <tool>/env, then tools/<tool>/post_install.sh if present (e.g.
# compiling CUDA kernels), then tools/<tool>/download.sh into <tool>/weights. Last, <tool>/.installed records
# the tool_hash the server checks before it starts a worker. All environments share the uv cache in
# $INSTALL_DIR/uv-cache and hardlink their files from it, so equal packages of several tools take space once.
# Needs uv (in the image; on the host: https://docs.astral.sh/uv/). Usually run via `scripts/docker.sh install`.
set -euo pipefail
source "$(dirname "$0")/lib/config.sh"
export UV_CACHE_DIR=$INSTALL_DIR/uv-cache UV_LINK_MODE=hardlink

all_tools() {
  local f; for f in "$ROOT"/tools/*/download.sh; do basename "$(dirname "$f")"; done
}

# state TOOL: installed, outdated, incomplete (a failed or running installation) or "not installed"
state() {
  local dir=$INSTALL_DIR/$1
  if [[ -f $dir/.installed ]]; then
    [[ $(cat "$dir/.installed") == "$(tool_hash "$1")" ]] && echo installed || echo outdated
  elif [[ -d $dir ]]; then echo incomplete
  else echo "not installed"; fi
}

install_tool() {
  local t=$1 dir=$INSTALL_DIR/$1 src=$ROOT/tools/$1
  [[ -x $src/download.sh ]] || { echo "ERROR: unknown tool $t (no tools/$t/download.sh)" >&2; return 1; }
  mkdir -p "$dir"
  # the server holds a shared lock while a worker of the tool runs, and starts none while this one is held
  exec 9> "$dir/.lock"
  flock -n 9 || { echo "[install] $t: waiting for its running worker to exit"; flock 9; }
  rm -f "$dir/.installed"
  echo "[install] $t: environment -> $dir/env"
  UV_PROJECT_ENVIRONMENT=$dir/env uv sync --frozen --no-dev --project "$src"
  if [[ -x $src/post_install.sh ]]; then
    echo "[install] $t: post_install.sh"
    "$src/post_install.sh" "$dir/env/bin/python" "$dir/build"
  fi
  echo "[install] $t: weights -> $dir/weights"
  WEIGHTS_DIR=$dir/weights "$src/download.sh"
  tool_hash "$t" > "$dir/.installed"
  exec 9>&-
  echo "[install] $t: installed ($(du -sh "$dir" | cut -f1))"
}

case ${1:-} in
  --list)
    for t in $(all_tools); do printf '%-18s %s\n' "$t" "$(state "$t")"; done
    exit ;;
  --all) mapfile -t tools < <(all_tools) ;;
  --outdated)
    tools=(); for t in $(all_tools); do [[ $(state "$t") != outdated ]] || tools+=("$t"); done
    [[ ${#tools[@]} -gt 0 ]] || { echo "all installed tools are up to date"; exit; } ;;
  "" | -*) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; [[ -n ${1:-} ]] || exit 0; exit 1 ;;
  *) tools=("$@") ;;
esac

failed=()
for t in "${tools[@]}"; do
  echo "=== $t"
  install_tool "$t" || failed+=("$t")
done
if [[ ${#failed[@]} -gt 0 ]]; then echo "FAILED: ${failed[*]}" >&2; exit 1; fi
echo "installed in $INSTALL_DIR: ${tools[*]}"
