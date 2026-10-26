# Repo configuration for host scripts (source this file, do not run it).
#
# Sets ROOT (repo directory) and INSTALL_DIR, in this order: environment, INSTALL_DIR in <repo>/.env,
# <repo>/installed. A relative path is taken relative to the repo, like compose.yaml does.
# INSTALL_DIR holds one directory per installed tool (<tool>/env: uv environment, <tool>/weights: model
# weights) and the shared uv cache (uv-cache/).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# env_value KEY: value of KEY in <repo>/.env (last occurrence, surrounding quotes removed), empty if unset
env_value() {
  [[ -f $ROOT/.env ]] || return 0
  local v
  v=$(sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$ROOT/.env" | tail -n 1)
  v=${v%"${v##*[![:space:]]}"}
  v=${v#[\"\']}; v=${v%[\"\']}
  printf '%s' "$v"
}

# abs_path PATH: PATH relative to the repo made absolute, ~ expanded
abs_path() {
  local p=$1
  [[ $p == "~"* ]] && p="$HOME${p:1}"
  p=${p#./}
  [[ $p == /* ]] || p="$ROOT/$p"
  printf '%s' "$p"
}

# tool_hash TOOL: fingerprint of what an installation of TOOL depends on: every file in tools/TOOL except the
# Python code (pyproject.toml, uv.lock, download.sh, ...). Must match tool_hash() in
# src/gb10_assets/server/runners.py, which compares it with <INSTALL_DIR>/<tool>/.installed.
tool_hash() {
  (cd "$ROOT/tools/$1" && find . -maxdepth 1 -type f ! -name '*.py' -printf '%P\n' | LC_ALL=C sort \
    | xargs -d '\n' sha256sum) | sha256sum | cut -c1-16
}

if [[ -z ${INSTALL_DIR:-} ]]; then INSTALL_DIR=$(env_value INSTALL_DIR); fi
INSTALL_DIR=$(abs_path "${INSTALL_DIR:-installed}")
export INSTALL_DIR
