# Repo configuration for host scripts (source this file, do not run it).
#
# Sets ROOT (repo directory) and MODELS_DIR, in this order: environment, MODELS_DIR in <repo>/.env,
# <repo>/models. A relative path is taken relative to the repo, like compose.yaml does.

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

if [[ -z ${MODELS_DIR:-} ]]; then MODELS_DIR=$(env_value MODELS_DIR); fi
MODELS_DIR=$(abs_path "${MODELS_DIR:-models}")
export MODELS_DIR
