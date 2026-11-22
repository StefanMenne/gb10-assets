#!/usr/bin/env bash
# Docker helper for the gb10-assets server. Settings come from .env (see .env.example).
#
#   scripts/docker.sh init                create .env with a random token, create the installed/jobs directories
#   scripts/docker.sh init_without_token  the same, but without authentication (GB10_ASSETS_TOKEN=none)
#   scripts/docker.sh build               build the image (tags :<GB10_ASSETS_TAG> and :<version>)
#   scripts/docker.sh up | down | logs    start in the background, stop, follow the server log
#   scripts/docker.sh install TOOL ...    install tools (environment + weights) with the image, --all: every tool
#   scripts/docker.sh update              reinstall the installed tools that changed (after build)
#   scripts/docker.sh uninstall TOOL ...  delete a tool's environment and weights
#   scripts/docker.sh status              container state, server version and queue, installed tools (alias: list)
#   scripts/docker.sh install-shutdown    let the dashboard shut down the host (systemd units, sudo)
#   scripts/docker.sh uninstall-shutdown  remove them again
#   scripts/docker.sh power-on [shelly HOST[/SWITCH] | url URL | off]
#                                         how clients switch the host on (a smart plug), shown and checked without
#                                         arguments; restarts a running server
set -euo pipefail
ENV_INSTALL_DIR=${INSTALL_DIR:-}
source "$(dirname "$0")/lib/config.sh"
cd "$ROOT"

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -n 1)
IMAGE=${GB10_ASSETS_IMAGE:-$(env_value GB10_ASSETS_IMAGE)}; IMAGE=${IMAGE:-stefanmenne/gb10-assets}
TAG=${GB10_ASSETS_TAG:-$(env_value GB10_ASSETS_TAG)}; TAG=${TAG:-latest}

need_env() {
  [[ -f .env ]] || { echo "ERROR: .env missing, run: $0 init" >&2; exit 1; }
}

# confirm QUESTION: true only on an explicit yes; no terminal (EOF) counts as no
confirm() {
  local answer
  read -r -p "$1 [y/N] " answer || { echo; return 1; }
  [[ $answer == [yY] || $answer == [yY][eE][sS] ]]
}

jobs_dir() {
  local jobs; jobs=$(env_value JOBS_DIR)
  abs_path "${jobs:-jobs}"
}

# create the bind-mount directories as the host user; Docker would create missing ones owned by root
make_dirs() {
  mkdir -p "$INSTALL_DIR" "$(jobs_dir)"
}

UNIT=gb10-assets-shutdown

# json_field KEY < JSON: the first value of KEY (flat values only: numbers, strings, true/false/null)
json_field() {
  sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\).*/\1/p" | head -n 1
}

# set_env KEY VALUE: replace or append KEY=VALUE in .env (empty VALUE: remove the line); keeps its mode
set_env() {
  local tmp; tmp=$(mktemp .env.XXXXXX)
  grep -v "^[[:space:]]*$1[[:space:]]*=" .env > "$tmp" || true
  [[ -z $2 ]] || printf '%s=%s\n' "$1" "$2" >> "$tmp"
  chmod --reference=.env "$tmp"
  mv "$tmp" .env
}

# shelly_check HOST SWITCH: describe the plug and whether this host hangs on it; fails if it does not answer
shelly_check() {
  local base=http://$1 id=$2 info gen st on watts cfg scripts
  info=$(curl -fsS --max-time 5 "$base/shelly") || { echo "ERROR: no Shelly answers at $base/shelly" >&2; return 1; }
  gen=$(json_field gen <<< "$info"); gen=${gen:-1}
  if [[ $gen -ge 2 ]]; then
    st=$(curl -fsS --max-time 5 "$base/rpc/Switch.GetStatus?id=$id") \
      || { echo "ERROR: $base has no switch $id" >&2; return 1; }
    on=$(json_field output <<< "$st"); watts=$(json_field apower <<< "$st")
    cfg=$(curl -fsS --max-time 5 "$base/rpc/Switch.GetConfig?id=$id" || true)
    scripts=$(curl -fsS --max-time 5 "$base/rpc/Script.List" || true)
  else
    st=$(curl -fsS --max-time 5 "$base/relay/$id") || { echo "ERROR: $base has no relay $id" >&2; return 1; }
    on=$(json_field ison <<< "$st")
    watts=$(curl -fsS --max-time 5 "$base/status" | json_field power || true)
  fi
  echo "plug:      Shelly $(json_field model <<< "$info")$(json_field type <<< "$info") (Gen $gen) at $base," \
       "switch $id: $([[ $on == true ]] && echo on || echo off)${watts:+, $watts W}"
  if [[ -n $watts ]] && awk "BEGIN {exit !($watts < 10)}"; then
    echo "WARNING:   this host is running, but the plug measures only $watts W: is the host plugged into it?"
  fi
  if [[ $(json_field auto_off <<< "$cfg") == true ]] || grep -q '"running":true' <<< "$scripts"; then
    echo "after a shutdown the plug switches itself off (auto-off or a script on it: $base)"
  else
    echo "after a shutdown the plug stays on; clients then switch it off and on again to start the host"
  fi
}

cmd=${1:-}; shift || true
case $cmd in
  init | init_without_token)
    if [[ $cmd == init_without_token ]]; then
      echo "WARNING: without a token the server accepts every request. Every device that can reach the port"
      echo "can submit jobs, use the GPU, download all results and delete jobs. Use this only in a trusted"
      echo "network and never forward the port to the internet."
      confirm "Create .env without authentication?" || { echo "aborted"; exit 1; }
    fi
    write=1 backup=0
    if [[ -f .env ]]; then
      if confirm ".env exists. Overwrite it (token and settings are replaced, a backup goes to .env.bak)?"; then
        cp -p .env .env.bak; backup=1
      else
        write=0
        echo ".env not changed"
      fi
    fi
    if [[ $write == 1 ]]; then
      if [[ $cmd == init ]]; then
        command -v openssl > /dev/null || { echo "ERROR: openssl not found" >&2; exit 1; }
        token=$(openssl rand -hex 32)
      else
        token=none
      fi
      sed -e "s/^GB10_ASSETS_TOKEN=.*/GB10_ASSETS_TOKEN=$token/" \
          -e "s/^GB10_UID=.*/GB10_UID=$(id -u)/" -e "s/^GB10_GID=.*/GB10_GID=$(id -g)/" .env.example > .env
      chmod 600 .env
      if [[ $token == none ]]; then
        echo "created .env without authentication"
      else
        echo "created .env (token: grep GB10_ASSETS_TOKEN .env)"
      fi
      [[ $backup == 0 ]] || echo "previous settings: .env.bak; a running server keeps its old token until: $0 up"
    fi
    dir=${ENV_INSTALL_DIR:-$(env_value INSTALL_DIR)}; INSTALL_DIR=$(abs_path "${dir:-installed}")   # new .env
    make_dirs
    echo "tools are installed in: $INSTALL_DIR"
    ;;
  build)
    need_env
    docker compose build "$@"
    [[ $TAG == "$VERSION" ]] || docker tag "$IMAGE:$TAG" "$IMAGE:$VERSION"
    echo "built $IMAGE:$TAG and $IMAGE:$VERSION"
    ;;
  up)
    need_env; make_dirs
    docker compose up -d "$@"
    ;;
  down)
    docker compose down "$@"
    ;;
  logs)
    docker compose logs -f "$@"
    ;;
  install | update)
    need_env; make_dirs
    [[ $cmd == install ]] || set -- --outdated
    [[ $# -gt 0 ]] || { echo "usage: $0 install TOOL ... | --all   (tools: $0 status)" >&2; exit 1; }
    # with the GPU like the server: compiling CUDA extensions (ltx-video) links the driver's libcuda. HF_TOKEN
    # (gated models) only goes to this container, the server's does not get it (compose.yaml)
    HF_TOKEN=${HF_TOKEN:-$(env_value HF_TOKEN)} docker compose run --rm --no-deps -e HF_TOKEN server scripts/install.sh "$@"
    ;;
  uninstall)
    [[ $# -gt 0 ]] || { echo "usage: $0 uninstall TOOL ..." >&2; exit 1; }
    for t in "$@"; do
      dir=$INSTALL_DIR/$t
      [[ $t =~ ^[a-z0-9-]+$ && -d $dir ]] || { echo "ERROR: $t is not installed ($dir)" >&2; exit 1; }
      confirm "Delete $dir ($(du -sh "$dir" | cut -f1), environment and weights)?" || continue
      # the lock is held while a worker of the tool runs or an installation is going on
      flock -n "$dir/.lock" rm -rf "$dir" || { echo "ERROR: $t is in use, try again later" >&2; exit 1; }
      echo "uninstalled $t"
    done
    ;;
  install-shutdown)
    # the server in the container has no rights on the host: it creates jobs/.shutdown, a systemd path unit
    # (root) sees the file, deletes it and powers off. jobs/.shutdown-helper tells the server the units exist
    need_env; make_dirs
    jobs=$(jobs_dir)
    [[ $jobs != *[[:space:]]* ]] || { echo "ERROR: JOBS_DIR must not contain spaces: $jobs" >&2; exit 1; }
    rm -f "$jobs/.shutdown"
    sudo tee /etc/systemd/system/$UNIT.path > /dev/null <<UNIT_END
[Unit]
Description=gb10-assets: shut down the host when the dashboard asks

[Path]
PathExists=$jobs/.shutdown

[Install]
WantedBy=multi-user.target
UNIT_END
    sudo tee /etc/systemd/system/$UNIT.service > /dev/null <<UNIT_END
[Unit]
Description=gb10-assets: shut down the host

[Service]
Type=oneshot
ExecStart=/usr/bin/rm -f $jobs/.shutdown
ExecStart=/usr/bin/systemctl poweroff
UNIT_END
    sudo systemctl daemon-reload
    sudo systemctl enable --now $UNIT.path
    echo "installed by scripts/docker.sh install-shutdown" > "$jobs/.shutdown-helper"
    echo "the dashboard can shut down the host now (watching $jobs/.shutdown)"
    ;;
  uninstall-shutdown)
    rm -f "$(jobs_dir)/.shutdown-helper"
    sudo systemctl disable --now $UNIT.path 2>/dev/null || true
    sudo rm -f /etc/systemd/system/$UNIT.path /etc/systemd/system/$UNIT.service
    sudo systemctl daemon-reload
    echo "removed the shutdown units"
    ;;
  power-on)
    # the server only hands the setting out (GET /power-on); clients keep it and use it while the host is off
    need_env
    case ${1:-} in
      shelly)
        [[ -n ${2:-} && $2 != *[[:space:]]* ]] || { echo "usage: $0 power-on shelly HOST[/SWITCH]" >&2; exit 1; }
        host=${2#http://}; host=${host%/}; id=0
        [[ $host != */* ]] || { id=${host##*/}; host=${host%%/*}; }
        [[ $id =~ ^[0-9]+$ ]] || { echo "ERROR: switch must be a number: $2" >&2; exit 1; }
        shelly_check "$host" "$id"
        set_env POWER_ON "shelly://$host$([[ $id == 0 ]] || echo "/$id")" ;;
      url)
        [[ ${2:-} =~ ^https?://[^[:space:]]+$ ]] || { echo "usage: $0 power-on url http(s)://..." >&2; exit 1; }
        set_env POWER_ON "$2" ;;
      off)
        set_env POWER_ON "" ;;
      "")
        spec=$(env_value POWER_ON)
        if [[ -z $spec ]]; then echo "not set up: $0 power-on shelly HOST[/SWITCH] | url URL"; exit 0; fi
        echo "POWER_ON=$spec"
        if [[ $spec == shelly://* ]]; then
          spec=${spec#shelly://}; id=0
          [[ $spec != */* ]] || { id=${spec##*/}; spec=${spec%%/*}; }
          shelly_check "$spec" "$id"
        fi
        exit 0 ;;
      *) echo "usage: $0 power-on [shelly HOST[/SWITCH] | url URL | off]" >&2; exit 1 ;;
    esac
    echo "POWER_ON=$(env_value POWER_ON) in .env"
    if [[ -n $(docker compose ps -q server 2>/dev/null) ]]; then
      docker compose up -d
    else
      echo "start the server to apply it: $0 up"
    fi
    echo "Clients learn the command on their next connection and keep it: gb10-assets power-on (or the plugin's"
    echo "power_on) switches the host on while it is off. The host must start when its power returns"
    echo "(BIOS/UEFI: power on after AC power loss); test it once: shut down, wait until the plug is off, then"
    echo "run gb10-assets power-on on a client."
    ;;
  status | list)
    state=$(docker compose ps --all --format '{{.State}}, {{.Status}}' server 2>/dev/null || true)
    echo "container: ${state:-not created (start: $0 up)}"
    port=${GB10_ASSETS_PORT:-$(env_value GB10_ASSETS_PORT)}
    if health=$(curl -fsS --max-time 3 "http://localhost:${port:-8750}/health" 2>/dev/null); then
      # {"status":"ok","api_version":1,"version":"0.1.0","running":null,"queued":0,"worker":null}
      field() { sed -n "s/.*\"$1\":\"\{0,1\}\([^\",}]*\).*/\1/p" <<< "$health" | sed 's/^null$/-/'; }
      server=$(field version)
      echo "server:    version $server, API v$(field api_version)" \
           "$([[ $server == "$VERSION" ]] || echo "(repo: $VERSION, rebuild: $0 build && $0 up)")"
      echo "jobs:      running $(field running), queued $(field queued), worker $(field worker)"
      echo "dashboard: http://$(hostname):${port:-8750}/dashboard"
    else
      echo "server:    not reachable on port ${port:-8750}"
    fi
    echo
    scripts/install.sh --list
    ;;
  *)
    sed -n '/^set -euo/q; 2,$s/^# \{0,1\}//p' "$0"
    [[ -z $cmd ]] || exit 1
    ;;
esac
