#!/usr/bin/env bash
# Docker helper for the gb10-assets server. Settings come from .env (see .env.example).
#
#   scripts/docker.sh init                create .env with a random token, create the models/jobs directories
#   scripts/docker.sh init_without_token  the same, but without authentication (GB10_ASSETS_TOKEN=none)
#   scripts/docker.sh build               build the image (tags :<GB10_ASSETS_TAG> and :<version>)
#   scripts/docker.sh push                push both tags (docker login first)
#   scripts/docker.sh pull                pull the published image instead of building it
#   scripts/docker.sh up | down | logs    start in the background, stop, follow the server log
#   scripts/docker.sh download [tool ...] download model weights with the image (no uv needed on the host)
set -euo pipefail
ENV_MODELS_DIR=${MODELS_DIR:-}
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

# create the bind-mount directories as the host user; Docker would create missing ones owned by root
make_dirs() {
  local jobs; jobs=$(env_value JOBS_DIR)
  mkdir -p "$MODELS_DIR" "$(abs_path "${jobs:-jobs}")"
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
    dir=${ENV_MODELS_DIR:-$(env_value MODELS_DIR)}; MODELS_DIR=$(abs_path "${dir:-models}")   # from the new .env
    make_dirs
    echo "models: $MODELS_DIR"
    ;;
  build)
    need_env
    docker compose build "$@"
    [[ $TAG == "$VERSION" ]] || docker tag "$IMAGE:$TAG" "$IMAGE:$VERSION"
    echo "built $IMAGE:$TAG and $IMAGE:$VERSION"
    ;;
  push)
    docker push "$IMAGE:$TAG"
    [[ $TAG == "$VERSION" ]] || docker push "$IMAGE:$VERSION"
    ;;
  pull)
    need_env
    docker compose pull
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
  download)
    need_env; make_dirs
    docker compose run --rm --no-deps server scripts/download-all.sh "$@"
    ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    [[ -z $cmd ]] || exit 1
    ;;
esac
