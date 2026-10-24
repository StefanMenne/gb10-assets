#!/usr/bin/env bash
# Docker helper for the gb10-assets server. Settings come from .env (see .env.example).
#
#   scripts/docker.sh init                create .env with a random token, create the models/jobs directories
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

# create the bind-mount directories as the host user; Docker would create missing ones owned by root
make_dirs() {
  local jobs; jobs=$(env_value JOBS_DIR)
  mkdir -p "$MODELS_DIR" "$(abs_path "${jobs:-jobs}")"
}

cmd=${1:-}; shift || true
case $cmd in
  init)
    if [[ -f .env ]]; then
      echo ".env exists, not changed"
    else
      command -v openssl > /dev/null || { echo "ERROR: openssl not found" >&2; exit 1; }
      sed -e "s/^GB10_ASSETS_TOKEN=.*/GB10_ASSETS_TOKEN=$(openssl rand -hex 32)/" \
          -e "s/^GB10_UID=.*/GB10_UID=$(id -u)/" -e "s/^GB10_GID=.*/GB10_GID=$(id -g)/" .env.example > .env
      chmod 600 .env
      echo "created .env (token: grep GB10_ASSETS_TOKEN .env)"
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
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    [[ -z $cmd ]] || exit 1
    ;;
esac
