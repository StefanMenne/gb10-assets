# syntax=docker/dockerfile:1
# gb10-assets server image (linux/arm64, CUDA 13): the HTTP server plus one uv environment per tool in
# /opt/<tool>/.venv. Model weights are not part of the image; mount them at /models (see compose.yaml).
ARG BASE_IMAGE=nvidia/cuda:13.0.0-base-ubuntu24.04

FROM ghcr.io/astral-sh/uv:0.10.11 AS uv

# only the dependency manifests of the tools, so editing run.py does not invalidate the (large) torch layer
FROM ${BASE_IMAGE} AS tool-manifests
COPY tools /manifests
RUN find /manifests -type f ! -name pyproject.toml ! -name uv.lock -delete

FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# uv/uvx stay in the image: scripts/download-all.sh runs `hf` via uvx
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
RUN uv python install 3.12

# tool environments: /opt/<tool>/.venv, found by the server via TOOL_ENVS_DIR
COPY --from=tool-manifests /manifests /app/tools
RUN --mount=type=cache,target=/root/.cache/uv \
    for d in /app/tools/*/; do \
      [ -f "$d/pyproject.toml" ] || continue; \
      t=$(basename "$d"); echo "=== $t"; \
      UV_PROJECT_ENVIRONMENT=/opt/$t/.venv uv sync --frozen --no-dev --project "$d" || exit 1; \
    done

# server environment: /app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra server --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra server
COPY tools ./tools
COPY scripts ./scripts

# HOME=/tmp: the container usually runs as the host user (compose.yaml), which needs a writable HOME for
# the uv, Hugging Face and torch caches
ENV PATH=/app/.venv/bin:$PATH \
    HOME=/tmp \
    MODELS_DIR=/models \
    JOBS_DIR=/jobs \
    TOOLS_DIR=/app/tools \
    TOOL_ENVS_DIR=/opt
EXPOSE 8750
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8750/health', timeout=4)"
CMD ["gb10-assets", "server", "--host", "0.0.0.0", "--port", "8750"]
