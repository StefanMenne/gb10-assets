# syntax=docker/dockerfile:1
# gb10-assets server image (linux/arm64, CUDA 13): the HTTP server, the tool definitions (tools/) and what
# installing them needs (uv, Python, git, gcc). The tools themselves, their uv environments and model weights,
# are not part of the image: `scripts/docker.sh install <tool>` puts them into the directory mounted at
# /installed (see compose.yaml and scripts/install.sh).
ARG BASE_IMAGE=nvidia/cuda:13.0.0-base-ubuntu24.04

FROM ghcr.io/astral-sh/uv:0.10.11 AS uv

FROM ${BASE_IMAGE}
# git: uv fetches git dependencies (ltx-video: Lightricks/LTX-2); gcc: Triton compiles its kernel launchers at
# runtime, tools/*/post_install.sh compiles CUDA extensions
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# uv/uvx: scripts/install.sh creates the tool environments with them and runs `hf` via uvx
COPY --from=uv /uv /uvx /usr/local/bin/
# the Python of all environments, the server's and the tools' (whose environments link to it)
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1
RUN uv python install 3.12

# server environment: /app/.venv. The uv cache is a build cache mount (another file system), so copy, not hardlink.
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --frozen --no-dev --extra server --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv UV_LINK_MODE=copy uv sync --frozen --no-dev --extra server
COPY tools ./tools
COPY scripts ./scripts

# HOME=/tmp: the container usually runs as the host user (compose.yaml), which needs a writable HOME for
# the Hugging Face, Triton and torch caches
ENV PATH=/app/.venv/bin:$PATH \
    HOME=/tmp \
    INSTALL_DIR=/installed \
    JOBS_DIR=/jobs \
    TOOLS_DIR=/app/tools
EXPOSE 8750
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8750/health', timeout=4)"
CMD ["gb10-assets", "server", "--host", "0.0.0.0", "--port", "8750"]
