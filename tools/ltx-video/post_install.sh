#!/usr/bin/env bash
# Builds ltx-kernels (Lightricks, the NVFP4 GEMM that runs the NVFP4 transformer) and installs it into the
# ltx-video environment. It is not in pyproject.toml because it compiles CUDA code against the installed torch
# and needs a patch (below). scripts/install.sh runs this after every `uv sync` (which removes it again):
#
#   tools/ltx-video/post_install.sh PYTHON BUILD_DIR
#
# PYTHON is the environment's python, BUILD_DIR keeps the wheel per LTX revision and torch version, so only the
# first installation compiles (about two minutes, needs git). nvcc comes from pip in the CUDA version of torch
# (a system toolkit of another minor version clashes with the CUDA headers torch brings along); the blockwise
# FP8 extension links libcuda, which the NVIDIA container runtime provides (in Docker: run with the GPU).
set -euo pipefail

LTX_REV=e98f6d927d83c92ef2be6aec76c8440b19306e39   # keep in sync with ltx-core/ltx-pipelines in pyproject.toml
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$1
ENV_DIR=$(cd "$(dirname "$PY")/.." && pwd)
torch=$("$PY" -c 'import torch; print(torch.__version__)')
WHEELS=$2/ltx-kernels-${LTX_REV:0:12}-torch$torch

if ! compgen -G "$WHEELS/ltx_kernels-*.whl" > /dev/null; then
  SRC=$(mktemp -d)
  trap 'rm -rf "$SRC"' EXIT
  git clone -q --filter=blob:none --no-checkout https://github.com/Lightricks/LTX-2 "$SRC"
  git -C "$SRC" checkout -q "$LTX_REV"
  # GB10 is sm_121. Upstream builds the NVFP4 extension only for sm_100a/110a/120a, and an sm_120a cubin does not
  # run on 12.1 (kernels silently do nothing); map TORCH_CUDA_ARCH_LIST=12.1 to sm_121a instead.
  setup=$SRC/packages/ltx-kernels/setup.py
  sed -i 's/_NVFP4_ARCHES = \["100a", "110a", "120a"\]/_NVFP4_ARCHES = ["100a", "110a", "120a", "121a"]/' "$setup"
  "$PY" - "$setup" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = '        elif tok.startswith("12.0"):\n            sel.append("120a")\n'
assert old in s and "121a" in s, "setup.py changed upstream, update the patch in post_install.sh"
s = s.replace(old, old + '        elif tok.startswith("12.1"):\n            sel.append("121a")\n')
open(p, "w").write(s)
EOF

  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.1} MAX_JOBS=${MAX_JOBS:-$(nproc)} LTX_KERNELS_CACHE_DIR=$SRC/cache
  # build requirements of the no-isolation build; torch comes from the environment itself
  cuda=$("$PY" -c 'import torch; print(torch.version.cuda)')
  # nvvm (cicc) and crt are separate wheels and must match nvcc, otherwise ptxas rejects the PTX
  uv pip install -q --python "$PY" setuptools wheel ninja \
    "nvidia-cuda-nvcc==$cuda.*" "nvidia-cuda-cccl==$cuda.*" "nvidia-nvvm==$cuda.*" "nvidia-cuda-crt==$cuda.*"
  # the pip toolkit first, also for ptxas, which nvcc otherwise may take from a system toolkit on PATH
  CUDA_HOME=$("$PY" -c 'import nvidia, pathlib; print(pathlib.Path(nvidia.__path__[0]) / "cu13")')
  export CUDA_HOME PATH=$CUDA_HOME/bin:$PATH
  mkdir -p "$WHEELS"
  uv build -q --wheel --no-build-isolation --python "$PY" -o "$WHEELS" "$SRC/packages/ltx-kernels"
  # drop the build requirements again: back to exactly the locked environment
  UV_PROJECT_ENVIRONMENT=$ENV_DIR uv sync -q --frozen --no-dev --project "$HERE"
fi
# --no-deps: triton comes with torch, nvidia-cutlass-dsl is only used by the datacenter-Blackwell VAE kernels
uv pip install -q --python "$PY" --no-deps --reinstall "$WHEELS"/ltx_kernels-*.whl
echo "ltx-kernels installed into $ENV_DIR (wheel: $WHEELS)"
