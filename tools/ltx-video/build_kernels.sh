#!/usr/bin/env bash
# Builds ltx-kernels (Lightricks, the NVFP4 GEMM that runs the NVFP4 transformer) as a wheel and installs it into
# the ltx-video environment. It is not in pyproject.toml because it compiles CUDA code against the installed torch
# and needs a patch (below), so run this again after every `uv sync` of the tool (uv sync removes it).
#
#   tools/ltx-video/build_kernels.sh [PYTHON] [WHEEL_DIR]
#
# PYTHON defaults to tools/ltx-video/.venv/bin/python, WHEEL_DIR (where the wheel is kept) to a temporary directory.
# Needs git and about two minutes; nvcc comes from pip in the CUDA version of torch (a system toolkit of another
# minor version clashes with the CUDA headers torch brings along). The blockwise FP8 extension links the libcuda
# stub of a system toolkit, so the Dockerfile builds the wheel in the CUDA devel image and installs only the wheel
# into the runtime image.
set -euo pipefail

LTX_REV=e98f6d927d83c92ef2be6aec76c8440b19306e39   # keep in sync with ltx-core/ltx-pipelines in pyproject.toml
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${1:-$HERE/.venv/bin/python}
WHEELS=${2:-$(mktemp -d)}
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
assert old in s and "121a" in s, "setup.py changed upstream, update the patch in build_kernels.sh"
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
uv build -q --wheel --no-build-isolation --python "$PY" -o "$WHEELS" "$SRC/packages/ltx-kernels"
# --no-deps: triton comes with torch, nvidia-cutlass-dsl is only used by the datacenter-Blackwell VAE kernels
uv pip install -q --python "$PY" --no-deps --reinstall "$WHEELS"/ltx_kernels-*.whl
echo "ltx-kernels installed into $("$PY" -c 'import sys; print(sys.prefix)') (wheel: $WHEELS)"
