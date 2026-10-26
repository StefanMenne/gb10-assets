#!/usr/bin/env bash
# Builds Blender as a Python module (bpy) and installs it into the blender environment. There is no bpy wheel
# for Linux aarch64 on PyPI, so the first installation compiles one (about 10-30 minutes, ~10 GB of temporary
# space under BUILD_DIR, needs git, git-lfs and gcc-14 from the image); later ones reuse it:
#
#   tools/blender/post_install.sh PYTHON BUILD_DIR
#
# Blender only is compiled: its libraries (USD, OpenVDB, Embree, OSL, OIDN, Python ...) come precompiled from
# lib-linux_arm64. Headless build (no X11/Wayland; EEVEE draws through EGL), Cycles with CUDA kernels for the
# GB10 (sm_121). cmake, ninja and nvcc come from pip, into a throwaway environment next to the sources.
set -euo pipefail

BLENDER_REV=d13f752e3b9c4f8c261cda552b1021f8bcc0382c   # v5.2.2
LIBS_REV=c1f0435fde33f7dc921584018a39b11eef78d98a      # lib-linux_arm64, branch blender-v5.2-release
CUDA=13.0
PY=$1
WHEELS=$2/bpy-${BLENDER_REV:0:12}

fetch() {  # fetch URL REV DIR: one commit, without history; git-lfs files come from the remote "origin"
  git init -q "$3"
  git -C "$3" remote add origin "$1"
  git -C "$3" lfs install --local > /dev/null
  git -C "$3" fetch -q --depth 1 origin "$2"
  git -C "$3" checkout -q FETCH_HEAD
}

if ! compgen -G "$WHEELS/bpy-*.whl" > /dev/null; then
  # on the mounted volume, not in /tmp: the container's writable layer would hold ~10 GB
  # kept when the build fails, so that a retry neither downloads nor compiles again; removed after a successful build
  SRC=$2/bpy-src
  if [[ $(cat "$SRC/.rev" 2>/dev/null) != "$BLENDER_REV $LIBS_REV" ]]; then
    rm -rf "$SRC"
    mkdir -p "$SRC"
    echo "[blender] sources -> $SRC"
    fetch https://projects.blender.org/blender/blender.git "$BLENDER_REV" "$SRC/blender"
    fetch https://projects.blender.org/blender/lib-linux_arm64.git "$LIBS_REV" "$SRC/blender/lib/linux_arm64"
    echo "$BLENDER_REV $LIBS_REV" > "$SRC/.rev"
  fi
  rm -rf "$SRC/tools" "$SRC/install"   # build/ stays: a retry continues the build

  # nvvm (cicc) and crt are separate wheels and must match nvcc; setuptools/wheel for make_bpy_wheel.py
  uv venv -q --python 3.13 "$SRC/tools"
  uv pip install -q --python "$SRC/tools/bin/python" "cmake==3.31.*" ninja setuptools wheel \
    "nvidia-cuda-nvcc==$CUDA.*" "nvidia-cuda-cccl==$CUDA.*" "nvidia-nvvm==$CUDA.*" "nvidia-cuda-crt==$CUDA.*" \
    "nvidia-cuda-runtime==$CUDA.*"
  CUDA_HOME=$("$SRC/tools/bin/python" -c 'import nvidia, pathlib; print(pathlib.Path(nvidia.__path__[0]) / "cu13")')
  export PATH=$SRC/tools/bin:$CUDA_HOME/bin:$PATH

  echo "[blender] building bpy (log: $SRC/build/build.log)"
  cmake -S "$SRC/blender" -B "$SRC/build" -G Ninja -Wno-dev \
    -C "$SRC/blender/build_files/cmake/config/bpy_module.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_INSTALL_PREFIX="$SRC/install" -DWITH_INSTALL_PORTABLE=ON \
    -DWITH_HEADLESS=ON -DWITH_VULKAN_BACKEND=OFF \
    -DWITH_CYCLES_DEVICE_CUDA=ON -DWITH_CYCLES_CUDA_BINARIES=ON -DCYCLES_CUDA_BINARIES_ARCH=sm_121 \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -DCUDA_HOST_COMPILER=/usr/bin/gcc-14 \
    -DWITH_CYCLES_DEVICE_OPTIX=OFF -DWITH_CYCLES_DEVICE_HIP=OFF -DWITH_CYCLES_DEVICE_ONEAPI=OFF \
    > "$SRC/cmake.log" || { tail -40 "$SRC/cmake.log" >&2; exit 1; }
  ninja -C "$SRC/build" install > "$SRC/build/build.log" || { grep -B5 -A20 -m3 'FAILED' "$SRC/build/build.log" >&2; exit 1; }
  mkdir -p "$WHEELS"
  "$SRC/tools/bin/python" "$SRC/blender/build_files/utils/make_bpy_wheel.py" "$SRC/install" --build-dir "$SRC/build" --output-dir "$WHEELS"
  rm -rf "$SRC"
fi
# --no-deps: its requirements are in pyproject.toml
uv pip install -q --python "$PY" --no-deps --reinstall "$WHEELS"/bpy-*.whl
echo "bpy installed (wheel: $WHEELS)"
