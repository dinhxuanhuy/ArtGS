#!/usr/bin/env bash
# Bootstrap a fresh Colab VM for ArtGS. Run ON THE VM (as root), from the synced repo:
#   bash /content/ArtGS/scripts/bootstrap_colab.sh
# Idempotent: every step is skipped when already done on this VM.
#
# Env overrides:
#   ARTGS_REMOTE   rclone path of the persistent project root (default "nckh:" = Drive folder
#                  NCKH-2027; scripts/colab_up.sh installs rclone + config on the VM).
#                  Unset/"none" or remote unreachable -> no persistent cache (build + HF download).
#   DATASETS       datasets to fetch (default "paris"; "" = none). See configs/datasets.json.
#   FORCE_BUILD=1  ignore cached wheels and rebuild the CUDA extensions.
#
# Layout on the VM:
#   /content/venv     Python 3.10 venv (torch 2.1.2+cu121)   /content/data   datasets (local disk)
#   /content/ArtGS    code (rsynced)  data -> /content/data  /content/build  extension builds
# Persistent (Drive, via rclone):  <remote>data/archives/*.zip   dataset archives (sha256-verified)
#                                  <remote>_env_cache/wheels/<key>/  built CUDA-extension wheels
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/content/venv
DATA=/content/data
BUILD=/content/build
REMOTE="${ARTGS_REMOTE:-nckh:}"
ARCH_DIR=/content/_archives
DATASETS="${DATASETS-paris}"
PY_VER=3.10
TORCH_VER=2.1.2
TV_VER=0.16.2
CU=cu121
P3D_REF=75ebeeaea0908c5527e7b1e305fbc7681382db47   # tag V0.7.8 (= local env)
TCNN_COMMIT=3daa6e5b8b8e0fa17971a1a20acfbb3754494b38
CUDA_LABEL=nvidia/label/cuda-12.1.1
TIMING=/content/bootstrap_timing.tsv

export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 UV_LINK_MODE=copy

T0=$(date +%s); STEP_T=$T0; STEP=""
: > "$TIMING"
step() {
  local now; now=$(date +%s)
  [ -n "$STEP" ] && printf '%s\t%d\n' "$STEP" $((now - STEP_T)) | tee -a "$TIMING" | awk -F'\t' '{printf "[bootstrap]   done: %s (%ss)\n",$1,$2}'
  STEP="$1"; STEP_T=$now
  [ "$1" != "END" ] && echo "[bootstrap] === $1"
  return 0
}
die() { echo "[bootstrap] ERROR: $*" >&2; exit 1; }

[ -f "$REPO/train.py" ] || die "run from the synced repo (expected $REPO/train.py)"
nvidia-smi >/dev/null 2>&1 || die "no GPU on this runtime"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
SM=${CAP/./}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "[bootstrap] GPU: $GPU (sm_$SM)"

RC=(rclone --transfers 8 --checkers 16 --drive-chunk-size 64M -q)
if [ "$REMOTE" = none ] || ! command -v rclone >/dev/null || ! timeout 60 rclone lsf --max-depth 1 "$REMOTE" >/dev/null 2>&1; then
  echo "[bootstrap] WARN: persistent remote '$REMOTE' unavailable -> no persistent cache"
  REMOTE=""
else
  echo "[bootstrap] persistent remote: $REMOTE"
fi

# ---------------------------------------------------------------- venv + torch
step "venv (python $PY_VER)"
command -v uv >/dev/null || pip install -q uv
if ! "$VENV/bin/python" -c "import sys; assert sys.version_info[:2] == (3, 10)" 2>/dev/null; then
  uv venv -q --seed --clear -p "$PY_VER" "$VENV"   # missing or broken (e.g. dangling interpreter)
fi
PY="$VENV/bin/python"
UVPIP=(uv pip install -q --python "$PY")

step "torch $TORCH_VER+$CU"
if ! "$PY" -c "import torch,sys; sys.exit(torch.__version__!='$TORCH_VER+$CU')" 2>/dev/null; then
  "${UVPIP[@]}" "torch==$TORCH_VER" "torchvision==$TV_VER" --index-url "https://download.pytorch.org/whl/$CU"
fi

step "requirements-colab.txt"
"${UVPIP[@]}" -r "$REPO/requirements-colab.txt"

# ---------------------------------------------------------------- CUDA extensions
# Cache key = python/torch/cuda/arch + hash of every extension source we build.
src_hash() {
  {
    git -C "$REPO/submodules/diff-gaussian-rasterization" rev-parse HEAD 2>/dev/null \
      || find "$REPO/submodules/diff-gaussian-rasterization" -path '*/build' -prune -o -type f \( -name '*.cu' -o -name '*.h' -o -name '*.cpp' -o -name '*.py' \) -print0 | sort -z | xargs -0 sha256sum
    git -C "$REPO/submodules/simple-knn" rev-parse HEAD 2>/dev/null \
      || find "$REPO/submodules/simple-knn" -path '*/build' -prune -o -type f \( -name '*.cu' -o -name '*.h' -o -name '*.cpp' -o -name '*.py' \) -print0 | sort -z | xargs -0 sha256sum
    find "$REPO/utils/pointnet_lib/src" "$REPO/utils/pointnet_lib/setup.py" -type f -print0 | sort -z | xargs -0 cat
    echo "$P3D_REF $TCNN_COMMIT"
  } | sha256sum | cut -c1-8
}
KEY="py${PY_VER/./}-torch${TORCH_VER}-${CU}-sm${SM}-$(src_hash)"
WHEELS="$BUILD/wheels"
echo "[bootstrap] extension cache key: $KEY"

ext_ok() {
  "$PY" - <<'EOF' 2>/dev/null
import torch, diff_gaussian_rasterization, simple_knn._C, pointnet2_cuda, pytorch3d._C, tinycudann
EOF
}

step "CUDA extensions"
if ext_ok && [ -z "${FORCE_BUILD:-}" ]; then
  echo "[bootstrap] extensions already installed"
elif [ -z "${FORCE_BUILD:-}" ] && [ -n "$REMOTE" ] \
     && rclone lsf "${REMOTE}_env_cache/wheels/$KEY/" 2>/dev/null | grep -qx COMPLETE; then
  echo "[bootstrap] installing cached wheels from ${REMOTE}_env_cache/wheels/$KEY"
  rm -rf "$WHEELS"; mkdir -p "$WHEELS"
  "${RC[@]}" copy "${REMOTE}_env_cache/wheels/$KEY/" "$WHEELS/" --include '*.whl'
  "${UVPIP[@]}" --no-deps --reinstall "$WHEELS"/*.whl
else
  echo "[bootstrap] no cached wheels for $KEY -> building (slow)"
  step "build: CUDA $CU toolkit + gcc-11"
  # nvcc 12.1 rejects gcc > 12 (Colab ships 13), and with gcc 12 it fails on torch's pybind11
  # headers ("cast.h: expected template-name before <"). gcc 11 works (same as local env).
  command -v g++-11 >/dev/null || { apt-get -qq update >/dev/null && apt-get -qq install -y g++-11 >/dev/null; }
  MM=/content/micromamba/bin/micromamba
  if [ ! -x "$MM" ]; then
    mkdir -p /content/micromamba
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /content/micromamba bin/micromamba
  fi
  CUDA_HOME=/content/cuda121
  if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    "$MM" create -y -q -r /content/micromamba/root -p "$CUDA_HOME" -c "$CUDA_LABEL" \
      cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl >/dev/null
  fi
  "$CUDA_HOME/bin/nvcc" --version | tail -1
  export CUDA_HOME PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
  export CC=gcc-11 CXX=g++-11 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
  export TORCH_CUDA_ARCH_LIST="$CAP" TCNN_CUDA_ARCHITECTURES="$SM" FORCE_CUDA=1
  export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
  rm -rf "$WHEELS"; mkdir -p "$WHEELS" "$BUILD"
  PIPW=("$PY" -m pip wheel -v --no-build-isolation --no-deps -w "$WHEELS")

  build_one() {  # name, src
    local s; s=$(date +%s)
    echo "[bootstrap]   building $1"
    "${PIPW[@]}" "$2" > "$BUILD/$1.log" 2>&1 || { grep -E -B2 -A8 "error|Error" "$BUILD/$1.log" | grep -v Werror | head -60; die "build of $1 failed (log: $BUILD/$1.log)"; }
    printf 'build: %s\t%d\n' "$1" $(( $(date +%s) - s )) | tee -a "$TIMING" | awk -F'\t' '{printf "[bootstrap]   %s (%ss)\n",$1,$2}'
  }
  step "build: extensions"
  for d in diff-gaussian-rasterization simple-knn; do
    rm -rf "$BUILD/$d"; cp -r "$REPO/submodules/$d" "$BUILD/$d"; rm -rf "$BUILD/$d"/build "$BUILD/$d"/*.egg-info
    build_one "$d" "$BUILD/$d"
  done
  rm -rf "$BUILD/pointnet_lib"; cp -r "$REPO/utils/pointnet_lib" "$BUILD/pointnet_lib"
  rm -rf "$BUILD/pointnet_lib"/{build,dist,*.egg-info}
  build_one pointnet2 "$BUILD/pointnet_lib"
  build_one pytorch3d "git+https://github.com/facebookresearch/pytorch3d.git@$P3D_REF"
  if [ ! -d "$BUILD/tiny-cuda-nn/.git" ]; then
    git clone -q https://github.com/NVlabs/tiny-cuda-nn.git "$BUILD/tiny-cuda-nn"
  fi
  git -C "$BUILD/tiny-cuda-nn" checkout -q "$TCNN_COMMIT"
  git -C "$BUILD/tiny-cuda-nn" submodule -q update --init --recursive
  # setuptools >= 81 dropped pkg_resources (third-party build file, not ArtGS code)
  sed -i 's/^from pkg_resources import parse_version$/from packaging.version import Version as parse_version/' \
    "$BUILD/tiny-cuda-nn/bindings/torch/setup.py"
  build_one tinycudann "$BUILD/tiny-cuda-nn/bindings/torch"

  step "build: install + save wheels"
  "${UVPIP[@]}" --no-deps --reinstall "$WHEELS"/*.whl
  ext_ok || die "extensions built but failed to import"
  { echo "key=$KEY"; echo "gpu=$GPU"; echo "built=$(date -u +%FT%TZ)"; "$CUDA_HOME/bin/nvcc" --version | tail -1; } > "$WHEELS/BUILD_INFO"
  if [ -n "$REMOTE" ]; then
    dst="${REMOTE}_env_cache/wheels/$KEY"
    "${RC[@]}" copy "$WHEELS/" "$dst/"            # wheels + BUILD_INFO first ...
    touch "$WHEELS/COMPLETE"; "${RC[@]}" copy "$WHEELS/COMPLETE" "$dst/"   # ... marker last
    echo "[bootstrap] wheels saved to $dst"
  fi
fi
ext_ok || die "CUDA extensions not importable"

# ---------------------------------------------------------------- data
step "datasets: ${DATASETS:-none}"
mkdir -p "$DATA"
if [ -L "$REPO/data" ] || [ ! -e "$REPO/data" ]; then
  ln -sfn "$DATA" "$REPO/data"
else
  echo "[bootstrap] WARN: $REPO/data is a real dir, not linking to $DATA"
fi
if [ -n "$DATASETS" ]; then
  mkdir -p "$ARCH_DIR"
  # shellcheck disable=SC2086
  NEED=$("$PY" - $DATASETS <<'EOF'
import json, sys, pathlib
m = json.load(open("/content/ArtGS/configs/datasets.json"))
done = lambda d: (pathlib.Path("/content/data") / m["datasets"][d]["dest"] / ".artgs_complete.json").exists()
print(" ".join(sorted({m["datasets"][d]["archive"] for d in sys.argv[1:] if not done(d)})))
EOF
)
  for a in $NEED; do   # pull archives from Drive when available (else load_data.py downloads)
    [ -n "$REMOTE" ] && "${RC[@]}" copy "${REMOTE}data/archives/$a" "$ARCH_DIR/" 2>/dev/null && echo "[bootstrap] pulled $a from $REMOTE"
  done
  # shellcheck disable=SC2086
  "$PY" "$REPO/scripts/load_data.py" $DATASETS --root "$DATA" --downloads "$ARCH_DIR"
  for a in $NEED; do   # push freshly downloaded archives so the next VM gets them from Drive
    if [ -n "$REMOTE" ] && [ -f "$ARCH_DIR/$a" ] && ! rclone lsf "${REMOTE}data/archives/" 2>/dev/null | grep -qx "$a"; then
      "${RC[@]}" copy "$ARCH_DIR/$a" "${REMOTE}data/archives/" && echo "[bootstrap] saved $a to $REMOTE"
    fi
  done
  rm -rf "$ARCH_DIR"
fi

# ---------------------------------------------------------------- verify
step "verify_env.py"
# opencv/open3d need libGL, glib, gomp: present on Colab images, installed here only if missing
"$PY" -c "import cv2, open3d" 2>/dev/null \
  || { apt-get -qq update >/dev/null && apt-get -qq install -y libgl1 libglib2.0-0t64 libgomp1 >/dev/null; }
( cd "$REPO" && PYTHONNOUSERSITE=1 "$PY" verify_env.py | tail -3 )

step "END"
printf 'TOTAL\t%d\n' $(( $(date +%s) - T0 )) | tee -a "$TIMING"
echo "[bootstrap] ready. activate with: source $VENV/bin/activate"
