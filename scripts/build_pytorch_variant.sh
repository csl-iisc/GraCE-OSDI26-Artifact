#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
  echo "Usage: $0 --variant <vanilla|cgct|cgct-pi|full|cgct-copy-debug|cgct-pi-copy-debug> [--skip-triton]"
}

VARIANT=""
SKIP_TRITON=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --skip-triton) SKIP_TRITON=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$VARIANT" ]]; then
  usage
  exit 1
fi

ENV_NAME="$(variant_env "$VARIANT")"
PYTORCH_BRANCH="$(variant_pytorch_branch "$VARIANT")"

if [[ "$SKIP_TRITON" -eq 0 ]]; then
  bash "$SCRIPT_DIR/build_triton.sh" --variant "$VARIANT"
fi

activate_env "$ENV_NAME"

log_step "Building PyTorch for variant=$VARIANT env=$ENV_NAME branch=$PYTORCH_BRANCH"

cd "$PYTORCH_DIR"
git checkout "$PYTORCH_BRANCH"
git submodule sync
git submodule update --init --recursive

pip install -r requirements.txt
pip install depyf ninja cmake==3.28.3

if [[ -d build ]]; then
  log_warn "Removing existing PyTorch build directory"
  rm -rf build
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"

if [[ "$CUDA_SHORT" != "124" && -z "${MAGMA_HOME:-}" ]]; then
  export USE_MAGMA="${USE_MAGMA:-0}"
  log_warn "CUDA_SHORT=$CUDA_SHORT and MAGMA_HOME unset. Building PyTorch with USE_MAGMA=$USE_MAGMA"
fi

python setup.py install

log_ok "PyTorch build complete for $VARIANT"

# torchvision/torchaudio installation. Keep pinned.
DEPS_ROOT="$ARTIFACT_ROOT/third_party"

cd "$DEPS_ROOT"

if [[ ! -d vision ]]; then
  git clone https://github.com/pytorch/vision.git
fi
cd vision
git checkout 06a925c32b49fd0455e265097fa7ca623cec4154
rm -rf build
python setup.py install
cd "$DEPS_ROOT"

if [[ ! -d audio ]]; then
  git clone https://github.com/pytorch/audio.git
fi
cd audio
git checkout b6d4675c7aedc53ba04f3f55786aac1de32be6b4
rm -rf build
python setup.py install

log_ok "torchvision and torchaudio installed for $VARIANT"