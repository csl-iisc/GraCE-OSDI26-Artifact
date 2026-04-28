#!/usr/bin/env bash
set -euo pipefail

CHECK_TORCH=0
CONDA_ENV=""

usage() {
  cat <<'EOF'
Usage:
  scripts/check_cuda_stack.sh [--torch] [--env <conda-env>]

Default:
  Check NVIDIA driver, CUDA compiler, CUDA_HOME, CUDA libraries, Python, and conda.

Options:
  --torch
      Also check PyTorch import and CUDA visibility through torch.

  --env <conda-env>
      Activate this conda environment before running the optional torch check.

Examples:
  bash scripts/check_cuda_stack.sh
  bash scripts/check_cuda_stack.sh --torch --env grace-full
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch)
      CHECK_TORCH=1
      shift
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "=== NVIDIA driver visibility ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[ERROR] nvidia-smi not found. Did you start Docker with --gpus all?"
  exit 1
fi

echo
echo "=== CUDA compiler ==="
if command -v nvcc >/dev/null 2>&1; then
  which nvcc
  nvcc --version
else
  echo "[ERROR] nvcc not found. Use a CUDA devel image, not runtime/base."
  exit 1
fi

echo
echo "=== CUDA_HOME ==="
echo "${CUDA_HOME:-unset}"

if [[ "${CUDA_HOME:-}" != "/usr/local/cuda" ]]; then
  echo "[WARN] CUDA_HOME is not /usr/local/cuda"
fi

echo
echo "=== CUDA runtime libraries ==="
ls -l /usr/local/cuda/lib64/libcudart.so* || {
  echo "[ERROR] Could not find libcudart under /usr/local/cuda/lib64"
  exit 1
}

echo
echo "=== Python availability ==="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
else
  echo "[WARN] python not found in current PATH"
fi

echo
echo "=== Conda availability ==="
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"

if [[ -f "$CONDA_SH" ]]; then
  echo "conda.sh: $CONDA_SH"
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda --version
else
  echo "[WARN] conda.sh not found at $CONDA_SH"
fi

if [[ "$CHECK_TORCH" == "1" ]]; then
  echo
  echo "=== Optional PyTorch check ==="

  if [[ -n "$CONDA_ENV" ]]; then
    if [[ ! -f "$CONDA_SH" ]]; then
      echo "[ERROR] Cannot activate env '$CONDA_ENV': conda.sh not found at $CONDA_SH"
      exit 1
    fi

    # shellcheck source=/dev/null
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    echo "Activated conda env: $CONDA_ENV"
  fi

  python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("cuda device 0:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("[ERROR] torch.cuda.is_available() returned False")
PY
else
  echo
  echo "=== PyTorch check skipped ==="
  echo "Run this later after building/installing PyTorch:"
  echo "  bash scripts/check_cuda_stack.sh --torch --env grace-full"
fi

echo
echo "[OK] CUDA stack check complete"