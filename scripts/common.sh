#!/usr/bin/env bash

# Shared config for GraCE OSDI'26 artifact scripts.
# This file is intended to be sourced, not executed.

set -euo pipefail

ARTIFACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

PYTORCH_DIR="${PYTORCH_DIR:-$ARTIFACT_ROOT/third_party/pytorch}"
TRITON_DIR="${TRITON_DIR:-$ARTIFACT_ROOT/third_party/triton}"
TORCHBENCH_DIR="${TORCHBENCH_DIR:-$ARTIFACT_ROOT/third_party/torchbenchmark}"
BENCHMARK_RUNNER_DIR="${BENCHMARK_RUNNER_DIR:-$ARTIFACT_ROOT/benchmark_runner}"

CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_SHORT="${CUDA_SHORT:-124}"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Variant names used throughout artifact.
ALL_VARIANTS=(vanilla cgct cgct-pi full)
DEBUG_VARIANTS=(cgct-copy-debug cgct-pi-copy-debug)

variant_env() {
  case "$1" in
    vanilla) echo "grace-vanilla" ;;
    cgct) echo "grace-cgct" ;;
    cgct-pi) echo "grace-cgct-pi" ;;
    full) echo "grace-full" ;;
    cgct-copy-debug) echo "grace-cgct-copy-debug" ;;
    cgct-pi-copy-debug) echo "grace-cgct-pi-copy-debug" ;;
    *) echo "[ERROR] Unknown variant: $1" >&2; return 1 ;;
  esac
}

variant_pytorch_branch() {
  case "$1" in
    vanilla) echo "osdi26/baseline-pytorch2" ;;
    cgct) echo "osdi26/grace-cgct" ;;
    cgct-pi) echo "osdi26/grace-cgct-pi" ;;
    full) echo "osdi26/grace-full" ;;
    cgct-copy-debug) echo "osdi26/grace-cgct-copy-debug" ;;
    cgct-pi-copy-debug) echo "osdi26/grace-cgct-pi-copy-debug" ;;
    *) echo "[ERROR] Unknown variant: $1" >&2; return 1 ;;
  esac
}

variant_triton_branch() {
  case "$1" in
    vanilla) echo "osdi26/baseline-triton" ;;
    cgct) echo "osdi26/baseline-triton" ;;
    cgct-pi) echo "osdi26/grace-indirection" ;;
    full) echo "osdi26/grace-indirection" ;;
    cgct-copy-debug) echo "osdi26/baseline-triton" ;;
    cgct-pi-copy-debug) echo "osdi26/grace-indirection" ;;
    *) echo "[ERROR] Unknown variant: $1" >&2; return 1 ;;
  esac
}

log_step() { echo -e "\033[1;36m>>> $*\033[0m"; }
log_ok() { echo -e "\033[1;32m>>> $*\033[0m"; }
log_warn() { echo -e "\033[1;33m>>> $*\033[0m"; }
log_err() { echo -e "\033[1;31m>>> $*\033[0m" >&2; }

require_conda() {
  if [[ ! -f "$CONDA_SH" ]]; then
    log_err "conda.sh not found at $CONDA_SH"
    exit 1
  fi
}

activate_env() {
  local env_name="$1"
  require_conda
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$env_name"
}