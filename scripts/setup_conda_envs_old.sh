#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_conda
source "$CONDA_SH"

PY_VER="${PY_VER:-3.10.12}"

ENVS=(
  grace-vanilla
  grace-cgct
  grace-cgct-pi
  grace-full
  grace-cgct-copy-debug
  grace-cgct-pi-copy-debug
)

for env_name in "${ENVS[@]}"; do
  if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    log_warn "Conda env already exists: $env_name"
  else
    log_step "Creating conda env: $env_name"
    conda create -y -n "$env_name" "python=$PY_VER" pip -c conda-forge -c pytorch
  fi

  conda activate "$env_name"

  log_step "Installing common build/runtime packages in $env_name"
  conda install -y -c conda-forge libstdcxx-ng=12

  log_step "Installing cudnn=8.9.2 in $env_name"
  conda install -c pkgs/main cudnn=8.9.2.26=cuda12_0

  if [[ "$CUDA_SHORT" == "124" ]]; then # fix this
    conda install -y -c pytorch magma-cuda124
  else
    log_warn "Skipping conda MAGMA for CUDA_SHORT=$CUDA_SHORT. Use USE_MAGMA=0 or provide MAGMA_HOME."
  fi

  python -m pip install --force-reinstall \
    "pip==24.3.1" \
    "setuptools==80.9.0" \
    "wheel==0.45.1"

  python -m pip install \
    ninja \
    cmake==3.28.3 \
    pyyaml \
    pandas \
    matplotlib \
    numpy==1.23.5 \
    pillow \
    numba \
    depyf \
    scipy \
    tqdm \
    mkl-static \
    mkl-include

  conda deactivate
done

log_ok "All conda environments are ready."