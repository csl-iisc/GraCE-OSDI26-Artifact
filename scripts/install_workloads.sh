#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

WORKLOAD_FILE="${WORKLOAD_FILE:-$ARTIFACT_ROOT/configs/workloads_single_gpu_25_with_acronym.txt}"
ENVS_TO_INSTALL="${ENVS_TO_INSTALL:-grace-vanilla grace-cgct grace-cgct-pi grace-full grace-cgct-copy-debug grace-cgct-pi-copy-debug}"
INSTALL_CANARY="${INSTALL_CANARY:-1}"

if [[ ! -f "$WORKLOAD_FILE" ]]; then
  log_err "Workload file not found: $WORKLOAD_FILE"
  exit 1
fi

mapfile -t TORCHBENCH_MODELS < <(
  awk '
    NF == 4 && $2 == "torchbench" { print $1 }
    NF >= 5 && $3 == "torchbench" { print $2 }
  ' "$WORKLOAD_FILE" | sort -u
)

log_step "TorchBench models to install:"
printf '  %s\n' "${TORCHBENCH_MODELS[@]}"

for env_name in $ENVS_TO_INSTALL; do
  activate_env "$env_name"

  log_step "Installing TorchBench workloads in env=$env_name"
  cd "$TORCHBENCH_DIR"

  for model in "${TORCHBENCH_MODELS[@]}"; do
    if [[ "$model" == "DALLE2_pytorch" ]]; then
      if [[ "$INSTALL_CANARY" == "1" ]]; then
        log_step "Installing canary TorchBench model: $model"
        python3 install.py --canary model DALLE2_pytorch || python3 install.py --canary DALLE2_pytorch
      else
        log_warn "Skipping canary model DALLE2_pytorch because INSTALL_CANARY=0"
      fi
    else
      log_step "Installing TorchBench model: $model"
      python3 install.py "$model"
    fi
  done

  pip install -e .

done

log_ok "TorchBench workload installation complete."