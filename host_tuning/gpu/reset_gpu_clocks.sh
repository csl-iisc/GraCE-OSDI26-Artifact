#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root
require_cmd nvidia-smi

GPU_IDS="${GPU_IDS:-all}"
RESET_PERSISTENCE="${RESET_PERSISTENCE:-0}"

query_gpu_ids() {
  if [[ "${GPU_IDS}" == "all" ]]; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ' '
  else
    echo "${GPU_IDS}" | tr ',' ' '
  fi
}

for gpu in $(query_gpu_ids); do
  log_info "Resetting GPU ${gpu} graphics clock lock"
  nvidia-smi -i "${gpu}" -rgc || true

  log_info "Resetting GPU ${gpu} memory clock lock"
  nvidia-smi -i "${gpu}" -rmc || true
done

if [[ "${RESET_PERSISTENCE}" == "1" ]]; then
  log_info "Disabling NVIDIA persistence mode"
  nvidia-smi -pm 0 || true
fi

echo
log_info "GPU clock state after reset:"
nvidia-smi --query-gpu=index,name,persistence_mode,clocks.gr,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv || true

log_ok "GPU clocks reset."