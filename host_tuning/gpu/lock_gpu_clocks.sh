#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root
require_cmd nvidia-smi

GPU_IDS="${GPU_IDS:-all}"
GPU_CLOCK_MHZ="${GPU_CLOCK_MHZ:-1410}"
GPU_MEM_CLOCK_MHZ="${GPU_MEM_CLOCK_MHZ:-}"

ensure_backup_dir

BACKUP_FILE="${GRACE_TUNING_BACKUP_DIR}/gpu_clocks.env"

query_gpu_ids() {
  if [[ "${GPU_IDS}" == "all" ]]; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ' '
  else
    echo "${GPU_IDS}" | tr ',' ' '
  fi
}

log_info "Saving GPU state to ${BACKUP_FILE}"
{
  echo "GPU_IDS='${GPU_IDS}'"
  echo "GPU_CLOCK_MHZ='${GPU_CLOCK_MHZ}'"
  echo "GPU_MEM_CLOCK_MHZ='${GPU_MEM_CLOCK_MHZ}'"
  echo "TIMESTAMP='$(date --iso-8601=seconds)'"
  echo "NVIDIA_SMI_QUERY='$(nvidia-smi --query-gpu=index,name,persistence_mode,clocks.gr,clocks.mem --format=csv,noheader 2>/dev/null | tr '\n' ';')'"
} > "${BACKUP_FILE}"

log_info "Enabling NVIDIA persistence mode"
nvidia-smi -pm 1

for gpu in $(query_gpu_ids); do
  log_info "Locking GPU ${gpu} graphics clock to ${GPU_CLOCK_MHZ} MHz"
  nvidia-smi -i "${gpu}" -lgc "${GPU_CLOCK_MHZ}"

  if [[ -n "${GPU_MEM_CLOCK_MHZ}" ]]; then
    log_info "Locking GPU ${gpu} memory clock to ${GPU_MEM_CLOCK_MHZ} MHz"
    # This may not be supported on all GPUs/drivers. Fail loudly because the user asked for it.
    nvidia-smi -i "${gpu}" --lock-memory-clocks="${GPU_MEM_CLOCK_MHZ},${GPU_MEM_CLOCK_MHZ}"
  fi
done

echo
log_info "GPU clock state after lock:"
nvidia-smi --query-gpu=index,name,persistence_mode,clocks.gr,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv

log_ok "GPU clocks locked."