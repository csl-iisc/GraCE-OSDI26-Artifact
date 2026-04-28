#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root

PROFILE="${HOST_TUNING_PROFILE:-${SCRIPT_DIR}/profiles/example_intel_h100.env}"
load_profile "${PROFILE}"

RESULTS_HOST_STATE_DIR="${RESULTS_HOST_STATE_DIR:-${GRACE_REPO_ROOT}/results/host_state}"
mkdir -p "${RESULTS_HOST_STATE_DIR}"

echo "============================================================"
echo "GraCE host tuning restore: Intel + H100"
echo "============================================================"
echo
echo "This script restores HOST state after benchmarking."
echo

confirm_or_exit "Type YES to restore host power/cpuset settings: "

# Stop/release Docker cpuset first if present. It may restore to the CPU set
# that was online at setup time. We later widen systemd CPU access again after
# re-enabling HT to handle HT-disabled tuning flows.
if [[ "${ENABLE_DOCKER_CPUSET:-1}" == "1" ]]; then
  CPUSET_RESTORE="${SCRIPT_DIR}/cpuset/restore_docker_bench_cpuset.sh"
  if [[ -x "${CPUSET_RESTORE}" ]]; then
    log_info "Restoring Docker-aware benchmark cpuset"
    bash "${CPUSET_RESTORE}" || true
  else
    log_warn "Docker cpuset restore script not found/executable: ${CPUSET_RESTORE}"
  fi
fi

if [[ "${DISABLE_HT:-1}" == "1" ]]; then
  HT_CPUS="${HT_CPUS:-}" bash "${SCRIPT_DIR}/cpu/enable_hyperthreading.sh" || true
fi

if [[ "${DISABLE_C6:-1}" == "1" ]]; then
  bash "${SCRIPT_DIR}/cpu/enable_c6.sh" || true
fi

if [[ "${SKIP_CPU_FREQ_TUNING:-0}" != "1" ]]; then
  bash "${SCRIPT_DIR}/cpu/intel_restore_cpu_freq.sh" || true
fi

if [[ "${SKIP_GPU_TUNING:-0}" != "1" ]]; then
  GPU_IDS="${GPU_IDS:-all}" bash "${SCRIPT_DIR}/gpu/reset_gpu_clocks.sh" || true
fi

# Important: if cpuset setup was done after HT had been disabled, its backup may
# only include the then-online CPUs. After enabling HT, explicitly restore broad
# CPU access to all currently online CPUs for normal host units.
if command -v systemctl >/dev/null 2>&1; then
  ALL_CPUS="$(online_cpu_list)"
  ALL_MEMS="$(online_numa_nodes)"
  log_info "Restoring broad systemd CPU access after HT restore: CPUs=${ALL_CPUS}, MEMs=${ALL_MEMS}"

  for unit in init.scope system.slice user.slice bench.slice; do
    systemctl set-property --runtime "${unit}" \
      "AllowedCPUs=${ALL_CPUS}" \
      "AllowedMemoryNodes=${ALL_MEMS}" >/dev/null 2>&1 || true
  done
fi

log_info "Recording host state after restore"
python3 "${SCRIPT_DIR}/check_host_state.py" \
  --output "${RESULTS_HOST_STATE_DIR}/host_state_after_restore.json" || true

echo
log_ok "Host tuning restored."
echo
echo "Host-state log:"
echo "  ${RESULTS_HOST_STATE_DIR}/host_state_after_restore.json"