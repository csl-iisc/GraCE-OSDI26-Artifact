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
echo "GraCE host tuning: Intel + H100"
echo "============================================================"
echo
echo "This script mutates HOST state:"
echo "  - locks GPU clocks"
echo "  - pins CPU frequency"
echo "  - optionally disables C6"
echo "  - optionally disables hyperthreading"
echo "  - optionally configures Docker-aware cpuset isolation"
echo
echo "Profile:"
echo "  ${PROFILE}"
echo
echo "Current online CPUs:"
echo "  $(online_cpu_list)"
echo
echo "Current online NUMA nodes:"
echo "  $(online_numa_nodes)"
echo

confirm_or_exit "Type YES to tune this host for GraCE benchmarking: "

log_info "Recording host state before tuning"
python3 "${SCRIPT_DIR}/check_host_state.py" \
  --output "${RESULTS_HOST_STATE_DIR}/host_state_before_tune.json" || true

if [[ "${SKIP_GPU_TUNING:-0}" != "1" ]]; then
  GPU_IDS="${GPU_IDS:-all}" \
  GPU_CLOCK_MHZ="${GPU_CLOCK_MHZ:-1410}" \
  GPU_MEM_CLOCK_MHZ="${GPU_MEM_CLOCK_MHZ:-}" \
  bash "${SCRIPT_DIR}/gpu/lock_gpu_clocks.sh"
else
  log_warn "Skipping GPU tuning because SKIP_GPU_TUNING=1"
fi

if [[ "${SKIP_CPU_FREQ_TUNING:-0}" != "1" ]]; then
  INTEL_TARGET_FREQ_KHZ="${INTEL_TARGET_FREQ_KHZ:-2800000}" \
  DISABLE_TURBO_OS="${DISABLE_TURBO_OS:-1}" \
  bash "${SCRIPT_DIR}/cpu/intel_pin_cpu_freq.sh"
else
  log_warn "Skipping CPU frequency tuning because SKIP_CPU_FREQ_TUNING=1"
fi

if [[ "${DISABLE_C6:-1}" == "1" ]]; then
  bash "${SCRIPT_DIR}/cpu/disable_c6.sh"
else
  log_warn "Skipping C6 disable because DISABLE_C6=${DISABLE_C6:-0}"
fi

if [[ "${DISABLE_HT:-1}" == "1" ]]; then
  HT_CPUS="${HT_CPUS:-}" bash "${SCRIPT_DIR}/cpu/disable_hyperthreading.sh"
else
  log_warn "Keeping hyperthreading enabled because DISABLE_HT=0"
fi

if [[ "${ENABLE_DOCKER_CPUSET:-1}" == "1" ]]; then
  CPUSET_SETUP="${SCRIPT_DIR}/cpuset/setup_docker_bench_cpuset_full.sh"
  if [[ ! -x "${CPUSET_SETUP}" ]]; then
    die "Docker cpuset setup script not found/executable: ${CPUSET_SETUP}"
  fi

  log_info "Configuring Docker-aware benchmark cpuset"
  DOCKER_IMAGE="${DOCKER_IMAGE:-grace-osdi26:cuda124-prebuilt}" \
  ARTIFACT_ROOT_IN_CONTAINER="${ARTIFACT_ROOT_IN_CONTAINER:-/workspace/grace-osdi-26-artifact}" \
  REPO_ROOT="${GRACE_REPO_ROOT}" \
  bash "${CPUSET_SETUP}"
else
  log_warn "Skipping Docker cpuset isolation because ENABLE_DOCKER_CPUSET=0"
fi

log_info "Recording host state after tuning"
python3 "${SCRIPT_DIR}/check_host_state.py" \
  --output "${RESULTS_HOST_STATE_DIR}/host_state_after_tune.json" || true

echo
log_ok "Host tuning complete."
echo
echo "Host-state logs:"
echo "  ${RESULTS_HOST_STATE_DIR}/host_state_before_tune.json"
echo "  ${RESULTS_HOST_STATE_DIR}/host_state_after_tune.json"
echo
echo "After experiments, restore with:"
echo "  sudo ${SCRIPT_DIR}/restore_host_intel_h100.sh"