#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root

BACKUP_FILE="${GRACE_TUNING_BACKUP_DIR}/cpu_freq_backup.tsv"
INTEL_PSTATE_BACKUP="${GRACE_TUNING_BACKUP_DIR}/intel_pstate.env"

log_info "Restoring Intel CPU frequency policy"

if [[ -f "${INTEL_PSTATE_BACKUP}" ]]; then
  # shellcheck disable=SC1090
  source "${INTEL_PSTATE_BACKUP}"

  if [[ -f /sys/devices/system/cpu/intel_pstate/status && -n "${INTEL_PSTATE_STATUS:-}" ]]; then
    log_info "Restoring intel_pstate/status=${INTEL_PSTATE_STATUS}"
    echo "${INTEL_PSTATE_STATUS}" > /sys/devices/system/cpu/intel_pstate/status || true
  fi

  if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo && -n "${INTEL_NO_TURBO:-}" ]]; then
    log_info "Restoring intel_pstate/no_turbo=${INTEL_NO_TURBO}"
    echo "${INTEL_NO_TURBO}" > /sys/devices/system/cpu/intel_pstate/no_turbo || true
  fi
else
  log_warn "No intel_pstate backup found: ${INTEL_PSTATE_BACKUP}"
  if [[ -f /sys/devices/system/cpu/intel_pstate/status ]]; then
    log_info "Fallback: setting intel_pstate/status=active"
    echo active > /sys/devices/system/cpu/intel_pstate/status || true
  fi
fi

if [[ -f "${BACKUP_FILE}" ]]; then
  log_info "Restoring per-CPU frequency policy from ${BACKUP_FILE}"

  while IFS=$'\t' read -r cpu_id governor min_freq max_freq; do
    [[ -n "${cpu_id}" ]] || continue
    freq_dir="/sys/devices/system/cpu/cpu${cpu_id}/cpufreq"
    [[ -d "${freq_dir}" ]] || continue

    log_info "Restoring cpu${cpu_id}: governor=${governor}, min=${min_freq}, max=${max_freq}"

    if command -v cpufreq-set >/dev/null 2>&1; then
      [[ -n "${governor}" ]] && cpufreq-set -c "${cpu_id}" -g "${governor}" >/dev/null 2>&1 || true
      [[ -n "${min_freq}" && -n "${max_freq}" ]] && cpufreq-set -c "${cpu_id}" -d "${min_freq}" -u "${max_freq}" >/dev/null 2>&1 || true
    else
      [[ -f "${freq_dir}/scaling_governor" && -n "${governor}" ]] && echo "${governor}" > "${freq_dir}/scaling_governor" || true
      [[ -f "${freq_dir}/scaling_min_freq" && -n "${min_freq}" ]] && echo "${min_freq}" > "${freq_dir}/scaling_min_freq" || true
      [[ -f "${freq_dir}/scaling_max_freq" && -n "${max_freq}" ]] && echo "${max_freq}" > "${freq_dir}/scaling_max_freq" || true
    fi
  done < "${BACKUP_FILE}"
else
  log_warn "No CPU frequency backup found: ${BACKUP_FILE}"
  log_info "Fallback: setting governor=powersave and min/max to hardware limits"

  if command -v cpupower >/dev/null 2>&1; then
    cpupower frequency-set -g powersave >/dev/null 2>&1 || true
  fi

  for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
    cpu_id="$(basename "${cpu_dir}" | sed 's/cpu//')"
    freq_dir="${cpu_dir}/cpufreq"
    [[ -d "${freq_dir}" ]] || continue

    min_hw="$(read_file_or_empty "${freq_dir}/cpuinfo_min_freq")"
    max_hw="$(read_file_or_empty "${freq_dir}/cpuinfo_max_freq")"

    if command -v cpufreq-set >/dev/null 2>&1; then
      cpufreq-set -c "${cpu_id}" -g powersave >/dev/null 2>&1 || true
      [[ -n "${min_hw}" && -n "${max_hw}" ]] && cpufreq-set -c "${cpu_id}" -d "${min_hw}" -u "${max_hw}" >/dev/null 2>&1 || true
    else
      [[ -f "${freq_dir}/scaling_governor" ]] && echo powersave > "${freq_dir}/scaling_governor" || true
      [[ -f "${freq_dir}/scaling_min_freq" && -n "${min_hw}" ]] && echo "${min_hw}" > "${freq_dir}/scaling_min_freq" || true
      [[ -f "${freq_dir}/scaling_max_freq" && -n "${max_hw}" ]] && echo "${max_hw}" > "${freq_dir}/scaling_max_freq" || true
    fi
  done
fi

echo
log_info "Frequency summary after restore:"
if command -v cpupower >/dev/null 2>&1; then
  cpupower frequency-info | grep -E "driver:|governor|current CPU frequency|hardware limits" || true
fi

log_ok "CPU frequency policy restored."