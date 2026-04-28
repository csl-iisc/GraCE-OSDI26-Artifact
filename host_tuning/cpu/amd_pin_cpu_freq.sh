#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root

TARGET_FREQ_KHZ="${AMD_TARGET_FREQ_KHZ:-3100000}"

ensure_backup_dir
BACKUP_FILE="${GRACE_TUNING_BACKUP_DIR}/cpu_freq_backup.tsv"

log_info "Target AMD CPU frequency: ${TARGET_FREQ_KHZ} KHz"

log_info "Backing up current CPU frequency policy to ${BACKUP_FILE}"
: > "${BACKUP_FILE}"

for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
  cpu_id="$(basename "${cpu_dir}" | sed 's/cpu//')"
  freq_dir="${cpu_dir}/cpufreq"
  [[ -d "${freq_dir}" ]] || continue

  governor="$(read_file_or_empty "${freq_dir}/scaling_governor")"
  min_freq="$(read_file_or_empty "${freq_dir}/scaling_min_freq")"
  max_freq="$(read_file_or_empty "${freq_dir}/scaling_max_freq")"

  printf "%s\t%s\t%s\t%s\n" "${cpu_id}" "${governor}" "${min_freq}" "${max_freq}" >> "${BACKUP_FILE}"
done

if command -v cpupower >/dev/null 2>&1; then
  log_info "Setting CPU governor to performance"
  cpupower frequency-set -g performance >/dev/null || true
fi

for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
  cpu_id="$(basename "${cpu_dir}" | sed 's/cpu//')"
  freq_dir="${cpu_dir}/cpufreq"
  [[ -d "${freq_dir}" ]] || continue

  log_info "Pinning cpu${cpu_id} to ${TARGET_FREQ_KHZ} KHz"

  if command -v cpufreq-set >/dev/null 2>&1; then
    cpufreq-set -c "${cpu_id}" -g performance >/dev/null 2>&1 || true
    cpufreq-set -c "${cpu_id}" -f "${TARGET_FREQ_KHZ}" >/dev/null
  else
    [[ -f "${freq_dir}/scaling_governor" ]] && echo performance > "${freq_dir}/scaling_governor" || true
    [[ -f "${freq_dir}/scaling_min_freq" ]] && echo "${TARGET_FREQ_KHZ}" > "${freq_dir}/scaling_min_freq" || true
    [[ -f "${freq_dir}/scaling_max_freq" ]] && echo "${TARGET_FREQ_KHZ}" > "${freq_dir}/scaling_max_freq" || true
  fi
done

log_ok "AMD CPU frequency pinned."



# For restore on AMD, use the same restore script:

# sudo host_tuning/cpu/intel_restore_cpu_freq.sh