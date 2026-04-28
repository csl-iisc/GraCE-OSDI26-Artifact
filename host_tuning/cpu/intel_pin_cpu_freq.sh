#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root
require_cmd python3

TARGET_FREQ_KHZ="${INTEL_TARGET_FREQ_KHZ:-2800000}"
DISABLE_TURBO_OS="${DISABLE_TURBO_OS:-0}"

ensure_backup_dir

BACKUP_FILE="${GRACE_TUNING_BACKUP_DIR}/cpu_freq_backup.tsv"
INTEL_PSTATE_BACKUP="${GRACE_TUNING_BACKUP_DIR}/intel_pstate.env"

log_info "Target Intel CPU frequency: ${TARGET_FREQ_KHZ} KHz"

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

{
  echo "TIMESTAMP='$(date --iso-8601=seconds)'"
  echo "INTEL_PSTATE_STATUS='$(read_file_or_empty /sys/devices/system/cpu/intel_pstate/status)'"
  echo "INTEL_NO_TURBO='$(read_file_or_empty /sys/devices/system/cpu/intel_pstate/no_turbo)'"
} > "${INTEL_PSTATE_BACKUP}"

if [[ -f /sys/devices/system/cpu/intel_pstate/status ]]; then
  current_status="$(cat /sys/devices/system/cpu/intel_pstate/status)"
  if [[ "${current_status}" != "passive" ]]; then
    log_info "Switching intel_pstate to passive mode"
    echo passive > /sys/devices/system/cpu/intel_pstate/status
  fi
fi

if [[ "${DISABLE_TURBO_OS}" == "1" && -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
  log_info "Disabling Intel turbo at OS level"
  echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
fi

if command -v cpupower >/dev/null 2>&1; then
  log_info "Setting CPU governor to performance via cpupower"
  cpupower frequency-set -g performance >/dev/null || true
else
  log_warn "cpupower not found; falling back to direct sysfs/cpufreq-set where possible"
fi

for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
  cpu_id="$(basename "${cpu_dir}" | sed 's/cpu//')"
  freq_dir="${cpu_dir}/cpufreq"
  [[ -d "${freq_dir}" ]] || continue

  log_info "Pinning cpu${cpu_id} min=max=${TARGET_FREQ_KHZ} KHz"

  if command -v cpufreq-set >/dev/null 2>&1; then
    cpufreq-set -c "${cpu_id}" -g performance >/dev/null 2>&1 || true
    cpufreq-set -c "${cpu_id}" -d "${TARGET_FREQ_KHZ}" -u "${TARGET_FREQ_KHZ}" >/dev/null
  else
    [[ -f "${freq_dir}/scaling_governor" ]] && echo performance > "${freq_dir}/scaling_governor" || true
    [[ -f "${freq_dir}/scaling_min_freq" ]] && echo "${TARGET_FREQ_KHZ}" > "${freq_dir}/scaling_min_freq" || true
    [[ -f "${freq_dir}/scaling_max_freq" ]] && echo "${TARGET_FREQ_KHZ}" > "${freq_dir}/scaling_max_freq" || true
  fi
done

echo
log_info "Frequency summary:"
if command -v cpupower >/dev/null 2>&1; then
  cpupower frequency-info | grep -E "driver:|governor|current CPU frequency|hardware limits" || true
else
  python3 - <<'PY'
from pathlib import Path
vals = []
for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
    try:
        vals.append(int(p.read_text().strip()))
    except Exception:
        pass
if vals:
    print(f"cur_freq_khz_min={min(vals)} max={max(vals)} unique_count={len(set(vals))}")
PY
fi

log_ok "Intel CPU frequency pinned."