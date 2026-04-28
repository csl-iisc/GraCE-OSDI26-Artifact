#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root

ensure_backup_dir

BACKUP_FILE="${GRACE_TUNING_BACKUP_DIR}/c6_state_backup.tsv"
: > "${BACKUP_FILE}"

log_info "Disabling C6 cpuidle state on all online CPUs"

found=0

for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
  cpu_name="$(basename "${cpu_dir}")"

  for state_dir in "${cpu_dir}"/cpuidle/state*; do
    [[ -d "${state_dir}" ]] || continue
    name_file="${state_dir}/name"
    disable_file="${state_dir}/disable"
    [[ -f "${name_file}" && -f "${disable_file}" ]] || continue

    state_name="$(cat "${name_file}")"

    if [[ "${state_name}" == "C6" || "${state_name}" == *"C6"* ]]; then
      old_value="$(cat "${disable_file}")"
      printf "%s\t%s\t%s\n" "${disable_file}" "${old_value}" "${state_name}" >> "${BACKUP_FILE}"

      log_info "Disabling ${state_name} for ${cpu_name}: ${disable_file}"
      echo 1 > "${disable_file}"
      found=1
    fi
  done
done

if [[ "${found}" == "0" ]]; then
  log_warn "No C6 cpuidle states found."
else
  log_ok "C6 disabled."
fi