#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root

BACKUP_FILE="${GRACE_TUNING_BACKUP_DIR}/c6_state_backup.tsv"

if [[ -f "${BACKUP_FILE}" ]]; then
  log_info "Restoring C6 cpuidle state from ${BACKUP_FILE}"

  while IFS=$'\t' read -r disable_file old_value state_name; do
    [[ -n "${disable_file}" ]] || continue
    if [[ -f "${disable_file}" ]]; then
      log_info "Restoring ${state_name}: ${disable_file}=${old_value}"
      echo "${old_value}" > "${disable_file}" || true
    fi
  done < "${BACKUP_FILE}"

  log_ok "C6 state restored from backup."
  exit 0
fi

log_warn "No C6 backup found. Fallback: enabling all C6 states."

for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
  for state_dir in "${cpu_dir}"/cpuidle/state*; do
    [[ -d "${state_dir}" ]] || continue
    name_file="${state_dir}/name"
    disable_file="${state_dir}/disable"
    [[ -f "${name_file}" && -f "${disable_file}" ]] || continue

    state_name="$(cat "${name_file}")"
    if [[ "${state_name}" == "C6" || "${state_name}" == *"C6"* ]]; then
      log_info "Enabling ${state_name}: ${disable_file}"
      echo 0 > "${disable_file}" || true
    fi
  done
done

log_ok "C6 enabled."