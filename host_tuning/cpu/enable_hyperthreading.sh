#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root
require_cmd python3

BACKUP_FILE="${GRACE_TUNING_BACKUP_DIR}/ht_disabled_cpus.txt"
HT_CPUS="${HT_CPUS:-}"

if [[ -f "${BACKUP_FILE}" ]]; then
  HT_CPUS="$(cat "${BACKUP_FILE}")"
fi

if [[ -z "${HT_CPUS}" ]]; then
  log_warn "No HT backup found and HT_CPUS not specified. Enabling all offline CPUs with an online file."
  HT_CPUS="$(
    python3 <<'PY'
from pathlib import Path
cpus = []
for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/online"):
    cpu = int(p.parent.name[3:])
    try:
        if p.read_text().strip() == "0":
            cpus.append(cpu)
    except Exception:
        pass
print(",".join(map(str, sorted(cpus))))
PY
  )"
fi

HT_CPUS="$(compact_cpu_list "${HT_CPUS}")"

if [[ -z "${HT_CPUS}" ]]; then
  log_info "No CPUs to online."
  exit 0
fi

log_info "Re-enabling CPUs: ${HT_CPUS}"

for cpu in $(expand_cpu_list "${HT_CPUS}"); do
  online_file="/sys/devices/system/cpu/cpu${cpu}/online"
  if [[ ! -f "${online_file}" ]]; then
    log_warn "CPU ${cpu} does not have an online file; skipping"
    continue
  fi

  current="$(cat "${online_file}")"
  if [[ "${current}" == "1" ]]; then
    log_info "CPU ${cpu} already online"
    continue
  fi

  log_info "Onlining CPU ${cpu}"
  echo 1 > "${online_file}"
done

echo
log_info "Online CPUs after enabling HT:"
cat /sys/devices/system/cpu/online

log_ok "Hyperthreading sibling CPUs enabled."