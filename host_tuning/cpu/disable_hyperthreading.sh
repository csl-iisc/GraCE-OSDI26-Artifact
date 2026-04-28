#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

trap 'log_err "$0 failed at line $LINENO"' ERR

require_root
require_cmd python3

ensure_backup_dir

BACKUP_FILE="${GRACE_TUNING_BACKUP_DIR}/ht_disabled_cpus.txt"
HT_CPUS="${HT_CPUS:-}"

detect_ht_siblings() {
  python3 <<'PY'
from pathlib import Path

def expand_cpu_list(s):
    out = set()
    for part in s.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)

siblings_to_disable = set()

for cpu_path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
    cpu = int(cpu_path.name[3:])
    online_file = cpu_path / "online"
    if online_file.exists():
        try:
            if online_file.read_text().strip() != "1":
                continue
        except Exception:
            continue

    sib_file = cpu_path / "topology" / "thread_siblings_list"
    if not sib_file.exists():
        continue

    siblings = expand_cpu_list(sib_file.read_text().strip())
    if len(siblings) <= 1:
        continue

    primary = min(siblings)
    for s in siblings:
        if s != primary:
            siblings_to_disable.add(s)

print(",".join(map(str, sorted(siblings_to_disable))))
PY
}

if [[ -z "${HT_CPUS}" ]]; then
  HT_CPUS="$(detect_ht_siblings)"
fi

HT_CPUS="$(compact_cpu_list "${HT_CPUS}")"

if [[ -z "${HT_CPUS}" ]]; then
  log_warn "No hyperthread sibling CPUs detected or specified."
  exit 0
fi

log_info "Hyperthread sibling CPUs selected for offlining: ${HT_CPUS}"
echo "${HT_CPUS}" > "${BACKUP_FILE}"

for cpu in $(expand_cpu_list "${HT_CPUS}"); do
  online_file="/sys/devices/system/cpu/cpu${cpu}/online"
  if [[ ! -f "${online_file}" ]]; then
    log_warn "CPU ${cpu} does not have an online file; skipping"
    continue
  fi

  current="$(cat "${online_file}")"
  if [[ "${current}" == "0" ]]; then
    log_info "CPU ${cpu} already offline"
    continue
  fi

  log_info "Offlining CPU ${cpu}"
  echo 0 > "${online_file}"
done

echo
log_info "Online CPUs after disabling HT:"
cat /sys/devices/system/cpu/online

log_ok "Hyperthreading sibling CPUs disabled."