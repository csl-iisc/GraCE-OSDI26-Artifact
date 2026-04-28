#!/usr/bin/env bash

# Common helpers for GraCE host tuning scripts.
# This file is sourced by other scripts; do not execute directly.

set -euo pipefail

GRACE_HOST_TUNING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRACE_REPO_ROOT="$(cd -- "${GRACE_HOST_TUNING_DIR}/.." && pwd)"

GRACE_TUNING_BACKUP_DIR="${GRACE_TUNING_BACKUP_DIR:-/run/grace_host_tuning_backup}"

log_info() {
  echo "[INFO] $*"
}

log_ok() {
  echo "[OK] $*"
}

log_warn() {
  echo "[WARN] $*" >&2
}

log_err() {
  echo "[ERROR] $*" >&2
}

die() {
  log_err "$*"
  exit 1
}

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    die "This script must be run as root. Use sudo."
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

maybe_require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_warn "Optional command not found: $1"
    return 1
  fi
  return 0
}

ensure_backup_dir() {
  mkdir -p "${GRACE_TUNING_BACKUP_DIR}"
}

read_file_or_empty() {
  local f="$1"
  if [[ -f "${f}" ]]; then
    cat "${f}"
  fi
}

write_file_checked() {
  local f="$1"
  local value="$2"
  if [[ ! -e "${f}" ]]; then
    log_warn "File does not exist, skipping: ${f}"
    return 0
  fi
  echo "${value}" > "${f}"
}

load_profile() {
  local profile="${1:-}"
  if [[ -z "${profile}" ]]; then
    return 0
  fi
  if [[ ! -f "${profile}" ]]; then
    die "Profile not found: ${profile}"
  fi
  log_info "Loading host tuning profile: ${profile}"
  # shellcheck disable=SC1090
  source "${profile}"
}

expand_cpu_list() {
  python3 - "$1" <<'PY'
import sys

def expand(s):
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

print(" ".join(map(str, expand(sys.argv[1]))))
PY
}

compact_cpu_list() {
  python3 - "$1" <<'PY'
import sys

def expand(s):
    out = set()
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)

def compact(vals):
    if not vals:
        return ""
    ranges = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)

print(compact(expand(sys.argv[1])))
PY
}

online_cpu_list() {
  cat /sys/devices/system/cpu/online
}

online_numa_nodes() {
  if [[ -f /sys/devices/system/node/online ]]; then
    cat /sys/devices/system/node/online
  else
    echo "0"
  fi
}

confirm_or_exit() {
  local prompt="${1:-Type YES to continue: }"
  if [[ "${ASSUME_YES:-0}" == "1" ]]; then
    return 0
  fi
  local ans
  read -r -p "${prompt}" ans
  if [[ "${ans}" != "YES" ]]; then
    die "Aborted."
  fi
}