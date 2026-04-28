#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO" >&2' ERR

# -----------------------------------------------------------------------------
# GraCE OSDI'26 artifact: Docker-aware benchmark CPU isolation
#
# This script configures host-side systemd/cgroup-v2 CPU isolation for Docker.
#
# What it does:
#   1. Shows CPU/NUMA/GPU locality information.
#   2. Prompts the reviewer for benchmark CPUs and NUMA memory nodes.
#   3. Creates/configures bench.slice with benchmark CPUs.
#   4. Restricts init.scope/system.slice/user.slice away from benchmark CPUs.
#   5. Prints the exact docker run command using:
#        --cgroup-parent=bench.slice
#        --cpuset-cpus=<bench CPUs>
#        --cpuset-mems=<bench mem nodes>
#
# Why this is needed:
#   Starting docker from a shell that is already in /sys/fs/cgroup/bench does
#   not make the container inherit that cpuset. Docker containers are created
#   by dockerd/systemd, not directly as children of the invoking shell.
#
# Expected host:
#   - Linux with cgroup v2
#   - systemd
#   - Docker using the systemd cgroup driver
#
# This is a HOST script. Do not run it inside the Docker container.
# -----------------------------------------------------------------------------

ROOT_CG="/sys/fs/cgroup"
BACKUP_DIR="${BACKUP_DIR:-/run/grace_docker_cpuset_backup}"

BENCH_SLICE="${BENCH_SLICE:-bench.slice}"
BENCH_SLICE_UNIT="/run/systemd/system/${BENCH_SLICE}"

ARTIFACT_ROOT_IN_CONTAINER="${ARTIFACT_ROOT_IN_CONTAINER:-/workspace/grace-osdi-26-artifact}"
DOCKER_IMAGE="${DOCKER_IMAGE:-grace-osdi26:cuda124-prebuilt}"

# Default volume source: current directory, assumed artifact repo root.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"

TARGET_UNITS=("init.scope" "system.slice" "user.slice")

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

info() {
  echo "[INFO] $*"
}

ok() {
  echo "[OK] $*"
}

need_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    die "Run this script as root: sudo $0"
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

read_file_or_empty() {
  local f="$1"
  if [[ -f "$f" ]]; then
    cat "$f"
  fi
}

python_cpulist_op() {
  python3 - "$@" <<'PY'
import sys

def expand_list(s):
    out = set()
    s = (s or "").strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out

def compact(vals):
    vals = sorted(set(vals))
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

op = sys.argv[1]

if op == "subtract":
    all_cpus = expand_list(sys.argv[2])
    bench_cpus = expand_list(sys.argv[3])
    print(compact(all_cpus - bench_cpus))

elif op == "validate_subset":
    all_cpus = expand_list(sys.argv[2])
    bench_cpus = expand_list(sys.argv[3])
    bad = sorted(bench_cpus - all_cpus)
    if bad:
        print("BAD:" + compact(bad))
        sys.exit(2)
    print("OK")

elif op == "compact":
    print(compact(expand_list(sys.argv[2])))

elif op == "count":
    print(len(expand_list(sys.argv[2])))

else:
    raise SystemExit(f"unknown op: {op}")
PY
}

infer_mems_from_cpus() {
  local cpus="$1"
  python3 - "$cpus" <<'PY'
import subprocess
import sys

def expand_list(s):
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
    return out

def compact(vals):
    vals = sorted(set(vals))
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

selected = expand_list(sys.argv[1])
nodes = set()

try:
    out = subprocess.check_output(
        ["lscpu", "--all", "--parse=CPU,NODE,ONLINE"],
        text=True,
    )
    for line in out.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu_s, node_s, online_s = line.split(",")[:3]
        if online_s.strip().upper() != "Y":
            continue
        cpu = int(cpu_s)
        node = int(node_s)
        if cpu in selected:
            nodes.add(node)
except Exception:
    pass

print(compact(nodes) if nodes else "0")
PY
}

print_cpu_numa_summary() {
  echo
  echo "============================================================"
  echo "CPU / NUMA topology"
  echo "============================================================"

  echo
  echo "Online CPUs:"
  echo "  $(cat /sys/devices/system/cpu/online)"

  if [[ -f /sys/devices/system/node/online ]]; then
    echo "Online NUMA nodes:"
    echo "  $(cat /sys/devices/system/node/online)"
  fi

  echo
  echo "CPU list by NUMA node:"
  python3 <<'PY'
import subprocess
from collections import defaultdict

nodes = defaultdict(list)

try:
    out = subprocess.check_output(
        ["lscpu", "--all", "--parse=CPU,NODE,SOCKET,CORE,ONLINE"],
        text=True,
    )
    for line in out.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu_s, node_s, socket_s, core_s, online_s = line.split(",")[:5]
        if online_s.strip().upper() != "Y":
            continue
        nodes[int(node_s)].append(int(cpu_s))
except Exception as e:
    print(f"  Could not parse lscpu topology: {e}")
    raise SystemExit(0)

def compact(vals):
    vals = sorted(set(vals))
    if not vals:
        return ""
    out = []
    s = p = vals[0]
    for x in vals[1:]:
        if x == p + 1:
            p = x
        else:
            out.append((s, p))
            s = p = x
    out.append((s, p))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)

for node in sorted(nodes):
    print(f"  NUMA node {node}: {compact(nodes[node])}")
PY

  if command -v numactl >/dev/null 2>&1; then
    echo
    echo "numactl --hardware:"
    numactl --hardware || true
  fi
}

normalize_nvidia_bus_id() {
  local raw="$1"
  # nvidia-smi may print 00000000:17:00.0, while sysfs usually uses 0000:17:00.0.
  python3 - "$raw" <<'PY'
import sys
s = sys.argv[1].strip()
parts = s.split(":")
if len(parts) == 3 and len(parts[0]) == 8:
    s = parts[0][-4:] + ":" + parts[1] + ":" + parts[2]
print(s.lower())
PY
}

print_gpu_locality_summary() {
  echo
  echo "============================================================"
  echo "GPU locality"
  echo "============================================================"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found; skipping GPU locality summary."
    return
  fi

  echo
  echo "nvidia-smi topo -m:"
  nvidia-smi topo -m || true

  echo
  echo "Per-GPU PCI locality from sysfs:"
  local line idx name bus sys_bus sys_path numa local_cpus
  while IFS=',' read -r idx name bus; do
    idx="$(echo "$idx" | xargs)"
    name="$(echo "$name" | xargs)"
    bus="$(echo "$bus" | xargs)"
    sys_bus="$(normalize_nvidia_bus_id "$bus")"
    sys_path="/sys/bus/pci/devices/${sys_bus}"

    numa="$(read_file_or_empty "${sys_path}/numa_node")"
    local_cpus="$(read_file_or_empty "${sys_path}/local_cpulist")"

    [[ -z "$numa" ]] && numa="unknown"
    [[ -z "$local_cpus" ]] && local_cpus="unknown"

    echo "  GPU ${idx}: ${name}"
    echo "    PCI bus id     : ${bus}"
    echo "    sysfs device   : ${sys_path}"
    echo "    NUMA node      : ${numa}"
    echo "    local_cpulist  : ${local_cpus}"
  done < <(nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader 2>/dev/null || true)
}

docker_cgroup_driver() {
  docker info --format '{{.CgroupDriver}}' 2>/dev/null || true
}

docker_cgroup_version() {
  docker info --format '{{.CgroupVersion}}' 2>/dev/null || true
}

create_runtime_bench_slice_unit() {
  mkdir -p /run/systemd/system

  if [[ ! -f "$BENCH_SLICE_UNIT" ]]; then
    cat > "$BENCH_SLICE_UNIT" <<EOF
[Unit]
Description=GraCE benchmark Docker slice

[Slice]
EOF
    systemctl daemon-reload
  fi

  systemctl start "$BENCH_SLICE" || true
}

set_unit_allowed() {
  local unit="$1"
  local cpus="$2"
  local mems="$3"

  info "Setting ${unit}: AllowedCPUs=${cpus}, AllowedMemoryNodes=${mems}"
  systemctl set-property --runtime "$unit" \
    "AllowedCPUs=${cpus}" \
    "AllowedMemoryNodes=${mems}" >/dev/null
}

show_unit_allowed() {
  local unit="$1"
  echo
  echo "${unit}:"
  systemctl show "$unit" -p AllowedCPUs -p AllowedMemoryNodes || true
}

write_backup() {
  local all_cpus="$1"
  local all_mems="$2"
  local bench_cpus="$3"
  local bench_mems="$4"
  local system_cpus="$5"
  local system_mems="$6"
  local driver="$7"
  local cgv="$8"

  mkdir -p "$BACKUP_DIR"

  cat > "${BACKUP_DIR}/state.env" <<EOF
ALL_CPUS='${all_cpus}'
ALL_MEMS='${all_mems}'
BENCH_CPUS='${bench_cpus}'
BENCH_MEMS='${bench_mems}'
SYSTEM_CPUS='${system_cpus}'
SYSTEM_MEMS='${system_mems}'
BENCH_SLICE='${BENCH_SLICE}'
DOCKER_CGROUP_DRIVER='${driver}'
DOCKER_CGROUP_VERSION='${cgv}'
TIMESTAMP='$(date --iso-8601=seconds)'
EOF

  for unit in "$BENCH_SLICE" "${TARGET_UNITS[@]}"; do
    systemctl show "$unit" -p AllowedCPUs -p AllowedMemoryNodes \
      > "${BACKUP_DIR}/${unit}.show" 2>/dev/null || true
  done
}

write_repo_docker_cpuset_env() {
  local bench_cpus="$1"
  local bench_mems="$2"
  local system_cpus="$3"
  local system_mems="$4"
  local driver="$5"
  local cgv="$6"

  local generated_dir="${REPO_ROOT}/docker/.generated"
  local env_file="${generated_dir}/docker_cpuset.env"

  mkdir -p "${generated_dir}"

  local cgroup_parent
  if [[ "${driver}" == "systemd" ]]; then
    cgroup_parent="${BENCH_SLICE}"
  else
    cgroup_parent="/bench"
  fi

  cat > "${env_file}" <<EOF
# Auto-generated by host_tuning/cpuset/setup_docker_bench_cpuset_full.sh
# Do not edit by hand unless you know what you are doing.
#
# This file lets docker/run_cuda*.sh automatically launch containers inside
# the host benchmark cpuset configured for GraCE artifact evaluation.

USE_CPUSET=1
CGROUP_PARENT='${cgroup_parent}'
BENCH_CPUS='${bench_cpus}'
BENCH_MEMS='${bench_mems}'
SYSTEM_CPUS='${system_cpus}'
SYSTEM_MEMS='${system_mems}'
DOCKER_CGROUP_DRIVER='${driver}'
DOCKER_CGROUP_VERSION='${cgv}'
BENCH_SLICE='${BENCH_SLICE}'
ARTIFACT_ROOT_IN_CONTAINER='${ARTIFACT_ROOT_IN_CONTAINER}'
GENERATED_AT='$(date --iso-8601=seconds)'
EOF

  chmod 0644 "${env_file}"

  echo
  echo "Saved Docker cpuset environment to:"
  echo "  ${env_file}"
}

print_docker_command() {
  local bench_cpus="$1"
  local bench_mems="$2"
  local driver="$3"

  local cgroup_parent
  if [[ "$driver" == "systemd" ]]; then
    cgroup_parent="$BENCH_SLICE"
  else
    cgroup_parent="/bench"
  fi

  mkdir -p "$REPO_ROOT/results" "$REPO_ROOT/figures" "$REPO_ROOT/tables"

  cat > "${BACKUP_DIR}/docker_run_grace.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

docker run --rm -it \\
  --gpus all \\
  --ipc=host \\
  --cgroup-parent=${cgroup_parent} \\
  --cpuset-cpus=${bench_cpus} \\
  --cpuset-mems=${bench_mems} \\
  --cap-add=SYS_NICE \\
  --security-opt seccomp=unconfined \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  -e USE_NUMACTL=0 \\
  -v "\$(pwd)/results:${ARTIFACT_ROOT_IN_CONTAINER}/results" \\
  -v "\$(pwd)/figures:${ARTIFACT_ROOT_IN_CONTAINER}/figures" \\
  -v "\$(pwd)/tables:${ARTIFACT_ROOT_IN_CONTAINER}/tables" \\
  ${DOCKER_IMAGE}
EOF

  chmod +x "${BACKUP_DIR}/docker_run_grace.sh"

  echo
  echo "============================================================"
  echo "Docker command"
  echo "============================================================"
  echo
  echo "Run this from the artifact repository root:"
  echo
  cat "${BACKUP_DIR}/docker_run_grace.sh"
  echo
  echo "The command has also been saved to:"
  echo "  ${BACKUP_DIR}/docker_run_grace.sh"
  echo
  echo "Alternatively, you can also use docker/run_cuda*.sh scripts which" 
  echo "automatically source the generated environment file to run the "
  echo "container with the correct cpuset settings."
}

main() {
  need_root
  require_cmd python3
  require_cmd systemctl
  require_cmd lscpu
  require_cmd docker

  [[ -d "$ROOT_CG" ]] || die "${ROOT_CG} does not exist."
  [[ -f "${ROOT_CG}/cgroup.controllers" ]] || die "cgroup v2 root not detected at ${ROOT_CG}."
  grep -qw cpuset "${ROOT_CG}/cgroup.controllers" || die "cpuset controller not available in cgroup v2."

  local driver cgv
  driver="$(docker_cgroup_driver)"
  cgv="$(docker_cgroup_version)"

  echo "============================================================"
  echo "Docker / cgroup configuration"
  echo "============================================================"
  echo "Docker cgroup driver : ${driver:-unknown}"
  echo "Docker cgroup version: ${cgv:-unknown}"

  if [[ "$driver" != "systemd" ]]; then
    die "This script targets Docker with the systemd cgroup driver. Current driver: ${driver:-unknown}"
  fi

  if [[ "$cgv" != "2" ]]; then
    die "This script targets cgroup v2. Current Docker cgroup version: ${cgv:-unknown}"
  fi

  print_cpu_numa_summary
  print_gpu_locality_summary

  local all_cpus all_mems
  all_cpus="$(cat /sys/devices/system/cpu/online)"
  all_mems="$(cat /sys/devices/system/node/online 2>/dev/null || echo 0)"

  echo
  echo "============================================================"
  echo "Benchmark CPU selection"
  echo "============================================================"
  echo "Online CPUs       : ${all_cpus}"
  echo "Online NUMA nodes : ${all_mems}"
  echo
  echo "Choose CPUs to reserve for the Docker benchmark container."
  echo "Example on our development H100 NVL machine with HT enabled:"
  echo "  8-31,64-95"
  echo
  echo "Example with HT disabled:"
  echo "  8-31"
  echo

  local bench_cpus
  read -r -p "Benchmark CPUs to isolate: " bench_cpus
  bench_cpus="$(python_cpulist_op compact "$bench_cpus")"

  [[ -n "$bench_cpus" ]] || die "No benchmark CPUs selected."

  local validation
  validation="$(python_cpulist_op validate_subset "$all_cpus" "$bench_cpus" || true)"
  if [[ "$validation" == BAD:* ]]; then
    die "Selected benchmark CPUs are not online: ${validation#BAD:}"
  fi

  local system_cpus
  system_cpus="$(python_cpulist_op subtract "$all_cpus" "$bench_cpus")"
  [[ -n "$system_cpus" ]] || die "System CPU set would be empty. Choose fewer benchmark CPUs."

  local default_bench_mems bench_mems
  default_bench_mems="$(infer_mems_from_cpus "$bench_cpus")"

  echo
  echo "Inferred NUMA memory nodes from selected CPUs: ${default_bench_mems}"
  read -r -p "NUMA memory nodes for benchmark container [${default_bench_mems}]: " bench_mems
  bench_mems="${bench_mems:-$default_bench_mems}"

  local system_mems
  system_mems="$all_mems"

  echo
  echo "============================================================"
  echo "Planned cgroup isolation"
  echo "============================================================"
  echo "Benchmark slice : ${BENCH_SLICE}"
  echo "Benchmark CPUs  : ${bench_cpus}"
  echo "Benchmark MEMs  : ${bench_mems}"
  echo "System CPUs     : ${system_cpus}"
  echo "System MEMs     : ${system_mems}"
  echo
  echo "The following host units will be restricted away from benchmark CPUs:"
  printf '  %s\n' "${TARGET_UNITS[@]}"
  echo
  echo "This changes HOST scheduling policy until restore is run."
  echo

  if [[ "${ASSUME_YES:-0}" != "1" ]]; then
    local ans
    read -r -p "Type YES to apply host cpuset isolation: " ans
    [[ "$ans" == "YES" ]] || die "Aborted by user."
  fi

  write_backup "$all_cpus" "$all_mems" "$bench_cpus" "$bench_mems" "$system_cpus" "$system_mems" "$driver" "$cgv"

  write_repo_docker_cpuset_env \
    "$bench_cpus" \
    "$bench_mems" \
    "$system_cpus" \
    "$system_mems" \
    "$driver" \
    "$cgv"

  # Create the benchmark slice first, then move normal system/user work away.
  set_unit_allowed "$BENCH_SLICE" "$bench_cpus" "$bench_mems"

  for unit in "${TARGET_UNITS[@]}"; do
    set_unit_allowed "$unit" "$system_cpus" "$system_mems"
  done

  echo
  echo "============================================================"
  echo "Verification"
  echo "============================================================"
  show_unit_allowed "$BENCH_SLICE"
  for unit in "${TARGET_UNITS[@]}"; do
    show_unit_allowed "$unit"
  done

  print_docker_command "$bench_cpus" "$bench_mems" "$driver"

  echo
  ok "Docker benchmark cpuset isolation configured."
  echo
  echo "After experiments, restore the host with:"
  echo "  sudo host_tuning/cpuset/restore_docker_bench_cpuset.sh"
}

main "$@"