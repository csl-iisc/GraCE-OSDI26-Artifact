#!/usr/bin/env python3
"""
Read-only host-state checker for GraCE OSDI'26 artifact runs.

This script does not mutate the machine. It records CPU, NUMA, GPU, cgroup,
frequency, and cpuidle state so that paper-quality runs can be audited.

Example:
  python3 host_tuning/check_host_state.py --output results/host_state_after_tune.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_text(path: str | Path) -> Optional[str]:
    p = Path(path)
    try:
        return p.read_text().strip()
    except Exception:
        return None


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    try:
        out = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": out.returncode,
            "stdout": out.stdout.strip(),
            "stderr": out.stderr.strip(),
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "error": repr(e),
        }


def compact_list(vals: List[int]) -> str:
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


def parse_cpu_list(text: str) -> List[int]:
    vals = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(range(int(a), int(b) + 1))
        else:
            vals.append(int(part))
    return sorted(set(vals))


def collect_os() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "os_release": read_text("/etc/os-release"),
        "kernel_cmdline": read_text("/proc/cmdline"),
    }


def collect_lscpu() -> Dict[str, Any]:
    info = {
        "lscpu": run_cmd(["lscpu"]),
        "online_cpus": read_text("/sys/devices/system/cpu/online"),
        "present_cpus": read_text("/sys/devices/system/cpu/present"),
        "online_numa_nodes": read_text("/sys/devices/system/node/online"),
    }

    parsed = run_cmd(["lscpu", "--all", "--parse=CPU,NODE,SOCKET,CORE,ONLINE"])
    info["lscpu_parse"] = parsed

    nodes: Dict[str, List[int]] = defaultdict(list)
    physical_like = []
    sibling_like = []

    if parsed.get("returncode") == 0:
        for line in parsed.get("stdout", "").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            cpu_s, node_s, socket_s, core_s, online_s = parts[:5]
            if online_s.strip().upper() != "Y":
                continue
            cpu = int(cpu_s)
            node = int(node_s)
            core = int(core_s)
            nodes[str(node)].append(cpu)
            if cpu == core:
                physical_like.append(cpu)
            else:
                sibling_like.append(cpu)

    info["numa_node_cpulists"] = {
        node: compact_list(cpus) for node, cpus in sorted(nodes.items(), key=lambda x: int(x[0]))
    }
    info["cpu_eq_core_cpus"] = compact_list(physical_like)
    info["cpu_ne_core_cpus"] = compact_list(sibling_like)

    return info


def collect_process_affinity() -> Dict[str, Any]:
    out = {
        "pid": os.getpid(),
    }
    try:
        aff = sorted(os.sched_getaffinity(0))
        out["sched_getaffinity"] = aff
        out["sched_getaffinity_compact"] = compact_list(aff)
    except Exception as e:
        out["sched_getaffinity_error"] = repr(e)

    status = read_text(f"/proc/{os.getpid()}/status")
    if status:
        for line in status.splitlines():
            if line.startswith("Cpus_allowed_list") or line.startswith("Mems_allowed_list"):
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()

    return out


def collect_cpufreq() -> Dict[str, Any]:
    rows = []
    governors = Counter()
    min_freqs = Counter()
    max_freqs = Counter()
    cur_freqs = []

    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        freq_dir = cpu_dir / "cpufreq"
        if not freq_dir.is_dir():
            continue

        row = {
            "cpu": int(cpu_dir.name[3:]),
            "scaling_driver": read_text(freq_dir / "scaling_driver"),
            "scaling_governor": read_text(freq_dir / "scaling_governor"),
            "scaling_min_freq": read_text(freq_dir / "scaling_min_freq"),
            "scaling_max_freq": read_text(freq_dir / "scaling_max_freq"),
            "scaling_cur_freq": read_text(freq_dir / "scaling_cur_freq"),
            "cpuinfo_min_freq": read_text(freq_dir / "cpuinfo_min_freq"),
            "cpuinfo_max_freq": read_text(freq_dir / "cpuinfo_max_freq"),
        }
        rows.append(row)

        if row["scaling_governor"]:
            governors[row["scaling_governor"]] += 1
        if row["scaling_min_freq"]:
            min_freqs[row["scaling_min_freq"]] += 1
        if row["scaling_max_freq"]:
            max_freqs[row["scaling_max_freq"]] += 1
        if row["scaling_cur_freq"]:
            try:
                cur_freqs.append(int(row["scaling_cur_freq"]))
            except Exception:
                pass

    return {
        "intel_pstate_status": read_text("/sys/devices/system/cpu/intel_pstate/status"),
        "intel_pstate_no_turbo": read_text("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "governor_counts": dict(governors),
        "scaling_min_freq_counts": dict(min_freqs),
        "scaling_max_freq_counts": dict(max_freqs),
        "scaling_cur_freq_min": min(cur_freqs) if cur_freqs else None,
        "scaling_cur_freq_max": max(cur_freqs) if cur_freqs else None,
        "scaling_cur_freq_unique_count": len(set(cur_freqs)) if cur_freqs else None,
        "per_cpu": rows,
    }


def collect_cpuidle() -> Dict[str, Any]:
    c6_rows = []
    disabled_counts = Counter()

    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        for state_dir in sorted(cpu_dir.glob("cpuidle/state*")):
            name = read_text(state_dir / "name")
            disable = read_text(state_dir / "disable")
            if not name:
                continue
            if "C6" in name:
                row = {
                    "cpu": int(cpu_dir.name[3:]),
                    "state": state_dir.name,
                    "name": name,
                    "disable": disable,
                }
                c6_rows.append(row)
                disabled_counts[str(disable)] += 1

    return {
        "c6_disable_value_counts": dict(disabled_counts),
        "c6_states": c6_rows,
    }


def collect_nvidia() -> Dict[str, Any]:
    if subprocess.run(["bash", "-lc", "command -v nvidia-smi >/dev/null 2>&1"]).returncode != 0:
        return {"available": False}

    query = [
        "index",
        "name",
        "pci.bus_id",
        "persistence_mode",
        "clocks.gr",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
        "power.draw",
        "clocks_throttle_reasons.gpu_idle",
        "clocks_throttle_reasons.applications_clocks_setting",
        "clocks_throttle_reasons.hw_slowdown",
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.sw_power_cap",
    ]

    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(query)}",
        "--format=csv,noheader",
    ]
    q = run_cmd(cmd)

    gpus = []
    if q.get("returncode") == 0:
        for line in q.get("stdout", "").splitlines():
            vals = [x.strip() for x in line.split(",")]
            gpus.append(dict(zip(query, vals)))

    return {
        "available": True,
        "query": q,
        "gpus": gpus,
        "topo_m": run_cmd(["nvidia-smi", "topo", "-m"]),
        "clock_detail": run_cmd(["nvidia-smi", "-q", "-d", "CLOCK"]),
        "supported_clocks": run_cmd(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"]),
    }


def collect_docker() -> Dict[str, Any]:
    if subprocess.run(["bash", "-lc", "command -v docker >/dev/null 2>&1"]).returncode != 0:
        return {"available": False}

    return {
        "available": True,
        "info_cgroup_driver": run_cmd(["docker", "info", "--format", "{{.CgroupDriver}}"]),
        "info_cgroup_version": run_cmd(["docker", "info", "--format", "{{.CgroupVersion}}"]),
    }


def collect_systemd_slices() -> Dict[str, Any]:
    out = {}
    for unit in ["init.scope", "system.slice", "user.slice", "bench.slice"]:
        out[unit] = run_cmd(["systemctl", "show", unit, "-p", "AllowedCPUs", "-p", "AllowedMemoryNodes"])
    return out


def collect_all() -> Dict[str, Any]:
    return {
        "os": collect_os(),
        "cpu_topology": collect_lscpu(),
        "process_affinity": collect_process_affinity(),
        "cpufreq": collect_cpufreq(),
        "cpuidle": collect_cpuidle(),
        "nvidia": collect_nvidia(),
        "docker": collect_docker(),
        "systemd_slices": collect_systemd_slices(),
    }


def print_summary(state: Dict[str, Any]) -> None:
    print("============================================================")
    print("GraCE host state summary")
    print("============================================================")

    os_info = state["os"]
    print(f"Kernel: {os_info.get('release')}")
    print(f"Platform: {os_info.get('platform')}")

    topo = state["cpu_topology"]
    print(f"Online CPUs: {topo.get('online_cpus')}")
    print(f"Online NUMA nodes: {topo.get('online_numa_nodes')}")
    print("NUMA node CPU lists:")
    for node, cpus in topo.get("numa_node_cpulists", {}).items():
        print(f"  node {node}: {cpus}")

    aff = state["process_affinity"]
    print(f"Current process affinity: {aff.get('sched_getaffinity_compact')}")

    cf = state["cpufreq"]
    print(f"intel_pstate/status: {cf.get('intel_pstate_status')}")
    print(f"intel_pstate/no_turbo: {cf.get('intel_pstate_no_turbo')}")
    print(f"Governor counts: {cf.get('governor_counts')}")
    print(f"Min freq counts: {cf.get('scaling_min_freq_counts')}")
    print(f"Max freq counts: {cf.get('scaling_max_freq_counts')}")
    print(f"Cur freq min/max: {cf.get('scaling_cur_freq_min')} / {cf.get('scaling_cur_freq_max')}")

    ci = state["cpuidle"]
    print(f"C6 disable value counts: {ci.get('c6_disable_value_counts')}")

    nv = state["nvidia"]
    if nv.get("available"):
        print("GPUs:")
        for gpu in nv.get("gpus", []):
            print(
                "  GPU {index}: {name}, bus={bus}, persistence={persistence}, "
                "gr={gr}, sm={sm}, mem={mem}, temp={temp}, power={power}".format(
                    index=gpu.get("index", "unknown"),
                    name=gpu.get("name", "unknown"),
                    bus=gpu.get("pci.bus_id", "unknown"),
                    persistence=gpu.get("persistence_mode", "unknown"),
                    gr=gpu.get("clocks.gr", "unknown"),
                    sm=gpu.get("clocks.sm", "unknown"),
                    mem=gpu.get("clocks.mem", "unknown"),
                    temp=gpu.get("temperature.gpu", "unknown"),
                    power=gpu.get("power.draw", "unknown"),
                )
            )
    else:
        print("nvidia-smi: unavailable")

    docker = state["docker"]
    if docker.get("available"):
        print(f"Docker cgroup driver: {docker.get('info_cgroup_driver', {}).get('stdout')}")
        print(f"Docker cgroup version: {docker.get('info_cgroup_version', {}).get('stdout')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    state = collect_all()

    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(state, indent=2, sort_keys=True))
      print(f"[OK] Wrote host state JSON: {args.output}")

    if not args.quiet:
        print_summary(state)


if __name__ == "__main__":
    main()