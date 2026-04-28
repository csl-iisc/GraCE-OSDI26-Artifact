# Docker-Aware CPU Isolation

This directory contains the cgroup/systemd scripts used to isolate benchmark CPUs for Docker-based GraCE artifact runs.

The main scripts are:

```text
setup_docker_bench_cpuset_full.sh
restore_docker_bench_cpuset.sh
```

These scripts are host-side scripts. Run them on the bare-metal host, not inside the Docker container.

---

## Required Docker flags

After running the setup script, start Docker with:

```bash
--cgroup-parent=bench.slice
--cpuset-cpus=<benchmark-cpus>
--cpuset-mems=<benchmark-numa-node>
```

For example:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --cgroup-parent=bench.slice \
  --cpuset-cpus=8-31,64-95 \
  --cpuset-mems=0 \
  --cap-add=SYS_NICE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e USE_NUMACTL=0 \
  -v "$(pwd)/results:/workspace/grace-osdi-26-artifact/results" \
  -v "$(pwd)/figures:/workspace/grace-osdi-26-artifact/figures" \
  -v "$(pwd)/tables:/workspace/grace-osdi-26-artifact/tables" \
  grace-osdi26:cuda124-prebuilt
```

The important flags are:

| Flag | Meaning |
|---|---|
| `--cgroup-parent=bench.slice` | Places the container under the benchmark systemd slice. |
| `--cpuset-cpus=...` | Restricts the container to the benchmark CPUs. |
| `--cpuset-mems=...` | Restricts container memory allocation to the selected NUMA node. |
| `-e USE_NUMACTL=0` | Disables container-side `numactl`, since Docker already applies CPU/memory placement. |
| `--cap-add=SYS_NICE` | Allows benchmark processes to use scheduling/affinity operations where needed. |
| `--security-opt seccomp=unconfined` | Avoids seccomp blocking low-level profiling or scheduling operations. |

The setup script also writes Docker launch metadata that is automatically consumed
by:

```
docker/run_cuda124.sh
docker/run_cuda128.sh
```
After tuning, you can start the container with one of:
```
docker/run_cuda124.sh
# or
docker/run_cuda128.sh
```
Instead of manually copying and using the docker run command

---

## Setup

Run from the artifact repository root:

```bash
sudo host_tuning/cpuset/setup_docker_bench_cpuset_full.sh
```

The script prints:

- Docker cgroup driver and version;
- online CPUs;
- online NUMA nodes;
- CPU list by NUMA node;
- `numactl --hardware`, if available;
- `nvidia-smi topo -m`;
- per-GPU PCI locality from sysfs.

It then prompts for:

1. benchmark CPUs;
2. benchmark NUMA memory nodes.

Typical CPU choices on the authors' Intel + H100 host are:

```text
# HT enabled
8-31,64-95

# HT disabled
8-31
```

The script configures:

```text
bench.slice     -> benchmark CPUs
init.scope      -> system CPUs
system.slice    -> system CPUs
user.slice      -> system CPUs
```

Normal host sessions and system services are therefore restricted away from the benchmark CPUs.

---

## Restore

After experiments finish:

```bash
sudo host_tuning/cpuset/restore_docker_bench_cpuset.sh
```

This restores broad CPU and NUMA access for:

```text
init.scope
system.slice
user.slice
bench.slice
```

If a Docker container is still running under `bench.slice`, stop it before restoring for the cleanest state.

---

## Verification

Inside the Docker container:

```bash
grep Cpus_allowed_list /proc/self/status
```

Expected output should contain the benchmark CPUs only, for example:

```text
Cpus_allowed_list:  8-31,64-95
```

Python verification:

```bash
python3 - <<'PY'
import os
print(sorted(os.sched_getaffinity(0)))
PY
```

From a normal host shell:

```bash
grep Cpus_allowed_list /proc/self/status
```

Expected output should exclude the benchmark CPUs, for example:

```text
Cpus_allowed_list:  0-7,32-63,96-127
```

---

## Requirements

This path assumes:

```text
Linux with cgroup v2
systemd
Docker using the systemd cgroup driver
```

Check with:

```bash
docker info | grep -E 'Cgroup Driver|Cgroup Version'
```

Expected:

```text
Cgroup Driver: systemd
Cgroup Version: 2
```

---

## Notes

- `--cpuset-cpus` alone constrains the container but does not isolate those CPUs from the host.
- `--cgroup-parent=bench.slice` is the key Docker flag for this setup.
- `-e USE_NUMACTL=0` is recommended because Docker already applies CPU and memory-node placement. Container-side `numactl --membind` can fail without additional privileges.
- This script changes host scheduling policy until the restore script is run.
