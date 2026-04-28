# Host Tuning for Low-Noise GraCE Performance Evaluation

This directory contains optional host-side tuning scripts used to reduce measurement noise when reproducing the performance results in the GraCE OSDI'26 artifact.

GraCE evaluates CUDA Graph execution behavior. Many reported results are sensitive to CPU-side launch overheads, CPU scheduling jitter, CPU idle-state latency, GPU clock variation, NUMA placement, and interference from unrelated host processes. The scripts in this directory tune a dedicated benchmarking host to make performance measurements more stable.

These scripts are **not required for functional validation** of the artifact. They are recommended for **paper-quality timing reproduction**.

---

## Important warning

These scripts mutate the **host machine**, not the Docker container.

They may:

- lock NVIDIA GPU clocks;
- change CPU frequency governors and CPU frequency limits;
- disable deep CPU idle states such as C6;
- disable hyperthreading / SMT sibling CPUs;
- configure systemd/cgroup-v2 CPU isolation for Docker containers;
- restrict normal host processes away from benchmark CPUs.

Run them only on a dedicated benchmarking machine where you have administrative control.

Do **not** run these scripts inside the artifact Docker container.

The Docker image controls the software stack. Host tuning controls the bare-metal machine underneath Docker.

---

## Directory layout

```text
host_tuning/
├── README.md
├── common.sh
├── tune_host_intel_h100.sh
├── restore_host_intel_h100.sh
├── check_host_state.py
├── profiles/
│   ├── README.md
│   ├── example_intel_h100.env
│   └── example_amd_epyc_a6000.env
├── gpu/
│   ├── README.md
│   ├── lock_gpu_clocks.sh
│   └── reset_gpu_clocks.sh
├── cpu/
│   ├── README.md
│   ├── intel_pin_cpu_freq.sh
│   ├── intel_restore_cpu_freq.sh
│   ├── amd_pin_cpu_freq.sh
│   ├── disable_hyperthreading.sh
│   ├── enable_hyperthreading.sh
│   ├── disable_c6.sh
│   └── enable_c6.sh
└── cpuset/
    ├── README.md
    ├── setup_docker_bench_cpuset_full.sh
    └── restore_docker_bench_cpuset.sh
```

### Main entry points

| Script | Purpose |
|---|---|
| `tune_host_intel_h100.sh` | Applies the recommended low-noise host setup for the Intel Xeon + H100 NVL artifact machine. |
| `restore_host_intel_h100.sh` | Restores the host after experiments complete. |
| `check_host_state.py` | Read-only checker that records CPU, GPU, NUMA, cgroup, frequency, and idle-state information. |

### Component directories

| Directory | Purpose |
|---|---|
| `gpu/` | NVIDIA persistence mode and GPU clock locking/reset. |
| `cpu/` | CPU frequency pinning, C-state control, hyperthreading control. |
| `cpuset/` | Docker-aware CPU isolation using systemd slices and cgroup v2. |
| `profiles/` | Machine-specific configuration defaults. |

---

## Supported host configuration

The primary artifact host used by the authors has:

```text
CPU:      Intel Xeon Platinum 8462Y+
GPUs:     2x NVIDIA H100 NVL, 95830 MiB each
OS:       Ubuntu 22.04.1 LTS x86_64
Kernel:   6.8.0-106-generic
Docker:   cgroup v2 with systemd cgroup driver
```

The scripts are written to be inspectable and configurable. They can be adapted to other Linux machines, but the default profile is intended for the Intel + H100 artifact host.

---

## Why host tuning matters for GraCE

GraCE targets CUDA Graph execution. CUDA Graphs reduce CPU-side kernel launch overhead by capturing and replaying GPU work. Therefore, CPU-side noise can directly affect the measurements.

Important sources of noise include:

- CPU frequency scaling;
- turbo boost behavior;
- deep CPU idle states such as C6;
- hyperthread sibling contention;
- kernel interrupts and background tasks;
- unrelated user or system processes scheduled on benchmark CPUs;
- GPU clock throttling or dynamic GPU clock changes;
- NUMA mismatch between benchmark CPU cores and GPUs.

For this reason, the artifact provides optional host-level tuning scripts.

---

## High-level workflow

The intended low-noise workflow is:

```text
1. Tune the host.
2. Start the Docker container using the command printed by the tuning script.
3. Run the artifact experiments inside Docker.
4. Generate figures and tables.
5. Restore the host.
```

Commands:

```bash
sudo host_tuning/tune_host_intel_h100.sh

# Use the Docker command printed by the tuning script.
# Or use docker/run_cuda124.sh or docker/run_cuda128.sh
# Tuning scripts prepares meta data which the docker/run*.sh scripts can read automatially
# Then, inside Docker:
bash scripts/run_single_gpu_experiments.sh
bash scripts/generate_figure9.sh # in a conda environment provided with the docker container
bash scripts/generate_figure10.sh # in a conda environment provided with the docker container

bash scripts/run_tp_experiments.sh
bash scripts/generate_tp_figure.sh  # in a conda environment provided with the docker container

bash scripts/run_table2_cgct_coverage_nsys.sh
bash scripts/generate_table2_cgct_coverage.sh # in a conda environment provided with the docker container

bash scripts/run_table3_pi_copy_debug.sh
bash scripts/generate_table3_pi_copy_debug.sh # in a conda environment provided with the docker container

# After experiments finish, on the host:
sudo host_tuning/restore_host_intel_h100.sh
```

---

## Quick start: tune host for the Intel + H100 setup

From the artifact repository root:

```bash
sudo host_tuning/tune_host_intel_h100.sh
```

The script will:

1. record the host state before tuning;
2. lock GPU clocks;
3. pin CPU frequencies;
4. disable deep C6 idle states;
5. optionally disable hyperthreading;
6. configure Docker-aware CPU isolation;
7. print the exact `docker run` command to use;
8. record the host state after tuning.

The script prompts before applying host-level changes.

To skip the prompt in automated environments:

```bash
ASSUME_YES=1 sudo -E host_tuning/tune_host_intel_h100.sh
```

---

## Restore host after experiments

After all experiments finish, restore the host:

```bash
sudo host_tuning/restore_host_intel_h100.sh
```

This will:

1. restore Docker/systemd cpuset policy;
2. re-enable hyperthreading if it was disabled;
3. re-enable C6 idle states if they were disabled;
4. restore CPU frequency policy;
5. reset GPU clock locks;
6. record the host state after restore.

If you only need to restore Docker cpuset isolation:

```bash
sudo host_tuning/cpuset/restore_docker_bench_cpuset.sh
```

---

## Host-state logs

The tuning scripts write host-state logs under:

```text
results/host_state/
```

Typical files:

```text
results/host_state/host_state_before_tune.json
results/host_state/host_state_after_tune.json
results/host_state/host_state_after_restore.json
```

These logs contain:

- OS and kernel version;
- CPU topology;
- NUMA topology;
- process CPU affinity;
- CPU frequency governor and min/max/current frequency;
- Intel pstate status;
- CPU C-state / C6 state;
- NVIDIA GPU clocks, persistence mode, temperatures, throttle reasons;
- Docker cgroup driver and cgroup version;
- systemd slice CPU restrictions.

To manually record host state:

```bash
python3 host_tuning/check_host_state.py \
  --output results/host_state/manual_host_state.json
```

---

## Configuration profiles

Machine-specific defaults live in:

```text
host_tuning/profiles/
```

The default profile for the artifact host is:

```text
host_tuning/profiles/example_intel_h100.env
```

The tuning script loads this profile by default:

```bash
sudo host_tuning/tune_host_intel_h100.sh
```

To use a different profile:

```bash
HOST_TUNING_PROFILE=host_tuning/profiles/example_amd_epyc_a6000.env \
sudo -E host_tuning/tune_host_intel_h100.sh
```

---

## Useful environment variables

The tuning scripts are configurable through environment variables.

### General

| Variable | Default | Meaning |
|---|---:|---|
| `ASSUME_YES` | `0` | If `1`, skip confirmation prompts. |
| `HOST_TUNING_PROFILE` | `profiles/example_intel_h100.env` | Profile to load. |
| `GRACE_TUNING_BACKUP_DIR` | `/run/grace_host_tuning_backup` | Backup directory for restore metadata. |
| `RESULTS_HOST_STATE_DIR` | `results/host_state` | Directory for host-state JSON files. |

### GPU

| Variable | Default | Meaning |
|---|---:|---|
| `GPU_IDS` | `all` | GPU IDs to tune. |
| `GPU_CLOCK_MHZ` | `1410` | Graphics clock lock. |
| `GPU_MEM_CLOCK_MHZ` | empty | Optional memory clock lock. |
| `SKIP_GPU_TUNING` | `0` | If `1`, skip GPU clock tuning. |

### CPU frequency

| Variable | Default | Meaning |
|---|---:|---|
| `INTEL_TARGET_FREQ_KHZ` | `2800000` | Intel CPU frequency target. |
| `AMD_TARGET_FREQ_KHZ` | `3100000` | AMD CPU frequency target. |
| `DISABLE_TURBO_OS` | `0` | If `1`, disable Intel turbo through sysfs when available. |
| `SKIP_CPU_FREQ_TUNING` | `0` | If `1`, skip CPU frequency tuning. |

### Hyperthreading / SMT

| Variable | Default | Meaning |
|---|---:|---|
| `DISABLE_HT` | `1` | If `1`, offline hyperthread sibling CPUs. |
| `HT_CPUS` | machine-specific | Explicit CPU list to offline/online. |

### Idle states

| Variable | Default | Meaning |
|---|---:|---|
| `DISABLE_C6` | `1` | If `1`, disable C6 idle states. |
| `DISABLE_IDLE_STATES` | script-specific | Optional comma-separated idle-state names for generalized scripts. |

### Docker cpuset

| Variable | Default | Meaning |
|---|---:|---|
| `ENABLE_DOCKER_CPUSET` | `1` | If `1`, configure Docker-aware CPU isolation. |
| `DOCKER_IMAGE` | `grace-osdi26:cuda124-prebuilt-v1` | Image name printed in Docker command. |
| `ARTIFACT_ROOT_IN_CONTAINER` | `/workspace/grace-osdi-26-artifact` | Artifact path inside Docker. |

Examples:

```bash
# Keep hyperthreading enabled.
DISABLE_HT=0 sudo -E host_tuning/tune_host_intel_h100.sh

# Skip cpuset isolation but still tune clocks/frequency.
ENABLE_DOCKER_CPUSET=0 sudo -E host_tuning/tune_host_intel_h100.sh

# Skip GPU tuning.
SKIP_GPU_TUNING=1 sudo -E host_tuning/tune_host_intel_h100.sh

# Use a different GPU graphics clock.
GPU_CLOCK_MHZ=1410 sudo -E host_tuning/tune_host_intel_h100.sh
```

---

## Safety notes

- Run on a dedicated benchmark machine.
- Keep an SSH session open when changing cgroups remotely.
- Always run the restore script after experiments.
- Do not run host tuning inside Docker.
- Do not make these settings permanent unless you intentionally maintain a dedicated benchmark server.
