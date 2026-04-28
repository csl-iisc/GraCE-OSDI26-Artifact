# Host Tuning Profiles

This directory contains machine-specific defaults for the host tuning scripts.

Profiles are Bash environment files sourced by the top-level tuning scripts. They define GPU clocks, CPU frequency targets, hyperthreading policy, C-state policy, Docker image names, and whether Docker-aware cpuset isolation is enabled.

---

## Available profiles

| Profile | Purpose |
|---|---|
| `example_intel_h100.env` | Default profile for the authors' Intel Xeon Platinum 8462Y+ + 2x NVIDIA H100 NVL host. |
| `example_amd_epyc_a6000.env` | Example profile for an AMD EPYC 9554 + NVIDIA RTX A6000 style host. |

---

## Default profile

The top-level script loads:

```text
host_tuning/profiles/example_intel_h100.env
```

by default:

```bash
sudo host_tuning/tune_host_intel_h100.sh
```

To explicitly select a profile; example with the AMD profile:

```bash
HOST_TUNING_PROFILE=host_tuning/profiles/example_amd_epyc_a6000.env \
sudo -E host_tuning/tune_host_intel_h100.sh
```

The Intel-named top-level script is the main orchestrator. 

---

## Common profile variables

### GPU variables

| Variable | Meaning |
|---|---|
| `GPU_IDS` | GPU IDs to tune. Use `all`, `0`, or comma-separated values such as `0,1`. |
| `GPU_CLOCK_MHZ` | NVIDIA graphics clock target. |
| `GPU_MEM_CLOCK_MHZ` | Optional NVIDIA memory clock target. Empty means do not lock memory clocks. |

### CPU frequency variables

| Variable | Meaning |
|---|---|
| `INTEL_TARGET_FREQ_KHZ` | Intel frequency target for `intel_pin_cpu_freq.sh`. |
| `AMD_TARGET_FREQ_KHZ` | AMD frequency target for `amd_pin_cpu_freq.sh`. |
| `DISABLE_TURBO_OS` | If `1`, disable Intel turbo through sysfs when available. |

### Hyperthreading and idle-state variables

| Variable | Meaning |
|---|---|
| `DISABLE_HT` | If `1`, offline hyperthread/SMT sibling CPUs. |
| `HT_CPUS` | Explicit CPU list to offline/online. Empty means auto-detect when possible. |
| `DISABLE_C6` | If `1`, disable C6 idle states. |
| `DISABLE_IDLE_STATES` | Optional comma-separated list for generalized idle-state scripts. |

### Docker/cpuset variables

| Variable | Meaning |
|---|---|
| `ENABLE_DOCKER_CPUSET` | If `1`, run Docker-aware cpuset setup. |
| `DOCKER_IMAGE` | Image name printed in the generated Docker command. |
| `ARTIFACT_ROOT_IN_CONTAINER` | Path to the artifact root inside Docker. |

---

## Override examples

Keep hyperthreading enabled:

```bash
DISABLE_HT=0 sudo -E host_tuning/tune_host_intel_h100.sh
```

Use a different GPU clock:

```bash
GPU_CLOCK_MHZ=1530 sudo -E host_tuning/tune_host_intel_h100.sh
```

Skip Docker cpuset isolation:

```bash
ENABLE_DOCKER_CPUSET=0 sudo -E host_tuning/tune_host_intel_h100.sh
```

---

## Adding a new machine profile

Create a new file:

```text
host_tuning/profiles/<machine_name>.env
```

Include at least:

```bash
GPU_IDS="all"
GPU_CLOCK_MHZ="<valid-gpu-clock>"
GPU_MEM_CLOCK_MHZ=""

INTEL_TARGET_FREQ_KHZ="<target>"   # for Intel systems
AMD_TARGET_FREQ_KHZ="<target>"     # for AMD systems

DISABLE_HT="1"
HT_CPUS=""
DISABLE_C6="1"

ENABLE_DOCKER_CPUSET="1"
DOCKER_IMAGE="grace-osdi26:cuda124-prebuilt"
ARTIFACT_ROOT_IN_CONTAINER="/workspace/grace-osdi-26-artifact"
```

Then run:

```bash
HOST_TUNING_PROFILE=host_tuning/profiles/<machine_name>.env \
sudo -E host_tuning/tune_host_intel_h100.sh
```


