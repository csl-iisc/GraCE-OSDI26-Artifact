# Docker workflow for the GraCE OSDI'26 artifact

This directory contains Dockerfiles and helper scripts for building and running the GraCE artifact environment.

The Docker image controls the software stack: CUDA, cuDNN, Conda environments, GraCE variants, TorchBench workloads, and analysis dependencies. Host-level performance stabilization is handled separately by `host_tuning/`.

---

## Directory layout

```text
docker/
├── README.md
├── Dockerfile.cuda124
├── Dockerfile.cuda128
├── build_image.sh
├── build_cuda124.sh
├── build_cuda128.sh
├── run_common.sh
├── run_cuda124.sh
├── run_cuda124_developer.sh
├── run_cuda128.sh
├── run_cuda128_developer.sh
└── .generated/
```

---

## Image variants

The artifact supports two image modes.

| Mode | Tag convention | Purpose |
|---|---|---|
| Base/developer | `grace-osdi26:cuda128-base` | Contains system dependencies, CUDA, Conda, and the artifact repository. GraCE variants and workloads are built manually inside the container. |
| Prebuilt/evaluator | `grace-osdi26:cuda128-prebuilt` | Fully built evaluator image. Conda environments, GraCE variants, and workloads are already installed. |

The same convention is used for CUDA 12.4:

```text
grace-osdi26:cuda124-base
grace-osdi26:cuda124-prebuilt
```

---

## Prebuilt image

Prebuilt images are available from Docker Hub:

```bash
docker pull abhishekghosh1998/grace-osdi26:cuda128-prebuilt
```

After pulling, either retag it to the local convention:

```bash
docker tag abhishekghosh1998/grace-osdi26:cuda128-prebuilt \
  grace-osdi26:cuda128-prebuilt
```

or run with `IMAGE=...`:

```bash
IMAGE=abhishekghosh1998/grace-osdi26:cuda128-prebuilt docker/run_cuda128.sh
```

---

## Build images locally

Build from the artifact repository root.
### CUDA 12.8 base image

```bash
docker/build_cuda128.sh --base
```

Equivalent explicit form:

```bash
docker/build_cuda128.sh \
  --base \
  --image grace-osdi26:cuda128-base \
  --install-nsys 1 \
  --max-jobs 32
```

### CUDA 12.8 prebuilt image

```bash
docker/build_cuda128.sh --prebuilt
```

Equivalent explicit form:

```bash
docker/build_cuda128.sh \
  --prebuilt \
  --image grace-osdi26:cuda128-prebuilt \
  --install-nsys 1 \
  --max-jobs 32
```

The prebuilt image sets:

```text
BUILD_ARTIFACT=1
```

inside the Docker build, which runs the artifact setup scripts during image construction.

### CUDA 12.4 base image

```bash
docker/build_cuda124.sh --base
```

Equivalent explicit form:

```bash
docker/build_cuda124.sh \
  --base \
  --image grace-osdi26:cuda124-base \
  --install-nsys 1 \
  --max-jobs 32
```

### CUDA 12.4 prebuilt image

```bash
docker/build_cuda124.sh --prebuilt
```

Equivalent explicit form:

```bash
docker/build_cuda124.sh \
  --prebuilt \
  --image grace-osdi26:cuda124-prebuilt \
  --install-nsys 1 \
  --max-jobs 32
```
---

## Build script options

All options are handled by `docker/build_image.sh`.

```bash
docker/build_image.sh --cuda <cuda124|cuda128> [options]
```

Important options:

| Option | Meaning |
|---|---|
| `--base` | Build a base/developer image. Sets `BUILD_ARTIFACT=0`. |
| `--prebuilt` | Build a fully prebuilt image. Sets `BUILD_ARTIFACT=1`. |
| `--image <tag>` | Override the output image tag. |
| `--install-nsys <0|1>` | Install Nsight Systems. Default: `1`. |
| `--max-jobs <N>` | Build parallelism passed to Dockerfile. Default: `32`. |
| `--progress <mode>` | Docker build progress mode. Default: `plain`. |
| `--no-cache` | Disable Docker layer cache. |
| `--push` | Push the image after a successful build. |

Examples:

```bash
# Build a base image without Nsight Systems.
docker/build_cuda124.sh --base --install-nsys 0

# Build a prebuilt image with higher build parallelism.
docker/build_cuda124.sh --prebuilt --max-jobs 64

# Build and push a custom registry tag.
docker/build_cuda124.sh --prebuilt \
  --image abhishekghosh1998/grace-osdi26:cuda124-prebuilt \
  --push
```

---

## Run images

There are two run modes.

### Evaluator / prebuilt mode

Use this mode for normal artifact evaluation:

```bash
docker/run_cuda124.sh
```

This uses:

```text
grace-osdi26:cuda124-prebuilt
```

by default and bind-mounts only output directories:

```text
results/
figures/
tables/
```

This keeps the prebuilt repository inside the image intact while allowing generated outputs to appear on the host.

### Developer mode

Use this mode when actively editing the repository on the host:

```bash
docker/run_cuda124_developer.sh
```

This uses:

```text
grace-osdi26:cuda124-base
```

by default and bind-mounts the full host repository at:

```text
/workspace/grace-osdi-26-artifact
```

In this mode, changes made on the host are visible inside the container, and generated results are written directly into the host checkout.

### CUDA 12.8 run scripts

Same for CUDA 12.8:

```bash
docker/run_cuda128.sh
docker/run_cuda128_developer.sh
```

---

## Low-noise Docker run with host cpuset isolation

For paper-quality performance reproduction, run the optional host tuning flow first:

```bash
sudo host_tuning/tune_host_intel_h100.sh
```

or directly:

```bash
sudo host_tuning/cpuset/setup_docker_bench_cpuset_full.sh
```

The cpuset setup script creates a systemd benchmark slice, usually:

```text
bench.slice
```

and records selected benchmark CPUs and NUMA memory nodes.

The run scripts automatically detect cpuset metadata from:

```text
docker/.generated/docker_cpuset.env
/run/grace_docker_cpuset_backup/state.env
```

If detected, `docker/run_cuda124.sh` automatically adds:

```bash
--cgroup-parent=bench.slice
--cpuset-cpus=<benchmark-cpus>
--cpuset-mems=<benchmark-numa-node>
--cap-add=SYS_NICE
--security-opt seccomp=unconfined
-e USE_NUMACTL=0
```

Therefore the normal reviewer command remains:

```bash
docker/run_cuda124.sh
```

If host tuning has been configured, it runs isolated. If not, it runs in the portable default mode.

---

## Forcing or disabling cpuset mode

### Force cpuset mode manually

```bash
USE_CPUSET=1 \
BENCH_CPUS=8-31,64-95 \
BENCH_MEMS=0 \
CGROUP_PARENT=bench.slice \
docker/run_cuda124.sh
```

### Disable cpuset mode even if host tuning metadata exists

```bash
USE_CPUSET=0 docker/run_cuda124.sh
```

### Autodetect mode

This is the default:

```bash
USE_CPUSET=auto docker/run_cuda124.sh
```

---

## Verify isolation inside Docker

Inside the container:

```bash
grep Cpus_allowed_list /proc/self/status
```

Expected with cpuset isolation enabled:

```text
Cpus_allowed_list:  8-31,64-95
```

Python check:

```bash
python3 - <<'PY'
import os
print(sorted(os.sched_getaffinity(0)))
PY
```

---

## Why `USE_NUMACTL=0` is used

Inside Docker, `numactl --membind` may fail with:

```text
set_mempolicy: Operation not permitted
```

For Docker-based runs, CPU and NUMA placement are applied at the Docker/cgroup layer using:

```bash
--cpuset-cpus=...
--cpuset-mems=...
```

Therefore the run scripts pass:

```bash
-e USE_NUMACTL=0
```

by default.

---

## Output contract

Evaluator/prebuilt mode bind-mounts:

```text
results/ -> /workspace/grace-osdi-26-artifact/results
figures/ -> /workspace/grace-osdi-26-artifact/figures
tables/  -> /workspace/grace-osdi-26-artifact/tables
```

This means all experiment outputs, generated figures, and generated tables are visible on the host after the container exits.

Developer mode bind-mounts the entire repository, so all modifications and outputs are shared directly with the host checkout.

---

## Common workflows

### Pull prebuilt image and run artifact

```bash
docker pull abhishekghosh1998/grace-osdi26:cuda124-prebuilt
docker tag abhishekghosh1998/grace-osdi26:cuda124-prebuilt \
  grace-osdi26:cuda124-prebuilt
docker/run_cuda124.sh
```

### Build prebuilt image locally and run artifact

```bash
docker/build_cuda124.sh --prebuilt
docker/run_cuda124.sh
```

### Build base image and develop inside container

```bash
docker/build_cuda124.sh --base
docker/run_cuda124_developer.sh
```

Inside the base container, run the setup scripts manually:

```bash
bash scripts/setup_conda_envs.sh
bash scripts/build_all_variants.sh
bash scripts/install_workloads.sh
```

### Low-noise artifact reproduction

```bash
sudo host_tuning/tune_host_intel_h100.sh

docker/run_cuda124.sh

# run experiments inside Docker

sudo host_tuning/restore_host_intel_h100.sh
```

---
