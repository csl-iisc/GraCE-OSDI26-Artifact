# GraCE OSDI'26 Artifact

This repository contains the artifact for the OSDI'26 paper:

> [**GraCE: Unlocking CUDA Graphs with Compiler Support for ML
  Workloads**](https://www.usenix.org/conference/osdi26/presentation/ghosh)

The artifact is designed around a Docker-based workflow. The recommended path
for artifact evaluation is to use the prebuilt Docker image, run the smoke test,
and then run the provided experiment and analysis scripts to regenerate the figures and tables in the paper.

GraCE evaluates CUDA Graph execution behavior. Since CUDA Graph replay reduces
CPU-side kernel launch overhead, measured speedups are sensitive to host CPU
frequency scaling, idle states, GPU clocks, NUMA placement, and background
system noise. The artifact therefore includes optional host-tuning scripts for
paper-quality performance reproduction.

---

## Table of contents

1. [Artifact contents](#artifact-contents)
2. [Hardware requirements](#hardware-requirements)
3. [Host software requirements](#host-software-requirements)
    - [Installing host dependencies](#installing-host-dependencies)
4. [Getting started: 30-minute validation](#getting-started-30-minute-validation)
5. [Detailed instructions](#detailed-instructions)
   - [Docker images](#docker-images)
   - [Building the Docker image from source](#building-the-docker-image-from-source)
   - [Optional host tuning for low-noise runs](#optional-host-tuning-for-low-noise-runs)
   - [Full experiment workflow](#full-experiment-workflow)
   - [Result and output layout](#result-and-output-layout)
   - [Useful environment variables](#useful-environment-variables)
   - [Workload configuration](#workload-configuration)
   - [Benchmark runner](#benchmark-runner)
6. [Subdirectory documentation](#subdirectory-documentation)
7. [Artifact claims supported by this repository](#artifact-claims-supported-by-this-repository)
8. [TL;DR full flow](#tldr-full-flow)

---

## Artifact contents

```text
grace-osdi26-artifact/
├── benchmark_runner/   # Lower-level benchmark launcher
├── configs/            # Workload lists used by experiments
├── docker/             # Dockerfiles and Docker run/build scripts
├── figures/            # Generated paper figures
├── host_tuning/        # Optional host-side low-noise benchmarking scripts
├── results/            # Raw and processed experiment outputs
├── scripts/            # Build, experiment, and analysis wrappers
├── tables/             # Generated paper tables
└── third_party/        # Source submodules: PyTorch, Triton, TorchBench, etc.
```

Detailed READMEs are provided in the subdirectories and explain each component
in more detail.

---

## Hardware requirements

### Evaluation systems used in the paper

The paper uses two hardware setups.

**Single-GPU experiments.** The single-GPU results were collected on:

```text
GPU:        NVIDIA H100 NVL
GPU memory: 94 GB HBM3
CPU:        64-core Intel Xeon Platinum 8462Y+
            128 logical CPUs with hyperthreading enabled
Memory:     512 GB DDR5
Software:   PyTorch 2.4, CUDA 12.8, cuDNN 8.9.2
Driver:     NVIDIA 575.57.08
````

**Multi-GPU tensor-parallel experiments.** The tensor-parallel results were
collected on a separate multi-GPU system:

```text
GPUs:         4x NVIDIA H100
GPU memory:  80 GB per GPU
Interconnect: high-bandwidth NVLink
CUDA:         12.4
```

The artifact does not require these exact machines to run. Any recent x86-64
Linux machine with a CUDA-capable NVIDIA GPU, a compatible NVIDIA driver, Docker,
and NVIDIA Container Toolkit should be able to run the functional checks and
many of the experiments. However, the exact performance numbers in the paper are
hardware- and system-dependent. For closest reproduction, use H100-class GPUs,
a recent [NVIDIA driver compatible with the CUDA 
version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#id7) 
of the selected Docker image, and the optional host-tuning workflow 
described later in this README.

Tensor-parallel experiments require at least as many visible GPUs as the chosen
tensor-parallel degree. 


## Host software requirements

The artifact keeps Python, PyTorch, Triton, CUDA user-space libraries, and
benchmark dependencies inside Docker. The host only needs the software required
to run GPU-enabled Docker containers:


1. Git
2. Docker Engine
3. NVIDIA GPU driver
4. NVIDIA Container Toolkit

A full CUDA Toolkit installation on the host is not required for the prebuilt
Docker workflow. The important requirement is that the [host NVIDIA driver is
new enough for the CUDA runtime]((https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#id7)) 
used inside the selected Docker image.

Plan for at least 150 GB of free host disk space for one prebuilt Docker image,
and more if keeping both CUDA 12.8 and CUDA 12.4 images locally. Building the
prebuilt Docker image from source can take about 2--3 hours on an H100-class
host. The full experiment workflow should be treated as a long-running run; a
safe end-to-end allocation is about 24 hours, depending on GPU count, repeat
count, Nsight Systems collection.

### Installing host dependencies

Install Docker Engine using the official Docker documentation:

* Docker Engine for Ubuntu: [https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)

Verify Docker:

```bash
docker --version
docker run --rm hello-world
```

Install the NVIDIA driver using your cluster/system administrator's preferred
method or the official NVIDIA/Ubuntu documentation:

* Ubuntu NVIDIA driver installation:
  [https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)
* NVIDIA CUDA Installation Guide for Linux:
  [https://docs.nvidia.com/cuda/cuda-installation-guide-linux/](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

Verify the driver:

```bash
nvidia-smi
```

Install NVIDIA Container Toolkit using the official NVIDIA documentation:

* NVIDIA Container Toolkit:
  [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Verify that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this command prints the GPU list, the host is ready to run the GraCE Docker
image.

---

## Getting started: 30-minute validation

This path validates that the artifact works. It uses the prebuilt Docker image
and runs a small smoke test.

### Step 1: clone the repository

```bash
git clone https://github.com/csl-iisc/GraCE-OSDI26-Artifact grace-osdi26-artifact
cd grace-osdi26-artifact
```

Synchronize and initialize submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The main source submodules are:

```text
third_party/pytorch
third_party/triton
third_party/torchbenchmark
```

If a submodule checkout is inconsistent, reset it to the artifact-recorded
commit:

```bash
git submodule update --init --recursive --force
```

The GraCE implementation is contained in the source submodules. In particular,
`third_party/pytorch` contains the modified PyTorch implementation, with
separate `osdi26/grace-*` branches corresponding to the incremental GraCE
optimizations used in the artifact. `third_party/triton` contains the Triton
changes needed by GraCE, primarily on the `osdi26/grace-indirection` branch.
`third_party/torchbenchmark` contains the benchmark-suite used by the evaluation.

### Step 2: pull the prebuilt Docker image

Building the docker container from the source takes 2+ hours. We provide a 
prebuilt image that avoids the long source build, and can be quickly 
pulled and used for the smoke test (and running the experiments).

```bash
docker pull abhishekghosh1998/grace-osdi26:cuda128-prebuilt
```

Tag it with the local name expected by the run scripts:

```bash
docker tag abhishekghosh1998/grace-osdi26:cuda128-prebuilt \
  grace-osdi26:cuda128-prebuilt
```

A CUDA 12.4 image is also available:

```bash
docker pull abhishekghosh1998/grace-osdi26:cuda124-prebuilt
docker tag abhishekghosh1998/grace-osdi26:cuda124-prebuilt \
  grace-osdi26:cuda124-prebuilt
```

The CUDA 12.8 image is the default artifact path. The CUDA 12.4 image is 
provided as an additional prebuilt environment.


### Step 3: start the Docker container

For the CUDA 12.8 image:

```bash
docker/run_cuda128.sh
```

For CUDA 12.4:

```bash
docker/run_cuda124.sh
```

The run scripts bind-mount output directories from the host:

```text
results/
figures/
tables/
```

Therefore, experiment outputs produced inside Docker remain available on the
host.

For development, use the full-repository bind mount:

```bash
docker/run_cuda128_developer.sh
```

### Step 4: run the smoke test

Inside the Docker container:

```bash
bash scripts/smoke_test.sh
```

The smoke test checks:


1. CUDA visibility
2. PyTorch CUDA allocation
3. GraCE conda environment activation
4. Benchmark runner functionality
5. Result-file writing


Expected final output:

```text
[OK] Workload run completed.
```

Smoke-test results are written under:

```text
results/raw/smoke/<run_id>/
```

At this point, the artifact is functionally validated.

For the complete end-to-end command sequence, jump to
[TL;DR full flow](#tldr-full-flow).

### Step 5: optional ownership fix for host outputs

Docker containers often write bind-mounted outputs as `root`. If the generated
files are not editable on the host, run:

```bash
sudo chown -R "$USER:$USER" results figures tables
```

---

## Detailed instructions

The getting-started path above validates that the artifact is functional. This
section describes the full workflow for Docker images, optional source builds,
low-noise host tuning, complete experiment execution, and regeneration of the
paper figures and tables.

### Docker images

The artifact uses two classes of Docker images.

#### Prebuilt image

Recommended for artifact evaluation:

```text
grace-osdi26:cuda128-prebuilt
```

This image already contains:


1. Conda environments
2. Built PyTorch/GraCE variants
3. Built Triton variants
4. TorchBench/model dependencies
5. Benchmark scripts
6. Analysis dependencies
7. Nsight Systems, if included in the build


Run it with:

```bash
docker/run_cuda128.sh
```

#### Base/developer image

Useful for source rebuilds and development:

```text
grace-osdi26:cuda128-base
```

Run it with:

```bash
docker/run_cuda128_developer.sh
```

The base image contains the only system-level build environment needed to 
compile and install the artifact stack, but it does **not** contain the fully 
built GraCE variants. In particular, it includes CUDA development environment,
Conda, Compilers and build tools, Git and source-control utilities, Python build
 dependencies, Common system libraries required by PyTorch/Triton/TorchBench 
 builds. 

Inside the base image, use the scripts under `scripts/` to create conda
environments and build the variants.

See [`docker/README.md`](./docker/README.md) for Docker-specific details.

### Building the Docker image from source

Building from source is optional for artifact evaluation. It can take about
2 hours depending on CPU count, network speed, and storage performance.

To build the CUDA 12.8 prebuilt image:

```bash
docker/build_cuda128.sh --prebuilt
```

To build the CUDA 12.8 base image:

```bash
docker/build_cuda128.sh --base
```

For CUDA 12.4:

```bash
docker/build_cuda124.sh --prebuilt
docker/build_cuda124.sh --base
```

The generic build helper is:

```bash
docker/build_image.sh
```

If using the base/developer image, build the software stack inside Docker:

```bash
bash scripts/setup_conda_envs.sh
bash scripts/build_all_variants.sh
bash scripts/install_workloads.sh
```

The prebuilt image already performs these steps.
See [`docker/README.md`](./docker/README.md) for Docker-specific details.


### Optional host tuning for low-noise runs

Host tuning is optional for functional validation but recommended for
paper-quality performance reproduction.

GraCE evaluates CUDA Graph execution. CUDA Graph replay reduces CPU-side
launch overhead, so host-side variability can affect measurements. Important
sources of noise include, CPU frequency scaling, deep CPU idle states such as C6,
hyperthread sibling interference, NUMA mismatch, GPU clock variation, background 
host processes

To tune the host before launching Docker:

```bash
sudo host_tuning/tune_host_intel_h100.sh
```

The tuning script configures CPU/GPU stability settings and Docker-aware cpuset
isolation. It also writes Docker launch metadata that the Docker run scripts can
read automatically:

```text
docker/.generated/docker_cpuset.env
```

After tuning, launch Docker normally:

```bash
docker/run_cuda128.sh
```

or use the explicit Docker command printed by the tuning script.

After all experiments finish, restore the host:

```bash
sudo host_tuning/restore_host_intel_h100.sh
```

See [`host_tuning/README.md`](./host_tuning/README.md) for details and safety notes.

### Full experiment workflow

All commands in this section are run inside the Docker container unless stated
otherwise.

#### 1. Smoke test

```bash
bash scripts/smoke_test.sh
```

#### 2. Single-GPU experiments

Run the experiments:

```bash
bash scripts/run_single_gpu_experiments.sh
```

Activate a conda environment in the container:
```bash
conda activate grace-full
```

Generate Figures 9 and 10:

```bash
bash scripts/generate_figure9.sh
bash scripts/generate_figure10.sh
```

or:

```bash
bash scripts/generate_single_gpu_figures.sh
```

Expected outputs:

```text
figures/figure9.pdf
figures/figure10.pdf
```

#### 3. Tensor-parallel experiments

Run:

```bash
bash scripts/run_tp_experiments.sh
```

Generate the TP scaling figure:

```bash
conda activate grace-full
bash scripts/generate_tp_figure.sh
```

Expected output:

```text
figures/figure11.pdf
```

You can control TP sizes with:

```bash
TP_LIST="1 2 4" bash scripts/run_tp_experiments.sh
```

#### 4. Table 2: CUDA Graph coverage

Collect Nsight Systems traces:

```bash
bash scripts/run_table2_cgct_coverage_nsys.sh
```

Generate the table:

```bash
conda activate grace-full
bash scripts/generate_table2_cgct_coverage.sh
```

Expected output:

```text
tables/table2_cgct_coverage.csv
```

This step requires `nsys` inside the Docker image.

#### 5. Table 3: data-copy movement

Run the debug-copy experiment:

```bash
bash scripts/run_table3_pi_copy_debug.sh
```

Generate the table:

```bash
conda activate grace-full
bash scripts/generate_table3_pi_copy_debug.sh
```

Expected output:

```text
tables/table3_pi_copy_debug.csv
```

### Result and output layout

Raw results are timestamped:

```text
results/raw/<experiment>/<run_id>/
```

Examples:

```text
results/raw/single_gpu/20260502-201906/
results/raw/tp/20260502-223735/
results/raw/table2_cgct_coverage/20260502-095126/
results/raw/table3_pi_copy_debug/20260502-100909/
```

Each experiment root maintains:

```text
latest.txt
```

Analysis wrappers use `latest.txt` by default.

Processed outputs are written under:

```text
results/processed/
```

Paper-facing outputs are written under:

```text
figures/
tables/
```

Typical outputs:

```text
figures/figure9.pdf
figures/figure10.pdf
figures/figure11.pdf
tables/table2_cgct_coverage.csv
tables/table3_pi_copy_debug.csv
```

### Useful environment variables

Most experiment scripts support environment-variable overrides.

| Variable | Meaning |
|---|---|
| `RUN_ID` | Use a fixed run ID instead of a timestamp |
| `REPEAT` | Number of benchmark repetitions |
| `WORKLOADS` | Override workload list |
| `RESULTS_ROOT` | Override raw output root |
| `TIMEOUT_SECS` | Add per-run timeout |
| `TP_LIST` | TP sizes for tensor-parallel experiments |
| `CUDA_VISIBLE_DEVICES` | Select GPU IDs |
| `USE_NUMACTL` | Enable/disable container-side `numactl` |

Examples:

```bash
RUN_ID=test REPEAT=3 bash scripts/run_single_gpu_experiments.sh
TP_LIST="1 2" REPEAT=10 bash scripts/run_tp_experiments.sh
```

See [`scripts/README.md`](./scripts/README.md) for details.

### Workload configuration

Workloads are listed under:

```text
configs/
```

Important files:

```text
workloads_smoke.txt
workloads_single_gpu_25_with_acronym.txt
workloads_tp.txt
workloads_table2_cgct_coverage.txt
workloads_table3_pi_copy_debug.txt
```

Each line has either:

```text
<Model> <Suite> <Mode> <BatchSize>
```

or:

```text
<Acronym> <Model> <Suite> <Mode> <BatchSize>
```

See [`configs/README.md`](./configs/README.md) for details.

### Benchmark runner

The lower-level benchmark launcher is:

```text
benchmark_runner/run_workloads.sh
```

The higher-level scripts in `scripts/` call it automatically. Most users should
not need to invoke it directly.

See [`benchmark_runner/README.md`](./benchmark_runner/README.md) for details.

---

## Subdirectory documentation

More detailed documentation is available in:

```text
benchmark_runner/README.md
configs/README.md
docker/README.md
host_tuning/README.md
scripts/README.md
scripts/analysis/README.md
```

---

## Artifact claims supported by this repository

This artifact supports the following claims:

1. GraCE improves CUDA Graph coverage over the baseline PyTorch CUDA Graph
   path on selected deep learning workloads.
2. GraCE reduces data-copy movement required by CUDA Graph execution.
3. GraCE improves performance on selected single-GPU workloads.
4. GraCE improves performance under selected tensor-parallel workloads.
5. The reported figures and tables can be regenerated using the provided
   workload lists, scripts, Docker images, and analysis pipeline.

Exact numerical values may vary with GPU model, CPU topology, driver version,
host noise, and clock/power settings.

---

## TL;DR full flow

```bash
# ---------------------------------------------------------------------------
# Host: clone source.
# ---------------------------------------------------------------------------
git clone https://github.com/csl-iisc/GraCE-OSDI26-Artifact grace-osdi26-artifact
cd grace-osdi26-artifact
git submodule sync --recursive
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# Host: pull prebuilt image.
# ---------------------------------------------------------------------------
docker pull abhishekghosh1998/grace-osdi26:cuda128-prebuilt
docker tag abhishekghosh1998/grace-osdi26:cuda128-prebuilt \
  grace-osdi26:cuda128-prebuilt

# ---------------------------------------------------------------------------
# Host: optional, for paper-quality performance reproduction.
# ---------------------------------------------------------------------------
sudo host_tuning/tune_host_intel_h100.sh

# ---------------------------------------------------------------------------
# Host: start Docker.
# ---------------------------------------------------------------------------
docker/run_cuda128.sh

# ---------------------------------------------------------------------------
# Inside Docker: validate artifact.
# ---------------------------------------------------------------------------
bash scripts/smoke_test.sh

# ---------------------------------------------------------------------------
# Inside Docker: run experiments and generate outputs.
# ---------------------------------------------------------------------------
bash scripts/run_single_gpu_experiments.sh
bash scripts/generate_figure9.sh # in a conda environment provided with the docker container
bash scripts/generate_figure10.sh # in a conda environment provided with the docker container

bash scripts/run_tp_experiments.sh
bash scripts/generate_tp_figure.sh # in a conda environment provided with the docker container

bash scripts/run_table2_cgct_coverage_nsys.sh
bash scripts/generate_table2_cgct_coverage.sh # in a conda environment provided with the docker container

bash scripts/run_table3_pi_copy_debug.sh
bash scripts/generate_table3_pi_copy_debug.sh # in a conda environment provided with the docker container

# ---------------------------------------------------------------------------
# Host: restore after experiments.
# ---------------------------------------------------------------------------
sudo host_tuning/restore_host_intel_h100.sh

# Optional: fix bind-mounted output ownership on host.
sudo chown -R "$USER:$USER" results figures tables
```
