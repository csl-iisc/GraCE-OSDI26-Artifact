# Scripts

This directory contains the main automation scripts for building the GraCE
software stack, running artifact experiments, and generating paper figures and
tables.

Most artifact evaluators using the **prebuilt Docker image** only need the
experiment and analysis scripts. The build/setup scripts are mainly for users
who start from the base Docker image or want to rebuild PyTorch/Triton variants
from source.

---

## Directory layout

```text
scripts/
├── analysis/
├── common.sh
├── setup_conda_envs.sh
├── build_*.sh
├── install_*.sh
├── run_*.sh
├── generate_*.sh
├── check_cuda_stack.sh
└── smoke_test.sh
````

---

## Script categories

## 1. Shared helpers

| Script      | Purpose                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------- |
| `common.sh` | Shared paths, variant mappings, logging helpers, and conda activation helpers used by the other scripts |

Most scripts source `common.sh`, so they should be run from a correctly prepared
artifact checkout/container.

---

## 2. Build and setup scripts

These scripts are mainly needed when using the **base/developer Docker image**
or rebuilding the artifact stack from source.

| Script                     | Purpose                                                              |
| -------------------------- | -------------------------------------------------------------------- |
| `setup_conda_envs.sh`      | Creates the GraCE conda environments                               |
| `build_triton.sh`          | Builds the Triton variant for a selected GraCE configuration       |
| `build_pytorch_variant.sh` | Builds and installs one PyTorch/GraCE variant                      |
| `build_all_variants.sh`    | Builds all GraCE variants                                          |
| `install_workloads.sh`     | Installs TorchBench/model dependencies                               |
| `install_nsys.sh`          | Installs Nsight Systems inside the image/container                   |
| `check_cuda_stack.sh`      | Checks CUDA, NVIDIA driver visibility, and PyTorch CUDA availability |

If you are using the prebuilt image, these steps should already be done.

Typical developer flow:

```bash
bash scripts/setup_conda_envs.sh
bash scripts/build_all_variants.sh
bash scripts/install_workloads.sh
```

---

## 3. Smoke test

| Script          | Purpose                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `smoke_test.sh` | Runs a small sanity check to verify that the container, CUDA stack, conda environments, and benchmark runner are working |

Recommended before long experiments:

```bash
bash scripts/smoke_test.sh
```

---

## 4. Experiment collection scripts

These scripts run the raw experiments and write results under `results/raw/`.

| Script                             | Output directory                             | Purpose                                                         |
| ---------------------------------- | -------------------------------------------- | --------------------------------------------------------------- |
| `run_single_gpu_experiments.sh`    | `results/raw/single_gpu/<run_id>/`           | Runs the single-GPU experiments used by Figures 9 and 10        |
| `run_tp_experiments.sh`            | `results/raw/tp/<run_id>/`                   | Runs tensor-parallel experiments used by the TP scaling figure 11 |
| `run_table2_cgct_coverage_nsys.sh` | `results/raw/table2_cgct_coverage/<run_id>/` | Runs Nsight Systems collection for Table 2 CUDA Graph coverage  |
| `run_table3_pi_copy_debug.sh`      | `results/raw/table3_pi_copy_debug/<run_id>/` | Runs debug builds/log collection for Table 3 data-copy movement |

Each experiment script writes a timestamped run directory and updates:

```text
results/raw/<experiment>/latest.txt
```

The analysis scripts use `latest.txt` by default.

Example:

```bash
bash scripts/run_single_gpu_experiments.sh
bash scripts/run_tp_experiments.sh
```

---

## 5. Figure and table generation scripts

These scripts post-process the raw experiment outputs and write paper-facing
artifacts under `figures/`, `tables/`, and `results/processed/`.

| Script                             | Purpose                                                         |
| ---------------------------------- | --------------------------------------------------------------- |
| `generate_figure9.sh`              | Generates Figure 9 from single-GPU results                      |
| `generate_figure10.sh`             | Generates Figure 10 from single-GPU results                     |
| `generate_single_gpu_figures.sh`   | Convenience wrapper for Figures 9 and 10                        |
| `generate_tp_figure.sh`            | Generates the TP scaling figure 11                              |
| `generate_table2_cgct_coverage.sh` | Processes Nsight traces and generates Table 2 coverage data     |
| `generate_table3_pi_copy_debug.sh` | Parses debug logs and generates Table 3 data-copy movement data |

Example:

```bash
bash scripts/generate_figure9.sh
bash scripts/generate_figure10.sh
bash scripts/generate_tp_figure.sh
bash scripts/generate_table2_cgct_coverage.sh
bash scripts/generate_table3_pi_copy_debug.sh
```

Use one of the conda enviroments in the docker container to generate the figures and tables.

```bash
conda activate grace-full
```
---

## 6. Analysis implementation

The Python implementations live in:

```text
scripts/analysis/
```

The shell wrappers in this directory should be preferred for normal use because
they resolve paths and latest run IDs automatically.

For debugging, the Python scripts can be called directly:

```bash
python3 scripts/analysis/plot_figure9.py --repo-root .
python3 scripts/analysis/make_tp_plot.py --repo-root . --run-id latest
```

See `scripts/analysis/README.md` for details.

---

## Recommended artifact workflow

Inside the prebuilt Docker container:

```bash
bash scripts/smoke_test.sh

bash scripts/run_single_gpu_experiments.sh
bash scripts/generate_figure9.sh  # in a conda environment
bash scripts/generate_figure10.sh

bash scripts/run_tp_experiments.sh
bash scripts/generate_tp_figure.sh  # in a conda environment

bash scripts/run_table2_cgct_coverage_nsys.sh
bash scripts/generate_table2_cgct_coverage.sh # in a conda environment

bash scripts/run_table3_pi_copy_debug.sh
bash scripts/generate_table3_pi_copy_debug.sh # in a conda environment
```

For quick validation, start with only:

```bash
bash scripts/smoke_test.sh
```

---

## Useful environment variables

Most experiment scripts support environment-variable overrides.

| Variable               | Meaning                                           |
| ---------------------- | ------------------------------------------------- |
| `WORKLOADS`            | Override the workload list file                   |
| `RESULTS_ROOT`         | Override the raw result output root               |
| `RUN_ID`               | Use a fixed run ID instead of a timestamp         |
| `REPEAT`               | Override benchmark repetitions                    |
| `TIMEOUT_SECS`         | Add a timeout around each run case                |
| `TP_LIST`              | Tensor parallel sizes for `run_tp_experiments.sh` |
| `INCLUDE_ABLATIONS`    | Include extra TP ablation variants                |
| `USE_NUMACTL`          | Enable/disable `numactl` in benchmark runner      |
| `CUDA_VISIBLE_DEVICES` | Select GPUs used by the run                       |

Example:

```bash
RUN_ID=test REPEAT=3 bash scripts/run_single_gpu_experiments.sh
```

---

## Notes

* The prebuilt Docker image should already contain all conda environments,
  PyTorch/GraCE variants, Triton builds, and workload dependencies.
* Build scripts are mostly for developers or evaluators who want to reproduce
  the image build from the base Dockerfile.
* Experiment scripts generate raw timestamped results.
* Generation scripts consume raw results and produce paper-facing outputs.
* For reproducibility, prefer the wrapper scripts in this directory over direct
  invocation of lower-level benchmark or analysis scripts.

