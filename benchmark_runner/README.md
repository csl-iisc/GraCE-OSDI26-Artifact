# Benchmark Runner

This directory contains the workload launcher used by the GraCE artifact
experiment scripts.

The main entry point is:

```bash
benchmark_runner/run_workloads.sh
````

It selects the appropriate benchmark implementation, runs the requested models,
and writes raw CSV outputs under the experiment-specific results directory.

---

## Directory layout

```text
benchmark_runner/
├── run_workloads.sh
└── my_benchmarks/
    ├── dynamo-separate-eager-report-more-stats/
    ├── dynamo-separate-eager-report-more-stats-nvtx/
    ├── dynamo-separate-eager-report-more-stats-nccl-support/
    └── dynamo-separate-eager-report-more-stats-nccl-support-nvtx/
```

---

## Benchmark variants

| Directory                                                   | Used for                                        |
| ----------------------------------------------------------- | ----------------------------------------------- |
| `dynamo-separate-eager-report-more-stats`                   | Single-GPU runs without Nsight profiling        |
| `dynamo-separate-eager-report-more-stats-nvtx`              | Single-GPU runs with Nsight/NVTX profiling      |
| `dynamo-separate-eager-report-more-stats-nccl-support`      | Tensor-parallel runs without Nsight profiling   |
| `dynamo-separate-eager-report-more-stats-nccl-support-nvtx` | Tensor-parallel runs with Nsight/NVTX profiling |

`run_workloads.sh` chooses the correct directory automatically from:

* `--tp`
* `--profile`

---

## Usage

Normally, users should not call `run_workloads.sh` directly. It is invoked by
the higher-level experiment scripts in `scripts/`, for example:

```bash
bash scripts/run_single_gpu_experiments.sh
bash scripts/run_tp_experiments.sh
bash scripts/run_table2_cgct_coverage_nsys.sh
bash scripts/run_table3_pi_copy_debug.sh
```

For debugging, it can be run manually:

```bash
bash benchmark_runner/run_workloads.sh \
  --variant full \
  --graph-mode cg \
  --tp 1 \
  --workloads configs/workloads_smoke.txt \
  --output-dir results/raw/debug/full/cg \
  --repeat 3 \
  --profile none
```

---

## Important arguments

| Argument       | Meaning                                                                    |
| -------------- | -------------------------------------------------------------------------- |
| `--variant`    | GraCE/PyTorch variant to run: `vanilla`, `cgct`, `cgct-pi`, `full`, etc. |
| `--graph-mode` | Execution mode: `no-cg`, `cg`, or `eager`                                  |
| `--tp`         | Tensor parallel size: `1`, `2`, `4`, or `8`                                |
| `--workloads`  | Workload list file                                                         |
| `--output-dir` | Directory where CSVs, logs, metadata, and Nsight traces are written        |
| `--repeat`     | Number of benchmark repetitions                                            |
| `--profile`    | `none` or `nsys`                                                           |

---

## Workload file format

Each workload line may contain either four or five fields.

Without acronym:

```text
<Model> <Suite> <Mode> <BatchSize>
```

With acronym:

```text
<Acronym> <Model> <Suite> <Mode> <BatchSize>
```

Example:

```text
XLNET-I XLNetLMHeadModel huggingface inference 1
ST speech_transformer torchbench inference 1
VM vision_maskrcnn torchbench inference 1
```

Supported suites are:

```text
huggingface
torchbench
timm
```

The script normalizes `TIMM`/`timm` to `timm_models`.

---

## Outputs

Each run writes:

```text
<output-dir>/
├── metadata.txt
├── run.txt
├── nsys/
└── *.csv
```

For TP runs, per-rank CSV files are produced:

```text
*_rank_0.csv
*_rank_1.csv
...
```

These raw outputs are consumed by the analysis scripts under `scripts/analysis/`
to generate paper figures and tables.

