# Analysis Scripts

This directory contains the post-processing scripts used to convert raw GraCE
experiment outputs into paper figures and tables.

The scripts read results from:

```text
results/raw/
````

and write processed outputs to:

```text
results/processed/
figures/
tables/
```

---

## Directory contents

| File                           | Purpose                                                                |
| ------------------------------ | ---------------------------------------------------------------------- |
| `common.py`                    | Shared parsing, formatting, and plotting helpers                       |
| `plot_figure9.py`              | Generates Figure 9 from single-GPU results                             |
| `plot_figure10.py`             | Generates Figure 10 from single-GPU results                            |
| `make_tp_plot.py`              | Builds the TP summary and generates the TP scaling plot, Figure 11                |
| `make_table2_cgct_coverage.py` | Processes Nsight traces and generates Table 2 CUDA Graph coverage data |
| `make_table3_pi_copy_debug.py` | Parses debug logs and generates Table 3 data-copy movement data        |

---

## Normal usage

Most users should run the wrapper scripts from the repository root instead of
calling these Python scripts directly:

```bash
bash scripts/generate_figure9.sh # in a conda environment provided with the docker container
bash scripts/generate_figure10.sh
bash scripts/generate_tp_figure.sh
bash scripts/generate_table2_cgct_coverage.sh
bash scripts/generate_table3_pi_copy_debug.sh
```

The wrapper scripts resolve the latest run IDs and pass the correct input/output
paths to the analysis scripts.

---

## Input convention

Experiment scripts write timestamped raw results under:

```text
results/raw/<experiment>/<run_id>/
```

For example:

```text
results/raw/single_gpu/20260502-201906/
results/raw/tp/20260502-223735/
results/raw/table2_cgct_coverage/20260502-095126/
results/raw/table3_pi_copy_debug/20260502-100909/
```

Each experiment directory also maintains:

```text
latest.txt
```

which points to the most recent run ID. The analysis wrappers use this by
default.

---

## Output convention

Processed summaries are written under:

```text
results/processed/<experiment>/<run_id>/
```

Paper-facing outputs are written under:

```text
figures/
tables/
```

Typical outputs include:

```text
figures/figure9.pdf
figures/figure10.pdf
figures/figure11.pdf
tables/table2_cgct_coverage.csv
tables/table3_pi_copy_debug.csv
```

Run-specific copies may also be written under:

```text
figures/<run_id>/
tables/<run_id>/
```

---

## Direct invocation

The Python scripts can also be run directly for debugging. For example:

```bash
python3 scripts/analysis/plot_figure9.py --repo-root .
python3 scripts/analysis/plot_figure10.py --repo-root .
python3 scripts/analysis/make_tp_plot.py --repo-root . --run-id latest
python3 scripts/analysis/make_table2_cgct_coverage.py --repo-root . --run-id latest
python3 scripts/analysis/make_table3_pi_copy_debug.py --repo-root . --run-id latest
```

Use `--help` on each script to see available options:

```bash
python3 scripts/analysis/make_tp_plot.py --help
```

---

## Notes

* These scripts assume the raw result directory layout produced by the artifact
  experiment scripts.
* Figure scripts preserve the paper-style plotting format, model acronyms, and
  legend layout.
* Table 2 depends on Nsight Systems traces collected by
  `run_table2_cgct_coverage_nsys.sh`.
* Table 3 depends on debug logs collected by
  `run_table3_pi_copy_debug.sh`.
* For reproducibility, prefer using the wrapper scripts in `scripts/` rather
  than manually invoking the analysis scripts.

