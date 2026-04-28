# Configs

This directory contains workload lists used by the GraCE artifact experiment
scripts.

Each file specifies the models, benchmark suites, execution modes, and batch
sizes to run.

---

## Files

| File | Used by |
|---|---|
| `workloads_smoke.txt` | Quick smoke test |
| `workloads_single_gpu_25.txt` | Full single-GPU experiment without acronyms |
| `workloads_single_gpu_25_with_acronym.txt` | Full single-GPU experiment with paper-style acronyms |
| `workloads_tp.txt` | Tensor-parallel experiments |
| `workloads_table2_cgct_coverage.txt` | Table 2 CUDA Graph coverage collection |
| `workloads_table3_pi_copy_debug.txt` | Table 3 data-copy debug collection |

---

## Format

Workload lines use either four or five fields.

Without acronym:

```text
<Model> <Suite> <Mode> <BatchSize>
````

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

Blank lines and lines starting with `#` are ignored.

---

## Suites

Supported suite names are:

```text
huggingface
torchbench
timm
```

The benchmark runner normalizes `timm`/`TIMM` to `timm_models`.

---

## Notes

These files are consumed by the scripts under `scripts/`, which call:

```bash
benchmark_runner/run_workloads.sh
```

Most users should not need to edit these files unless they want to add, remove,
or debug specific workloads.

