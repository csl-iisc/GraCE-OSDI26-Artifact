# GPU Host Tuning

This directory contains NVIDIA GPU clock tuning scripts used for low-noise GraCE performance evaluation.

The scripts are host-side scripts. Run them on the bare-metal host, not inside Docker.

---

## Scripts

| Script | Purpose |
|---|---|
| `lock_gpu_clocks.sh` | Enables NVIDIA persistence mode and locks GPU graphics clocks. Optionally locks memory clocks. |
| `reset_gpu_clocks.sh` | Resets GPU graphics and memory clock locks. Optionally disables persistence mode. |

---

## Why GPU clock tuning matters

GPU clocks can change dynamically due to power management, thermal behavior, or boost policy. For performance experiments, especially those comparing CUDA Graph behavior, dynamic GPU clocks can introduce run-to-run variability.

Locking GPU clocks makes the GPU execution environment more stable.

---

## Lock GPU clocks

From the artifact repository root:

```bash
sudo host_tuning/gpu/lock_gpu_clocks.sh
```

Default behavior:

- sets persistence mode with `nvidia-smi -pm 1`;
- locks graphics clock using `nvidia-smi -lgc`;
- records a small backup/state file under `/run/grace_host_tuning_backup`.

Default clock for the authors' H100 NVL host:

```text
GPU_CLOCK_MHZ=1410  # ~80% of the peak
```

Override example:

```bash
GPU_CLOCK_MHZ=1410 sudo -E host_tuning/gpu/lock_gpu_clocks.sh
```

Tune selected GPUs only:

```bash
GPU_IDS=0 sudo -E host_tuning/gpu/lock_gpu_clocks.sh
```

Tune all GPUs:

```bash
GPU_IDS=all sudo -E host_tuning/gpu/lock_gpu_clocks.sh
```

---

## Optional memory clock lock

Memory clock locking is disabled by default.

If you know a valid memory clock for your GPU and driver, set:

```bash
GPU_MEM_CLOCK_MHZ=<clock> sudo -E host_tuning/gpu/lock_gpu_clocks.sh
```

Check supported clocks with:

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
```

On some systems or drivers, memory clock locking may not be supported or may require additional permissions.

---

## Reset GPU clocks

After experiments:

```bash
sudo host_tuning/gpu/reset_gpu_clocks.sh
```

This runs:

```text
nvidia-smi -rgc
nvidia-smi -rmc
```

for the selected GPUs.

By default, persistence mode is left enabled. To disable persistence mode too:

```bash
RESET_PERSISTENCE=1 sudo -E host_tuning/gpu/reset_gpu_clocks.sh
```

---

## Verification

Check current GPU clocks:

```bash
nvidia-smi --query-gpu=index,name,persistence_mode,clocks.gr,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv
```

Detailed clock information:

```bash
nvidia-smi -q -d CLOCK
```

Supported clock settings:

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
```

Check topology:

```bash
nvidia-smi topo -m
```

---

## Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `GPU_IDS` | `all` | GPU IDs to tune. Use `all`, `0`, or comma-separated values such as `0,1`. |
| `GPU_CLOCK_MHZ` | `1410` | Graphics clock target in MHz. |
| `GPU_MEM_CLOCK_MHZ` | empty | Optional memory clock target in MHz. |
| `RESET_PERSISTENCE` | `0` | If `1`, `reset_gpu_clocks.sh` disables persistence mode after resetting clocks. |

---

## Safety notes

- Use a valid clock for the GPU model and driver.
- Keep the GPU below thermal and power throttling limits.
- Reset clock locks after experiments.
- These settings affect the whole host, not only Docker.
