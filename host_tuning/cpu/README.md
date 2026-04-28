# CPU Host Tuning

This directory contains CPU-side host tuning utilities for GraCE performance evaluation.

The scripts control:

- CPU frequency pinning, Intel pstate behavior;
- CPU governors;
- deep idle states such as C6;
- hyperthreading / SMT sibling CPUs.

These scripts are host-side scripts. Run them on the bare-metal host, not inside Docker.

---

## Scripts

| Script | Purpose |
|---|---|
| `intel_pin_cpu_freq.sh` | Pins Intel CPU min/max frequency and sets a stable governor. |
| `intel_restore_cpu_freq.sh` | Restores CPU frequency policy from backup. Also works as the generic restore path for `amd_pin_cpu_freq.sh` if the backup file exists. |
| `amd_pin_cpu_freq.sh` | Pins AMD CPU frequency using the configured AMD target. |
| `disable_hyperthreading.sh` | Offlines hyperthread/SMT sibling CPUs. |
| `enable_hyperthreading.sh` | Re-onlines CPUs previously disabled by `disable_hyperthreading.sh`. |
| `disable_c6.sh` | Disables C6 idle states through cpuidle sysfs. |
| `enable_c6.sh` | Restores C6 idle states from backup. |

---

## Why CPU tuning matters

GraCE evaluates CUDA Graph behavior. CUDA Graphs reduce CPU-side launch overhead, so CPU-side jitter can strongly affect measured performance.

Important CPU-side noise sources include:

- frequency scaling;
- turbo boost transitions;
- deep idle state wake-up latency;
- hyperthread sibling interference;
- background processes scheduled on benchmark CPUs.

The CPU scripts reduce these sources of variation.

---

## Intel CPU frequency pinning

Main script:

```bash
sudo host_tuning/cpu/intel_pin_cpu_freq.sh
```

Default behavior:

- switches `intel_pstate` to passive mode when available;
- sets governor to `performance`;
- pins `scaling_min_freq` and `scaling_max_freq` to `INTEL_TARGET_FREQ_KHZ`.

Default target for the authors' machine:

```text
INTEL_TARGET_FREQ_KHZ=2800000
```

Override example:

```bash
INTEL_TARGET_FREQ_KHZ=2800000 sudo -E host_tuning/cpu/intel_pin_cpu_freq.sh
```

Optional OS-level turbo disable:

```bash
DISABLE_TURBO_OS=1 sudo -E host_tuning/cpu/intel_pin_cpu_freq.sh
```

On the authors' Intel Xeon Platinum 8462Y+ machine, turbo was disabled at BIOS level. 

Restore:

```bash
sudo host_tuning/cpu/intel_restore_cpu_freq.sh
```

---

## AMD CPU frequency pinning

Main script:

```bash
sudo host_tuning/cpu/amd_pin_cpu_freq.sh
```

Default target:

```text
AMD_TARGET_FREQ_KHZ=3100000
```

Recommended governor for benchmark stability:

```text
performance
```

Example:

```bash
AMD_TARGET_FREQ_KHZ=3100000 sudo -E host_tuning/cpu/amd_pin_cpu_freq.sh
```

Restore using the generic backup path:

```bash
sudo host_tuning/cpu/intel_restore_cpu_freq.sh
```

The restore script name is Intel-oriented, but it restores from the generic CPU frequency backup file and works for the AMD pinning path as well.

---

## Hyperthreading / SMT control

Disable sibling logical CPUs:

```bash
sudo host_tuning/cpu/disable_hyperthreading.sh
```

Restore them:

```bash
sudo host_tuning/cpu/enable_hyperthreading.sh
```

The script can auto-detect sibling CPUs from:

```text
/sys/devices/system/cpu/cpu*/topology/thread_siblings_list
```

You can also specify explicit CPUs:

```bash
HT_CPUS=64-127 sudo -E host_tuning/cpu/disable_hyperthreading.sh
```

For the authors' HT-enabled Intel + H100 host, sibling CPUs were:

```text
64-127
```

When HT is disabled, online CPUs become:

```text
0-63
```

---

## C-states and C6 control

Disable C6:

```bash
sudo host_tuning/cpu/disable_c6.sh
```

Restore C6:

```bash
sudo host_tuning/cpu/enable_c6.sh
```

The scripts operate through:

```text
/sys/devices/system/cpu/cpu*/cpuidle/state*/disable
```

On Intel server CPUs, the important deep idle state is usually named:

```text
C6
```

On AMD systems, equivalent deep idle states may appear as:

```text
C6
CC6
PC6
C2
```

For AMD, inspect the actual states first:

```bash
cpupower idle-info
```

or:

```bash
for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
  echo "=== $s ==="
  cat "$s/name" 2>/dev/null || true
  cat "$s/latency" 2>/dev/null || true
  cat "$s/disable" 2>/dev/null || true
  echo
 done
```

---

## Verification

After CPU tuning, verify governors and frequency limits:

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq | sort | uniq -c
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq | sort | uniq -c
```

For the authors' Intel machine, expected output after tuning is similar to:

```text
64 performance
64 2800000
64 2800000
```

Check online CPUs:

```bash
cat /sys/devices/system/cpu/online
```

Check C6 disable state:

```bash
for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
  echo -n "$(cat $s/name 2>/dev/null): "
  cat $s/disable 2>/dev/null || true
 done
```

---

## Safety and restore

Always restore after benchmark runs:

```bash
sudo host_tuning/cpu/enable_hyperthreading.sh
sudo host_tuning/cpu/enable_c6.sh
sudo host_tuning/cpu/intel_restore_cpu_freq.sh
```

The higher-level restore script runs these for the default Intel + H100 path:

```bash
sudo host_tuning/restore_host_intel_h100.sh
```

Keep an SSH session open when changing CPU topology or cgroups remotely.
