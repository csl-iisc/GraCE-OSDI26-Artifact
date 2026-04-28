#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

WORKLOADS="${WORKLOADS:-$ARTIFACT_ROOT/configs/workloads_table3_pi_copy_debug.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-$ARTIFACT_ROOT/results/raw/table3_pi_copy_debug}"
REPEAT="${REPEAT:-3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"

run_case() {
  local variant="$1"
  local graph_mode="$2"

  local env_name
  env_name="$(variant_env "$variant")"

  activate_env "$env_name"

  local out_dir="$RESULTS_ROOT/$RUN_ID/$variant/$graph_mode"
  mkdir -p "$out_dir"

  log_step "Table 3 debug run: variant=$variant graph_mode=$graph_mode env=$env_name"

  local cmd=(
    bash "$BENCHMARK_RUNNER_DIR/run_workloads.sh"
    --variant "$variant"
    --graph-mode "$graph_mode"
    --tp 1
    --workloads "$WORKLOADS"
    --output-dir "$out_dir"
    --repeat "$REPEAT"
    --profile none
  )

  if [[ "$TIMEOUT_SECS" -gt 0 ]]; then
    timeout "$TIMEOUT_SECS" "${cmd[@]}" 2>&1 | tee "$out_dir/run.txt"
  else
    "${cmd[@]}" 2>&1 | tee "$out_dir/run.txt"
  fi
}

# Data-copy debug print before PI.
run_case cgct-copy-debug cg

# Data-copy debug print after PI.
run_case cgct-pi-copy-debug cg

# # Full GraCE log for SCG-disabled CUDA Graph counts.
# run_case full cg # I guess this you can directly get from the full logs

mkdir -p "$RESULTS_ROOT"
printf '%s\n' "$RUN_ID" > "$RESULTS_ROOT/latest.txt"

log_ok "Table 3 debug-log collection complete."
echo "Results: $RESULTS_ROOT/$RUN_ID"
echo "Latest:  $RESULTS_ROOT/latest.txt"
echo "Next: run analysis/make_table3.py on this directory."