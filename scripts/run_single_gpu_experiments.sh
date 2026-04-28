#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

WORKLOADS="${WORKLOADS:-$ARTIFACT_ROOT/configs/workloads_single_gpu_25_with_acronym.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-$ARTIFACT_ROOT/results/raw/single_gpu}"
REPEAT="${REPEAT:-100}"
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

  log_step "Single-GPU run: variant=$variant graph_mode=$graph_mode env=$env_name"

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

run_case vanilla no-cg
run_case vanilla cg
run_case cgct cg
run_case cgct-pi cg
run_case full cg

mkdir -p "$RESULTS_ROOT"
printf '%s\n' "$RUN_ID" > "$RESULTS_ROOT/latest.txt"

log_ok "Single-GPU experiments complete."
echo "Results: $RESULTS_ROOT/$RUN_ID"
echo "Latest:  $RESULTS_ROOT/latest.txt"