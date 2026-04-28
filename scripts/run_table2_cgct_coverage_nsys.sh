#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

WORKLOADS="${WORKLOADS:-$ARTIFACT_ROOT/configs/workloads_table2_cgct_coverage.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-$ARTIFACT_ROOT/results/raw/table2_cgct_coverage}"
REPEAT="${REPEAT:-2}" 
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"

if ! command -v nsys >/dev/null 2>&1; then
  log_err "nsys not found. Install Nsight Systems in the Docker image before running Table 2 collection."
  exit 1
fi

run_case() {
  local variant="$1"
  local graph_mode="$2"

  local env_name
  env_name="$(variant_env "$variant")"

  activate_env "$env_name"

  local out_dir="$RESULTS_ROOT/$RUN_ID/$variant/$graph_mode"
  mkdir -p "$out_dir"

  log_step "Table 2 nsys run: variant=$variant graph_mode=$graph_mode"

  local cmd=(
    bash "$BENCHMARK_RUNNER_DIR/run_workloads.sh"
    --variant "$variant"
    --graph-mode "$graph_mode"
    --tp 1
    --workloads "$WORKLOADS"
    --output-dir "$out_dir"
    --repeat "$REPEAT"
    --profile nsys
  )

  if [[ "$TIMEOUT_SECS" -gt 0 ]]; then
    timeout "$TIMEOUT_SECS" "${cmd[@]}" 2>&1 | tee "$out_dir/run.txt"
  else
    "${cmd[@]}" 2>&1 | tee "$out_dir/run.txt"
  fi
}

# To get baseline PyTorch2 no-CG kernel 
run_case vanilla no-cg

# # Baseline PyTorch2-CG coverage.
# run_case vanilla cg

# # GraCE-CGCT coverage.
# run_case cgct cg

mkdir -p "$RESULTS_ROOT"
printf '%s\n' "$RUN_ID" > "$RESULTS_ROOT/latest.txt"

log_ok "Table 2 Nsight collection complete."
echo "Results: $RESULTS_ROOT/$RUN_ID"
echo "Latest:  $RESULTS_ROOT/latest.txt"
echo "Next: run analysis/make_table2.py on this directory."