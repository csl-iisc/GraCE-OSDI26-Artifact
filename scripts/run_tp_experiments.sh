#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

WORKLOADS="${WORKLOADS:-$ARTIFACT_ROOT/configs/workloads_tp.txt}"
RESULTS_ROOT="${RESULTS_ROOT:-$ARTIFACT_ROOT/results/raw/tp}"
REPEAT="${REPEAT:-100}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TP_LIST="${TP_LIST:-1 2 4}"
INCLUDE_ABLATIONS="${INCLUDE_ABLATIONS:-0}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"

run_case() {
  local tp="$1"
  local variant="$2"
  local graph_mode="$3"

  local env_name
  env_name="$(variant_env "$variant")"

  activate_env "$env_name"

  local out_dir="$RESULTS_ROOT/$RUN_ID/TP-$tp/$variant/$graph_mode"
  mkdir -p "$out_dir"

  log_step "TP run: TP=$tp variant=$variant graph_mode=$graph_mode env=$env_name"

  local cmd=(
    bash "$BENCHMARK_RUNNER_DIR/run_workloads.sh"
    --variant "$variant"
    --graph-mode "$graph_mode"
    --tp "$tp"
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

for tp in $TP_LIST; do
  run_case "$tp" vanilla no-cg
  run_case "$tp" vanilla cg
  run_case "$tp" full cg

  if [[ "$INCLUDE_ABLATIONS" == "1" ]]; then
    run_case "$tp" cgct cg
    run_case "$tp" cgct-pi cg
  fi
done

mkdir -p "$RESULTS_ROOT"
printf '%s\n' "$RUN_ID" > "$RESULTS_ROOT/latest.txt"

log_ok "TP experiments complete."
echo "Results: $RESULTS_ROOT/$RUN_ID"
echo "Latest:  $RESULTS_ROOT/latest.txt"