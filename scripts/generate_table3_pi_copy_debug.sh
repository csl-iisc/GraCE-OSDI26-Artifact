#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RESULTS_ROOT="${RESULTS_ROOT:-$ARTIFACT_ROOT/results/raw/table3_pi_copy_debug}"
WORKLOADS="${WORKLOADS:-$ARTIFACT_ROOT/configs/workloads_table3_pi_copy_debug.txt}"
RUN_ID="${RUN_ID:-latest}"

if [[ "$RUN_ID" == "latest" ]]; then
  LATEST_FILE="$RESULTS_ROOT/latest.txt"
  if [[ ! -f "$LATEST_FILE" ]]; then
    log_err "latest.txt not found: $LATEST_FILE"
  fi
  RUN_ID="$(tr -d '[:space:]' < "$LATEST_FILE")"
fi

RUN_DIR="$RESULTS_ROOT/$RUN_ID"

if [[ ! -d "$RUN_DIR" ]]; then
  log_err "Table 3 run directory not found: $RUN_DIR"
fi

if [[ ! -f "$WORKLOADS" ]]; then
  log_err "Workload file not found: $WORKLOADS"
fi

DIRECT_LOG="$RUN_DIR/cgct-copy-debug/cg/run.txt"
INDIRECT_LOG="$RUN_DIR/cgct-pi-copy-debug/cg/run.txt"

if [[ ! -f "$DIRECT_LOG" ]]; then
  log_err "Direct/cgct-copy-debug log not found: $DIRECT_LOG"
fi

if [[ ! -f "$INDIRECT_LOG" ]]; then
  log_err "Indirect/cgct-pi-copy-debug log not found: $INDIRECT_LOG"
fi

log_step "Generating Table 3 PI data-copy reduction table"
log_step "Run id: $RUN_ID"

python "$ARTIFACT_ROOT/scripts/analysis/make_table3_pi_copy_debug.py" \
  --repo-root "$ARTIFACT_ROOT" \
  --run-id "$RUN_ID" \
  --workloads "$WORKLOADS"

log_ok "Table 3 generation complete."
echo "Raw run:      $RUN_DIR"
echo "Processed:    $ARTIFACT_ROOT/results/processed/table3_pi_copy_debug/$RUN_ID"
echo "Stable table: $ARTIFACT_ROOT/tables/table3.csv"
