#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RESULTS_ROOT="${RESULTS_ROOT:-$ARTIFACT_ROOT/results/raw/table2_cgct_coverage}"
WORKLOADS="${WORKLOADS:-$ARTIFACT_ROOT/configs/workloads_table2_cgct_coverage.txt}"
RUN_ID="${RUN_ID:-latest}"
FORCE="${FORCE:-0}"

if [[ "$RUN_ID" == "latest" ]]; then
  LATEST_FILE="$RESULTS_ROOT/latest.txt"
  if [[ ! -f "$LATEST_FILE" ]]; then
    log_err "latest.txt not found: $LATEST_FILE"
  fi
  RUN_ID="$(cat "$LATEST_FILE" | tr -d '[:space:]')"
fi

RUN_DIR="$RESULTS_ROOT/$RUN_ID"

if [[ ! -d "$RUN_DIR" ]]; then
  log_err "Table 2 run directory not found: $RUN_DIR"
fi

if [[ ! -f "$WORKLOADS" ]]; then
  log_err "Workload file not found: $WORKLOADS"
fi

if ! command -v nsys >/dev/null 2>&1; then
  log_err "nsys not found. Install Nsight Systems before generating Table 2 traces."
fi

process_case() {
  local variant="$1"
  local graph_mode="$2"

  local case_dir="$RUN_DIR/$variant/$graph_mode"
  local nsys_dir="$case_dir/nsys"
  local sqlite_dir="$case_dir/sqlite"
  local trace_dir="$case_dir/cuda_kern_exec_trace"

  if [[ ! -d "$nsys_dir" ]]; then
    log_err "Missing nsys directory: $nsys_dir"
  fi

  mkdir -p "$sqlite_dir" "$trace_dir"

  log_step "Processing Table 2 case: variant=$variant graph_mode=$graph_mode"
  log_step "Generating sqlite files from: $nsys_dir"

  shopt -s nullglob
  local reps=("$nsys_dir"/*.nsys-rep)
  shopt -u nullglob

  if [[ "${#reps[@]}" -eq 0 ]]; then
    log_err "No .nsys-rep files found in $nsys_dir"
  fi

  for rep in "${reps[@]}"; do
    local base
    base="$(basename "$rep" .nsys-rep)"
    local sqlite="$sqlite_dir/$base.sqlite"

    if [[ "$FORCE" == "1" || ! -s "$sqlite" ]]; then
      echo ">>> nsys export: $rep -> $sqlite"
      nsys export \
        --type sqlite \
        --force-overwrite true \
        --output "$sqlite" \
        "$rep"
    else
      echo ">>> sqlite exists, skipping: $sqlite"
    fi
  done

  log_step "Generating cuda_kern_exec_trace CSV files from: $sqlite_dir"

  shopt -s nullglob
  local sqlites=("$sqlite_dir"/*.sqlite)
  shopt -u nullglob

  if [[ "${#sqlites[@]}" -eq 0 ]]; then
    log_err "No .sqlite files found in $sqlite_dir"
  fi

  for sqlite in "${sqlites[@]}"; do
    local base
    base="$(basename "$sqlite" .sqlite)"
    local expected_trace="$trace_dir/${base}_cuda_kern_exec_trace.csv"

    if [[ "$FORCE" == "1" || ! -s "$expected_trace" ]]; then
      echo ">>> nsys stats: $sqlite -> $expected_trace"
      rm -f "$trace_dir/${base}"*cuda_kern_exec_trace*.csv
      nsys stats \
        -r cuda_kern_exec_trace \
        "$sqlite" \
        -o "$trace_dir/$base"
    else
      echo ">>> cuda_kern_exec_trace exists, skipping: $expected_trace"
    fi
  done
}

process_case vanilla no-cg
process_case vanilla cg
process_case cgct cg

log_step "Generating Table 2 CSV/text outputs"

python "$ARTIFACT_ROOT/scripts/analysis/make_table2_cgct_coverage.py" \
  --repo-root "$ARTIFACT_ROOT" \
  --run-id "$RUN_ID" \
  --workloads "$WORKLOADS"

log_ok "Table 2 generation complete."
echo "Raw run:      $RUN_DIR"
echo "Processed:    $ARTIFACT_ROOT/results/processed/table2_cgct_coverage/$RUN_ID"
echo "Stable table: $ARTIFACT_ROOT/tables/table2.csv"