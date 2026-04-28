#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SMOKE_WORKLOAD="$ARTIFACT_ROOT/configs/workloads_smoke.txt"
cat > "$SMOKE_WORKLOAD" <<'EOF'
ST speech_transformer torchbench inference 1
EOF

RESULTS_ROOT="${RESULTS_ROOT:-$ARTIFACT_ROOT/results/raw/smoke}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
REPEAT="${REPEAT:-3}"

activate_env grace-full

out_dir="$RESULTS_ROOT/$RUN_ID/full/cg"
mkdir -p "$out_dir"

bash "$BENCHMARK_RUNNER_DIR/run_workloads.sh" \
  --variant full \
  --graph-mode cg \
  --tp 1 \
  --workloads "$SMOKE_WORKLOAD" \
  --output-dir "$out_dir" \
  --repeat "$REPEAT" \
  --profile none \
  2>&1 | tee "$out_dir/run.txt"

log_ok "Smoke test complete."
echo "Results: $out_dir"