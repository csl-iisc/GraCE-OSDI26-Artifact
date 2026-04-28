#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RUN_ID="${RUN_ID:-latest}"
TP_VALUES="${TP_VALUES:-1 2 4}"
SAVE_SVG="${SAVE_SVG:-0}"

cmd=(
  python "$ARTIFACT_ROOT/scripts/analysis/make_tp_plot.py"
  --repo-root "$ARTIFACT_ROOT"
  --run-id "$RUN_ID"
  --tp-values "$TP_VALUES"
)

if [[ "$SAVE_SVG" == "1" ]]; then
  cmd+=(--save-svg)
fi

"${cmd[@]}"

log_ok "TP figure generation complete."
echo "Figure: $ARTIFACT_ROOT/figures/figure11.pdf"
