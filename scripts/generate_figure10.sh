#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERROR] $0 failed at line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="${1:-latest}"

python "$REPO_ROOT/scripts/analysis/plot_figure10.py" \
  --repo-root "$REPO_ROOT" \
  --run-id "$RUN_ID"
