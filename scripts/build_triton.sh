#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
  echo "Usage: $0 --variant <vanilla|cgct|cgct-pi|full|debug-copy-vanilla|debug-copy-cgct-pi>"
}

VARIANT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$VARIANT" ]]; then
  usage
  exit 1
fi

ENV_NAME="$(variant_env "$VARIANT")"
TRITON_BRANCH="$(variant_triton_branch "$VARIANT")"

activate_env "$ENV_NAME"

log_step "Building Triton for variant=$VARIANT env=$ENV_NAME branch=$TRITON_BRANCH"

cd "$TRITON_DIR"
git checkout "$TRITON_BRANCH"
git submodule update --init --recursive || true

pip install ninja cmake==3.28.3 wheel
pip install -e python --verbose

log_ok "Triton build complete for $VARIANT"