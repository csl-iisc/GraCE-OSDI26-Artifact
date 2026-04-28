#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

VARIANTS_TO_BUILD="${VARIANTS_TO_BUILD:-vanilla cgct cgct-pi full cgct-copy-debug cgct-pi-copy-debug}"

for variant in $VARIANTS_TO_BUILD; do
  log_step "Building variant: $variant"
  bash "$SCRIPT_DIR/build_pytorch_variant.sh" --variant "$variant"
done

log_ok "Requested variants built: $VARIANTS_TO_BUILD"

# question is who drives it?