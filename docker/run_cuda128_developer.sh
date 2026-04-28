#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_IMAGE="${DEFAULT_IMAGE:-grace-osdi26:cuda128-base}" \
DEVELOPER_MODE=1 \
exec "${SCRIPT_DIR}/run_common.sh" "$@"
