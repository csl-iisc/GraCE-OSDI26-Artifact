#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO" >&2' ERR

# Debug Docker build wrapper for GraCE OSDI'26 artifact.
#
# This script is intentionally verbose. It:
#   - forces BuildKit plain progress
#   - writes complete build output to docker/logs/
#   - uses a BuildKit builder with a large per-step log buffer
#   - preserves the real docker build exit status even when piping through tee
#   - prints useful host/repo/build metadata before starting
#
# Example:
#   bash docker/build_image_debug.sh --cuda 128 --prebuilt --max-jobs 32
#
# Useful env overrides:
#   USE_LARGE_LOG_BUILDER=1
#   BUILDKIT_STEP_LOG_MAX_SIZE=1073741824
#   BUILDKIT_STEP_LOG_MAX_SPEED=104857600
#   BUILDER_NAME=grace-large-log-builder-debug
#   NO_CACHE=1
#   LOAD_IMAGE=1

usage() {
  cat <<'EOF'
Usage:
  docker/build_image_debug.sh [options]

Options:
  --cuda <124|128>          CUDA image family. Default: 124
  --prebuilt                Build full prebuilt artifact image. Sets BUILD_ARTIFACT=1
  --base                    Build base/developer image. Sets BUILD_ARTIFACT=0
  --image <name:tag>        Override output image tag
  --dockerfile <path>       Override Dockerfile path
  --max-jobs <N>            Set MAX_JOBS build arg. Default: 32
  --install-nsys <0|1>      Set INSTALL_NSYS build arg. Default: 1
  --no-cache                Pass --no-cache to docker buildx build
  --pull                    Pass --pull to docker buildx build
  --no-large-log-builder    Do not create/use custom BuildKit large-log builder
  --no-load                 Do not --load image into local docker image store
  --build-arg KEY=VALUE     Add an extra Docker build arg
  --help                    Show this help

Examples:
  bash docker/build_image_debug.sh --cuda 128 --prebuilt --max-jobs 32

  NO_CACHE=1 bash docker/build_image_debug.sh --cuda 128 --base

  BUILDKIT_STEP_LOG_MAX_SIZE=-1 \
  BUILDKIT_STEP_LOG_MAX_SPEED=-1 \
  bash docker/build_image_debug.sh --cuda 128 --prebuilt

Logs:
  docker/logs/build_cuda<CUDASHORT>_<kind>_<timestamp>.log
EOF
}

log() {
  echo "[INFO] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_SHORT="${CUDA_SHORT:-124}"
KIND="${KIND:-prebuilt}"
BUILD_ARTIFACT="${BUILD_ARTIFACT:-1}"
INSTALL_NSYS="${INSTALL_NSYS:-1}"
MAX_JOBS="${MAX_JOBS:-32}"
IMAGE=""
DOCKERFILE=""
NO_CACHE="${NO_CACHE:-0}"
PULL="${PULL:-0}"
LOAD_IMAGE="${LOAD_IMAGE:-1}"

USE_LARGE_LOG_BUILDER="${USE_LARGE_LOG_BUILDER:-1}"
BUILDER_NAME="${BUILDER_NAME:-grace-large-log-builder-debug}"
BUILDKIT_STEP_LOG_MAX_SIZE="${BUILDKIT_STEP_LOG_MAX_SIZE:-1073741824}"
BUILDKIT_STEP_LOG_MAX_SPEED="${BUILDKIT_STEP_LOG_MAX_SPEED:-104857600}"

EXTRA_BUILD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA_SHORT="$2"
      shift 2
      ;;
    --prebuilt)
      KIND="prebuilt"
      BUILD_ARTIFACT=1
      shift
      ;;
    --base)
      KIND="base"
      BUILD_ARTIFACT=0
      shift
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE="$2"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    --install-nsys)
      INSTALL_NSYS="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --pull)
      PULL=1
      shift
      ;;
    --no-large-log-builder)
      USE_LARGE_LOG_BUILDER=0
      shift
      ;;
    --no-load)
      LOAD_IMAGE=0
      shift
      ;;
    --build-arg)
      EXTRA_BUILD_ARGS+=(--build-arg "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

case "$CUDA_SHORT" in
  124|12.4|cuda124)
    CUDA_SHORT=124
    CUDA_LABEL="cuda124"
    ;;
  128|12.8|cuda128)
    CUDA_SHORT=128
    CUDA_LABEL="cuda128"
    ;;
  *)
    die "Unsupported CUDA value: $CUDA_SHORT. Expected 124 or 128."
    ;;
esac

if [[ -z "$DOCKERFILE" ]]; then
  DOCKERFILE="$ARTIFACT_ROOT/docker/Dockerfile.${CUDA_LABEL}"
fi

if [[ -z "$IMAGE" ]]; then
  if [[ "$BUILD_ARTIFACT" == "1" ]]; then
    IMAGE="grace-osdi26:${CUDA_LABEL}-prebuilt"
  else
    IMAGE="grace-osdi26:${CUDA_LABEL}-base"
  fi
fi

[[ -f "$DOCKERFILE" ]] || die "Dockerfile not found: $DOCKERFILE"

mkdir -p "$ARTIFACT_ROOT/docker/logs"

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="$ARTIFACT_ROOT/docker/logs/build_${CUDA_LABEL}_${KIND}_${timestamp}.log"
cmd_file="$ARTIFACT_ROOT/docker/logs/build_${CUDA_LABEL}_${KIND}_${timestamp}.cmd.sh"

command -v docker >/dev/null 2>&1 || die "docker not found"
docker buildx version >/dev/null 2>&1 || die "docker buildx is not available"

BUILDER_ARGS=()
if [[ "$USE_LARGE_LOG_BUILDER" == "1" ]]; then
  if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    log "Creating BuildKit builder with large step log limits: $BUILDER_NAME"
    docker buildx create \
      --name "$BUILDER_NAME" \
      --driver docker-container \
      --driver-opt "env.BUILDKIT_STEP_LOG_MAX_SIZE=${BUILDKIT_STEP_LOG_MAX_SIZE}" \
      --driver-opt "env.BUILDKIT_STEP_LOG_MAX_SPEED=${BUILDKIT_STEP_LOG_MAX_SPEED}" \
      --use >/dev/null
  else
    log "Using existing BuildKit builder: $BUILDER_NAME"
    docker buildx use "$BUILDER_NAME"
  fi

  docker buildx inspect --bootstrap "$BUILDER_NAME" >/dev/null
  BUILDER_ARGS=(--builder "$BUILDER_NAME")
fi

BUILD_ARGS=(
  --progress=plain
  -f "$DOCKERFILE"
  --build-arg "BUILD_ARTIFACT=${BUILD_ARTIFACT}"
  --build-arg "INSTALL_NSYS=${INSTALL_NSYS}"
  --build-arg "MAX_JOBS=${MAX_JOBS}"
  -t "$IMAGE"
)

if [[ "$LOAD_IMAGE" == "1" ]]; then
  BUILD_ARGS+=(--load)
fi

if [[ "$NO_CACHE" == "1" ]]; then
  BUILD_ARGS+=(--no-cache)
fi

if [[ "$PULL" == "1" ]]; then
  BUILD_ARGS+=(--pull)
fi

BUILD_ARGS+=("${EXTRA_BUILD_ARGS[@]}")
BUILD_ARGS+=("$ARTIFACT_ROOT")

echo "============================================================"
echo "GraCE debug Docker build"
echo "============================================================"
echo "Artifact root:              $ARTIFACT_ROOT"
echo "Dockerfile:                 $DOCKERFILE"
echo "CUDA label:                 $CUDA_LABEL"
echo "Kind:                       $KIND"
echo "Image:                      $IMAGE"
echo "BUILD_ARTIFACT:             $BUILD_ARTIFACT"
echo "INSTALL_NSYS:               $INSTALL_NSYS"
echo "MAX_JOBS:                   $MAX_JOBS"
echo "NO_CACHE:                   $NO_CACHE"
echo "PULL:                       $PULL"
echo "LOAD_IMAGE:                 $LOAD_IMAGE"
echo "Use large log builder:      $USE_LARGE_LOG_BUILDER"
echo "Builder name:               ${BUILDER_NAME}"
echo "BuildKit log max size:      ${BUILDKIT_STEP_LOG_MAX_SIZE}"
echo "BuildKit log max speed:     ${BUILDKIT_STEP_LOG_MAX_SPEED}"
echo "Log file:                   $log_file"
echo "Command file:               $cmd_file"
echo "============================================================"
echo

{
  echo "============================================================"
  echo "Host/build metadata"
  echo "============================================================"
  date
  echo
  echo "uname:"
  uname -a || true
  echo
  echo "docker version:"
  docker version || true
  echo
  echo "docker buildx version:"
  docker buildx version || true
  echo
  echo "docker buildx ls:"
  docker buildx ls || true
  echo
  echo "nvidia-smi:"
  nvidia-smi || true
  echo
  echo "git status:"
  git -C "$ARTIFACT_ROOT" status --short || true
  echo
  echo "git submodule status:"
  git -C "$ARTIFACT_ROOT" submodule status --recursive || true
  echo
  echo "Dockerfile digest:"
  sha256sum "$DOCKERFILE" || true
  echo
} | tee "$log_file"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$ARTIFACT_ROOT"
  printf 'DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain docker buildx build'
  for arg in "${BUILDER_ARGS[@]}" "${BUILD_ARGS[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
} > "$cmd_file"
chmod +x "$cmd_file"

echo | tee -a "$log_file"
echo "============================================================" | tee -a "$log_file"
echo "Build command" | tee -a "$log_file"
echo "============================================================" | tee -a "$log_file"
cat "$cmd_file" | tee -a "$log_file"
echo | tee -a "$log_file"

set +e
DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain \
docker buildx build \
  "${BUILDER_ARGS[@]}" \
  "${BUILD_ARGS[@]}" \
  2>&1 | tee -a "$log_file"

status="${PIPESTATUS[0]}"
set -e

echo | tee -a "$log_file"
echo "============================================================" | tee -a "$log_file"
if [[ "$status" -eq 0 ]]; then
  echo "[OK] Docker build succeeded" | tee -a "$log_file"
  echo "Image: $IMAGE" | tee -a "$log_file"
else
  echo "[ERROR] Docker build failed with status $status" | tee -a "$log_file"
  echo "[INFO] Full log: $log_file" | tee -a "$log_file"
  echo "[INFO] Re-run command: $cmd_file" | tee -a "$log_file"
  echo | tee -a "$log_file"
  echo "[INFO] Last 200 log lines:" | tee -a "$log_file"
  tail -n 200 "$log_file" | tee -a "$log_file"
fi
echo "============================================================" | tee -a "$log_file"

exit "$status"
