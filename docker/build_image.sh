#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO" >&2' ERR

usage() {
  cat <<'USAGE'
Usage:
  docker/build_image.sh --cuda <cuda124|cuda128> [options]

Options:
  --base                    Build base/developer image only. Sets BUILD_ARTIFACT=0.
  --prebuilt                Build fully prebuilt evaluator image. Sets BUILD_ARTIFACT=1. Default.
  --image <tag>             Docker image tag to produce.
  --install-nsys <0|1>      Install Nsight Systems. Default: 1.
  --max-jobs <N>            Build parallelism passed to Dockerfile. Default: 32.
  --progress <mode>         Docker build progress mode. Default: plain.
  --no-cache                Disable Docker build cache.
  --push                    Push image after a successful build.
  -h, --help                Show this help.

Examples:
  docker/build_image.sh --cuda cuda124 --base \
    --image grace-osdi26:cuda124-base

  docker/build_image.sh --cuda cuda124 --prebuilt \
    --image grace-osdi26:cuda124-prebuilt

  INSTALL_NSYS=0 MAX_JOBS=64 docker/build_image.sh --cuda cuda124 --base
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CUDA_FLAVOR="${CUDA_FLAVOR:-}"
MODE="${MODE:-prebuilt}"
IMAGE="${IMAGE:-}"
INSTALL_NSYS="${INSTALL_NSYS:-1}"
MAX_JOBS="${MAX_JOBS:-32}"
PROGRESS="${PROGRESS:-plain}"
NO_CACHE=0
PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA_FLAVOR="$2"; shift 2 ;;
    --base) MODE="base"; shift ;;
    --prebuilt) MODE="prebuilt"; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --install-nsys) INSTALL_NSYS="$2"; shift 2 ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --progress) PROGRESS="$2"; shift 2 ;;
    --no-cache) NO_CACHE=1; shift ;;
    --push) PUSH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${CUDA_FLAVOR}" ]]; then
  echo "[ERROR] Missing --cuda <cuda124|cuda128>" >&2
  usage
  exit 1
fi

case "${CUDA_FLAVOR}" in
  cuda124)
    DOCKERFILE="${REPO_ROOT}/docker/Dockerfile.cuda124"
    DEFAULT_BASE_IMAGE="grace-osdi26:cuda124-base"
    DEFAULT_PREBUILT_IMAGE="grace-osdi26:cuda124-prebuilt"
    ;;
  cuda128)
    DOCKERFILE="${REPO_ROOT}/docker/Dockerfile.cuda128"
    DEFAULT_BASE_IMAGE="grace-osdi26:cuda128-base"
    DEFAULT_PREBUILT_IMAGE="grace-osdi26:cuda128-prebuilt"
    ;;
  *)
    echo "[ERROR] Unknown CUDA flavor: ${CUDA_FLAVOR}" >&2
    usage
    exit 1
    ;;
esac

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "[ERROR] Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

case "${MODE}" in
  base)
    BUILD_ARTIFACT=0
    IMAGE="${IMAGE:-${DEFAULT_BASE_IMAGE}}"
    ;;
  prebuilt)
    BUILD_ARTIFACT=1
    IMAGE="${IMAGE:-${DEFAULT_PREBUILT_IMAGE}}"
    ;;
  *)
    echo "[ERROR] Unknown MODE=${MODE}; expected base or prebuilt" >&2
    exit 1
    ;;
esac

extra_args=()
if [[ "${NO_CACHE}" == "1" ]]; then
  extra_args+=(--no-cache)
fi

echo "============================================================"
echo "GraCE Docker build"
echo "============================================================"
echo "CUDA flavor    : ${CUDA_FLAVOR}"
echo "Dockerfile     : ${DOCKERFILE}"
echo "Mode           : ${MODE}"
echo "BUILD_ARTIFACT : ${BUILD_ARTIFACT}"
echo "INSTALL_NSYS   : ${INSTALL_NSYS}"
echo "MAX_JOBS       : ${MAX_JOBS}"
echo "Image          : ${IMAGE}"
echo "Repo root      : ${REPO_ROOT}"
echo "============================================================"

DOCKER_BUILDKIT=1 docker build \
  --progress="${PROGRESS}" \
  "${extra_args[@]}" \
  -f "${DOCKERFILE}" \
  --build-arg BUILD_ARTIFACT="${BUILD_ARTIFACT}" \
  --build-arg INSTALL_NSYS="${INSTALL_NSYS}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  -t "${IMAGE}" \
  "${REPO_ROOT}"

if [[ "${PUSH}" == "1" ]]; then
  docker push "${IMAGE}"
fi

echo
echo "[OK] Built image: ${IMAGE}"
