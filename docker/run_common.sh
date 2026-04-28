#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO" >&2' ERR

# Shared Docker runner for GraCE artifact images.
# This script is normally invoked by run_cuda124.sh, run_cuda124_developer.sh,
# run_cuda128.sh, or run_cuda128_developer.sh.

usage() {
  cat <<'USAGE'
Usage:
  docker/run_common.sh [-- command...]

Environment variables:
  IMAGE                 Docker image tag. Required unless DEFAULT_IMAGE is set.
  DEFAULT_IMAGE         Fallback image tag used by wrapper scripts.
  DEVELOPER_MODE        1: bind-mount full repository. 0: mount outputs only.
  USE_CPUSET            auto|0|1. Default: auto.
  BENCH_CPUS            CPU list for isolated mode, e.g. 8-31,64-95.
  BENCH_MEMS            NUMA memory node list for isolated mode, e.g. 0.
  CGROUP_PARENT         Docker cgroup parent. Default from host tuning: bench.slice.
  USE_NUMACTL           Passed into container. Default: 0.
  CONTAINER_ARTIFACT_ROOT  Default: /workspace/grace-osdi-26-artifact.
  EXTRA_DOCKER_ARGS     Optional extra raw docker args, split by shell words.

Cpuset autodetection:
  If USE_CPUSET=auto, this script detects host_tuning state from either:
    docker/.generated/docker_cpuset.env
    /run/grace_docker_cpuset_backup/state.env

Examples:
  docker/run_cuda124.sh
  USE_CPUSET=0 docker/run_cuda124.sh
  USE_CPUSET=1 BENCH_CPUS=8-31,64-95 BENCH_MEMS=0 docker/run_cuda124.sh
  docker/run_cuda124_developer.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-${DEFAULT_IMAGE:-}}"
if [[ -z "${IMAGE}" ]]; then
  echo "[ERROR] IMAGE or DEFAULT_IMAGE must be set" >&2
  usage
  exit 1
fi

DEVELOPER_MODE="${DEVELOPER_MODE:-0}"
USE_CPUSET="${USE_CPUSET:-auto}"
USE_NUMACTL="${USE_NUMACTL:-0}"
CONTAINER_ARTIFACT_ROOT="${CONTAINER_ARTIFACT_ROOT:-/workspace/grace-osdi-26-artifact}"

mkdir -p "${REPO_ROOT}/results" "${REPO_ROOT}/figures" "${REPO_ROOT}/tables"

# Source generated cpuset metadata if available. The Docker-aware host tuning
# script writes /run/grace_docker_cpuset_backup/state.env. A repo-local file
# can also be used for non-root convenience.
GENERATED_ENV="${REPO_ROOT}/docker/.generated/docker_cpuset.env"
HOST_TUNING_ENV="/run/grace_docker_cpuset_backup/state.env"

CPUSET_SOURCE="none"
if [[ "${USE_CPUSET}" == "auto" || "${USE_CPUSET}" == "1" ]]; then
  if [[ -f "${GENERATED_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${GENERATED_ENV}"
    CPUSET_SOURCE="${GENERATED_ENV}"
  elif [[ -r "${HOST_TUNING_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${HOST_TUNING_ENV}"
    CPUSET_SOURCE="${HOST_TUNING_ENV}"
  fi
fi

# The host_tuning state.env uses BENCH_CPUS, BENCH_MEMS, and BENCH_SLICE.
CGROUP_PARENT="${CGROUP_PARENT:-${BENCH_SLICE:-bench.slice}}"
BENCH_CPUS="${BENCH_CPUS:-}"
BENCH_MEMS="${BENCH_MEMS:-}"

USE_CPUSET_EFFECTIVE=0
case "${USE_CPUSET}" in
  0|false|False|no|NO)
    USE_CPUSET_EFFECTIVE=0
    ;;
  1|true|True|yes|YES)
    if [[ -z "${BENCH_CPUS}" || -z "${BENCH_MEMS}" ]]; then
      echo "[ERROR] USE_CPUSET=1 but BENCH_CPUS or BENCH_MEMS is missing." >&2
      echo "        Run host_tuning/cpuset/setup_docker_bench_cpuset_full.sh or set them manually." >&2
      exit 1
    fi
    USE_CPUSET_EFFECTIVE=1
    ;;
  auto)
    if [[ -n "${BENCH_CPUS}" && -n "${BENCH_MEMS}" ]]; then
      USE_CPUSET_EFFECTIVE=1
    else
      USE_CPUSET_EFFECTIVE=0
    fi
    ;;
  *)
    echo "[ERROR] Unknown USE_CPUSET=${USE_CPUSET}; expected auto, 0, or 1" >&2
    exit 1
    ;;
esac

docker_args=(
  --rm
  -it
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e "USE_NUMACTL=${USE_NUMACTL}"
)

if [[ "${USE_CPUSET_EFFECTIVE}" == "1" ]]; then
  docker_args+=(
    --cgroup-parent="${CGROUP_PARENT}"
    --cpuset-cpus="${BENCH_CPUS}"
    --cpuset-mems="${BENCH_MEMS}"
    --cap-add=SYS_NICE
    --security-opt seccomp=unconfined
  )
fi

if [[ -n "${EXTRA_DOCKER_ARGS:-}" ]]; then
  # Intentional shell splitting for advanced users.
  # Example: EXTRA_DOCKER_ARGS='--name grace-dev --shm-size=64g'
  read -r -a extra_args <<< "${EXTRA_DOCKER_ARGS}"
  docker_args+=("${extra_args[@]}")
fi

if [[ "${DEVELOPER_MODE}" == "1" ]]; then
  docker_args+=(
    -v "${REPO_ROOT}:${CONTAINER_ARTIFACT_ROOT}"
  )
else
  docker_args+=(
    -v "${REPO_ROOT}/results:${CONTAINER_ARTIFACT_ROOT}/results"
    -v "${REPO_ROOT}/figures:${CONTAINER_ARTIFACT_ROOT}/figures"
    -v "${REPO_ROOT}/tables:${CONTAINER_ARTIFACT_ROOT}/tables"
  )
fi

docker_args+=(
  -w "${CONTAINER_ARTIFACT_ROOT}"
  "${IMAGE}"
)

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  docker_args+=("$@")
fi

echo "============================================================"
echo "GraCE Docker run"
echo "============================================================"
echo "Image              : ${IMAGE}"
echo "Developer mode     : ${DEVELOPER_MODE}"
echo "Repo root          : ${REPO_ROOT}"
echo "Container root     : ${CONTAINER_ARTIFACT_ROOT}"
echo "USE_CPUSET request : ${USE_CPUSET}"
echo "USE_CPUSET active  : ${USE_CPUSET_EFFECTIVE}"
echo "Cpuset source      : ${CPUSET_SOURCE}"
if [[ "${USE_CPUSET_EFFECTIVE}" == "1" ]]; then
  echo "Cgroup parent      : ${CGROUP_PARENT}"
  echo "Benchmark CPUs     : ${BENCH_CPUS}"
  echo "Benchmark MEMs     : ${BENCH_MEMS}"
fi
echo "USE_NUMACTL        : ${USE_NUMACTL}"
echo "============================================================"

echo "+ docker run ${docker_args[*]}"
exec docker run "${docker_args[@]}"
