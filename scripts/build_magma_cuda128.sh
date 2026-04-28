#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO" >&2' ERR

MAGMA_VERSION="${MAGMA_VERSION:-2.6.1}"
DESIRED_CUDA="${DESIRED_CUDA:-12.8}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MAGMA_INSTALL_DIR="${MAGMA_INSTALL_DIR:-/usr/local/magma-cuda128}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# Pin this. Do not track floating main in an artifact build.
# You can replace this with the exact PyTorch commit you want to trust.
PYTORCH_MAGMA_REF="${PYTORCH_MAGMA_REF:-f983c5ca0bceeda0dd4e1f9d74da151c058df7e6}"

TMP_ROOT="${TMP_ROOT:-/tmp/grace-magma-build}"
PYTORCH_TMP="${TMP_ROOT}/pytorch-magma-src"

# H100-only build:
MAGMA_CUDA_ARCH_LIST="${MAGMA_CUDA_ARCH_LIST:--gencode arch=compute_90,code=sm_90}"

# Broader Ampere/Hopper support, if desired:
# MAGMA_CUDA_ARCH_LIST="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

echo "============================================================"
echo "Building MAGMA ${MAGMA_VERSION} for CUDA ${DESIRED_CUDA}"
echo "Using PyTorch .ci/magma orchestration"
echo "PYTORCH_MAGMA_REF:   ${PYTORCH_MAGMA_REF}"
echo "CUDA_HOME:           ${CUDA_HOME}"
echo "MAGMA_INSTALL_DIR:   ${MAGMA_INSTALL_DIR}"
echo "MAX_JOBS:            ${MAX_JOBS}"
echo "MAGMA_CUDA_ARCH_LIST:${MAGMA_CUDA_ARCH_LIST}"
echo "============================================================"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "[ERROR] nvcc not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

CUDA_VERSION="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"
if [[ "${CUDA_VERSION}" != "${DESIRED_CUDA}" ]]; then
  echo "[ERROR] CUDA version mismatch. Expected ${DESIRED_CUDA}, got ${CUDA_VERSION}" >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git \
  ca-certificates \
  curl \
  patch \
  tar \
  bzip2 \
  xz-utils \
  coreutils \
  build-essential \
  gcc \
  g++ \
  make \
  cmake \
  python3 \
  python3-pip \
  python-is-python3

rm -rf "${TMP_ROOT}"
mkdir -p "${TMP_ROOT}"

echo ">>> Cloning PyTorch temporarily for MAGMA packaging files"
git clone --filter=blob:none --no-checkout https://github.com/pytorch/pytorch.git "${PYTORCH_TMP}"
cd "${PYTORCH_TMP}"
git checkout "${PYTORCH_MAGMA_REF}"

# Keep the checkout small if sparse-checkout is available.
git sparse-checkout init --cone || true
git sparse-checkout set .ci/magma || true
git checkout "${PYTORCH_MAGMA_REF}" -- .ci/magma || true

MAGMA_CI_DIR="${PYTORCH_TMP}/.ci/magma"
BUILD_MAGMA_SH="${MAGMA_CI_DIR}/build_magma.sh"
PACKAGE_FILES="${MAGMA_CI_DIR}/package_files"

if [[ ! -f "${BUILD_MAGMA_SH}" ]]; then
  echo "[ERROR] PyTorch MAGMA build script not found: ${BUILD_MAGMA_SH}" >&2
  exit 1
fi

if [[ ! -d "${PACKAGE_FILES}" ]]; then
  echo "[ERROR] PyTorch MAGMA package_files not found: ${PACKAGE_FILES}" >&2
  exit 1
fi

echo ">>> PyTorch MAGMA files found"
ls -lh "${MAGMA_CI_DIR}"
ls -lh "${PACKAGE_FILES}"

# PyTorch's package_files/build.sh normally uses all visible CPUs.
# Limit it for Docker-build stability.
sed -i \
  's/make -j$(getconf _NPROCESSORS_CONF)/make -j${MAX_JOBS:-$(getconf _NPROCESSORS_CONF)}/' \
  "${PACKAGE_FILES}/build.sh"

echo ">>> Running PyTorch MAGMA build_magma.sh directly inside this CUDA ${DESIRED_CUDA} image"

cd "${MAGMA_CI_DIR}"

PACKAGE_NAME="magma-cuda128" \
DESIRED_CUDA="${DESIRED_CUDA}" \
CUDA_ARCH_LIST="${MAGMA_CUDA_ARCH_LIST}" \
MAX_JOBS="${MAX_JOBS}" \
bash "${BUILD_MAGMA_SH}"

PKG="${MAGMA_CI_DIR}/output/linux-64/magma-cuda128-${MAGMA_VERSION}-1.tar.bz2"

if [[ ! -f "${PKG}" ]]; then
  echo "[ERROR] Expected MAGMA package was not produced: ${PKG}" >&2
  echo "[ERROR] Available output:" >&2
  find "${MAGMA_CI_DIR}/output" -maxdepth 3 -type f -print >&2 || true
  exit 1
fi

echo ">>> Installing MAGMA package to ${MAGMA_INSTALL_DIR}"
rm -rf "${MAGMA_INSTALL_DIR}"
mkdir -p "${MAGMA_INSTALL_DIR}"
tar -xjf "${PKG}" -C "${MAGMA_INSTALL_DIR}"

echo ">>> MAGMA install sanity check"

test -f "${MAGMA_INSTALL_DIR}/lib/libmagma.a"
test -f "${MAGMA_INSTALL_DIR}/include/magma.h"
test -f "${MAGMA_INSTALL_DIR}/include/magma_v2.h"
test -f "${MAGMA_INSTALL_DIR}/include/magma_config.h"

lib_size="$(stat -c%s "${MAGMA_INSTALL_DIR}/lib/libmagma.a")"
if [[ "${lib_size}" -lt 1000000 ]]; then
  echo "[ERROR] libmagma.a is suspiciously small: ${lib_size} bytes" >&2
  exit 1
fi

mkdir -p /usr/local/include /usr/local/lib
cp -r "${MAGMA_INSTALL_DIR}/include/"* /usr/local/include/
ln -sf "${MAGMA_INSTALL_DIR}/lib/libmagma.a" /usr/local/lib/libmagma.a

ldconfig

echo ">>> Cleaning temporary PyTorch clone"
rm -rf "${TMP_ROOT}"

echo
echo "[OK] MAGMA ${MAGMA_VERSION} installed using PyTorch .ci/magma orchestration"
echo "MAGMA_HOME=${MAGMA_INSTALL_DIR}"
ls -lh "${MAGMA_INSTALL_DIR}/lib/libmagma.a"