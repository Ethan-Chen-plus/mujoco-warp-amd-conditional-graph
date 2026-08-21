#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
TOOLCHAIN="${ROOT}/../../envs/mujoco-warp-hip-toolchain"
ROCM_PATH="${ROCM_PATH:-}"
HIPCC="${HIPCC:-}"

if [[ -z "${HIPCC}" ]]; then
  HIPCC="$(command -v hipcc || true)"
fi
if [[ -z "${HIPCC}" && -x "${TOOLCHAIN}/usr/bin/hipcc" ]]; then
  HIPCC="${TOOLCHAIN}/usr/bin/hipcc"
fi
if [[ -z "${HIPCC}" ]]; then
  echo "hipcc was not found. Install ROCm or set HIPCC=/path/to/hipcc." >&2
  exit 2
fi

if [[ -z "${ROCM_PATH}" ]]; then
  if [[ -d /opt/rocm ]]; then
    ROCM_PATH=/opt/rocm
  else
    ROCM_PATH="$(cd "$(dirname "${HIPCC}")/.." && pwd)"
  fi
fi

# The bundled toolchain is rooted at `.../mujoco-warp-hip-toolchain/usr`,
# while users naturally pass its parent directory as ROCM_PATH. Normalize that
# layout so the same command works with the bundled and system ROCm trees.
if [[ ! -x "${ROCM_PATH}/bin/hipcc" && -x "${ROCM_PATH}/usr/bin/hipcc" ]]; then
  ROCM_PATH="${ROCM_PATH}/usr"
fi

# AMD Core SDK wheels expose the active tree below the Python environment.
# Normalize that layout so hipcc resolves its clang, headers, and runtime from
# one consistent ROCM_PATH instead of falling back to an older system tree.
if [[ -n "${ROCM_PATH}" && ! -x "${ROCM_PATH}/lib/llvm/bin/clang++" ]]; then
  for candidate in \
    "${ROCM_PATH}"/lib/python*/site-packages/_rocm_sdk_devel \
    "${ROCM_PATH}"/lib/python*/site-packages/rocm_sdk_devel; do
    if [[ -x "${candidate}/lib/llvm/bin/clang++" ]]; then
      ROCM_PATH="${candidate}"
      break
    fi
  done
fi

# Apply the same normalization to an explicitly supplied HIP_PATH. This keeps
# the compiler, headers, and device libraries on the same ROCm SDK when a
# Python-installed SDK environment is passed by the caller.
if [[ -n "${HIP_PATH:-}" && ! -f "${HIP_PATH}/include/hip/hip_runtime.h" ]]; then
  for candidate in \
    "${HIP_PATH}"/lib/python*/site-packages/_rocm_sdk_devel \
    "${HIP_PATH}"/lib/python*/site-packages/rocm_sdk_devel; do
    if [[ -f "${candidate}/include/hip/hip_runtime.h" ]]; then
      HIP_PATH="${candidate}"
      break
    fi
  done
fi

HIP_CLR_LIB="${HIP_CLR_LIB:-}"
if [[ -n "$HIP_CLR_LIB" && -f "$HIP_CLR_LIB" ]]; then
  HIP_CLR_DIR="$(dirname "$HIP_CLR_LIB")"
else
  HIP_CLR_DIR="$HIP_CLR_LIB"
fi

export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
export HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"
export PATH="$(dirname "${HIPCC}"):${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${TOOLCHAIN}/usr/lib/llvm-17/bin:${PATH}"
export LD_LIBRARY_PATH="${HIP_CLR_DIR:+$HIP_CLR_DIR:}${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${TOOLCHAIN}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

DEVICE_LIB_PATH="${ROCM_DEVICE_LIB_PATH:-}"
if [[ -z "${DEVICE_LIB_PATH}" ]]; then
  for candidate in \
    "${ROCM_PATH}/lib/llvm/amdgcn/bitcode" \
    "${ROCM_PATH}/lib/llvm"/*/amdgcn/bitcode \
    "${ROCM_PATH}/lib/llvm/lib/clang/17/amdgcn/bitcode" \
    "${ROCM_PATH}/lib/llvm-17/lib/clang/17/amdgcn/bitcode" \
    "${TOOLCHAIN}/usr/lib/llvm-17/lib/clang/17/amdgcn/bitcode"; do
    if [[ -d "${candidate}" ]]; then
      DEVICE_LIB_PATH="${candidate}"
      break
    fi
  done
fi

mkdir -p "${BUILD_DIR}"
ARCH_ARGS=()
if [[ -n "${GPU_ARCH:-}" ]]; then
  ARCH_ARGS+=("--offload-arch=${GPU_ARCH}")
fi

LINK_ARGS=()
for candidate in "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64" "${ROCM_PATH}/lib/x86_64-linux-gnu" \
  "${ROCM_PATH}/../_rocm_sdk_libraries/lib" \
  "${TOOLCHAIN}/usr/lib/x86_64-linux-gnu"; do
  [[ -d "${candidate}" ]] && LINK_ARGS+=("-L${candidate}")
done
if [[ -n "$HIP_CLR_DIR" ]]; then
  LINK_ARGS=("-L${HIP_CLR_DIR}" "${LINK_ARGS[@]}")
fi
DEVICE_ARGS=()
[[ -n "${DEVICE_LIB_PATH}" ]] && DEVICE_ARGS+=("--rocm-device-lib-path=${DEVICE_LIB_PATH}")

HIP_INCLUDE="${HIP_INCLUDE:-${HIP_SDK_SRC:-}/include}"
CONDITIONAL_ARGS=()
if [[ -f "${HIP_INCLUDE}/hip/hip_runtime_api.h" ]]; then
  CONDITIONAL_ARGS+=("-I${HIP_INCLUDE}" "-DHIP_GRAPH_CONDITIONAL_EXT=1")
else
  echo "Set HIP_INCLUDE or HIP_SDK_SRC to the patched HIP headers for conditional tests." >&2
  exit 2
fi

for source in hip_graph_benchmark conditional_while_device conditional_while_benchmark; do
  EXTRA_ARGS=()
  if [[ "${source}" != hip_graph_benchmark ]]; then
    EXTRA_ARGS=("${CONDITIONAL_ARGS[@]}")
  fi
  "${HIPCC}" -O3 -std=c++17 "${ARCH_ARGS[@]}" "${DEVICE_ARGS[@]}" "${LINK_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    "${ROOT}/hip/${source}.cpp" \
    -o "${BUILD_DIR}/${source}"
  echo "Built ${BUILD_DIR}/${source} with ${HIPCC}"
done
