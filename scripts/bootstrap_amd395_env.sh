#!/usr/bin/env bash
set -euo pipefail

# Create an isolated Python environment and build the bundled ROCm Warp source.
# The script never modifies a pre-existing environment unless it is explicitly
# selected through ENV.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV:-${HOME}/envs/mujoco-warp-amd-py312}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
GPU_ARCH="${GPU_ARCH:-gfx1151}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if ! command -v hipcc >/dev/null 2>&1; then
  echo "hipcc was not found. Install ROCm or set PATH/ROCM_PATH first." >&2
  exit 2
fi

if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | grep -E "Name:.*gfx|Marketing Name" | head -n 8 || true
fi

if [[ ! -x "${ENV}/bin/python" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    micromamba create --yes --prefix "${ENV}" "python=${PYTHON_VERSION}" pip
  elif command -v python${PYTHON_VERSION} >/dev/null 2>&1; then
    python${PYTHON_VERSION} -m venv "${ENV}"
  else
    python3 -m venv "${ENV}"
  fi
fi

PY="${ENV}/bin/python"
"${PY}" -m pip install --upgrade pip setuptools wheel
"${PY}" -m pip install "mujoco==3.8.1" numpy absl-py "etils[epath]"

pushd "${ROOT}/upstream/warp" >/dev/null
ROCM_PATH="${ROCM_PATH}" "${PY}" build_lib.py \
  --no-cuda --hip-arch="${GPU_ARCH}" --rocm-path="${ROCM_PATH}"
ROCM_PATH="${ROCM_PATH}" "${PY}" -m pip install -e . --no-deps
popd >/dev/null

"${PY}" -m pip install -e "${ROOT}/upstream/mujoco_warp" --no-deps
"${PY}" -m pip install absl-py "etils[epath]" "mujoco==3.8.1" numpy

export PYTHONPATH="${ROOT}/upstream/warp:${ROOT}/upstream/mujoco_warp"
export LD_LIBRARY_PATH="${ROOT}/upstream/warp/warp/bin:${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH:-}"
"${PY}" "${ROOT}/scripts/check_mjwarp_import.py" \
  --output "${ROOT}/results/mujoco_warp_import_amd395.json"

echo "Environment ready: ${ENV}"
