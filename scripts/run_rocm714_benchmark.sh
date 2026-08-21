#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ROCM_ENV:-${ROCM714_ENV:-}}"
if [[ -z "${ENV}" ]]; then
  if [[ -d /opt/rocm ]]; then
    ENV=/opt/rocm
  else
    ENV="${HOME}/envs/mujoco-warp-rocm714"
  fi
fi
SDK="${ROCM_SDK:-${ENV}/lib/python3.11/site-packages/_rocm_sdk_devel}"
LIB="${ROCM_LIB:-${ENV}/lib/python3.11/site-packages/_rocm_sdk_libraries}"
[[ -d "${SDK}" ]] || SDK="${ENV}"
[[ -d "${LIB}" ]] || LIB="${ENV}"
SCENARIO="${SCENARIO:-humanoid-contact}"
WORLDS="${WORLDS:-1024}"
STEPS="${STEPS:-1000}"
ROCM_TAG="${ROCM_TAG:-$(basename "${ENV}" | tr -cs '[:alnum:]' '_')}"
OUTPUT="${OUTPUT:-${ROOT}/results/hip_${SCENARIO}_${ROCM_TAG}.json}"
PYTHON="${PYTHON:-python3}"

if [[ ! -r /dev/kfd ]]; then
  echo "ROCm device access is unavailable: /dev/kfd is not readable." >&2
  echo "Add the account to the render group and start a new login session." >&2
  exit 3
fi
if ! id -nG | tr ' ' '\n' | grep -qx render; then
  echo "ROCm device access is unavailable: this session is not in render." >&2
  exit 3
fi

export PATH="${ENV}/bin:${ENV}/usr/bin:${SDK}/bin:${PATH}"
export ROCM_PATH="${ROCM_PATH:-${ENV}}"
export HIPCC="${HIPCC:-$(command -v hipcc || true)}"
export HIP_PATH="${HIP_PATH:-${SDK}}"
export HIP_PLATFORM=amd
export LD_LIBRARY_PATH="${SDK}/lib:${LIB}/lib:${ENV}/lib:${ENV}/lib64:${LD_LIBRARY_PATH:-}"

cd "${ROOT}"
"${PYTHON}" scripts/run_capability_probe.py --output "results/capability_probe_${ROCM_TAG}.json"
bash scripts/build_hip_benchmark.sh
"${ROOT}/build/hip_graph_benchmark" \
  --scenario "${SCENARIO}" \
  --worlds "${WORLDS}" \
  --steps "${STEPS}" \
  --output "${OUTPUT}"
"${PYTHON}" scripts/write_manifest.py --output results/source_manifest.json >/dev/null
echo "Benchmark result: ${OUTPUT}"
