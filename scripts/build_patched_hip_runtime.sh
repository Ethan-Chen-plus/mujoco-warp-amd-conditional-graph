#!/usr/bin/env bash
set -euo pipefail

# Apply the experimental HIP conditional-node patches and build amdhip64.
# HIP_CLR_SRC and HIP_SDK_SRC must be clean checkouts at the pinned ROCm 7.2.1
# baseline. The script accepts an already-patched checkout and is idempotent.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIP_CLR_SRC="${HIP_CLR_SRC:-}"
HIP_SDK_SRC="${HIP_SDK_SRC:-}"
CLR_BUILD="${CLR_BUILD:-${ROOT}/build/hip-clr}"
CLR_INSTALL="${CLR_INSTALL:-${ROOT}/build/hip-clr-install}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
JOBS="${JOBS:-2}"

if [[ -z "$HIP_CLR_SRC" || -z "$HIP_SDK_SRC" ]]; then
  echo "Set HIP_CLR_SRC and HIP_SDK_SRC to clean HIP/CLR and HIP SDK checkouts." >&2
  exit 2
fi

apply_once() {
  local repo="$1"
  local patch_file="$2"
  if git -C "$repo" apply --check "$patch_file" >/dev/null 2>&1; then
    git -C "$repo" apply "$patch_file"
  elif git -C "$repo" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "Assuming patch is already applied: $patch_file"
  else
    echo "Patch cannot apply cleanly or is not already applied: $patch_file" >&2
    exit 3
  fi
}

apply_once "$HIP_CLR_SRC" "$ROOT/patches/hip-clr-conditional.diff"
apply_once "$HIP_SDK_SRC" "$ROOT/patches/hip-sdk-conditional.diff"

cmake -S "$HIP_CLR_SRC" -B "$CLR_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DHIP_RUNTIME=rocclr \
  -DHIP_COMPILER=clang \
  -DHIP_OFFICIAL_BUILD=ON \
  -DHIP_ENABLE_ROCPROFILER_REGISTER=ON \
  -DHIP_COMMON_DIR="$HIP_SDK_SRC" \
  -DROCM_PATH="$ROCM_PATH" \
  -DCMAKE_INSTALL_PREFIX="$CLR_INSTALL"

cmake --build "$CLR_BUILD" --target amdhip64 -j"$JOBS"
cmake --install "$CLR_BUILD"

echo "Runtime built: $CLR_BUILD/hipamd/lib/libamdhip64.so"
sha256sum "$CLR_BUILD/hipamd/lib/libamdhip64.so"
