#!/usr/bin/env bash
set -euo pipefail

# The public benchmark now measures the patched native HIP conditional node
# against the eager solver path. Keep this filename as a compatibility alias.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$ROOT/scripts/run_native_mjwarp_benchmark.sh" "$@"
