#!/usr/bin/env bash
set -euo pipefail

# Run the paired full-physics benchmark on an AMD ROCm host.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV="${ENV:-/home/aup/envs/mujoco-warp-amd-py312}"
MODEL="${MODEL:-$ROOT/upstream/mujoco_warp/benchmarks/humanoid/humanoid.xml}"
WORLDS="${WORLDS:-1024}"
STEPS="${STEPS:-1000}"
OUT="${OUT:-$ROOT/results/mjwarp_amd_benchmark}"

mkdir -p "$OUT"
export PYTHONPATH="$ROOT/upstream/warp:$ROOT/upstream/mujoco_warp"
export LD_LIBRARY_PATH="$ROOT/upstream/warp/warp/bin:/opt/rocm/lib:/opt/rocm/lib64"
export MUJOCO_GL=egl
export WP_HIP_GRAPH_ENABLE=0
export MJW_HIP_SINGLE_STREAM=1
export WP_HIP_CONDITIONAL_MAX_ITERS=100

for mode in baseline emulation; do
  if [[ "$mode" == emulation ]]; then
    export WP_HIP_CONDITIONAL_EMULATION=1
  else
    export WP_HIP_CONDITIONAL_EMULATION=0
  fi
  "$ENV/bin/python" "$ROOT/upstream/mujoco_warp/mujoco_warp/testspeed.py" "$MODEL" \
    --function=step --nworld="$WORLDS" --nstep="$STEPS" \
    --nconmax=128 --njmax=128 --device=cuda:0 --format=json \
    > "$OUT/${mode}.log" 2>&1
done

export WP_HIP_CONDITIONAL_EMULATION=1
"$ENV/bin/python" "$ROOT/scripts/inspect_mjwarp_state.py" "$MODEL" \
  --nworld=256 --steps=100 > "$OUT/correctness.log" 2>&1

"$ENV/bin/python" "$ROOT/scripts/collect_mjwarp_benchmark.py" \
  --baseline "$OUT/baseline.log" \
  --emulation "$OUT/emulation.log" \
  --correctness "$OUT/correctness.log" \
  --worlds "$WORLDS" --steps "$STEPS" \
  --output "$OUT/summary.json"

sha256sum "$OUT"/*.log "$OUT/summary.json" > "$OUT/SHA256SUMS"
printf 'Wrote %s\n' "$OUT/summary.json"
