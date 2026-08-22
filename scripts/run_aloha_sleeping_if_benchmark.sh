#!/usr/bin/env bash
set -euo pipefail

# Compare eager broadphase rebuild, static graph reuse, and native HIP
# conditional capture_if on the ALOHA pot scene.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV="${ENV:-/home/aup/envs/mujoco-warp-amd-py312}"
MODEL="${MODEL:-$ROOT/upstream/mujoco_warp/benchmarks/aloha/scene_pot.xml}"
WORLDS="${WORLDS:-64}"
WARMUP="${WARMUP:-8}"
STEPS="${STEPS:-500}"
WAKE_EVERY="${WAKE_EVERY:-40}"
GRAVITY_MODE="${GRAVITY_MODE:-default}"
WAKE_MODE="${WAKE_MODE:-motion}"
STIMULUS="${STIMULUS:-joint_target}"
FIXTURE="${FIXTURE:-dynamic}"
NCONMAX="${NCONMAX:-8192}"
NJMAX="${NJMAX:-4096}"
OUT="${OUT:-$ROOT/results/aloha_sleeping_if_amd}"
HIP_CLR_LIB="${HIP_CLR_LIB:-}"
ASSET_CHECK="${ASSET_CHECK:-1}"

if [[ -n "$HIP_CLR_LIB" && -f "$HIP_CLR_LIB" ]]; then
  HIP_CLR_DIR="$(dirname "$HIP_CLR_LIB")"
else
  HIP_CLR_DIR="$HIP_CLR_LIB"
fi

mkdir -p "$OUT"
if [[ ! -f "$MODEL" ]]; then
  printf 'ALOHA model is missing: %s\n' "$MODEL" >&2
  exit 2
fi
if [[ "$ASSET_CHECK" == "1" && ! -f "$(dirname "$MODEL")/assets/extrusion_2040_880.stl" ]]; then
  printf 'ALOHA assets are missing. Run scripts/fetch_aloha_assets.sh first.\n' >&2
  exit 2
fi
export PYTHONPATH="$ROOT/upstream/warp:$ROOT/upstream/mujoco_warp"
export LD_LIBRARY_PATH="${HIP_CLR_DIR:+$HIP_CLR_DIR:}$ROOT/upstream/warp/warp/bin:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export MJW_HIP_SINGLE_STREAM=1
export WP_HIP_GRAPH_ENABLE=0
export WP_HIP_CONDITIONAL_MAX_ITERS=100

for variant in eager static native; do
  "$ENV/bin/python" "$ROOT/scripts/run_mjwarp_sleeping_benchmark.py" \
    "$MODEL" \
    --variant="$variant" \
    --worlds="$WORLDS" \
    --warmup="$WARMUP" \
    --steps="$STEPS" \
    --wake-every="$WAKE_EVERY" \
    --gravity="$GRAVITY_MODE" \
    --wake-mode="$WAKE_MODE" \
    --stimulus="$STIMULUS" \
    --fixture="$FIXTURE" \
    --nconmax="$NCONMAX" \
    --njmax="$NJMAX" \
    --output="$OUT/${variant}.json"
done

"$ENV/bin/python" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
rows = [json.loads((out / f"{variant}.json").read_text()) for variant in ("eager", "static", "native")]
eager = next(row for row in rows if row["variant"] == "eager")
static = next(row for row in rows if row["variant"] == "static")

def state_error(row, reference):
  return {
    "qpos_max_abs": float(np.max(np.abs(np.asarray(row["final_qpos"]) - np.asarray(reference["final_qpos"]))
                                 )),
    "qvel_max_abs": float(np.max(np.abs(np.asarray(row["final_qvel"]) - np.asarray(reference["final_qvel"]))
                                 )),
  }

def matched_state_error(row, reference):
  qpos = np.asarray(row["final_qpos"], dtype=np.float64)
  qvel = np.asarray(row["final_qvel"], dtype=np.float64)
  ref_qpos = np.asarray(reference["final_qpos"], dtype=np.float64)
  ref_qvel = np.asarray(reference["final_qvel"], dtype=np.float64)
  count = min(len(qpos), len(ref_qpos))
  if count == 0:
    return {"matched_worlds": 0}
  pair_cost = np.maximum(
    np.max(np.abs(qpos[:, None, :] - ref_qpos[None, :, :]), axis=2),
    np.max(np.abs(qvel[:, None, :] - ref_qvel[None, :, :]), axis=2),
  )
  remaining_rows = set(range(len(qpos)))
  remaining_cols = set(range(len(ref_qpos)))
  pairs = []
  for _ in range(count):
    best = min(
      ((pair_cost[i, j], i, j) for i in remaining_rows for j in remaining_cols),
      key=lambda item: item[0],
    )
    _, i, j = best
    pairs.append((i, j))
    remaining_rows.remove(i)
    remaining_cols.remove(j)
  qpos_errors = [float(np.max(np.abs(qpos[i] - ref_qpos[j]))) for i, j in pairs]
  qvel_errors = [float(np.max(np.abs(qvel[i] - ref_qvel[j]))) for i, j in pairs]
  return {
    "matched_worlds": len(pairs),
    "matched_qpos_max_abs": max(qpos_errors),
    "matched_qpos_mean_max_abs": float(np.mean(qpos_errors)),
    "matched_qvel_max_abs": max(qvel_errors),
    "matched_qvel_mean_max_abs": float(np.mean(qvel_errors)),
    "matched_state_max_abs": max(max(qpos_errors), max(qvel_errors)),
  }

summary = {
  "schema": "mujoco-warp-amd-sleeping-if-summary-v1",
  "variants": rows,
  "native_vs_eager_speedup": rows[2]["world_steps_per_second"] / eager["world_steps_per_second"],
  "native_vs_static_speedup": rows[2]["world_steps_per_second"] / rows[1]["world_steps_per_second"],
  "all_final_state_sha256": {row["variant"]: row["final_state_sha256"] for row in rows},
  "state_error_vs_eager": {row["variant"]: state_error(row, eager) for row in rows},
  "state_error_vs_static": {row["variant"]: state_error(row, static) for row in rows},
  "matched_state_error_vs_eager": {row["variant"]: matched_state_error(row, eager) for row in rows},
  "matched_state_error_vs_static": {row["variant"]: matched_state_error(row, static) for row in rows},
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

"$ENV/bin/python" "$ROOT/scripts/summarize_aloha_sleeping_if.py" \
  --input "$OUT" --output "$OUT/summary.json"

sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS"
printf 'Wrote %s\n' "$OUT/summary.json"
