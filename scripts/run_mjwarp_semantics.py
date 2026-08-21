#!/usr/bin/env python3
"""Run the pinned MuJoCo-Warp step path on a deterministic CPU reference device."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUMANOID_XML = ROOT / "upstream/mujoco_warp/mujoco_warp/test_data/humanoid/humanoid.xml"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def run_variant(mujoco, wp, mjw, mjm, *, graph_conditional: bool, worlds: int, iterations: int, warmup: int, steps: int):
    mjd = mujoco.MjData(mjm)
    model = mjw.put_model(mjm)
    model.opt.iterations = iterations
    model.opt.graph_conditional = graph_conditional
    data = mjw.put_data(mjm, mjd, nworld=worlds, nconmax=256, njmax=512)

    for _ in range(warmup):
        mjw.step(model, data)
        wp.synchronize()

    latencies = []
    for _ in range(steps):
        start = time.perf_counter()
        mjw.step(model, data)
        wp.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)

    return {
        "graph_conditional": graph_conditional,
        "p50_step_ms": percentile(latencies, 0.50),
        "p95_step_ms": percentile(latencies, 0.95),
        "mean_step_ms": statistics.fmean(latencies),
        "qpos": data.qpos.numpy().tolist(),
        "qvel": data.qvel.numpy().tolist(),
        "time": data.time.numpy().tolist(),
    }


def max_abs_difference(left, right) -> float:
    values = []
    for a, b in zip(left, right):
        values.append(abs(a - b))
    return max(values, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    if args.device != "cpu":
        raise SystemExit("This semantic reference defaults to CPU; use the HIP benchmark for AMD measurements.")

    sys.path.insert(0, str(ROOT / "upstream/mujoco_warp"))
    import mujoco  # type: ignore
    import warp as wp  # type: ignore

    # The pinned AMD branch expects the newer Warp Device.is_hip property. The
    # installed CPU reference Warp predates that property, so expose the
    # truthful CPU value without changing the pinned upstream source.
    device_type = type(wp.get_device())
    compatibility_shim = not hasattr(device_type, "is_hip")
    if compatibility_shim:
        device_type.is_hip = property(lambda device: False)
    wp.set_device("cpu")

    import mujoco_warp as mjw  # type: ignore

    mjm = mujoco.MjModel.from_xml_path(str(HUMANOID_XML))
    fixed = run_variant(
        mujoco,
        wp,
        mjw,
        mjm,
        graph_conditional=False,
        worlds=args.worlds,
        iterations=args.iterations,
        warmup=args.warmup,
        steps=args.steps,
    )
    conditional_path = run_variant(
        mujoco,
        wp,
        mjw,
        mjm,
        graph_conditional=True,
        worlds=args.worlds,
        iterations=args.iterations,
        warmup=args.warmup,
        steps=args.steps,
    )

    qpos_error = max_abs_difference(
        [value for row in fixed["qpos"] for value in row],
        [value for row in conditional_path["qpos"] for value in row],
    )
    qvel_error = max_abs_difference(
        [value for row in fixed["qvel"] for value in row],
        [value for row in conditional_path["qvel"] for value in row],
    )
    time_error = max_abs_difference(fixed["time"], conditional_path["time"])
    result = {
        "schema": "mujoco-warp-amd-cpu-semantics-v1",
        "execution": "cpu_semantics_only",
        "device": args.device,
        "python": platform.python_version(),
        "mujoco_xml": str(HUMANOID_XML.relative_to(ROOT)),
        "worlds": args.worlds,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "steps": args.steps,
        "compatibility_shim_device_is_hip": compatibility_shim,
        "variants": {
            "fixed_iteration": {key: value for key, value in fixed.items() if key not in {"qpos", "qvel", "time"}},
            "graph_conditional_path": {
                key: value for key, value in conditional_path.items() if key not in {"qpos", "qvel", "time"}
            },
        },
        "max_abs_qpos_error": qpos_error,
        "max_abs_qvel_error": qvel_error,
        "max_abs_time_error": time_error,
        "numerically_equivalent": max(qpos_error, qvel_error, time_error) == 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
