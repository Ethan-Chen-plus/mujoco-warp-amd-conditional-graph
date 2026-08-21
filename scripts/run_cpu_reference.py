#!/usr/bin/env python3
"""Validate tier dispatch and work accounting without a GPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


TIERS = (1, 4, 10, 40, 100)


def required_iterations(world: int, step: int, scenario: str, cap: int) -> int:
    phase = (world * 1103515245 + step * 12345 + len(scenario) * 97) & 0x7FFFFFFF
    if scenario == "humanoid-contact":
        raw = 1 + (phase % 100)
    elif scenario == "aloha-sleeping":
        raw = 1 if phase % 10 else 40 + phase % 61
    else:
        raw = 1 + phase % 20
    return min(cap, raw)


def tier_for(needed: int, cap: int) -> int:
    for tier in TIERS:
        if tier >= needed:
            return min(tier, cap)
    return cap


def state_after(needed: int, budget: int) -> float:
    value = 1.0
    for iteration in range(budget):
        if iteration < needed:
            value = math.fma(value, 0.997, 0.003) if hasattr(math, "fma") else value * 0.997 + 0.003
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="humanoid-contact", choices=("humanoid-contact", "aloha-sleeping"))
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_iter <= 0 or args.worlds <= 0 or args.steps <= 0:
        raise SystemExit("worlds, steps, and max-iter must be positive")
    started = time.perf_counter()
    needed = [required_iterations(w, s, args.scenario, args.max_iter) for s in range(args.steps) for w in range(args.worlds)]
    selected = [tier_for(n, args.max_iter) for n in needed]
    eager_ops = args.steps * args.worlds * args.max_iter
    adaptive_ops = sum(selected)
    errors = [abs(state_after(n, args.max_iter) - state_after(n, tier)) for n, tier in zip(needed, selected)]
    hist = {str(tier): selected.count(tier) for tier in TIERS if tier <= args.max_iter}
    record = {
        "schema": "mujoco-warp-amd-cpu-reference-v1",
        "scenario": args.scenario,
        "worlds": args.worlds,
        "steps": args.steps,
        "max_iterations": args.max_iter,
        "tiers": [tier for tier in TIERS if tier <= args.max_iter] + ([args.max_iter] if args.max_iter not in TIERS else []),
        "selected_tier_histogram": hist,
        "eager_iteration_work": eager_ops,
        "host_adaptive_iteration_work": adaptive_ops,
        "ideal_work_reduction": 1.0 - adaptive_ops / eager_ops,
        "max_abs_state_error": max(errors),
        "numerically_equivalent": max(errors) == 0.0,
        "execution": "cpu_reference_only",
        "elapsed_seconds": time.perf_counter() - started,
    }
    # Keep the provenance hash stable across reruns; wall-clock timing is a
    # measurement field, not part of the deterministic result identity.
    encoded = json.dumps(
        {key: value for key, value in record.items() if key != "elapsed_seconds"},
        sort_keys=True,
    ).encode()
    record["result_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
