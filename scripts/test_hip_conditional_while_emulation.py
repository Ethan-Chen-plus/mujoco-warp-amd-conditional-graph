#!/usr/bin/env python3
"""Validate Warp's opt-in HIP device-gated conditional-while backend.

The test uses one captured HIP graph.  Each expanded body checks the device
condition before changing state, then updates the condition on the device.
This is the same contract used by the MuJoCo-Warp solver's convergence flags.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def increment_if_active(state: wp.array[int], condition: wp.array[int]):
  tid = wp.tid()
  if condition[0] != 0:
    state[tid] += 1


@wp.kernel
def stop_at_target(state: wp.array[int], target: int, condition: wp.array[int]):
  if wp.tid() == 0 and state[0] >= target:
    condition[0] = 0


def body(state, target, condition_arr):
  wp.launch(increment_if_active, dim=1, inputs=[state, condition_arr])
  wp.launch(stop_at_target, dim=1, inputs=[state, target, condition_arr])


def run_case(device: str, initial_condition: int, target: int, max_iterations: int) -> int:
  state = wp.zeros(1, dtype=wp.int32, device=device)
  condition = wp.array([initial_condition], dtype=wp.int32, device=device)
  with wp.ScopedCapture(device=device) as capture:
    wp.capture_while(
      condition,
      while_body=body,
      max_iterations=max_iterations,
      state=state,
      target=target,
      condition_arr=condition,
    )
  wp.capture_launch(capture.graph)
  wp.synchronize_device(device)
  return int(state.numpy()[0])


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--max-iterations", type=int, default=8)
  args = parser.parse_args()

  os.environ.setdefault("WP_HIP_CONDITIONAL_EMULATION", "1")
  wp.init()
  device = wp.get_device(args.device)
  if not device.is_hip:
    raise RuntimeError(f"This validation requires a HIP device, got {device}")
  if wp.is_conditional_graph_supported():
    raise RuntimeError("The test must exercise the HIP compatibility path, not native CUDA nodes")
  if not wp.is_conditional_graph_emulated():
    raise RuntimeError("WP_HIP_CONDITIONAL_EMULATION=1 was not enabled before Warp initialization")

  active_result = run_case(args.device, initial_condition=1, target=3, max_iterations=args.max_iterations)
  inactive_result = run_case(args.device, initial_condition=0, target=3, max_iterations=args.max_iterations)
  expected = {"active": 3, "inactive": 0}
  result = {
    "backend": "hip-device-gated-conditional-while",
    "native_conditional_graph_api": False,
    "emulation_enabled": True,
    "device": str(device),
    "arch": getattr(device, "arch", None),
    "max_iterations": args.max_iterations,
    "active_result": active_result,
    "inactive_result": inactive_result,
    "expected": expected,
    "numerically_equivalent": active_result == expected["active"] and inactive_result == expected["inactive"],
    "python": sys.version,
    "platform": platform.platform(),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2))
  return 0 if result["numerically_equivalent"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
