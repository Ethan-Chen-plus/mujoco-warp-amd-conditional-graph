"""Run a minimal Warp GPU kernel on the target AMD device.

The smoke test validates the source-built Warp HIP backend independently of
MuJoCo-Warp. It records the visible Warp devices, selected device, output
values, and numerical error in a small JSON artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def add_one(values: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)):
  """Add one to every element of a device array."""
  index = wp.tid()
  output[index] = values[index] + 1.0


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path)
  parser.add_argument("--count", type=int, default=256)
  args = parser.parse_args()

  wp.init()
  devices = [str(device) for device in wp.get_devices()]
  result: dict[str, object] = {
      "schema": "mujoco-warp-amd-warp-gpu-smoke-v1",
      "devices": devices,
      "count": args.count,
  }

  try:
    device = wp.get_device("cuda:0")
    values = np.arange(args.count, dtype=np.float32)
    device_values = wp.array(values, dtype=wp.float32, device=device)
    device_output = wp.zeros(args.count, dtype=wp.float32, device=device)
    wp.launch(add_one, dim=args.count, inputs=[device_values, device_output], device=device)
    wp.synchronize_device(device)
    observed = device_output.numpy()
    expected = values + 1.0
    result.update(
        {
            "status": "ok",
            "device": str(device),
            "max_abs_error": float(np.max(np.abs(observed - expected))),
            "first_values": observed[:8].tolist(),
            "last_values": observed[-8:].tolist(),
        }
    )
  except Exception as exc:
    result.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
    if args.output:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    raise

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
