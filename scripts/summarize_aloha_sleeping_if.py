#!/usr/bin/env python3
"""Summarize paired ALOHA conditional-branch benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_model_path(value: object) -> str:
  """Keep evidence portable when a runner recorded its absolute checkout path."""
  text = str(value)
  marker = "/upstream/mujoco_warp/"
  if marker in text:
    return "upstream/mujoco_warp/" + text.split(marker, 1)[1]
  return text


def load_variant(directory: Path, variant: str) -> dict:
  path = directory / f"{variant}.json"
  row = json.loads(path.read_text(encoding="utf-8"))
  row["model"] = normalize_model_path(row.get("model", ""))
  row["fixture"] = row.get("fixture") or "dynamic"
  return row


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  rows = {variant: load_variant(args.input, variant) for variant in ("eager", "static", "native")}
  eager = rows["eager"]
  static = rows["static"]
  native = rows["native"]
  summary = {
    "schema": "mujoco-warp-amd-sleeping-if-summary-v2",
    "protocol": {
      "same_model": all(row["model"] == eager["model"] for row in rows.values()),
      "same_worlds": len({row["worlds"] for row in rows.values()}) == 1,
      "same_steps": len({row["steps"] for row in rows.values()}) == 1,
      "stimulus": {variant: row["stimulus"] for variant, row in rows.items()},
      "fixture": {variant: row.get("fixture", "dynamic") for variant, row in rows.items()},
      "wake_mode": {variant: row["wake_mode"] for variant, row in rows.items()},
    },
    "variants": rows,
    "native_vs_eager_speedup": native["world_steps_per_second"] / eager["world_steps_per_second"],
    "native_vs_static_speedup": native["world_steps_per_second"] / static["world_steps_per_second"],
    "native_wake_predicate_fraction": native["wake_predicate_fraction"],
    "all_variants_converged": all(
      row["converged_worlds"] == row["worlds"] for row in rows.values()
    ),
    "state_digest": {
      variant: row["final_state_sha256"] for variant, row in rows.items()
    },
    "interpretation": (
      "The native capture_if path executed with the device wake predicate. "
      "A native speedup is expected only after the workload reaches a stable "
      "sleeping interval; a wake fraction near one means the ALOHA scene is "
      "still moving and the broadphase must be rebuilt."
    ),
  }
  args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({
    "output": str(args.output),
    "native_vs_eager_speedup": summary["native_vs_eager_speedup"],
    "native_vs_static_speedup": summary["native_vs_static_speedup"],
    "native_wake_predicate_fraction": summary["native_wake_predicate_fraction"],
    "all_variants_converged": summary["all_variants_converged"],
  }, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
