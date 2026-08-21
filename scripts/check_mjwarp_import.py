"""Record MuJoCo and MuJoCo-Warp import compatibility for one environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def package_version(name: str) -> str | None:
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return None


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  result: dict[str, object] = {
      "schema": "mujoco-warp-amd-import-probe-v1",
      "python": __import__("sys").version.split()[0],
      "mujoco_distribution": package_version("mujoco"),
      "mujoco_warp_distribution": package_version("mujoco-warp"),
  }
  try:
    import mujoco

    result["mujoco_module_version"] = getattr(mujoco, "__version__", None)
    result["mujoco_import"] = "ok"
  except Exception as exc:
    result.update(
        {
            "mujoco_import": "error",
            "mujoco_error_type": type(exc).__name__,
            "mujoco_error": str(exc),
        }
    )

  try:
    import mujoco_warp

    result.update({"mujoco_warp_import": "ok", "mujoco_warp_module": str(mujoco_warp.__file__)})
  except Exception as exc:
    result.update(
        {
            "mujoco_warp_import": "error",
            "mujoco_warp_error_type": type(exc).__name__,
            "mujoco_warp_error": str(exc),
        }
    )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
