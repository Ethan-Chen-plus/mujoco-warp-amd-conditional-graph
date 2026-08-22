#!/usr/bin/env python3
"""Extract conditional-graph call sites from the pinned MuJoCo-Warp sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "solver": {
        "path": "upstream/mujoco_warp/mujoco_warp/_src/solver.py",
        "patterns": [
            "wp.capture_while",
            "graph_conditional",
            "wp.get_device().is_hip",
            "MJW_SOLVER_NCHECK",
        ],
    },
    "forward": {
        "path": "upstream/mujoco_warp/mujoco_warp/_src/forward.py",
        "patterns": [
            "WP_HIP_GRAPH_ENABLE",
            "_HIP_GRAPH_ITER_SEQUENCE",
            "wp.ScopedCapture",
            "hipGraphLaunch",
        ],
    },
    "broadphase": {
        "path": "upstream/mujoco_warp/mujoco_warp/_src/collision_driver.py",
        "patterns": [
            "_run_broadphase",
            "Reuse cached broadphase result",
            "m.opt.broadphase",
            "capture_if",
        ],
    },
    "warp_conditional_backend": {
        "path": "upstream/warp/warp/native/warp.cu",
        "patterns": [
            "__HIP_PLATFORM_AMD__",
            "cudaGraphConditionalHandle",
            "wp_cuda_graph_insert_if_else",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def matches(path: Path, patterns: list[str]) -> list[dict[str, object]]:
    lines = path.read_text(errors="replace").splitlines()
    found = []
    for line_number, line in enumerate(lines, start=1):
        for pattern in patterns:
            if pattern in line:
                start = max(0, line_number - 2)
                end = min(len(lines), line_number + 1)
                found.append(
                    {
                        "pattern": pattern,
                        "line": line_number,
                        "snippet": lines[start:end],
                    }
                )
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/source_callsite_map.json")
    args = parser.parse_args()

    records = {}
    for name, target in TARGETS.items():
        path = ROOT / str(target["path"])
        records[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "matches": matches(path, list(target["patterns"])),
        }

    broadphase_matches = records["broadphase"]["matches"]
    records["broadphase"]["interpretation"] = (
        "The collision driver contains the opt-in device wake predicate and capture_if adapter. "
        "Verify the native runtime and ALOHA benchmark before reporting performance."
        if not any(item["pattern"] == "capture_if" for item in broadphase_matches)
        else "A capture_if token is present; verify its device predicate semantics and benchmark artifact."
    )

    result = {
        "schema": "mujoco-warp-amd-source-callsite-v1",
        "upstream_revisions": {
            "mujoco_warp": revision(ROOT / "upstream/mujoco_warp"),
            "warp": revision(ROOT / "upstream/warp"),
        },
        "targets": records,
        "priority": {
            "p0": "solver capture_while and its HIP convergence fallback",
            "p1": "experimental sleeping broadphase cache with a native capture_if node",
            "deferred": "capture_switch; no main-path use found in the pinned snapshot",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
