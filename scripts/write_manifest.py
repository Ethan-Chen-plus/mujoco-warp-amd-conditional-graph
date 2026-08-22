#!/usr/bin/env python3
"""Write source, executable, and result hashes for a reproducible PoC."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


KNOWN_UPSTREAM_REVISIONS = {
    "mujoco_warp": "9229bb9d1a698c9464df862a915b46899720338c",
    "warp": "8ca65dd5f8a444785408ecaa956bac0d2c427d6f",
}


def git_revision(path: Path, fallback: str) -> str:
    try:
        top_level = Path(
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        # The public bundle may itself be the nearest Git worktree for the
        # vendored source snapshots. In that case its HEAD is not the upstream
        # source revision; use the pinned revision recorded by the bundle.
        if top_level == ROOT.resolve():
            return fallback
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/source_manifest.json")
    args = parser.parse_args()
    files = []
    for path in sorted((ROOT / "hip").glob("*.cpp")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in sorted((ROOT / "scripts").glob("*.py")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in sorted((ROOT / "scripts").glob("*.sh")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    source_paths = [
        ROOT / "upstream/mujoco_warp/mujoco_warp/_src/cli.py",
        ROOT / "upstream/mujoco_warp/mujoco_warp/_src/forward.py",
        ROOT / "upstream/mujoco_warp/mujoco_warp/_src/io.py",
        ROOT / "upstream/mujoco_warp/mujoco_warp/_src/solver.py",
        ROOT / "upstream/mujoco_warp/mujoco_warp/_src/warp_util.py",
        ROOT / "upstream/warp/warp/_src/context.py",
        ROOT / "upstream/warp/warp/__init__.py",
        ROOT / "upstream/warp/warp/_src/build_dll.py",
        ROOT / "upstream/warp/warp/native/cuda_util.cpp",
        ROOT / "upstream/warp/warp/native/hip_util.h",
        ROOT / "upstream/warp/warp/native/warp.cu",
    ]
    for path in source_paths:
        if path.exists():
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    binary = ROOT / "build/hip_graph_benchmark"
    if binary.exists():
        files.append({"path": str(binary.relative_to(ROOT)), "sha256": sha256(binary), "bytes": binary.stat().st_size})
    for path in sorted((ROOT / "results").glob("hip_execution_attempt*.txt")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in sorted((ROOT / "results").glob("conditional_while_device*.txt")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for pattern in (
        "hip_*.json",
        "warp_gpu_smoke*.json",
        "mujoco_warp_import*.json",
        "mjwarp_*.json",
        "mjwarp_cpu_semantics.json",
        "capability_probe*.json",
        "native_conditional_*.json",
    ):
        for path in sorted((ROOT / "results").glob(pattern)):
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in sorted((ROOT / "results/mjwarp_native_conditional_amd395").glob("*.json")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in sorted((ROOT / "results/aloha_sleeping_if_amd395_revalidation").glob("*.json")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in sorted((ROOT / "results/aloha_sleeping_if_amd395_manual_branch").glob("*.json")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in sorted((ROOT / "upstream/warp/warp/bin").glob("warp*.so")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": "mujoco-warp-amd-source-manifest-v1",
        "upstream": {
            "mujoco_warp": {
                "repository": "https://github.com/AMD-Ecosystem/mujoco_warp",
                "ref": "amd-integration",
                "commit": git_revision(
                    ROOT / "upstream/mujoco_warp", KNOWN_UPSTREAM_REVISIONS["mujoco_warp"]
                ),
            },
            "warp": {
                "repository": "https://github.com/AMD-Ecosystem/warp",
                "ref": "fet/hip-graph-capture",
                "commit": git_revision(ROOT / "upstream/warp", KNOWN_UPSTREAM_REVISIONS["warp"]),
            },
        },
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
