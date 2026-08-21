#!/usr/bin/env python3
"""Validate the CPU-visible structure and frozen evidence of this repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    path = ROOT / relative
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {relative}")
    return value


def validate_required_files() -> None:
    required = [
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "CITATION.cff",
        "scripts/bootstrap_amd395_env.sh",
        "scripts/run_mjwarp_amd_benchmark.sh",
        "docs/technical-report.md",
        "docs/native-conditional-node-plan.md",
        "patches/warp-hip-conditional.diff",
        "patches/mujoco-warp-amd.diff",
        "results/mjwarp_humanoid_conditional_rocm721.json",
        "results/mujoco_warp_import_rocm721_py312.json",
        "results/source_manifest.json",
        "results/SHA256SUMS",
    ]
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))


def validate_primary_result() -> None:
    result = load_json("results/mjwarp_humanoid_conditional_rocm721.json")
    comparison = result["comparison"]
    correctness = comparison["correctness"]
    speedup = float(comparison["throughput_speedup"])
    qpos_error = float(correctness["max_abs_qpos_error"])
    qvel_error = float(correctness["max_abs_qvel_error"])
    if result.get("benchmark_status") != "frozen_primary_run":
        raise ValueError("primary result is not marked frozen_primary_run")
    if speedup <= 1.0:
        raise ValueError(f"primary speedup is not above 1.0: {speedup}")
    if qpos_error > 1e-5 or qvel_error > 1e-5:
        raise ValueError(f"state error exceeds tolerance: {qpos_error}, {qvel_error}")
    if correctness.get("numerically_equivalent") is not True:
        raise ValueError("primary correctness gate is false")


def validate_import_probe() -> None:
    probe = load_json("results/mujoco_warp_import_rocm721_py312.json")
    expected = {
        "mujoco_distribution": "3.8.1",
        "mujoco_warp_distribution": "3.8.1",
        "mujoco_module_version": "3.8.1",
        "mujoco_import": "ok",
        "mujoco_warp_import": "ok",
    }
    for key, value in expected.items():
        if probe.get(key) != value:
            raise ValueError(f"import probe mismatch for {key}: {probe.get(key)!r}")


def validate_manifest() -> None:
    manifest = load_json("results/source_manifest.json")
    upstream = manifest["upstream"]
    if upstream["warp"]["commit"] != "8ca65dd5f8a444785408ecaa956bac0d2c427d6f":
        raise ValueError("unexpected Warp base commit")
    if upstream["mujoco_warp"]["commit"] != "9229bb9d1a698c9464df862a915b46899720338c":
        raise ValueError("unexpected MuJoCo-Warp base commit")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest file is missing: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"manifest hash mismatch: {entry['path']}")


def validate_sha_file() -> None:
    checksum_file = ROOT / "results/SHA256SUMS"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative_path = relative.removeprefix("./")
        path = ROOT / relative_path
        if relative_path == "results/SHA256SUMS" or not path.is_file():
            raise ValueError(f"invalid SHA entry: {relative}")
        if sha256(path) != digest:
            raise ValueError(f"SHA256 mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sha", action="store_true")
    args = parser.parse_args()
    validate_required_files()
    validate_primary_result()
    validate_import_probe()
    validate_manifest()
    if not args.skip_sha:
        validate_sha_file()
    print("public bundle validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
