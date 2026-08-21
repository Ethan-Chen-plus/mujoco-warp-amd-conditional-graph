#!/usr/bin/env python3
"""Record local HIP and MuJoCo-Warp capability without claiming a GPU result."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_TOOLCHAIN = Path(
    os.environ.get("ROCM_TOOLCHAIN", str(ROOT / ".toolchain"))
)


def rocm_roots() -> list[Path]:
    roots: list[Path] = []
    configured = Path(os.environ["ROCM_PATH"]) if os.environ.get("ROCM_PATH") else None
    candidates = [configured, Path("/opt/rocm"), LOCAL_TOOLCHAIN / "usr", LOCAL_TOOLCHAIN]
    for base in candidates:
        if base is None or not base.exists():
            continue
        if (base / "include/hip/hip_runtime.h").is_file():
            roots.append(base)
            continue
        if (base / "usr/include/hip/hip_runtime.h").is_file():
            roots.append(base / "usr")
            continue
        for candidate in sorted(base.glob("lib/python*/site-packages/_rocm_sdk_devel")):
            if (candidate / "include/hip/hip_runtime.h").is_file():
                roots.append(candidate)
        for candidate in sorted(base.glob("lib/python*/site-packages/rocm_sdk_devel")):
            if (candidate / "include/hip/hip_runtime.h").is_file():
                roots.append(candidate)
    return list(dict.fromkeys(roots))


def command_path(name: str) -> str | None:
    explicit = os.environ.get(name.upper())
    if explicit and Path(explicit).exists():
        return explicit
    for root in rocm_roots():
        for candidate in (root / "bin" / name, root / "usr/bin" / name):
            if candidate.exists():
                return str(candidate)
    return shutil.which(name)


def run_command(command: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return result.returncode, (result.stdout + result.stderr).strip()


def header_paths() -> list[Path]:
    paths: list[Path] = []
    for base in rocm_roots():
        paths.extend([base / "include/hip/hip_runtime_api.h", base / "include/hip/hip_runtime.h"])
    paths.extend([Path("/usr/include/hip/hip_runtime_api.h"), Path("/usr/include/hip/hip_runtime.h")])
    return list(dict.fromkeys(paths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kfd = Path("/dev/kfd")
    render_nodes = sorted(Path("/dev/dri").glob("renderD*"))
    header_records = []
    conditional_header_hits = []
    for path in header_paths():
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        hits = re.findall(r"hipGraph(?:Conditional|AddConditional)[A-Za-z0-9_]*", text)
        header_records.append(str(path))
        conditional_header_hits.extend(hits)

    rocminfo = command_path("rocminfo")
    hipcc = command_path("hipcc")
    probe_env = os.environ.copy()
    roots = rocm_roots()
    if roots:
        root = roots[0]
        probe_env["PATH"] = f"{root}/bin:{root}/usr/bin:{root}/lib/llvm/bin:{probe_env['PATH']}"
        probe_env["HIP_PATH"] = str(root)
    rocminfo_result = None
    if rocminfo or (LOCAL_TOOLCHAIN / "usr/bin/rocminfo").exists():
        path = rocminfo or str(LOCAL_TOOLCHAIN / "usr/bin/rocminfo")
        code, output = run_command([path], probe_env)
        rocminfo_result = {"returncode": code, "output_head": output[:4000]}

    record = {
        "schema": "mujoco-warp-amd-capability-v1",
        "root": str(ROOT),
        "user": {"uid": os.getuid(), "groups": os.getgroups()},
        "devices": {
            "kfd_exists": kfd.exists(),
            "kfd_read_write": os.access(kfd, os.R_OK | os.W_OK),
            "render_nodes": [str(p) for p in render_nodes],
            "render_read_write": {str(p): os.access(p, os.R_OK | os.W_OK) for p in render_nodes},
        },
        "tools": {"hipcc": hipcc, "rocminfo": rocminfo},
        "rocm_roots": [str(root) for root in rocm_roots()],
        "headers": {
            "checked": header_records,
            "conditional_symbols": sorted(set(conditional_header_hits)),
            "conditional_graph_api_detected": bool(conditional_header_hits),
        },
        "rocminfo": rocminfo_result,
        "interpretation": {
            "native_hip_conditional_graph": bool(conditional_header_hits),
            "static_hip_graph_can_be_tested": bool(hipcc),
            "gpu_measurement_ready": bool(hipcc and kfd.exists() and os.access(kfd, os.R_OK | os.W_OK)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
