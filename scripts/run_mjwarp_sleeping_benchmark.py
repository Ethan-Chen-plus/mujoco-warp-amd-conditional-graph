#!/usr/bin/env python3
"""Benchmark ALOHA sleeping/wakeup broadphase variants on a GPU device.

The benchmark uses the same MJCF, batched controls, warmup, and rollout for
three modes:

* ``eager``: rebuild the broadphase on every step;
* ``static``: capture the native solver graph and rebuild the broadphase every
  replay;
* ``native``: use the same solver graph and add the sleeping broadphase
  branches, rebuilding only when the wake predicate requests it.

The result is a performance and numerical-evidence artifact. It does not
claim task success; it measures physics throughput and state evolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def _control_values(mjm, phase: int, wake_every: int, *, enable_motion: bool) -> np.ndarray:
  """Create a bounded actuator target for one rollout step.

  ``motion`` mode changes one actuator target so the device-side geometry
  detector must wake the broadphase. ``manual`` mode keeps the scene
  quiescent and only exercises explicit predicate toggles.
  """
  if mjm.nu == 0:
    return np.empty((0,), dtype=np.float32)

  limits = np.asarray(mjm.actuator_ctrlrange, dtype=np.float32)
  if limits.shape != (mjm.nu, 2):
    return np.zeros((mjm.nu,), dtype=np.float32)

  lower = limits[:, 0]
  upper = limits[:, 1]
  baseline = np.clip(np.zeros_like(lower), lower, upper)
  values = baseline.copy()
  if enable_motion and wake_every > 0 and (phase // wake_every) % 2 == 1:
    # Move one bounded joint target during the wake interval. A single target
    # keeps the stimulus deterministic and avoids changing the task geometry.
    values[0] = lower[0] + 0.75 * (upper[0] - lower[0])
  return values


def _state_digest(data) -> str:
  """Return a stable digest of the final batched qpos and qvel arrays."""
  digest = hashlib.sha256()
  digest.update(np.asarray(data.qpos.numpy()).tobytes())
  digest.update(np.asarray(data.qvel.numpy()).tobytes())
  return digest.hexdigest()


def _portable_model_path(path: Path) -> str:
  """Store a checkout-relative model path in public benchmark evidence."""
  text = str(path)
  marker = "/upstream/mujoco_warp/"
  if marker in text:
    return "upstream/mujoco_warp/" + text.split(marker, 1)[1]
  return text


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("model", type=Path)
  parser.add_argument("--variant", choices=("eager", "static", "native"), required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--worlds", type=int, default=64)
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--wake-every", type=int, default=40)
  parser.add_argument("--gravity", choices=("default", "zero"), default="default")
  parser.add_argument("--wake-mode", choices=("motion", "manual"), default="motion")
  parser.add_argument("--stimulus", choices=("joint_target", "none"), default="joint_target")
  parser.add_argument(
    "--fixture",
    choices=("dynamic", "freeze_after_warmup"),
    default="dynamic",
    help="Use the normal dynamic rollout or freeze qvel after warmup for branch-overhead measurement.",
  )
  parser.add_argument("--nconmax", type=int, default=4096)
  parser.add_argument("--njmax", type=int, default=2048)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  # Keep the baselines comparable. Both variants use the already validated
  # native capture_while solver path; only native enables the sleeping cache
  # and its capture_if branch.
  os.environ["WP_HIP_CONDITIONAL_NATIVE"] = "1"
  os.environ["WP_HIP_CONDITIONAL_EMULATION"] = "0"
  # Keep this benchmark on MuJoCo-Warp's stable step path. The native solver
  # still inserts capture_while and the sleeping branch inserts capture_if;
  # the separate full-step auto-capture path is measured by another script.
  os.environ["WP_HIP_GRAPH_ENABLE"] = "0"
  os.environ["MJW_HIP_SLEEPING_CAPTURE_IF"] = "1" if args.variant == "native" else "0"
  os.environ["MJW_HIP_BVH_CACHE"] = "1" if args.variant == "native" else "0"
  os.environ["MJW_HIP_SLEEPING_MANUAL_PREDICATE"] = "1" if args.wake_mode == "manual" else "0"

  import mujoco  # type: ignore
  import mujoco_warp as mjw  # type: ignore
  import warp as wp  # type: ignore

  wp.init()
  with wp.ScopedDevice(args.device):
    model_path = args.model.resolve()
    # MuJoCo resolves compiler meshdir paths relative to the current working
    # directory, so load the MJCF from its own directory for portable assets.
    os.chdir(model_path.parent)
    mjm = mujoco.MjModel.from_xml_path(model_path.name)
    mjdata = mujoco.MjData(mjm)
    mujoco.mj_resetData(mjm, mjdata)
    if args.gravity == "zero":
      mjm.opt.gravity[:] = 0.0
    model = mjw.put_model(mjm)
    # Lock the solver graph mode after model construction. The eager variant
    # is the direct-step reference; static and native share capture_while so
    # the P1 comparison isolates the additional capture_if branch.
    model.opt.graph_conditional = args.variant in ("static", "native")
    data = mjw.put_data(
      mjm,
      mjdata,
      nworld=args.worlds,
      nconmax=args.nconmax,
      njmax=args.njmax,
    )

    control_buffer = wp.zeros((max(mjm.nu, 1),), dtype=wp.float32, device=args.device)

    def set_controls(values: np.ndarray) -> None:
      if mjm.nu:
        control_buffer.assign(values)
        # The public Data API exposes ctrl as a batched array. A host update
        # before graph replay is intentional: it models a new control packet.
        data.ctrl.assign(np.broadcast_to(values, (args.worlds, mjm.nu)).copy())

    enable_motion = args.stimulus == "joint_target"
    zero_controls = _control_values(mjm, 0, args.wake_every, enable_motion=enable_motion)
    set_controls(zero_controls)
    for _ in range(args.warmup):
      mjw.step(model, data)
    wp.synchronize()

    if args.fixture == "freeze_after_warmup":
      if args.stimulus != "none":
        raise ValueError("freeze_after_warmup requires --stimulus=none")
      # Start timing from a settled ALOHA state. This isolates broadphase
      # branch reuse from ongoing joint motion and contact settling.
      data.qvel.zero_()
      if hasattr(data, "qacc"):
        data.qacc.zero_()
      if hasattr(data, "qfrc_applied"):
        data.qfrc_applied.zero_()
      if hasattr(data, "ctrl"):
        data.ctrl.zero_()
      wp.synchronize()

    if args.variant == "native" and args.wake_mode == "manual":
      data._bvh_wake_predicate.fill_(1)
    # Finish the warmup before timing the stable step path.
    wp.synchronize()
    graph_replay = False

    ncollision = []
    solver_niter = []
    wake_predicate = []
    started = time.perf_counter()
    for step in range(args.steps):
      set_controls(_control_values(mjm, step, args.wake_every, enable_motion=enable_motion))
      if args.variant == "native" and args.wake_mode == "manual":
        wake_value = 1 if args.wake_every <= 0 or step % args.wake_every == 0 else 0
        data._bvh_wake_predicate.fill_(wake_value)
        wp.synchronize()
      mjw.step(model, data)
      wp.synchronize()
      ncollision.append(int(np.max(data.ncollision.numpy())))
      solver_niter.append(float(np.mean(data.solver_niter.numpy())))
      if hasattr(data, "_bvh_wake_predicate"):
        wake_predicate.append(int(np.asarray(data._bvh_wake_predicate.numpy())[0]))
    elapsed = time.perf_counter() - started

    qpos = np.asarray(data.qpos.numpy())
    qvel = np.asarray(data.qvel.numpy())
    result = {
      "schema": "mujoco-warp-amd-sleeping-if-v1",
      "variant": args.variant,
      "model": _portable_model_path(args.model),
      "device": args.device,
      "python": platform.python_version(),
      "worlds": args.worlds,
      "nconmax": args.nconmax,
      "njmax": args.njmax,
      "warmup_steps": args.warmup,
      "steps": args.steps,
      "wake_every": args.wake_every,
      "wake_mode": args.wake_mode,
      "stimulus": args.stimulus,
      "fixture": args.fixture,
      "gravity": args.gravity,
      "graph_replay": graph_replay,
      "solver_graph_conditional": bool(model.opt.graph_conditional),
      "warp_conditional_supported": bool(wp.is_conditional_graph_supported()),
      "warp_conditional_emulated": bool(wp.is_conditional_graph_emulated()),
      "conditional_graph_path": (
        "capture_while_plus_capture_if"
        if args.variant == "native"
        else "capture_while_only"
        if args.variant == "static"
        else "eager_solver_loop"
      ),
      "elapsed_s": elapsed,
      "world_steps_per_second": args.worlds * args.steps / elapsed,
      "ncollision_mean": float(np.mean(ncollision)),
      "ncollision_p95": float(np.percentile(ncollision, 95)),
      "solver_niter_mean": float(np.mean(solver_niter)),
      "solver_niter_p95": float(np.percentile(solver_niter, 95)),
      "wake_predicate_steps": int(sum(wake_predicate)),
      "wake_predicate_fraction": float(np.mean(wake_predicate)) if wake_predicate else None,
      "converged_worlds": int(np.sum(~np.any(np.isnan(qpos), axis=1))),
      "final_state_sha256": _state_digest(data),
      "final_qpos": qpos.tolist(),
      "final_qvel": qvel.tolist(),
    }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({
    key: result[key]
    for key in (
      "variant",
      "worlds",
      "steps",
      "elapsed_s",
      "world_steps_per_second",
      "ncollision_p95",
      "solver_niter_p95",
      "wake_predicate_fraction",
      "converged_worlds",
      "final_state_sha256",
    )
  }, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
