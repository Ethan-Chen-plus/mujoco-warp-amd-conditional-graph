"""Print a short MJWarp state trace for the AMD conditional-while PoC."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import warp as wp
import mujoco_warp as mjw


def run_mode(xml: Path, mode: str, nworld: int, steps: int):
  os.environ["WP_HIP_CONDITIONAL_EMULATION"] = "1" if mode == "emulation" else "0"
  mjm = mujoco.MjModel.from_xml_path(str(xml))
  mjd = mujoco.MjData(mjm)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  print(f"mode={mode} graph_conditional={bool(m.opt.graph_conditional)}")

  mjw.step(m, d)
  wp.synchronize_device(wp.get_device())
  print(f"  warmup nacon={int(d.nacon.numpy()[0])} ncollision={int(d.ncollision.numpy()[0])} nefc={int(d.nefc.numpy()[0])}")

  capture = None
  if mode == "emulation":
    with wp.ScopedCapture() as capture_scope:
      mjw.step(m, d)
    capture = capture_scope
    wp.synchronize_device(wp.get_device())
    print(f"  capture nacon={int(d.nacon.numpy()[0])} ncollision={int(d.ncollision.numpy()[0])} nefc={int(d.nefc.numpy()[0])}")

  for i in range(steps):
    if capture is not None and capture.graph is not None:
      wp.capture_launch(capture.graph)
    else:
      mjw.step(m, d)
    wp.synchronize_device(wp.get_device())
    print(f"  step={i + 1} nacon={int(d.nacon.numpy()[0])} ncollision={int(d.ncollision.numpy()[0])} nefc={int(d.nefc.numpy()[0])}")
  return d.qpos.numpy().copy(), d.qvel.numpy().copy()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("xml", type=Path)
  parser.add_argument("--nworld", type=int, default=8)
  parser.add_argument("--steps", type=int, default=3)
  args = parser.parse_args()
  wp.init()
  states = {mode: run_mode(args.xml, mode, args.nworld, args.steps) for mode in ("baseline", "emulation")}
  qpos_error = abs(states["baseline"][0] - states["emulation"][0]).max()
  qvel_error = abs(states["baseline"][1] - states["emulation"][1]).max()
  print(f"comparison max_abs_qpos_error={qpos_error:.9g} max_abs_qvel_error={qvel_error:.9g}")


if __name__ == "__main__":
  main()
