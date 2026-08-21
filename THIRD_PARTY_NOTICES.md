# Third-Party Notices

This repository distributes source snapshots and modifications derived from
the following Apache-2.0 projects:

- NVIDIA Warp: `upstream/warp/`
- MuJoCo-Warp: `upstream/mujoco_warp/`

The original license files remain in their respective source directories. The
top-level scripts, benchmark harness, documentation and experiment records in
this repository are provided under the Apache License 2.0; see `LICENSE`.

The public repository does not include the machine-specific compiled Warp
shared libraries or local build cache. They must be built on the target AMD
host with the installed ROCm toolchain.
