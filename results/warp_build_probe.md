# Warp HIP Build Probe

Date: 2026-08-20

Command:

```bash
/data/Data14TB/envs/perceptive-cbf-rl-official/bin/python build_lib.py \
  --rocm-path=/data/Data14TB/envs/mujoco-warp-hip-toolchain/usr \
  --hip-arch=gfx1151 \
  --quick
```

Result: exit code `1`.

The host-side LLVM component `warp-clang.so` was rebuilt successfully, but the
HIP device backend was not produced. The isolated toolchain reports ROCm
5.7.1, while the pinned AMD Warp source requires ROCm 7.0 or newer. The same
build also stops on the older HIP headers because `HIP_MEMCPY2D` is missing.

This is a toolchain compatibility result, not a MuJoCo-Warp algorithm result.
The source remains unchanged. A functional AMD device run requires a ROCm 7.x
HIP compiler and headers, followed by a fresh build and the GPU benchmark in
`docs/benchmark-protocol.md`.

Artifacts:

- `upstream/warp/warp/bin/warp-clang.so`
  - SHA256: `b64e1f4632189dfca967771e1af6a4f58d4dfe46c4ecfcce6e99b961540f379b`
- The HIP benchmark hash from this older probe is superseded by the ROCm 7.14
  rebuild below.

## ROCm 7.14 SDK probe

Date: 2026-08-20

The isolated AMD Core SDK environment at
`/data/Data14TB/envs/mujoco-warp-rocm714` provides HIP 7.14.60850 and was
used to rebuild the pinned Warp snapshot for `gfx1151`:

```bash
ENV=/data/Data14TB/envs/mujoco-warp-rocm714
SDK="$ENV/lib/python3.11/site-packages/_rocm_sdk_devel"
PATH="$ENV/bin:$PATH" ROCM_PATH="$SDK" HIP_PATH="$SDK" \
  /data/Data14TB/envs/perceptive-cbf-rl-official/bin/python build_lib.py \
  --rocm-path="$SDK" --hip-arch=gfx1151 --quick
```

Result: exit code `0`. Both the host and HIP device artifacts were produced:

- `upstream/warp/warp/bin/warp.so`
  - SHA256: `7fa7ab81da5f4f6bea86079c800977a87824c7e9421ed4f5c4d91760f0fa8839`
- `upstream/warp/warp/bin/warp-clang.so`
  - SHA256: `b64e1f4632189dfca967771e1af6a4f58d4dfe46c4ecfcce6e99b961540f379b`

- `build/hip_graph_benchmark`
  - SHA256: `e64bf2196c522d6638ba55f328729bc03fef79b7e4d390f112b6106cc786df56`

The Python package imports successfully from the rebuilt tree. This removes
the old ROCm 5.7 toolchain blocker, but it does not imply GPU execution: the
current session cannot open `/dev/kfd` because the user is not in the `render`
group. The HIP benchmark therefore still exits before device enumeration and
has no throughput result.

## AMD395 target build and GPU verification

Date: 2026-08-21

The actual target is the LAN AMD Ryzen AI Max+ 395, not the local RTX PRO 6000.
It runs system ROCm 7.2.1 and reports `gfx1151`. The pinned Warp source was
rebuilt on that target with:

```bash
PATH=/opt/rocm/bin:$PATH ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
  /home/aup/envs/openpi-amd-jax010/bin/python build_lib.py \
  --rocm-path=/opt/rocm --hip-arch=gfx1151 --no-use-libmathdx \
  --quick --no-build-llvm --no-standalone --hipcc-options="-O0" -j4
```

Result: exit code `0`.

- `upstream/warp/warp/bin/warp.so` SHA256:
  `9e29f254ca4ab85078919cf9930d6e350c76db3d8f70e7489e096b2eea84f150`
- `build/hip_graph_benchmark` SHA256:
  `2a368e41fd959766fe117a9f543c713d8dbe44d6e48ea8c7ec646f848a1b8b91`
- `results/hip_humanoid-contact_rocm721.json` SHA256:
  `5802bf4b54a5cf64b349218fcbf537268a55a8108ce01f8859f7e2af03db705a`
- `results/warp_gpu_smoke_rocm721.json` SHA256:
  `9d55e88f9deb5f9673e8175ed9c29fa2b7d3f9ee0d165f4f41319f9f46d85985`

The Warp GPU smoke compiled and executed a kernel on `cuda:0` (Warp's HIP
alias for the AMD device), with zero numerical error. The separate HIP graph
benchmark completed with zero state error and recorded the full performance
comparison in `docs/benchmark-protocol.md`.
