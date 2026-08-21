# Device-Gated `capture_while` for MuJoCo-Warp on AMD ROCm

## Abstract

This report presents a reproducible MuJoCo-Warp conditional-solver PoC on an
AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`, ROCm 7.2.1). The target
path is the Newton constraint solver's `capture_while` loop. Because the
target ROCm runtime exposes ordinary HIP graph capture but no
`hipGraphConditionalHandle` API, the implementation uses one fixed-unroll HIP
graph whose solver kernels gate completed worlds from device-resident state.

The same MuJoCo 3.8.1 humanoid contact workload is run through an eager HIP
solver loop and the device-gated graph. The frozen primary run reports
673,563 versus 493,084 worlds/s (`1.366x`) with state errors below `1e-6`.

## Software and hardware

| Item | Value |
| --- | --- |
| CPU/GPU | AMD Ryzen AI Max+ 395 / Radeon 8060S |
| GPU target | `gfx1151` |
| ROCm | 7.2.1 |
| Python | 3.12.13 |
| MuJoCo | 3.8.1 |
| MuJoCo-Warp | 3.8.1 source snapshot |
| Warp | 1.13.0+rocm.0, HIP backend |
| Environment | `/home/aup/envs/mujoco-warp-amd-py312` |

The original JAX environment was left independent with MuJoCo 3.4.0. The
version mismatch is resolved for MJWarp by the new MicroMamba environment; the
existing environment is not changed.

## Implementation

The implementation is split across the pinned source snapshots:

- `warp/_src/context.py` exposes an explicit HIP emulation mode and expands
  `capture_while` into one fixed-length graph without host condition reads.
- `mujoco_warp/_src/io.py` enables the mode only when explicitly requested and
  allocates solver scratch outside the capture window.
- `mujoco_warp/_src/solver.py` resets device counters with capture-safe work,
  passes the MuJoCo iteration cap, and gates solver work using `ctx.done`.
- `mujoco_warp/_src/forward.py` keeps the capture path on one stream when
  required by the HIP runtime.
- `mujoco_warp/_src/cli.py` performs warm-up before capture and replays the
  resulting graph for steady-state measurement.

The mode is selected with:

```bash
WP_HIP_CONDITIONAL_EMULATION=1
```

This is a HIP device-gated compatibility implementation. It is not a native
conditional graph node because the target ROCm headers and runtime do not
provide the required ABI. The native boundary is recorded rather than hidden.

## Benchmark protocol

The fixed workload is the pinned humanoid MJCF with 1024 worlds and 1000
steps, `nconmax=128`, `njmax=128`, identical initial state, and one HIP stream.
The graph build and JIT phase are reported separately. Both variants must
finish with exit code 0 and converge all worlds.

Primary artifact: `results/mjwarp_humanoid_conditional_rocm721.json`.

| Variant | Runtime (s) | Throughput (worlds/s) |
| --- | ---: | ---: |
| Eager HIP solver loop | 2.076724 | 493,084 |
| Device-gated single HIP graph | 1.520274 | 673,563 |

The device-gated path reduces runtime by `26.79%`. A paired 256-world,
100-step state comparison reports:

```text
max_abs_qpos_error = 7.86967576e-7
max_abs_qvel_error = 9.53674316e-7
numerically_equivalent = true
```

The target telemetry snapshot records 45-47 C edge temperature, 100% GPU busy,
and 76% VRAM allocation. A long-lived policy service shared the GPU, so the
telemetry is labeled shared-host telemetry.

## Revalidation record

The later files
`results/mjwarp_humanoid_baseline_1024x1000_revalidation.log`,
`results/mjwarp_humanoid_emulation_1024x1000_revalidation.log`, and
`results/mjwarp_conditional_correctness_256x100_revalidation.log` are retained
as a second run. It completed correctly, with the same state tolerance, while
the policy service was actively using the Radeon. Its lower throughput is not
merged with the frozen primary result; it documents the effect of shared GPU
load and keeps the comparison auditable.

## Reproduction

The complete command is:

```bash
ROOT=/path/to/mujoco-warp-amd-conditional-graph
ENV=/home/aup/envs/mujoco-warp-amd-py312
ROCM_PATH=/opt/rocm GPU_ARCH=gfx1151 ROCM_TAG=rocm721 \
  ROOT="$ROOT" ENV="$ENV" bash "$ROOT/scripts/run_mjwarp_amd_benchmark.sh"
```

The script emits paired logs, the 256-world correctness log, a combined JSON
summary, and `SHA256SUMS`. `results/mujoco_warp_import_rocm721_py312.json`
records the successful full-package import in the 3.8.1 environment.

## Follow-up boundary

The P0 solver compatibility path and full MuJoCo 3.8.1 AMD395 throughput
comparison are complete. A native HIP conditional-node port depends on a
future ROCm runtime/API implementation. The sleeping broadphase `capture_if`
experiment is an independent P1 extension and is not included in this P0
claim.
