# Native HIP `capture_while` for MuJoCo-Warp on AMD ROCm

## Abstract

This report presents a reproducible MuJoCo-Warp conditional-solver port on an
AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`, ROCm 7.2.1). The target
path is the Newton constraint solver's `capture_while` loop. A matching HIP
SDK and HIP/CLR patch add an experimental `hipGraphConditionalHandle` ABI;
Warp inserts a native while node and updates its condition from device code.

The same MuJoCo 3.8.1 humanoid workload is run through an eager HIP solver
loop and the native conditional graph. The primary run reports 767,338 versus
491,345 worlds/s (`1.562x`) with state errors below `1e-6`.

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

The implementation is split across the pinned source snapshots and runtime
patches:

- `hipamd/src/hip_graph.cpp` implements conditional handle allocation,
  conditional node creation, body-graph attachment, and node parameter updates.
- `include/hip/hip_runtime_api.h` and `include/hip/hip_runtime.h` expose the
  experimental handle, node type, parameter block, and device setter.
- `warp/_src/context.py` exposes an explicit HIP native branch for
  `capture_while`, with eager and legacy compatibility modes remaining opt-in.
- `mujoco_warp/_src/io.py` enables the mode only when explicitly requested and
  allocates solver scratch outside the capture window.
- `mujoco_warp/_src/solver.py` resets device counters with capture-safe work,
  passes the MuJoCo iteration cap, and gates solver work using `ctx.done`.
- `mujoco_warp/_src/forward.py` keeps the capture path on one stream when
  required by the HIP runtime.
- `mujoco_warp/_src/cli.py` performs warm-up before capture and replays the
  resulting graph for steady-state measurement.
- `mujoco_warp/_src/collision_driver.py` maintains a device-side geometry
  snapshot, computes a wake predicate, and routes cached broadphase work
  through an opt-in native `capture_if` branch.

The native mode is selected with:

```bash
WP_HIP_CONDITIONAL_NATIVE=1
WP_HIP_CONDITIONAL_EMULATION=0
```

The runtime is experimental and must be built from the matching HIP/CLR and
HIP SDK patches. An unmodified ROCm installation stays on the eager path.

## Benchmark protocol

The fixed workload is the pinned humanoid MJCF with 1024 worlds and 1000
steps, `nconmax=128`, `njmax=128`, identical initial state, and one HIP stream.
The graph build and JIT phase are reported separately. Both variants must
finish with exit code 0 and converge all worlds.

Primary artifact: `results/mjwarp_native_conditional_amd395/summary.json`.

| Variant | Runtime (s) | Throughput (worlds/s) |
| --- | ---: | ---: |
| Eager HIP solver loop | 2.084074 | 491,345 |
| Native HIP conditional handle | 1.334484 | 767,338 |

The native path reduces runtime by `35.97%`. A paired 256-world, 100-step
state comparison reports:

```text
max_abs_qpos_error = 6.17932528e-7
max_abs_qvel_error = 9.53674316e-7
numerically_equivalent = true
```

The target telemetry snapshot records 45-47 C edge temperature, 100% GPU busy,
and 76% VRAM allocation. A long-lived policy service shared the GPU, so the
telemetry is labeled shared-host telemetry.

## Workload boundary

The earlier ALOHA pot result was measured before the sleeping/wakeup adapter was
implemented: 64 worlds and 500 steps, with native execution at 8,937 worlds/s
versus 19,117 worlds/s for eager execution. It remains a historical workload
boundary, not a measurement of the new three-way sleeping benchmark.

The new benchmark is defined by
`scripts/run_aloha_sleeping_if_benchmark.sh`. It compares eager rebuild,
static graph reuse, and native `capture_if` selection under identical control
stimuli, and writes the result only after the ALOHA assets and patched runtime
are available. The dynamic fixture is the physical integration check. A
separate `freeze_after_warmup` fixture measures graph-path overhead after the
state has been prepared, without presenting that fixture as a physical task
score.

The current AMD395 dynamic run converged all 32 worlds in all variants. The
native branch observed a wake predicate of 1.00 and therefore rebuilt the
broadphase on every timed step. Throughput was 1,278.16 worlds/s for eager,
1,332.28 for static, and 1,145.19 for native. This result confirms the
device-side predicate and conservative rebuild path; it does not claim a
sleeping speedup for a scene that has not reached a stable interval.

We also ran a manual predicate coverage fixture on AMD395. With 16 worlds and
128 timed steps, the predicate was false for 87.5% of timed steps; all worlds
converged, and native throughput was 1.015x eager and 1.008x static. This
artifact confirms both conditional branches execute in the patched runtime,
while remaining separate from the physical ALOHA result.

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
HIP_CLR_LIB=/path/to/patched/libamdhip64.so \
  ROOT="$ROOT" ENV="$ENV" HIP_CLR_LIB="$HIP_CLR_LIB" \
  bash "$ROOT/scripts/run_native_mjwarp_benchmark.sh"
```

The script emits paired logs, the 256-world correctness log, a combined JSON
summary, and `SHA256SUMS`. `results/mujoco_warp_import_rocm721_py312.json`
records the successful full-package import in the 3.8.1 environment.

## Follow-up boundary

The minimal native `capture_while` handle and the full MuJoCo 3.8.1 AMD395
throughput comparison are complete. The native sleeping broadphase `capture_if`
adapter is implemented as an independent P1 path. Its ALOHA result is kept
separate from the P0 claim. A future sleeping speedup requires a task fixture
that reaches a genuine stable interval and reports a lower wake fraction.
