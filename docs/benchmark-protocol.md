# AMD395 Benchmark Protocol

## Fixed configuration

- Target: AMD Ryzen AI Max+ 395, Radeon 8060S, `gfx1151`, ROCm 7.2.1.
- Environment: `/home/aup/envs/mujoco-warp-amd-py312`.
- MuJoCo: 3.8.1; MuJoCo-Warp: 3.8.1; Warp: `1.13.0+rocm.0`.
- MJCF: `upstream/mujoco_warp/benchmarks/humanoid/humanoid.xml`.
- Workload: 1024 worlds, 1000 steps, `nconmax=128`, `njmax=128`.
- Same initial state, controls, timestep, solver, tolerances, and iteration cap.
- `MJW_HIP_SINGLE_STREAM=1` for a stable, comparable HIP execution path.
- Warm-up and graph-build time are reported separately from steady-state time.

## Variants

### Eager HIP solver loop

`WP_HIP_CONDITIONAL_NATIVE=0` and `WP_HIP_CONDITIONAL_EMULATION=0` keep the
solver in its regular loop around HIP iteration kernels. This is the baseline.

### Native HIP conditional handle

`WP_HIP_CONDITIONAL_NATIVE=1` and `WP_HIP_CONDITIONAL_EMULATION=0` use the
patched HIP/CLR runtime. The solver creates a device-resident conditional
handle, attaches the captured body graph to a while node, and updates the
condition from device code. There is no host convergence copy during replay.

### Legacy fixed-unroll compatibility path

`WP_HIP_CONDITIONAL_EMULATION=1` captures a fixed upper bound and gates solver
kernels from device state. It remains useful for historical comparison but is
not the native result reported below.

## Correctness gates

1. Both variants complete with exit code 0.
2. `converged_worlds` equals the requested world count.
3. No NaN or Inf is present in the state or solver output.
4. A paired 256-world, 100-step run compares `qpos` and `qvel` against the
   eager reference with tolerance `1e-5`.
5. The exact environment, source revisions, command, logs, and SHA256 values
   are stored with the result.

The recorded state errors are:

```text
max_abs_qpos_error = 6.17932528e-7
max_abs_qvel_error = 9.53674316e-7
numerically_equivalent = true
```

## Primary native result

Artifact: `results/mjwarp_native_conditional_amd395/summary.json`.

| Variant | Runtime | Worlds/s |
| --- | ---: | ---: |
| Eager HIP solver loop | 2.084074 s | 491,345 |
| Native HIP conditional handle | 1.334484 s | 767,338 |

The native path is `1.562x` faster and reduces runtime by `35.97%` in this
fixed workload. The paired state errors are `6.18e-7` for `qpos`, `9.54e-7`
for `qvel`, and `0` for simulation time.

The larger ALOHA side result is intentionally separate: the native prototype
currently reaches 8,937 worlds/s versus 19,117 worlds/s for eager execution.
It identifies graph-build and replay overhead that must be reduced before
claiming a universal speedup.

The later files named `*_revalidation.log` are retained as a shared-GPU
repeat. They completed correctly but were slower because a long-lived policy
service was using the same Radeon. They must not be averaged into the frozen
primary result.

## Device telemetry

The primary run recorded 45-47 C edge temperature, 100% reported GPU busy,
and 76% VRAM allocation. A policy service was present on the shared target;
telemetry is therefore labeled shared-host telemetry.

## Reproduction command

```bash
ROOT=/path/to/mujoco-warp-amd-conditional-graph
ENV=/home/aup/envs/mujoco-warp-amd-py312
HIP_CLR_LIB=/path/to/patched/libamdhip64.so \
ROOT="$ROOT" ENV="$ENV" \
  bash "$ROOT/scripts/run_native_mjwarp_benchmark.sh"
```

The script runs both native and eager variants, a paired state comparison, and
emits `SHA256SUMS`. `scripts/run_mjwarp_amd_benchmark.sh` is a compatibility
alias for the same native benchmark.
