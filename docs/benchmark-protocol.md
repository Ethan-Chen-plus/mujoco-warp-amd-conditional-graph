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

`WP_HIP_CONDITIONAL_EMULATION=0` keeps the solver in its regular Python loop
around HIP iteration kernels. This is the baseline.

### Device-gated single HIP graph

`WP_HIP_CONDITIONAL_EMULATION=1` captures one fixed-length graph. Each solver
iteration reads device-resident convergence state and completed worlds skip
subsequent work. The graph has no host convergence copy during replay.

This is a project-level HIP compatibility implementation, not a native
`hipGraphConditionalHandle` node. ROCm 7.2.1 headers and runtime expose
ordinary graph capture but no conditional graph node ABI.

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
max_abs_qpos_error = 7.86967576e-7
max_abs_qvel_error = 9.53674316e-7
numerically_equivalent = true
```

## Primary result

Artifact: `results/mjwarp_humanoid_conditional_rocm721.json`.

| Variant | Runtime | Worlds/s |
| --- | ---: | ---: |
| Eager HIP solver loop | 2.076724 s | 493,084 |
| Device-gated single HIP graph | 1.520274 s | 673,563 |

The device-gated path is `1.366x` faster and reduces runtime by `26.79%` in
this fixed contact workload.

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
PYTHONPATH="$ROOT/upstream/warp:$ROOT/upstream/mujoco_warp" \
LD_LIBRARY_PATH="$ROOT/upstream/warp/warp/bin:/opt/rocm/lib:/opt/rocm/lib64" \
MUJOCO_GL=egl MJW_HIP_SINGLE_STREAM=1 WP_HIP_GRAPH_ENABLE=0 \
WP_HIP_CONDITIONAL_EMULATION=1 \
"$ENV/bin/python" \
"$ROOT/upstream/mujoco_warp/mujoco_warp/testspeed.py" \
"$ROOT/upstream/mujoco_warp/benchmarks/humanoid/humanoid.xml" \
--function=step --nworld=1024 --nstep=1000 --nconmax=128 --njmax=128 \
--device=cuda:0 --format=json
```

For both variants, use `scripts/run_mjwarp_amd_benchmark.sh`; it also runs the
correctness check and emits `SHA256SUMS`.
