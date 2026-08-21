# MuJoCo-Warp Conditional Graphs on AMD ROCm

[![Validate public bundle](https://github.com/Ethan-Chen-plus/mujoco-warp-amd-conditional-graph/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/Ethan-Chen-plus/mujoco-warp-amd-conditional-graph/actions/workflows/validate.yml)

An AMD-first, reproducible porting study for the MuJoCo-Warp Newton constraint
solver. The repository contains the modified Warp and MuJoCo-Warp source
snapshots, the HIP graph microbenchmark, the AMD395 benchmark protocol, frozen
results, and the patch set needed for upstream discussion.

The project answers one focused question:

> Can the solver's convergence loop move from host-dispatched HIP kernels into
> one replayable graph while preserving the physical state and improving
> throughput on AMD ROCm?

## Result at a glance

The target is an AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`) running
ROCm 7.2.1. On the fixed humanoid contact workload, the device-gated graph
path reaches **673,563 worlds/s**, compared with **493,084 worlds/s** for the
eager HIP loop:

| Variant | Runtime | Throughput | State check |
| --- | ---: | ---: | --- |
| Eager HIP solver loop | 2.076724 s | 493,084 worlds/s | reference |
| Device-gated single HIP graph | 1.520274 s | 673,563 worlds/s | `qpos/qvel < 1e-6` |

This is a **1.366x speedup** and a **26.79% runtime reduction** for the
reported configuration. The frozen result is
[`results/mjwarp_humanoid_conditional_rocm721.json`](results/mjwarp_humanoid_conditional_rocm721.json).

## What is implemented

The P0 implementation targets the solver `capture_while` path:

1. The solver convergence counter remains in device memory.
2. The solver body is captured once into a fixed-length HIP graph.
3. Kernels gate completed worlds using device-resident convergence state.
4. Steady-state replay does not copy the convergence condition to the host.
5. A paired eager run and a numerical state check guard the performance claim.

Enable it explicitly:

```bash
export WP_HIP_CONDITIONAL_EMULATION=1
```

The name `EMULATION` is deliberate. ROCm 7.2.1 exposes ordinary HIP graph
capture but does not expose CUDA's `hipGraphConditionalHandle` equivalent.
This repository therefore provides a working device-gated compatibility path,
not a falsely labelled native conditional graph node. The native integration
boundary and the exact next steps are documented in
[`docs/native-conditional-node-plan.md`](docs/native-conditional-node-plan.md).

## Reproduce on AMD395

The source tree is self-contained. It does not rely on the original nested Git
metadata. A clean AMD host needs ROCm, `hipcc`, Python 3.12, MuJoCo 3.8.1 and
the source-built Warp backend. The target environment used for the frozen run
was `/home/aup/envs/mujoco-warp-amd-py312`.

```bash
git clone https://github.com/Ethan-Chen-plus/mujoco-warp-amd-conditional-graph.git
cd mujoco-warp-amd-conditional-graph

# Check the host, create or reuse an isolated environment, and install the
# source snapshots. Set ENV to a writable path on the target machine.
ENV=/home/aup/envs/mujoco-warp-amd-py312 \
  bash scripts/bootstrap_amd395_env.sh

# Run both variants, correctness, diagnostics, and SHA generation.
ENV=/home/aup/envs/mujoco-warp-amd-py312 \
  bash scripts/run_mjwarp_amd_benchmark.sh
```

The benchmark script accepts `WORLDS`, `STEPS`, `MODEL`, and `OUT` overrides.
For example, a quick smoke run is:

```bash
ENV=/home/aup/envs/mujoco-warp-amd-py312 \
WORLDS=64 STEPS=20 \
  bash scripts/run_mjwarp_amd_benchmark.sh
```

The standard result must use the fixed protocol in
[`docs/benchmark-protocol.md`](docs/benchmark-protocol.md), not the smoke
configuration.

## Source-level reuse

The public tree includes the modified source files under `upstream/` and the
same changes as standalone patches under `patches/`:

```text
upstream/warp/                     AMD Warp source snapshot
upstream/mujoco_warp/              MuJoCo-Warp source snapshot
patches/warp-hip-conditional.diff  Warp graph/runtime patch
patches/mujoco-warp-amd.diff       MuJoCo-Warp solver integration patch
hip/hip_graph_benchmark.cpp        HIP graph microbenchmark
scripts/                            setup, benchmark, probe, manifest tools
docs/                               call chain, protocol, report, porting plan
results/                            JSON evidence, logs, probes, SHA records
```

The pinned upstream bases are recorded in
[`results/source_manifest.json`](results/source_manifest.json):

- AMD Warp: `8ca65dd5f8a444785408ecaa956bac0d2c427d6f`
- MuJoCo-Warp: `9229bb9d1a698c9464df862a915b46899720338c`

To inspect the patch without reading the entire snapshot:

```bash
less patches/warp-hip-conditional.diff
less patches/mujoco-warp-amd.diff
```

## Evidence package

- [`docs/technical-report.md`](docs/technical-report.md): method, benchmark,
  correctness and runtime boundary.
- [`docs/source-callchain.md`](docs/source-callchain.md): call-site map from
  `io.py` through `solver.py` and `warp/_src/context.py`.
- [`docs/priority.md`](docs/priority.md): why solver `capture_while` is P0 and
  sleeping broadphase `capture_if` is P1.
- [`results/mujoco_warp_import_rocm721_py312.json`](results/mujoco_warp_import_rocm721_py312.json):
  MuJoCo 3.8.1 and MuJoCo-Warp 3.8.1 import probe.
- [`results/logs/`](results/logs/): primary benchmark and correctness logs,
  plus the separately recorded shared-GPU revalidation.
- [`results/SHA256SUMS`](results/SHA256SUMS): checksums for the public evidence
  bundle.

## Native conditional graph follow-up

The missing `hipGraphConditionalHandle` ABI is a runtime capability boundary,
not a Python dependency problem. The project-side call site is isolated so a
future ROCm implementation can replace the fixed-unroll backend without
changing the benchmark or solver contract. The follow-up plan covers:

- a HIP capability probe;
- a minimal conditional-while adapter;
- solver convergence and early-exit tests;
- contact-heavy Humanoid/G1 throughput comparison;
- a separate sleeping-broadphase `capture_if` experiment.

See [`docs/native-conditional-node-plan.md`](docs/native-conditional-node-plan.md)
for the proposed interface and acceptance gates.

## Scope and licenses

This repository is an engineering PoC for the AMD ROCm porting discussion. It
does not claim that ROCm 7.2.1 already contains a native conditional graph ABI.
The upstream source snapshots retain their original Apache-2.0 notices. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing or
building the bundled sources.
