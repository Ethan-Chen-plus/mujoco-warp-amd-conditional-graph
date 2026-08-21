# MuJoCo-Warp Conditional Graphs on AMD ROCm

[![Validate public bundle](https://github.com/Ethan-Chen-plus/mujoco-warp-amd-conditional-graph/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/Ethan-Chen-plus/mujoco-warp-amd-conditional-graph/actions/workflows/validate.yml)

An AMD-first, reproducible port of the MuJoCo-Warp Newton constraint solver.
The repository contains the modified Warp and MuJoCo-Warp source snapshots,
the HIP/CLR runtime patch that provides an experimental native conditional
handle, direct device tests, the AMD395 benchmark protocol, frozen results, and
the patch set needed for upstream discussion.

The project answers one focused question:

> Can the solver's convergence loop move from host-dispatched HIP kernels into
> one replayable graph while preserving the physical state and improving
> throughput on AMD ROCm?

## Result at a glance

The target is an AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`) running
ROCm 7.2.1. The native path inserts a `hipGraphConditionalHandle` while node
through the patched HIP/CLR runtime and executes the MuJoCo-Warp solver body
inside that conditional graph.

On the fixed 1024-world humanoid workload, the native path reaches
**767,338 worlds/s**, compared with **491,345 worlds/s** for the eager HIP
solver loop:

| Variant | Runtime | Throughput | State check |
| --- | ---: | ---: | --- |
| Eager HIP solver loop | 2.084074 s | 491,345 worlds/s | reference |
| Native HIP conditional handle | 1.334484 s | 767,338 worlds/s | `qpos/qvel/time < 1e-6` |

This is a **1.562x throughput speedup** and a **35.97% runtime reduction** for
this fixed workload. The primary native evidence is
[`results/mjwarp_native_conditional_amd395/summary.json`](results/mjwarp_native_conditional_amd395/summary.json).

The result is workload-specific. On the larger ALOHA graph, the current
prototype reaches 8,937 worlds/s versus 19,117 worlds/s for eager execution;
that side result is retained in
[`results/mjwarp_native_aloha_amd395.json`](results/mjwarp_native_aloha_amd395.json)
as a boundary for the next optimization pass.

## What is implemented

The P0 native path targets the solver `capture_while` path:

1. HIP headers expose the experimental conditional handle, node type, and
   parameter block behind `HIP_GRAPH_CONDITIONAL_EXT`.
2. HIP/CLR allocates and tracks a device-resident handle, creates a while node,
   attaches the captured body graph, and supports graph execution.
3. A device setter uses an atomic write to update the handle without a host
   synchronization.
4. Warp's HIP native branch inserts the conditional node and compiles the
   helper setter kernel for the active `gfx1151` target.
5. MuJoCo-Warp captures and replays the solver through the native branch; the
   eager path remains available for comparison.

Enable the patched runtime path explicitly:

```bash
export WP_HIP_CONDITIONAL_NATIVE=1
export WP_HIP_CONDITIONAL_EMULATION=0
```

`WP_HIP_CONDITIONAL_EMULATION=1` is the older fixed-unroll compatibility path.
It is retained for source comparison but is not the primary native result. A
stock ROCm installation continues to use the eager path because it does not
contain these experimental ABI symbols.

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

# Build and run the native/eager comparison, correctness check, and SHA output.
ENV=/home/aup/envs/mujoco-warp-amd-py312 \
  bash scripts/run_native_mjwarp_benchmark.sh
```

The benchmark script accepts `WORLDS`, `STEPS`, `MODEL`, and `OUT` overrides.
For example, a quick smoke run is:

```bash
ENV=/home/aup/envs/mujoco-warp-amd-py312 \
WORLDS=64 STEPS=20 \
  bash scripts/run_native_mjwarp_benchmark.sh
```

The standard result must use the fixed protocol in
[`docs/benchmark-protocol.md`](docs/benchmark-protocol.md), not the smoke
configuration.

To build the matching runtime from source, apply the two runtime patches and
build `amdhip64` before rebuilding Warp:

```bash
HIP_CLR_SRC=/path/to/hipclr \
HIP_SDK_SRC=/path/to/hip \
ROCM_PATH=/opt/rocm \
  bash scripts/build_patched_hip_runtime.sh
```

The runtime patch is opt-in and must be paired with the public HIP headers and
the rebuilt Warp extension. The script does not modify the system ROCm tree.

Build and run the standalone handle checks against that runtime:

```bash
HIPCC=/opt/rocm/bin/hipcc \
HIP_SDK_SRC=/path/to/hip \
HIP_CLR_LIB=/path/to/libamdhip64.so \
GPU_ARCH=gfx1151 \
  bash scripts/build_hip_benchmark.sh

export LD_LIBRARY_PATH="$(dirname /path/to/libamdhip64.so):/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
./build/conditional_while_device
./build/conditional_while_benchmark
```

The first executable checks device-side iteration and the second compares the
conditional graph with a fixed graph. The benchmark reports the control-node
overhead separately from the MuJoCo-Warp result, so the solver speedup is only
meaningful when it amortizes this fixed cost.

Rebuild the Warp extension against the patched SDK with:

```bash
export HIP_PATH=/path/to/hip
export WP_ENABLE_HIP_CONDITIONAL_EXT=1
ROCM_PATH=/opt/rocm "$ENV/bin/python" upstream/warp/build_lib.py \
  --no-cuda --hip-arch=gfx1151 --rocm-path=/opt/rocm
```

## Source-level reuse

The public tree includes the modified source files under `upstream/` and the
same changes as standalone patches under `patches/`:

```text
upstream/warp/                     AMD Warp source snapshot
upstream/mujoco_warp/              MuJoCo-Warp source snapshot
patches/warp-hip-conditional.diff  Warp graph/runtime patch
patches/mujoco-warp-amd.diff       MuJoCo-Warp solver integration patch
patches/hip-clr-conditional.diff    HIP/CLR conditional-node runtime patch
patches/hip-sdk-conditional.diff    HIP SDK headers and device setter patch
hip/hip_graph_benchmark.cpp        HIP graph microbenchmark
hip/conditional_while_device.cpp   direct native handle functional test
hip/conditional_while_benchmark.cpp native handle overhead test
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
less patches/hip-clr-conditional.diff
less patches/hip-sdk-conditional.diff
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
- [`results/mjwarp_native_conditional_amd395/summary.json`](results/mjwarp_native_conditional_amd395/summary.json):
  native handle benchmark, state comparison, runtime and extension hashes.
- [`results/mjwarp_humanoid_conditional_rocm721.json`](results/mjwarp_humanoid_conditional_rocm721.json):
  retained fixed-unroll compatibility result for historical comparison.
- [`results/mjwarp_native_aloha_amd395.json`](results/mjwarp_native_aloha_amd395.json):
  larger-graph workload boundary.
- [`results/native_conditional_handle_benchmark_amd395.json`](results/native_conditional_handle_benchmark_amd395.json):
  direct conditional-node overhead measurement.
- [`results/logs/`](results/logs/): primary benchmark and correctness logs,
  plus the separately recorded shared-GPU revalidation.
- [`results/SHA256SUMS`](results/SHA256SUMS): checksums for the public evidence
  bundle.

## Next runtime milestones

The minimal native `capture_while` handle is implemented and verified by the
direct device test and the MuJoCo-Warp humanoid run. The next runtime milestone
is a native `capture_if` node for sleeping broadphase work, followed by mixed
convergence and wake/sleep workloads. The adapter keeps those extensions
isolated from the eager fallback and records a separate result for every
workload.

See [`docs/native-conditional-node-plan.md`](docs/native-conditional-node-plan.md)
for the proposed interface and acceptance gates.

## Scope and licenses

This repository is an engineering PoC for an experimental HIP/CLR conditional
graph extension on AMD ROCm 7.2.1. The extension is not part of an unmodified
ROCm installation. The upstream source snapshots retain their original
Apache-2.0 notices. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing or
building the bundled sources.
