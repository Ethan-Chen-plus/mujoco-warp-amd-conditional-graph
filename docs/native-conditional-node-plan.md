# Native HIP Conditional-Node Porting Plan

## Purpose

This document gives collaborators the smallest complete HIP conditional-graph
port: a device-resident handle, a `capture_while` node, a captured body graph,
and a MuJoCo-Warp solver integration. It intentionally avoids porting the whole
Warp graph subsystem first.

## Current runtime boundary

The AMD395 target uses ROCm 7.2.1. The stock headers and runtime do not expose
the required conditional-node ABI, so this repository carries a matching
experimental HIP SDK and HIP/CLR patch. The patched runtime exports the
CUDA-shaped conditional handle API, stores the handle state on device, and
executes the while body through the runtime conditional node. The stock runtime
remains supported through the eager path.

## Priority order

### P0: solver `capture_while` — complete

This is the highest-value node because every contact-rich Newton solve can
exercise it. The native adapter provides four operations:

```text
create conditional handle
create conditional node
attach body graph
update device predicate / replay graph
```

The adapter keeps the convergence predicate on device and does not require a
host synchronization between solver iterations. A direct device test reaches
the expected counter value, and the MuJoCo-Warp humanoid benchmark exercises
the same node from the solver graph.

### P1: sleeping broadphase `capture_if` — experimental adapter implemented

When sleeping is enabled and no body wakes, MuJoCo-Warp can skip an incremental
broadphase pass. The implementation uses a device-side geometry position and
orientation snapshot, an atomic wake predicate, and an opt-in `capture_if`
branch. The true branch rebuilds the broadphase; the false branch reuses the
cached collision context. The current ALOHA dynamic fixture wakes on every
timed step, which is the correct conservative outcome for an unsettled state.
The adapter is complete as an integration path, while a positive sleeping
speedup remains a separate workload-design and validation task. BVH/radix-sort
work remains outside the captured branch contract, so a later backend can
replace the brute-force rebuild without changing the adapter interface.

The feature is enabled only when both flags are set:

```bash
export MJW_HIP_SLEEPING_CAPTURE_IF=1
export WP_HIP_CONDITIONAL_NATIVE=1
```

The default remains the existing eager/cache behavior. Set
`MJW_HIP_BVH_CACHE=0` to create an eager rebuild baseline for the ALOHA
benchmark.

### Deferred: `capture_switch`

The pinned MuJoCo-Warp path does not depend on a switch node. It should not be
implemented before the solver loop and sleeping broadphase have evidence.

## Proposed adapter contract

Keep the public Warp-side interface stable and isolate HIP details in one
runtime adapter. A minimal implementation can expose:

```cpp
struct HipConditionalWhile {
  hipGraphConditionalHandle handle;
  hipGraph_t body_graph;
  hipGraphNode_t conditional_node;
};

HipConditionalWhile create_while(
    hipGraph_t parent,
    hipStream_t stream,
    hipDeviceptr_t predicate,
    int max_iterations);

void set_predicate(hipDeviceptr_t predicate);
void destroy(HipConditionalWhile* node);
```

The symbols are gated by `HIP_GRAPH_CONDITIONAL_EXT` at compile time and by
`WP_HIP_CONDITIONAL_NATIVE=1` at runtime. If the symbols are absent, Warp
selects the eager path instead of failing at import time. The older
`WP_HIP_CONDITIONAL_EMULATION=1` backend remains separate and is not used for
the native headline result.

## Acceptance gates

1. `hipGraphConditionalHandle` is present in headers and resolves in the
   runtime library.
2. A standalone conditional-while device test passes on `gfx1151`.
3. A solver convergence test shows the same final `qpos`, `qvel`, and solver
   status as the eager path within `1e-5`.
4. The ALOHA benchmark runs eager, static, and native variants with the same
   control stimulus and records wake/sleep collision statistics. A wake
   fraction near one is a valid conservative integration result, not a speedup.
5. Humanoid or Unitree G1 contact throughput is measured with identical world
   count, solver settings, warm-up, and graph replay policy.
6. The result includes logs, runtime diagnostics, source revisions and SHA256.
7. A positive sleeping speedup is accepted only when the task reaches a stable
   interval and the device wake fraction is reported.

## Measurement matrix

Run each variant under the same host policy:

| Variant | Purpose |
| --- | --- |
| eager HIP solver loop | baseline |
| fixed-unroll device-gated graph | legacy compatibility path |
| native conditional graph | patched HIP/CLR runtime path |

Report solver iteration distribution, graph build time, steady-state step time,
worlds/s, peak VRAM, and final state error. Do not average measurements from
different GPU contention conditions.

## What this repository already proves

The patched native handle reaches 767,338 worlds/s versus 491,345 worlds/s for
the eager loop on the fixed AMD395 humanoid run. The throughput ratio is 1.562x,
runtime reduction is 35.97%, and the maximum paired state error is below
`1e-6`. The direct handle overhead is 0.095 ms versus 0.022 ms for a static
graph, so the optimization is valuable when it removes enough host dispatch or
solver work to amortize this fixed cost. The ALOHA result is reported as a
separate wake/sleep measurement with its own checksum and is not folded into
the humanoid headline.
