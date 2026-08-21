# Native HIP Conditional-Node Porting Plan

## Purpose

This document gives collaborators a small, testable path from the current
device-gated compatibility implementation to a true HIP conditional graph
node. It intentionally avoids porting the whole Warp graph subsystem first.

## Current runtime boundary

The AMD395 target uses ROCm 7.2.1 and ordinary HIP graph capture. The installed
headers and runtime do not expose `hipGraphConditionalHandle` or an equivalent
conditional-node ABI. The current P0 path therefore captures a fixed upper
bound and lets solver kernels skip work after convergence. It is numerically
equivalent to the eager path for the reported workload and is already faster.

## Priority order

### P0: solver `capture_while`

This is the highest-value node because every contact-rich Newton solve can
exercise it. The first native adapter should provide four operations:

```text
create conditional handle
create conditional node
attach body graph
update device predicate / replay graph
```

The adapter must keep the convergence predicate on device and must not require
a host synchronization between solver iterations.

### P1: sleeping broadphase `capture_if`

When sleeping is enabled and no body wakes, MuJoCo-Warp can skip an incremental
broadphase pass. This should be evaluated on a deterministic ALOHA clutter or
stacked-object scene after P0. It needs a device wake predicate and a clear
separation between graph-captured kernels and BVH/radix-sort rebuilds.

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

The exact symbols must be gated by compile-time and runtime capability probes.
If the symbols are absent, the code must select the existing compatibility
path instead of failing at import time.

## Acceptance gates

1. `hipGraphConditionalHandle` is present in headers and resolves in the
   runtime library.
2. A standalone conditional-while microbenchmark passes on `gfx1151`.
3. A solver convergence test shows the same final `qpos`, `qvel`, and solver
   status as the eager path within `1e-5`.
4. A mixed-convergence workload demonstrates early exit, not merely fixed
   unrolling.
5. Humanoid or Unitree G1 contact throughput is measured with identical world
   count, solver settings, warm-up, and graph replay policy.
6. The result includes logs, runtime diagnostics, source revisions and SHA256.

## Measurement matrix

Run each variant under the same host policy:

| Variant | Purpose |
| --- | --- |
| eager HIP solver loop | baseline |
| fixed-unroll device-gated graph | current AMD compatibility path |
| native conditional graph | future HIP runtime path |

Report solver iteration distribution, graph build time, steady-state step time,
worlds/s, peak VRAM, and final state error. Do not average measurements from
different GPU contention conditions.

## What this repository already proves

The current fixed-unroll device-gated graph reaches 673,563 worlds/s versus
493,084 worlds/s for the eager loop on the frozen AMD395 humanoid run, with
`qpos` and `qvel` errors below `1e-6`. That result is the baseline for judging
the value of a native conditional node.
