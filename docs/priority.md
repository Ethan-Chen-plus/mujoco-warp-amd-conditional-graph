# Conditional-Graph Priority

## P0: solver `capture_while`

MuJoCo-Warp uses `capture_while` around the Newton/constraint solver. The loop
tracks the worlds that have not converged and stops before the XML iteration
cap when all worlds are done. Contact-rich locomotion, humanoid, and
manipulation workloads all exercise this path.

The AMD395 PoC now provides the native runtime path:

1. Keep `nsolving` and solver convergence flags on the device.
2. Create a HIP conditional handle and attach the solver body graph to a while
   node.
3. Update the handle from device code as worlds converge.
4. Compare against the eager solver on the same initial state.

The 1024-world contact benchmark completes with exit code 0, matches the
eager state within `1e-5`, and records the primary throughput comparison in
`results/mjwarp_native_conditional_amd395/summary.json`.

## Native HIP boundary

Stock ROCm 7.2.1 does not expose `hipGraphConditionalHandle` or an equivalent
conditional graph-node API. This repository adds the API in a matching HIP SDK
patch and implements it in HIP/CLR. The native path is explicitly selected by
`WP_HIP_CONDITIONAL_NATIVE=1`; an unmodified ROCm installation remains on the
eager path. The older fixed-unroll path is separately selected by
`WP_HIP_CONDITIONAL_EMULATION=1`.

## P1: sleeping broadphase `capture_if`

The incremental broadphase is the next independent graph experiment. When no
body wakes, a second broadphase pass can be skipped. This is useful for
clutter, stacked objects, and long sleeping periods, but it is not required by
the P0 solver deliverable. A future P1 experiment should:

- use a deterministic ALOHA clutter or stacked-object scene;
- keep BVH and radix-sort rebuilds outside the captured region;
- put the wake predicate on device;
- measure skip rate, physics state error, and end-to-end throughput.

## Deferred: `capture_switch`

No main MuJoCo-Warp path in the pinned source depends on a switch node, so it is
not a first implementation target.
