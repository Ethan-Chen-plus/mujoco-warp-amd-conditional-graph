# Upstream Evidence

The AMD porting baseline is pinned to these public sources:

- AMD-Ecosystem/mujoco_warp `amd-integration`:
  https://github.com/AMD-Ecosystem/mujoco_warp/tree/amd-integration
- AMD-Ecosystem/mujoco_warp PR #4:
  https://github.com/AMD-Ecosystem/mujoco_warp/pull/4
- AMD-Ecosystem/warp HIP graph support PR #15:
  https://github.com/AMD-Ecosystem/warp/pull/15
- ROCm/HIP conditional graph feature request #3905:
  https://github.com/ROCm/hip/issues/3905

PR #4 is the practical AMD fallback: it adds early-exit convergence checks,
pre-created streams, scratch reuse, BVH caching, and adaptive graph dispatch.
The implementation avoids claiming a HIP conditional node. PR #15 enables
ordinary HIP graph capture but keeps conditional capture self-skipping because
the HIP API is not available. Issue #3905 tracks the missing
`hipGraphConditionalHandle` capability.

The local benchmark therefore has two separate milestones:

1. HIP graph capture plus host-adaptive tier dispatch.
2. Native device-side conditional nodes after HIP exposes the required API.

The local snapshot used for this PoC is recorded in
`results/source_manifest.json`.
