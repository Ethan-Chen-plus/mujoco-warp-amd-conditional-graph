# Patch Set

The patches are generated against the pinned upstream commits recorded in
`results/source_manifest.json`.

```text
patches/warp-hip-conditional.diff
patches/mujoco-warp-amd.diff
patches/hip-clr-conditional.diff
patches/hip-sdk-conditional.diff
```

The full modified source snapshots are also included under `upstream/` so a
collaborator can inspect and run the result without reconstructing a patch.

To inspect the scope:

```bash
git apply --stat patches/warp-hip-conditional.diff
git apply --stat patches/mujoco-warp-amd.diff
git apply --stat patches/hip-clr-conditional.diff
git apply --stat patches/hip-sdk-conditional.diff
```

The HIP SDK patch adds the public handle and node types. The HIP/CLR patch
implements handle allocation, node creation, body-graph attachment, parameter
updates, device-side state access, and the capture-safe graph allocation path.
The Warp and MuJoCo-Warp patches connect that runtime API to the solver's
`capture_while` call site. They do not change the native CUDA path.
