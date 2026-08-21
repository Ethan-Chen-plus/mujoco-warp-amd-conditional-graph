# Patch Set

The two patches are generated against the pinned upstream commits recorded in
`results/source_manifest.json`.

```text
patches/warp-hip-conditional.diff
patches/mujoco-warp-amd.diff
```

The full modified source snapshots are also included under `upstream/` so a
collaborator can inspect and run the result without reconstructing a patch.

To inspect the scope:

```bash
git apply --stat patches/warp-hip-conditional.diff
git apply --stat patches/mujoco-warp-amd.diff
```

The patches do not change the native CUDA conditional-graph path. They add an
explicit HIP compatibility mode and the MuJoCo-Warp solver integration needed
to run it on AMD ROCm.
