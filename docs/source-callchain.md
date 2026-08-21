# MuJoCo-Warp Conditional-Graph Call Chain

This note maps the AMD395 implementation to the pinned source snapshots in
`upstream/`.

## P0 solver path

1. `mujoco_warp/_src/io.py` enables `graph_conditional` only when either the
   native backend reports support or an explicit HIP conditional flag is set.
2. `mujoco_warp/_src/solver.py::_solve` prepares the device-side
   `nsolving` counter and calls `wp.capture_while` with the MuJoCo iteration
   cap.
3. `warp/_src/context.py::capture_while` selects the HIP native branch when
   `WP_HIP_CONDITIONAL_NATIVE=1`. It inserts a conditional while node and
   captures the solver body graph without reading the condition on the host.
4. `solver.py::_solver_iteration` launches the existing linesearch, gradient,
   search, and `solve_done` kernels. Those kernels read `ctx.done`; completed
   worlds return early and decrement the device-side active-world counter only
   once.
5. `mujoco_warp/_src/cli.py` warms the solver before capture, then replays the
   captured graph for the steady-state benchmark.

The solver and graph share no host convergence check during replay. The native
path is a runtime conditional graph; the legacy compatibility path is a fixed
HIP graph with device-gated work.

## Runtime boundary

The stock HIP headers on the AMD395 target contain no conditional graph
symbols. The public runtime patch adds `hipGraphConditionalHandle`, the
conditional node parameter block, and the create/destroy/setter entry points.
The patched library is selected only with `WP_HIP_CONDITIONAL_NATIVE=1`; the
explicit emulation flag remains separate from native support and is reported in
every benchmark JSON.

## P1 sleeping broadphase

The pinned collision driver has broadphase cache logic, but no production
`capture_if` call site. A P1 port should make the wake/sleep predicate a
device value and keep the BVH/radix-sort rebuild outside the graph capture
window. No P1 result is mixed into the P0 benchmark.

## Source audit

Regenerate the call-site map with:

```bash
python3 scripts/inspect_mjwarp_source.py \
  --output results/source_callsite_map.json
```

The resulting JSON includes the source SHA256 values, line numbers, snippets,
and the native/emulation interpretation used by the report.
