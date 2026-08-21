# MuJoCo-Warp Conditional-Graph Call Chain

This note maps the AMD395 implementation to the pinned source snapshots in
`upstream/`.

## P0 solver path

1. `mujoco_warp/_src/io.py` enables `graph_conditional` only when either the
   native backend reports support or `WP_HIP_CONDITIONAL_EMULATION=1` is set.
2. `mujoco_warp/_src/solver.py::_solve` prepares the device-side
   `nsolving` counter and calls `wp.capture_while` with the MuJoCo iteration
   cap.
3. `warp/_src/context.py::capture_while` expands the solver body into one
   graph on HIP compatibility mode. It does not read the condition on the
   host.
4. `solver.py::_solver_iteration` launches the existing linesearch, gradient,
   search, and `solve_done` kernels. Those kernels read `ctx.done`; completed
   worlds return early and decrement the device-side active-world counter only
   once.
5. `mujoco_warp/_src/cli.py` warms the solver before capture, then replays the
   captured graph for the steady-state benchmark.

The solver and graph share no host convergence check during replay. The
compatibility path is thus a single HIP graph with device-gated work, rather
than a host-dispatched tier schedule.

## Runtime boundary

The HIP headers on the AMD395 target contain no `hipGraphConditional*` symbols.
`warp/_src/context.py::assert_conditional_graph_support` therefore keeps the
native capability false on HIP. The explicit emulation flag is separate from
native support and is reported in every benchmark JSON.

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
