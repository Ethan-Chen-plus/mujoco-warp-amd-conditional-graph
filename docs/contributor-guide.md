# Contributor Guide

## Fast orientation

Start with these files in order:

1. `README.md` for the result and one-command benchmark.
2. `docs/source-callchain.md` for the solver call chain.
3. `docs/technical-report.md` for the benchmark definition.
4. `docs/native-conditional-node-plan.md` for the conditional-node interface
   and acceptance gates.
5. `patches/` for the minimal source diff against the pinned upstream bases.

## Two implementation layers

The repository deliberately separates:

- **Warp layer**: graph capability detection and `capture_while` behavior in
  `upstream/warp/warp/_src/context.py`.
- **MuJoCo-Warp layer**: graph enablement, solver scratch allocation, device
  convergence state, warm-up and replay in `upstream/mujoco_warp/`.

When debugging a new ROCm version, test the Warp layer first. A failure in the
HIP graph capability probe should not be diagnosed as a MuJoCo solver error.

## Revalidation checklist

```bash
python3 -m py_compile scripts/*.py
bash -n scripts/*.sh
python3 scripts/check_mjwarp_import.py --help
python3 scripts/inspect_mjwarp_source.py \
  --output /tmp/source_callsite_map.json

python3 scripts/summarize_aloha_sleeping_if.py \
  --input results/aloha_sleeping_if_amd395_revalidation \
  --output /tmp/aloha_summary.json
```

On AMD, run the capability probe before the physical benchmark. Then compare
the eager and device-gated variants under the same GPU load and preserve the
generated logs.

For the ALOHA path, keep `WAKE_MODE=motion` for physical revalidation. Use
`FIXTURE=freeze_after_warmup` only as a separate branch-overhead measurement;
the result must retain its fixture name and wake fraction.

## Changing the solver path

Keep these invariants:

- the eager path remains available with
  `WP_HIP_CONDITIONAL_EMULATION=0`;
- compatibility mode is explicit and never silently enabled;
- convergence state remains device-resident during graph replay;
- a correctness artifact is generated for every performance result;
- native conditional support is reported separately from emulation.

## Reporting a new result

Add the following to the result directory:

- exact host and software versions;
- model path and workload dimensions;
- warm-up and steady-state timing;
- GPU contention policy;
- final-state error;
- process exit status;
- source and result SHA256 values.

For the sleeping broadphase result, also record `stimulus`, `wake_mode`,
`fixture`, `wake_predicate_fraction`, collision statistics, and solver
iteration statistics.

Do not replace the frozen primary result in place. Add a dated or explicitly
named revalidation artifact so comparisons remain auditable.
