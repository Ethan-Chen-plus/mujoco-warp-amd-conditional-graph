# Repository Map

## Runtime source

`upstream/warp/warp/_src/context.py` contains the HIP capability gate and the
device-gated `capture_while` implementation. `upstream/warp/warp/__init__.py`
exports the compatibility-mode query.

`upstream/mujoco_warp/mujoco_warp/_src/io.py` selects the graph mode and
allocates capture-safe scratch state. `solver.py` owns the device convergence
counter and passes the MuJoCo iteration cap. `forward.py` and `cli.py` keep the
capture and replay sequence on the stream policy required by the HIP runtime.

## Experiment source

- `hip/hip_graph_benchmark.cpp`: standalone HIP graph microbenchmark.
- `scripts/run_mjwarp_amd_benchmark.sh`: paired physical benchmark.
- `scripts/inspect_mjwarp_state.py`: eager-versus-device-gated state check.
- `scripts/collect_mjwarp_benchmark.py`: parses logs and writes the summary.
- `scripts/run_capability_probe.py`: records runtime graph capabilities.
- `scripts/write_manifest.py`: hashes source and evidence files.

## Evidence flow

```text
AMD395 host
  -> bootstrap_amd395_env.sh
  -> source-built Warp + MuJoCo-Warp 3.8.1
  -> run_mjwarp_amd_benchmark.sh
  -> baseline.log / device_gated.log / correctness.log
  -> summary.json + SHA256SUMS
```

The frozen primary artifact is kept separate from shared-GPU revalidation so a
later run cannot silently change the headline comparison.
