# AMD395 Execution Status

The reported GPU measurements come from the LAN AMD Ryzen AI Max+ 395, not
from the local RTX PRO 6000 controller host.

## Target access

- GPU: Radeon 8060S, `gfx1151`.
- ROCm: 7.2.1.
- `/dev/kfd`: readable and writable by the target account.
- Render node: `/dev/dri/renderD128`.
- Warp device: `cuda:0` (Warp's HIP backend alias).

## Environment alignment

The isolated MicroMamba environment is:

`/home/aup/envs/mujoco-warp-amd-py312`

The environment imports MuJoCo 3.8.1 and MuJoCo-Warp 3.8.1 successfully;
see `mujoco_warp_import_rocm721_py312.json`. The original JAX environment
remains on MuJoCo 3.4.0 and was not modified.

## Evidence

- `warp_gpu_smoke_rocm721.json`: real HIP kernel smoke, zero error.
- `mjwarp_humanoid_conditional_rocm721.json`: paired 1024-world, 1000-step
  physics throughput result.
- `mjwarp_conditional_correctness_256x100.log`: paired state check.
- `mujoco_warp_import_rocm721_py312.json`: successful full-package import.
- `source_manifest.json`: source, script, executable, and result hashes.

The primary run completed with exit code 0. A later shared-GPU repeat is stored
under the `*_revalidation.log` names and is intentionally separate from the
primary result.
