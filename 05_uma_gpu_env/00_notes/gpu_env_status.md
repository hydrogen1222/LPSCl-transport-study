# UMA GPU Environment Status

## Purpose

This directory is a clean, isolated workspace for a future CUDA-capable UMA environment.

It is intentionally separate from:

- `02_uma_workflows`
- `03_followup`
- `04_md_pilot`

## Current finding on 2026-04-09

The cluster does have a GPU node:

- `node3`
- `gpu:4`

But GPU jobs are not launching successfully at the scheduler level right now.

Two minimal probe jobs were submitted to `node3`, and both immediately ended with:

- `JobState=CANCELLED`
- `Reason=launch_failed_requeued_held`

So the current blocker is not yet the Python/CUDA package stack. The first blocker is that
`node3` cannot successfully launch even a simple GPU probe job from the current user account.

## Verified environment facts

- cluster Python: `3.10.8`
- existing UMA offline env: `/home/ctan/uma-offline-env`
- existing Torch in that env: `2.8.0+cpu`
- `torch.cuda.is_available() = False`

This means the current UMA environment is CPU-only.

## Practical conclusion

Do not start a large CUDA wheel download/upload yet.

Because the GPU node cannot currently launch jobs, a multi-GB CUDA offline bundle would likely be wasted bandwidth.

## Recommended next step

1. Confirm with cluster admins or lab seniors whether `node3` is currently usable for normal GPU Slurm jobs.
2. Once a trivial `nvidia-smi` batch job can run successfully on `node3`, continue here:
   - build a CUDA-capable Linux wheel bundle
   - create a new env such as `/home/ctan/uma-gpu-env`
   - run a tiny UMA GPU smoke test

## Planned clean paths

- local workspace: `05_uma_gpu_env`
- planned cluster env: `/home/ctan/uma-gpu-env`
- planned cluster test root: `/home/ctan/LPSCl_UMA_transport_project/05_uma_gpu_env_tests`
