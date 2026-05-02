# MD Benchmark 2026-04-07

## Purpose

This note records the first realistic MD timing checks for the new UMA MD pilot workflow.

## What was changed before benchmarking

- the old `uma_calc md` invocation was replaced by a direct Python driver
- implicit MD pre-relaxation was disabled
- Python stdout was made unbuffered so Slurm output becomes visible during execution
- `NVT` now receives Maxwell-Boltzmann velocity initialization before dynamics starts

## Cluster environment facts

- CPU node in use for the benchmark: `baifq-hpc141`
- available GPU node in the same partition:
  - `node3`
  - `gpu:4`
- current UMA environment:
  - `torch 2.8.0+cpu`
  - `torch.cuda.is_available() = False`

So the cluster has GPU hardware, but the current UMA offline environment cannot use it.

## CPU benchmark results

Representative structure:

- `gb_Sigma3_t3`

Benchmark settings:

- ensemble: `NVT`
- temperature: `600 K`
- timestep: `1.0 fs`
- threads: `8`
- `pre_relax = False`

### Benchmark A

- steps: `5`
- save interval: `1`
- walltime inside UMA: `207.01 s`

Approximate cost:

- about `41 s/step`

### Benchmark B

- steps: `20`
- save interval: `20`
- walltime inside UMA: `724.38 s`

Approximate cost:

- about `36 s/step`

## Extrapolated walltime

Using the `20`-step benchmark as the more realistic estimate:

- `5000` steps would cost about `724.38 * 250 = 181095 s`
- this is about `50.3 hours`
- in other words, about `2.1 days` for **one** `624`-atom GB structure

That is too slow for a serial `6`-structure pilot if every structure keeps the current `5000`-step target.

## Turbo-mode test

CPU `turbo` inference mode was also tested.

Result:

- failed almost immediately
- error:
  - `ValueError: No edges found in input system`

So CPU `turbo` is not a safe default for this structure at the moment.

## Practical conclusion

The current CPU-only MD route is usable, but only if the pilot scope is reduced.

Most realistic next choices:

1. keep `NVT 600 K`, but reduce to `1000` steps first
2. start from only `2` structures, not all `6`
3. only expand after checking whether Li motion is already visible on the shorter trajectory

## Recommended pilot order

1. `gb_Sigma3_t3`
2. `gb_Sigma3_t3_Li_vac_c1_s1`

If those two are informative, then add:

3. `gb_Sigma3_t1`
4. `gb_Sigma3_t1_Li_vac_c1_s1`
