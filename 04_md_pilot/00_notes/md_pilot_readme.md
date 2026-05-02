# MD Pilot Workflow

This root is a clean MD-only workflow and does not modify:

- `02_uma_workflows`
- `03_followup`

## Root paths

- workflow root: `04_md_pilot/01_md_workflows`
- notes: `04_md_pilot/00_notes`

## Purpose

- launch a small formal UMA MD pilot set for transport-related analysis
- keep the completed relaxation and single-point roots read-only
- compare neutral GB controls against selected vacancy-GB structures

## Selected structures

- neutral controls:
  - `gb_Sigma3_t3`
  - `gb_Sigma3_t1`
- vacancy-GB core cases:
  - `gb_Sigma3_t3_Li_vac_c1_s1`
  - `gb_Sigma3_t3_Li_vac_c2_s2`
- vacancy-GB contrast cases:
  - `gb_Sigma3_t1_Li_vac_c1_s1`
  - `gb_Sigma3_t2_Li_vac_c1_s2`

## MD settings

- code: UMA CPU inference
- ensemble: `NVT`
- temperature: `600 K`
- time step: `1.0 fs`
- number of steps: `5000`
- total simulated time per structure: `5 ps`
- save interval: `20`
- serial execution through the existing workflow driver
- current launcher uses a direct Python driver with:
  - `pre_relax = False`
  - unbuffered stdout
  - explicit Maxwell-Boltzmann velocity initialization for `NVT`

## Geometry source

- neutral structures are seeded from the finished production root `02_uma_workflows`
- vacancy structures are seeded from the finished follow-up root `03_followup/02_uma_continue_workflows`
- every selected structure in this new root keeps a local copy of:
  - `00_uma_relax/POSCAR`
  - `00_uma_relax/CONTCAR`

## Initial execution order

1. `gb_Sigma3_t3`
2. `gb_Sigma3_t1`
3. `gb_Sigma3_t3_Li_vac_c1_s1`
4. `gb_Sigma3_t3_Li_vac_c2_s2`
5. `gb_Sigma3_t1_Li_vac_c1_s1`
6. `gb_Sigma3_t2_Li_vac_c1_s2`

## Expected outputs

For each structure, the `02_uma_md_nvt` directory should contain:

- `POSCAR`
- MD trajectory and logs copied back from the UMA runner output directory
- `MD_DONE`

## Next decision point

After the first one or two structures finish, check:

- whether the MD is numerically stable
- whether Li motion is visible on the chosen time scale
- whether `600 K` and `5 ps` are enough for the first transport figures

## 2026-04-07 benchmark update

- the previous `uma_calc md` route was too opaque for debugging because:
  - it performed an implicit pre-relaxation step first
  - stdout was buffered, so Slurm logs looked empty for a long time
- the current pilot root now uses a direct MD runner instead
- CPU benchmark on `gb_Sigma3_t3`, `8` threads, `pre_relax = False`:
  - `5` steps, `save_interval = 1`: `207.01 s`
  - `20` steps, `save_interval = 20`: `724.38 s`
- rough walltime estimate from the `20`-step benchmark:
  - about `36 s/step`
  - about `50 hours` for a full `5000`-step run on one `624`-atom GB structure
- CPU `turbo` inference mode was tested and failed on this structure with:
  - `ValueError: No edges found in input system`
- cluster hardware note:
  - `node3` has `gpu:4`
  - but the current UMA offline environment is `torch 2.8.0+cpu`
  - therefore the present workflow cannot use CUDA yet
