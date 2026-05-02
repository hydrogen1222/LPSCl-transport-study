# LPSCl Transport Workflow

`done.cif` remains the single reference parent structure. The production route is now ML-first and aligned with the thesis goal: use machine learning to explore how defects and grain boundaries affect transport-related behavior in `Li6PS5Cl`.

## Workflow roots
- Clean transport production workflow: `02_uma_workflows`
- Legacy DFT benchmark data are kept outside this clean project

## Production stages in `02_uma_workflows`
1. `00_uma_relax`
   - UMA CPU geometry pre-optimization from `POSCAR`
   - default optimizer: `FIRE`
   - `fmax = 0.08`
   - `max_steps = 200`
   - CPU layout: `1 MPI + 8 OpenMP`
   - successful runs produce `CONTCAR`
2. `01_cp2k_singlepoint`
   - fixed-cell CP2K high-accuracy single-point calculation
   - `RUN_TYPE ENERGY_FORCE`
   - `PBE + TZVP-MOLOPT-PBE-GTH + GTH-PBE`
   - coordinates are injected from `../00_uma_relax/CONTCAR`
   - if UMA fails or the geometry sanity check fails, the workflow falls back to the original `POSCAR`
3. `02_uma_md_nvt` (optional, disabled by default)
   - UMA NVT MD production stage for later transport analysis
   - default temperature: `600 K`
   - default timestep: `1.0 fs`
   - default steps: `5000`
   - enabled only when the serial script is called with `--include-md`

## Research split
- bulk structures are no longer part of the default serial production workflow
- bulk DFT benchmark data are managed outside this clean project
- the new serial workflow only covers `gb_*` structures

## Shared settings
- reduced GB model uses `GB_C_REPEATS = 1`
- representative reduced GB size: `624` atoms for neutral `gb_Sigma3_t1`
- default pre-optimization submit script: `job_scripts/uma_preopt_cpu.sh`
- default CP2K single-point submit script: `job_scripts/cp2k_singlepoint_large_mem.sh`
- default UMA MD submit script: `job_scripts/uma_md_cpu.sh`
- default UMA offline env: `/home/ctan/uma-offline-env`
- default UMA model: `/home/ctan/uma-m-1p1.pt`
- `BASIS_MOLOPT`: `/home/ctan/cp2k/cp2k-2026.1/data/BASIS_MOLOPT`
- `POTENTIAL`: `/home/ctan/cp2k/cp2k-2026.1/data/POTENTIAL`

## Structure set
- bulk structures in manifest: `5`
- GB structures in manifest: `15`
- total structures in manifest: `20`
- workflow-managed GB structures: `15`

## Index files
- `00_notes/structure_manifest.csv`
- `00_notes/workflow_manifest.csv`
- `00_notes/cp2k_job_order.txt`

## Sequential submission
```bash
cd LPSCl_UMA_transport_project/02_uma_workflows
screen -S lpscl_transport
python3 job_scripts/serial_submit_cp2k.py
```

or

```bash
cd LPSCl_UMA_transport_project/02_uma_workflows
./job_scripts/run_serial_cp2k.sh
```

To include the optional UMA MD stage:

```bash
python3 job_scripts/serial_submit_cp2k.py --include-md
```

## Failure policy
- completed stages are skipped automatically
- if `00_uma_relax` fails or produces an unphysical geometry, the workflow writes `SKIP_PREOPT`
- `01_cp2k_singlepoint` then falls back to the original structure instead of stopping at the pre-optimization stage
- CP2K single-point failure stops the remaining stages for that structure
- failed stages are not resubmitted unless the user resets outputs or the state file

## Modeling note
- Li vacancies are modeled as charged defects with `CHARGE = -li_vac_count`
- all structures use `MULTIPLICITY 1`
- the production route is now `UMA -> CP2K single point -> optional UMA MD`, which matches the ML-first transport-study goal
