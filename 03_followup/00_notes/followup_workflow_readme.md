# Follow-up Workflow

This follow-up bundle is separate from the completed production root `02_uma_workflows` and keeps the original production data read-only.

## Root paths

- workflow root: `03_followup/02_uma_continue_workflows`
- notes and reports: `03_followup/00_notes`

## Purpose

- keep the first production workflow unchanged
- continue UMA only for the `12` vacancy-GB structures that were previously stopped by the stricter `0.08 eV/A` threshold
- run a cleaner CP2K single-point force-check with `36` MPI ranks
- prepare a smaller and better justified MD candidate set

## Stage meaning

- `00_uma_relax`
  - continuation UMA relaxation
  - starts from the previous workflow's `CONTCAR`
  - `FIRE`, `fmax = 0.15 eV/A`, `max_steps = 300`
- `01_cp2k_singlepoint`
  - CP2K single point with `36` MPI ranks
  - prints atomic forces to a dedicated `forces` file
- `02_uma_md_nvt`
  - kept disabled by default
  - will only be enabled after candidate screening

## Final status on 2026-04-06

- `00_uma_relax`: `12/12` completed
- `01_cp2k_singlepoint`: `12/12` completed
- `02_uma_md_nvt`: not started by design
- no remaining running jobs under user `ctan`
- no non-empty `slurm-*.err` files were found in this follow-up root
- no `SCF run NOT converged` messages were found in the follow-up CP2K single-point outputs

## Main outputs

- historical first-pass snapshot: `results_summary.csv`
- final follow-up summary: `followup_final_results.csv`
- MD screening note: `md_shortlist.md`
- geometry review note carried over from the previous pass: `t3_geometry_review.md`

## What the follow-up achieved

- all `12` vacancy-GB structures reached the relaxed UMA continuation target `fmax <= 0.15 eV/A`
- all `12` vacancy-GB structures completed CP2K single-point force-checks successfully
- CP2K force files are now available for every vacancy-GB structure
- the result set is now suitable for selecting a small UMA MD pilot set instead of launching MD on all structures

## Recommended next step

1. Use `followup_final_results.csv` as the main table for the vacancy-GB follow-up.
2. Use `md_shortlist.md` to choose a small MD pilot set.
3. When MD is started, include one or two neutral GB controls from the completed production root `02_uma_workflows`.
