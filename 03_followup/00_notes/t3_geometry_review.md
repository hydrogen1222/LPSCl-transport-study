# t3 Geometry Review

This note summarizes the manual sanity check that should be completed before making strong claims from the `t3` family.

## Key observations

- `t3` starts from a noticeably tighter initial geometry than `t1/t2`.
- The shortest pair in the initial `t3` structures is about `1.495 A (P-S)`, while `t1/t2` start near `2.059 A (P-S)`.
- After UMA relaxation, the shortest pair in the `t3` family moves back into a much more reasonable range:
  - `gb_Sigma3_t3`: `1.953 A (S-S)`
  - `gb_Sigma3_t3_Li_vac_c1_s1`: `1.945 A (S-S)`
  - `gb_Sigma3_t3_Li_vac_c1_s2`: `1.951 A (S-S)`
- The relaxation amplitude is also larger for the `t3` family:
  - `gb_Sigma3_t3`: `max_disp = 1.835 A`, `rms_disp = 0.663 A`
  - `gb_Sigma3_t3_Li_vac_c1_s1`: `max_disp = 1.934 A`, `rms_disp = 0.704 A`
  - `gb_Sigma3_t3_Li_vac_c1_s2`: `max_disp = 1.908 A`, `rms_disp = 0.722 A`

## Convergence status

- Neutral `t3` converged in UMA:
  - `172` steps
  - `fmax = 0.0734 eV/A`
- The two vacancy `t3` cases did not reach the `0.08 eV/A` threshold within `200` steps:
  - `gb_Sigma3_t3_Li_vac_c1_s1`: `fmax = 0.2012 eV/A`
  - `gb_Sigma3_t3_Li_vac_c1_s2`: `fmax = 0.1489 eV/A`
- Despite that, none of the `t3` jobs showed hard failures:
  - empty `slurm-*.err`
  - zero CP2K SCF failures
  - CP2K ended normally in all cases

## Energy trend

- `t3` is substantially lower in CP2K single-point energy than `t1/t2` within each corresponding group.
- This may indicate that `t3` is genuinely the most favorable translation family.
- It may also mean that `t3` explores a qualitatively different local reconstruction because its starting geometry is already more strained.

## Interpretation boundary

- At this stage, `t3` should be treated as physically plausible but deserving extra scrutiny.
- Do not immediately present the large `t3` energy preference as a final thermodynamic conclusion without a visual check and a DFT force check.

## Recommended next checks

1. Open the following three structures in OVITO or VESTA and confirm that no nonphysical bond rearrangement occurred near the interface:
   - `gb_Sigma3_t3`
   - `gb_Sigma3_t3_Li_vac_c1_s1`
   - `gb_Sigma3_t3_Li_vac_c1_s2`
2. Use the new follow-up workflow to continue UMA for the vacancy `t3` cases.
3. After continuation, rerun CP2K single points with the new 36-rank script that prints forces.
4. If the printed DFT forces on `t3` remain moderate and the geometry still looks reasonable, then `t3` can be promoted to a main-result family.
