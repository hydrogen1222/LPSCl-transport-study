# MD Shortlist

This note converts the completed follow-up results into a small UMA MD candidate set.

## Basis

- source table: `followup_final_results.csv`
- all `12` vacancy-GB follow-up structures completed both stages:
  - `00_uma_relax`
  - `01_cp2k_singlepoint`
- follow-up UMA final `fmax` range: about `0.118 ~ 0.149 eV/A`
- follow-up CP2K single-point checks:
  - all completed
  - no `SCF run NOT converged`
  - warning count `0` for the checked follow-up outputs

## Main observations

- `t3` is the lowest-energy translation in all four vacancy families.
- the energy gap between `t3` and the corresponding `t1/t2` structures is large, about `20 ~ 25 eV`
- the CP2K max force extracted from the `forces` file is also consistently lower for the `t3` family:
  - `t1/t2`: roughly `0.497 ~ 0.562 eV/A`
  - `t3`: roughly `0.322 ~ 0.411 eV/A`

## Recommended pilot set

### If only 2 MD runs are affordable

- `gb_Sigma3_t3_Li_vac_c1_s1`
  - best overall CP2K max force
  - lowest energy inside the `c1_s1` family
- `gb_Sigma3_t3_Li_vac_c2_s2`
  - lowest energy inside the `c2_s2` family
  - lower CP2K max force than `gb_Sigma3_t3_Li_vac_c2_s1`

### If 4 MD runs are affordable

- keep the two core `t3` cases above
- add `gb_Sigma3_t1_Li_vac_c1_s1`
  - best `t1` contrast case
- add `gb_Sigma3_t2_Li_vac_c1_s2`
  - best `t2` contrast case

This four-case set is the smallest set that still gives:

- one clearly favorable `t3` path for `c1`
- one clearly favorable `t3` path for `c2`
- one `t1` contrast
- one `t2` contrast

### If 6 MD runs are affordable

- add `gb_Sigma3_t3_Li_vac_c1_s2`
- add `gb_Sigma3_t3_Li_vac_c2_s1`

That expands the `t3` family coverage without opening all `12` structures at once.

## Practical recommendation

Do not start MD on all `12` vacancy-GB structures together.

Start from:

1. `gb_Sigma3_t3_Li_vac_c1_s1`
2. `gb_Sigma3_t3_Li_vac_c2_s2`

Then, if those two runs look stable and informative, add:

3. `gb_Sigma3_t1_Li_vac_c1_s1`
4. `gb_Sigma3_t2_Li_vac_c1_s2`

## Neutral controls

For the final transport discussion, the MD set should also include one or two neutral GB controls from the completed production root `02_uma_workflows`.

The most natural neutral controls are:

- `gb_Sigma3_t1`
- `gb_Sigma3_t3`

This lets the later analysis compare:

- neutral GB
- GB plus vacancy
- favorable `t3`
- contrast `t1/t2`
