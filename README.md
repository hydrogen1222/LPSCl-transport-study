# Li₆PS₅Cl Transport Properties: Grain Boundary & Li Vacancy Effects

> **Undergraduate thesis project** — Effect of grain boundaries and Li vacancies on transport properties of Li₆PS₅Cl, studied via Universal Machine Learning Potential (UMA) molecular dynamics and structural descriptor regression.

## What This Project Does

This project systematically investigates how Σ3 grain boundaries and Li vacancies affect Li-ion migration in the argyrodite solid electrolyte Li₆PS₅Cl. The workflow covers:

1. **Structure modeling** — 20 representative atomic structures (bulk, bulk+vacancy, GB, GB+vacancy), 414–624 atoms each
2. **UMA molecular dynamics** — 20 ps NVT-MD at 500/600/700/800/900 K using Meta's universal ML potential (100 formal trajectories)
3. **Transport analysis** — MSD → diffusion coefficient → Arrhenius fitting (5-point) → activation energy & Nernst-Einstein conductivity estimate (σ\_NE)
4. **Supplementary dynamics** — Li jump frequency statistics, Li probability density mapping
5. **ML structure–transport regression** — 19-dimensional structural descriptors → Ridge/RF/KNN → feature importance + SHAP analysis

## Key Results

| Structure Type | Ea (eV) | σ\_NE (mS/cm) | vs. Bulk |
|---|---:|---:|---:|
| Perfect bulk | 0.186 | 21.9 | 1× |
| Σ3 grain boundary (best: t2) | 0.137 | 133.3 | 6.1× |
| Σ3 grain boundary (range) | 0.137–0.175 | 51.6–133.3 | 2.4–6.1× |
| GB + vacancy (best: t1+c2s2) | 0.143 | 120.6 | 5.5× |
| GB + vacancy (worst: t3+c2s1) | 0.214 | 25.0 | suppressed |

Ridge regression MAE = 0.053 in log₁₀D (leave-one-structure-out, 20 structures). Top feature: Li–S coordination number.

**Key finding**: Grain boundaries consistently promote Li transport, but vacancy effects are highly site-dependent — some vacancy-GB combinations further enhance transport while others suppress it below the perfect bulk level.

## Directory Structure

```
├── done.cif                        # Optimized Li₆PS₅Cl reference cell (project starting point)
├── 00_notes/                       # Research notes and plans
├── 01_structures/                  # All 20 POSCAR structure files
├── 06_cloud_vm_gpu_bundle/         # Cloud GPU computation bundle
│   ├── 01_inputs/                  #   Input structures (POSCAR.start)
│   ├── 02_runtime/                 #   UMA model interface (install separately)
│   ├── 03_scripts/                 #   MD run scripts (bash + python)
│   └── 04_runs/md/                 #   MD results (XDATCAR excluded by .gitignore)
├── 07_ml_pipeline/                 # ML datasets
│   └── 01_datasets/
│       └── structure_features.csv  #   19-dim structural descriptors for all structures
├── 08_analysis/                    # Analysis scripts & results
│   ├── 01_scripts/                 #   Python analysis pipeline (v3, 10 steps)
│   ├── 02_results/                 #   CSV tables (diffusion, Arrhenius, ML, SHAP)
│   ├── 03_figures/                 #   Publication-ready figures (11 PNGs)
│   └── 04_origin_data/            #   Origin-ready CSV data for external plotting
│       └── rdf/                   #     RDF data split by atom pair type
├── chapters/                       # Thesis manuscript (Markdown, Chinese)
└── 项目目录导读手册.md              # Project guide (Chinese)
```

## Reproducing the Analysis

### Prerequisites

- Python 3.10+
- Dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `ase`, `pandas`, `shap`

### Run

```powershell
cd LPSCl_UMA_transport_project
python -m venv 08_analysis/.venv
& 08_analysis/.venv/Scripts/pip install numpy scipy matplotlib scikit-learn ase pandas shap
& 08_analysis/.venv/Scripts/python 08_analysis/01_scripts/run_all_analysis.py --project-root .
```

This regenerates all CSV tables, figures, and Origin data from the MD trajectory data. The pipeline uses `--run-filter "^md_\d+K_20000steps$"` by default to ensure only formal 20ps trajectories are included.

> **Note:** MD trajectory files (XDATCAR, ~281 MB) are excluded from this repository due to size. Contact the author if you need the raw trajectories.

## Citation

If you use this workflow or data, please cite:

```
晶界与锂空位对 Li₆PS₅Cl 输运性质的影响：基于通用机器学习势与结构描述符回归的研究
吉林大学化学学院本科毕业论文, 2026
```

## License

This project is released for academic and educational purposes. Please contact the author before commercial use.
