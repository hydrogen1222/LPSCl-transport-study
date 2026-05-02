"""
run_all_analysis.py — 一键运行所有本地分析流水线

执行顺序 (v3):
    1. compute_msd_all.py        → diffusion_table.csv, msd_curves.csv
    2. compute_arrhenius.py      → arrhenius_fit.csv
    3. compute_rdf.py            → rdf_data.csv
    4. run_ml_predictor.py       → ml_cv_results.csv, 特征重要性, 散点图
    5. plot_all_figures.py       → 所有论文图
    6. compute_li_density.py     → li_density_data.npz, fig_li_density_map.png
    7. compute_jump_stats.py     → jump_statistics.csv, fig_jump_frequency.png
    8. compute_vdos.py           → vdos_data.csv, fig_vdos_comparison.png
    9. compute_shap.py           → shap_values.csv, fig_shap_summary.png
   10. export_origin_data.py     → 04_origin_data/ (Origin 绘图用数据)

注意: 默认只处理 20ps (20000 steps) 正式轨迹，排除 50ps 长轨迹和多种子轨迹。
      通过 --run-filter 参数可以更改筛选规则。

用法 (在项目根目录):
    python 08_analysis/01_scripts/run_all_analysis.py

或指定路径:
    python 08_analysis/01_scripts/run_all_analysis.py \
        --project-root D:/毕业设计/LPSCl_UMA_transport_project

执行顺序:
    1. compute_msd_all.py     → diffusion_table.csv, msd_curves.csv
    2. compute_arrhenius.py   → arrhenius_fit.csv
    3. compute_rdf.py         → rdf_data.csv
    4. run_ml_predictor.py    → ml_cv_results.csv, 特征重要性, 散点图
    5. plot_all_figures.py    → 所有论文图

依赖:
    uv pip install numpy matplotlib scikit-learn ase
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]):
    """Run a step and check for errors."""
    print(f"\n{'=' * 60}")
    print(f"  Step: {name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(cmd[0]).rsplit("/", 1)[0] if "/" in str(cmd[0]) else ".")
    if result.returncode != 0:
        print(f"\n[WARN] Step '{name}' exited with code {result.returncode}")
        print(f"       Continuing with remaining steps...\n")
    else:
        print(f"\n[OK] Step '{name}' completed.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Root of LPSCl_UMA_transport_project")
    parser.add_argument("--run-filter", type=str,
                        default=r"^md_\d+K_20000steps$",
                        help="Regex to filter run directories. "
                             "Default: only formal 20ps runs")
    args = parser.parse_args()

    if args.project_root:
        root = args.project_root
    else:
        # Try to infer from script location
        script_dir = Path(__file__).resolve().parent
        root = script_dir.parents[1]  # 08_analysis/01_scripts -> project root

    # Paths
    md_root = root / "06_cloud_vm_gpu_bundle" / "04_runs" / "md"
    prepared_root = root / "06_cloud_vm_gpu_bundle" / "04_runs" / "prepared"
    inputs_root = root / "06_cloud_vm_gpu_bundle" / "01_inputs"
    features_csv = root / "07_ml_pipeline" / "01_datasets" / "structure_features.csv"
    scripts_dir = root / "08_analysis" / "01_scripts"
    results_dir = root / "08_analysis" / "02_results"
    figures_dir = root / "08_analysis" / "03_figures"
    ml_dir = results_dir / "ml_results"
    origin_dir = root / "08_analysis" / "04_origin_data"
    run_filter = args.run_filter

    python = sys.executable

    print("=" * 60)
    print("  LPSCl UMA Transport — Full Analysis Pipeline")
    print("=" * 60)
    print(f"  project_root = {root}")
    print(f"  md_root      = {md_root}")
    print(f"  python       = {python}")

    if not md_root.exists():
        print(f"\n[ERROR] MD root not found: {md_root}")
        print("Make sure you've synced the cloud GPU results to local.")
        return 1

    # Step 1: MSD + Diffusion
    run_step("MSD & Diffusion", [
        python, str(scripts_dir / "compute_msd_all.py"),
        "--md-root", str(md_root),
        "--output-dir", str(results_dir),
        "--run-filter", run_filter,
    ])

    # Step 2: Arrhenius fit
    diff_table = results_dir / "diffusion_table.csv"
    if diff_table.exists():
        run_step("Arrhenius Fitting", [
            python, str(scripts_dir / "compute_arrhenius.py"),
            "--diffusion-table", str(diff_table),
            "--volume-source", str(md_root),
            "--output-dir", str(results_dir),
        ])

    # Step 3: RDF
    run_step("RDF Analysis", [
        python, str(scripts_dir / "compute_rdf.py"),
        "--structures-root", str(prepared_root),
        "--fallback-root", str(inputs_root),
        "--output-dir", str(results_dir),
    ])

    # Step 4: ML Predictor
    if diff_table.exists() and features_csv.exists():
        arr_csv = results_dir / "arrhenius_fit.csv"
        cmd = [
            python, str(scripts_dir / "run_ml_predictor.py"),
            "--features", str(features_csv),
            "--labels", str(diff_table),
            "--output-dir", str(ml_dir),
        ]
        if arr_csv.exists():
            cmd.extend(["--arrhenius", str(arr_csv)])
        run_step("ML Predictor", cmd)

    # Step 5: Plot all figures
    run_step("Generate Figures", [
        python, str(scripts_dir / "plot_all_figures.py"),
        "--results-dir", str(results_dir),
        "--output-dir", str(figures_dir),
    ])

    # Also copy ML figures to figures dir
    for fig in (ml_dir).glob("fig_*.png") if ml_dir.exists() else []:
        import shutil
        shutil.copy2(fig, figures_dir / fig.name)

    # Step 6: Li probability density
    run_step("Li Density Map", [
        python, str(scripts_dir / "compute_li_density.py"),
        "--md-root", str(md_root),
        "--output-dir", str(results_dir),
        "--figures-dir", str(figures_dir),
    ])

    # Step 7: Jump statistics
    run_step("Jump Statistics", [
        python, str(scripts_dir / "compute_jump_stats.py"),
        "--md-root", str(md_root),
        "--output-dir", str(results_dir),
        "--figures-dir", str(figures_dir),
        "--run-filter", run_filter,
    ])

    # Step 8: VDOS
    run_step("VDOS Analysis", [
        python, str(scripts_dir / "compute_vdos.py"),
        "--md-root", str(md_root),
        "--output-dir", str(results_dir),
        "--figures-dir", str(figures_dir),
    ])

    # Step 9: SHAP analysis
    if diff_table.exists() and features_csv.exists():
        run_step("SHAP Analysis", [
            python, str(scripts_dir / "compute_shap.py"),
            "--features", str(features_csv),
            "--labels", str(diff_table),
            "--output-dir", str(ml_dir),
            "--figures-dir", str(figures_dir),
        ])

    # Step 10: Export Origin-ready data
    run_step("Export Origin Data", [
        python, str(scripts_dir / "export_origin_data.py"),
        "--project-root", str(root),
    ])

    print("\n" + "=" * 60)
    print("  ALL ANALYSIS COMPLETE (v3 — 10 steps)")
    print("=" * 60)
    print(f"  Results:    {results_dir}")
    print(f"  Figures:    {figures_dir}")
    print(f"  ML:         {ml_dir}")
    print(f"  Origin:     {origin_dir}")
    print(f"  Run filter: {run_filter}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
