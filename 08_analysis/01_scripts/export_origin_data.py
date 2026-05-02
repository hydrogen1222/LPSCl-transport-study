"""
export_origin_data.py — 导出所有分析结果为 Origin 可直接导入的 CSV/TXT 文件

为每张论文图生成对应的平坦 CSV 数据文件，用户可用 Origin 重绘。
同时导出高分辨率 Li 概率密度矩阵数据。

用法:
    python 08_analysis/01_scripts/export_origin_data.py [--project-root .]

输出目录: 08_analysis/04_origin_data/
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.project_root.resolve()
    results_dir = root / "08_analysis" / "02_results"
    origin_dir = root / "08_analysis" / "04_origin_data"
    origin_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  EXPORT ORIGIN DATA")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. fig_msd_comparison — MSD vs time curves
    # ----------------------------------------------------------------
    msd_csv = results_dir / "msd_curves.csv"
    if msd_csv.exists():
        import pandas as pd
        df = pd.read_csv(msd_csv)
        # Pivot: rows=time, columns=structure_temp
        for sid in df["structure_id"].unique():
            for temp in sorted(df["temperature_K"].unique()):
                sub = df[(df["structure_id"] == sid) & (df["temperature_K"] == temp)]
                if len(sub) == 0:
                    continue
                out_name = f"msd_{sid}_{temp}K.csv"
                sub[["time_ps", "msd_A2"]].to_csv(origin_dir / out_name, index=False)
        print(f"[OK] MSD curves -> 04_origin_data/msd_*.csv")

    # ----------------------------------------------------------------
    # 2. fig_arrhenius — 1000/T vs log10(D) for Arrhenius plot
    # ----------------------------------------------------------------
    diff_csv = results_dir / "diffusion_table.csv"
    arr_csv = results_dir / "arrhenius_fit.csv"
    if diff_csv.exists():
        df = pd.read_csv(diff_csv)
        arr_data = []
        for _, row in df.iterrows():
            arr_data.append({
                "structure_id": row["structure_id"],
                "temperature_K": int(row["temperature_K"]),
                "1000_over_T": 1000.0 / row["temperature_K"],
                "log10_D": row["log10_d_tracer"],
                "D_cm2_s": row["d_tracer_cm2_s"],
                "D_err": row.get("d_tracer_err_cm2_s", ""),
            })
        out = origin_dir / "arrhenius_plot_data.csv"
        pd.DataFrame(arr_data).to_csv(out, index=False)
        print(f"[OK] Arrhenius data -> {out.name}")

    # ----------------------------------------------------------------
    # 3. fig_sigma_bar + fig_ea_bar — bar chart data
    # ----------------------------------------------------------------
    if arr_csv.exists():
        arr = pd.read_csv(arr_csv)
        out = origin_dir / "ea_sigma_bar_data.csv"
        arr[["structure_id", "Ea_eV", "Arrhenius_R2",
             "sigma_NE_upper_300K_mS_cm", "D_tracer_300K_cm2_s"]].to_csv(out, index=False)
        print(f"[OK] Ea/sigma bar data -> {out.name}")

    # ----------------------------------------------------------------
    # 4. fig_rdf_comparison — RDF curves, split by pair type
    # ----------------------------------------------------------------
    rdf_csv = results_dir / "rdf_data.csv"
    if rdf_csv.exists():
        df = pd.read_csv(rdf_csv)
        rdf_dir = origin_dir / "rdf"
        rdf_dir.mkdir(exist_ok=True)

        # Per-structure per-pair files
        for sid in df["structure_id"].unique():
            for pair in df["pair"].unique():
                sub = df[(df["structure_id"] == sid) & (df["pair"] == pair)]
                if len(sub) == 0:
                    continue
                pair_safe = pair.replace("-", "_")
                out_name = f"rdf_{pair_safe}_{sid}.csv"
                sub[["r_A", "g_r"]].to_csv(rdf_dir / out_name, index=False)

        # Also generate a wide-format comparison file for each pair
        # (columns = r_A, bulk_ordered, gb_Sigma3_t1, ...) — easier for Origin
        for pair in df["pair"].unique():
            pair_df = df[df["pair"] == pair]
            pivot = pair_df.pivot_table(index="r_A", columns="structure_id",
                                        values="g_r", aggfunc="first")
            pair_safe = pair.replace("-", "_")
            pivot.to_csv(rdf_dir / f"rdf_{pair_safe}_all_structures.csv")

        print(f"[OK] RDF curves -> 04_origin_data/rdf/ (split by pair type)")

    # ----------------------------------------------------------------
    # 5. fig_ml_predicted_vs_actual + fig_ml_feature_importance
    # ----------------------------------------------------------------
    ml_dir = results_dir / "ml_results"
    for fn in ["ml_cv_results.csv", "ml_feature_importance.csv"]:
        src = ml_dir / fn
        if src.exists():
            import shutil
            shutil.copy2(src, origin_dir / fn)
    print(f"[OK] ML data -> 04_origin_data/ml_*.csv")

    # ----------------------------------------------------------------
    # 6. fig_jump_frequency — jump statistics
    # ----------------------------------------------------------------
    jump_csv = results_dir / "jump_statistics.csv"
    if jump_csv.exists():
        df = pd.read_csv(jump_csv)
        # Filter to only 20000-step formal runs
        formal = df[df["total_time_ps"] == 20.0].copy()
        out = origin_dir / "jump_frequency_data.csv"
        formal.to_csv(out, index=False)
        print(f"[OK] Jump data -> {out.name}")

    # ----------------------------------------------------------------
    # 7. fig_vdos_comparison — VDOS spectra
    # ----------------------------------------------------------------
    vdos_csv = results_dir / "vdos_data.csv"
    if vdos_csv.exists():
        df = pd.read_csv(vdos_csv)
        for sid in df["structure_id"].unique():
            sub = df[df["structure_id"] == sid]
            out_name = f"vdos_{sid}.csv"
            sub[["frequency_THz", "vdos"]].to_csv(origin_dir / out_name, index=False)
        print(f"[OK] VDOS curves -> 04_origin_data/vdos_*.csv")

    # ----------------------------------------------------------------
    # 8. fig_shap_summary — SHAP values
    # ----------------------------------------------------------------
    shap_csv = ml_dir / "shap_values.csv"
    if shap_csv.exists():
        import shutil
        shutil.copy2(shap_csv, origin_dir / "shap_values.csv")
        print(f"[OK] SHAP data -> 04_origin_data/shap_values.csv")

    # ----------------------------------------------------------------
    # 9. fig_li_density_map — Li probability density 2D maps
    #    Export as per-structure CSV matrices + metadata
    # ----------------------------------------------------------------
    density_npz = results_dir / "li_density_data.npz"
    if density_npz.exists():
        data = np.load(density_npz, allow_pickle=True)
        density_dir = origin_dir / "li_density"
        density_dir.mkdir(exist_ok=True)

        # npz contains arrays named by structure_id
        meta_rows = []
        for key in sorted(data.files):
            if key.endswith("_density"):
                sid = key.replace("_density", "")
                matrix = data[key]
                # Save as CSV matrix (rows=y bins, cols=x bins)
                mat_file = density_dir / f"density_{sid}.csv"
                np.savetxt(mat_file, matrix, delimiter=",", fmt="%.6f")
                meta_rows.append({
                    "structure_id": sid,
                    "matrix_file": f"density_{sid}.csv",
                    "shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                    "max_density": f"{matrix.max():.4f}",
                })
            elif key.endswith("_xedges"):
                sid = key.replace("_xedges", "")
                np.savetxt(density_dir / f"xedges_{sid}.csv", data[key], delimiter=",", fmt="%.6f")
            elif key.endswith("_yedges"):
                sid = key.replace("_yedges", "")
                np.savetxt(density_dir / f"yedges_{sid}.csv", data[key], delimiter=",", fmt="%.6f")

        if meta_rows:
            with open(density_dir / "metadata.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
                w.writeheader()
                w.writerows(meta_rows)
        print(f"[OK] Li density matrices -> 04_origin_data/li_density/")
    else:
        print(f"[SKIP] li_density_data.npz not found")

    # ----------------------------------------------------------------
    # 10. Temperature evolution data (from all formal runs)
    # ----------------------------------------------------------------
    md_root = root / "06_cloud_vm_gpu_bundle" / "04_runs" / "md"
    temp_rows = []
    for jp in sorted(md_root.rglob("uma_results.json")):
        rel = jp.relative_to(md_root)
        parts = list(rel.parts)
        sid, rn = parts[0], parts[1]
        if any(x in rn for x in ["1000steps", "200steps", "20steps", "smoke"]):
            continue
        m = re.search(r"(\d+)K", rn)
        if not m:
            continue
        target = int(m.group(1))
        with open(jp) as f:
            d = json.load(f)
        final_T = d.get("calculation", {}).get("md", {}).get("temperature")
        if final_T:
            temp_rows.append({
                "structure_id": sid,
                "run_name": rn,
                "target_T_K": target,
                "final_T_K": f"{final_T:.1f}",
                "deviation_pct": f"{abs(final_T - target) / target * 100:.1f}",
            })
    if temp_rows:
        out = origin_dir / "temperature_summary.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(temp_rows[0].keys()))
            w.writeheader()
            w.writerows(temp_rows)
        print(f"[OK] Temperature summary -> {out.name} ({len(temp_rows)} runs)")

    print()
    print("=" * 60)
    print(f"  ALL DATA EXPORTED TO: {origin_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
