"""
compute_shap.py — SHAP 特征重要性分析

用 shap.TreeExplainer 对 RandomForest 做 SHAP 分析，
生成 beeswarm plot 展示每个特征对每个样本预测的贡献方向。

用法:
    python 08_analysis/01_scripts/compute_shap.py \
        --features 07_ml_pipeline/01_datasets/structure_features.csv \
        --labels 08_analysis/02_results/diffusion_table.csv \
        --output-dir 08_analysis/02_results/ml_results \
        --figures-dir 08_analysis/03_figures

输出:
    - shap_values.csv           每个样本×特征的 SHAP 值
    - fig_shap_summary.png      SHAP beeswarm/bar plot
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True,
                        help="structure_features.csv")
    parser.add_argument("--labels", type=Path, required=True,
                        help="diffusion_table.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    import pandas as pd
    features_df = pd.read_csv(args.features)
    diff_df = pd.read_csv(args.labels)

    # Compute label: mean log10(D) across temperatures for each structure
    label_df = (diff_df.groupby("structure_id")["log10_d_tracer"]
                .mean().reset_index()
                .rename(columns={"log10_d_tracer": "mean_log10_D"}))

    # Merge
    merged = features_df.merge(label_df, on="structure_id", how="inner")
    print(f"[INFO] {len(merged)} structures with both features and labels")

    if len(merged) < 3:
        print("[ERROR] Too few samples for SHAP analysis")
        return 1

    # Separate features and target — only use numeric columns
    non_feature_cols = {"structure_id", "source_path", "formula", "structure_class",
                        "translation_state", "li_vac_region", "min_pair_label"}
    feature_cols = [c for c in features_df.columns
                    if c not in non_feature_cols
                    and features_df[c].dtype in ('int64', 'float64', 'bool')]
    X = merged[feature_cols].values.astype(float)
    y = merged["mean_log10_D"].values
    structure_ids = merged["structure_id"].values

    # ---- Standardize ----
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Train RandomForest on full data ----
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_scaled, y)
    print(f"[INFO] RandomForest trained, R^2 on training set: {rf.score(X_scaled, y):.4f}")

    # ---- SHAP analysis ----
    try:
        import shap
    except ImportError:
        print("[ERROR] shap package not installed. Run: pip install shap")
        return 1

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)
    print(f"[INFO] SHAP values computed: shape = {shap_values.shape}")

    # ---- Save SHAP values as CSV ----
    shap_rows = []
    for i in range(len(structure_ids)):
        row = {"structure_id": structure_ids[i]}
        for j, feat in enumerate(feature_cols):
            row[f"shap_{feat}"] = f"{shap_values[i, j]:.6f}"
        shap_rows.append(row)

    shap_csv = args.output_dir / "shap_values.csv"
    with open(shap_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(shap_rows[0].keys()))
        w.writeheader()
        w.writerows(shap_rows)
    print(f"[OK] {shap_csv}")

    # ---- Plot ----
    fig_dir = args.figures_dir or args.output_dir
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Create a nice two-panel figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # Left: SHAP bar plot (mean |SHAP|)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        top_n = min(12, len(feature_cols))
        top_idx = sorted_idx[:top_n]

        colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, top_n))
        ax1.barh(range(top_n), mean_abs_shap[top_idx][::-1],
                 color=colors_bar[::-1], edgecolor="white", height=0.7)
        ax1.set_yticks(range(top_n))
        ax1.set_yticklabels([feature_cols[i] for i in top_idx][::-1], fontsize=9)
        ax1.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax1.set_title("Feature Importance (SHAP)", fontsize=12, fontweight="bold")

        # Right: SHAP beeswarm-style dot plot
        for rank, feat_idx in enumerate(top_idx[:top_n]):
            feat_shap = shap_values[:, feat_idx]
            feat_raw = X[:, feat_idx]  # Use raw (unscaled) values for color
            feat_norm = (feat_raw - feat_raw.min()) / (feat_raw.max() - feat_raw.min() + 1e-10)

            y_pos = top_n - 1 - rank
            jitter = np.random.uniform(-0.2, 0.2, size=len(feat_shap))
            scatter = ax2.scatter(feat_shap, y_pos + jitter,
                                  c=feat_norm, cmap="RdBu_r",
                                  s=40, alpha=0.85, edgecolors="none",
                                  vmin=0, vmax=1)

        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels([feature_cols[i] for i in top_idx][::-1], fontsize=9)
        ax2.set_xlabel("SHAP value (impact on log₁₀D)", fontsize=11)
        ax2.set_title("SHAP Value Distribution", fontsize=12, fontweight="bold")
        ax2.axvline(0, color="gray", lw=0.5, ls=":")

        # Colorbar for feature value
        cbar = fig.colorbar(scatter, ax=ax2, shrink=0.6, pad=0.02)
        cbar.set_label("Feature value\n(low → high)", fontsize=9)

        fig_path = Path(fig_dir) / "fig_shap_summary.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Figure saved: {fig_path}")

    except ImportError as e:
        print(f"[WARN] Plotting failed: {e}")

    print("[DONE] SHAP analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
