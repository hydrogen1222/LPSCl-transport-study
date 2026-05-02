"""
run_ml_predictor.py — 结构 → 输运标签 ML 预测器

这是论文中"自己的 ML"部分。
输入：结构特征 + 扩散系数标签
输出：训练模型、交叉验证结果、特征重要性图、predicted vs actual 图

用法:
    python 08_analysis/01_scripts/run_ml_predictor.py \
        --features 07_ml_pipeline/01_datasets/structure_features.csv \
        --labels 08_analysis/02_results/diffusion_table.csv \
        --arrhenius 08_analysis/02_results/arrhenius_fit.csv \
        --output-dir 08_analysis/02_results/ml_results

输出:
    - ml_cv_results.csv           交叉验证结果
    - ml_feature_importance.csv   特征重要性
    - fig_ml_predicted_vs_actual.png   散点图
    - fig_ml_feature_importance.png    特征重要性条形图
    - ml_summary.md               结果摘要
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    HAS_SKL = True
except ImportError:
    HAS_SKL = False


# ---- Feature columns to use (from structure_features.csv) ----
FEATURE_COLS = [
    "natoms", "li_count", "vacancy_count",
    "volume_ang3", "li_number_density_per_A3", "li_fraction",
    "is_bulk", "is_gb",
    "li_s_nn_mean_A", "li_s_nn_std_A",
    "li_cl_nn_mean_A", "li_cl_nn_std_A",
    "li_li_nn_mean_A", "li_li_nn_std_A",
    "p_s_nn_mean_A", "p_s_nn_std_A",
    "li_s_coord_3p0_mean", "li_s_coord_3p0_std",
    "min_pair_distance_A",
]


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (ValueError, TypeError):
        return default


def build_dataset(features_path, labels_path, arrhenius_path=None):
    """Build X, y, groups, feature_names for ML training."""
    feat_rows = {r["structure_id"]: r for r in read_csv_rows(features_path)}

    # Aggregate diffusion data: per structure, average log10(D) across temperatures
    label_rows = read_csv_rows(labels_path)
    label_agg = defaultdict(list)
    for r in label_rows:
        sid = r["structure_id"]
        d = float(r["d_tracer_cm2_s"])
        if d > 0:
            label_agg[sid].append(np.log10(d))

    # Optionally merge Ea as a label
    ea_map = {}
    if arrhenius_path and arrhenius_path.exists():
        for r in read_csv_rows(arrhenius_path):
            ea_map[r["structure_id"]] = float(r["Ea_eV"])

    # Build arrays
    X_list, y_log10d_list, y_ea_list, groups, sids = [], [], [], [], []
    for sid, log10ds in sorted(label_agg.items()):
        if sid not in feat_rows:
            continue
        feat = feat_rows[sid]
        x_row = [safe_float(feat.get(col, 0.0)) for col in FEATURE_COLS]
        mean_log10d = np.mean(log10ds)

        X_list.append(x_row)
        y_log10d_list.append(mean_log10d)
        y_ea_list.append(ea_map.get(sid, np.nan))
        groups.append(sid)
        sids.append(sid)

    X = np.array(X_list)
    y_log10d = np.array(y_log10d_list)
    y_ea = np.array(y_ea_list)

    return X, y_log10d, y_ea, groups, sids, FEATURE_COLS


def run_leave_one_out(X, y, groups, model_class, model_kwargs=None, scaler=True):
    """Leave-one-structure-out cross-validation."""
    logo = LeaveOneGroupOut()
    y_pred = np.full_like(y, np.nan)
    groups_arr = np.array(groups)

    for train_idx, test_idx in logo.split(X, y, groups_arr):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        if scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        model = model_class(**(model_kwargs or {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)

    mask = ~np.isnan(y_pred)
    if mask.sum() < 2:
        return None, None, None
    mae = mean_absolute_error(y[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
    return mae, rmse, y_pred


def get_feature_importance(X, y, feature_names, model_class, model_kwargs=None):
    """Train on all data and extract feature importance."""
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    model = model_class(**(model_kwargs or {}))
    model.fit(X_sc, y)
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    return {}


def main():
    if not HAS_SKL:
        print("[ERROR] scikit-learn not installed. Run: uv pip install scikit-learn")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--arrhenius", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building ML dataset ===")
    X, y_log10d, y_ea, groups, sids, feat_names = build_dataset(
        args.features, args.labels, args.arrhenius
    )
    n_struct = len(set(groups))
    print(f"  Structures with labels: {n_struct}")
    print(f"  Feature dimensions: {X.shape[1]}")

    if n_struct < 4:
        print(f"[WARN] Only {n_struct} structures — too few for meaningful ML. "
              f"Need at least 5. Skipping model training.")
        # Still write a summary
        summary = (
            f"# ML Predictor Summary\n\n"
            f"**Status**: Insufficient data ({n_struct} structures, need ≥5)\n\n"
            f"Run more formal MD to expand the training set.\n"
        )
        (args.output_dir / "ml_summary.md").write_text(summary, encoding="utf-8")
        return

    # ---- Models ----
    models = {
        "Mean": (None, None),  # baseline
        "Ridge": (Ridge, {"alpha": 1.0}),
        "KNN": (KNeighborsRegressor, {"n_neighbors": min(3, n_struct - 1)}),
        "RandomForest": (RandomForestRegressor, {"n_estimators": 100, "random_state": 42, "max_depth": 3}),
        "GradientBoosting": (GradientBoostingRegressor, {"n_estimators": 50, "random_state": 42,
                                                          "max_depth": 2, "learning_rate": 0.1}),
    }

    print("\n=== Leave-one-structure-out CV on log10(D_tracer) ===\n")
    cv_results = []
    best_preds = {}
    best_mae = 999

    for name, (cls, kwargs) in models.items():
        if cls is None:
            # Mean baseline
            y_pred = np.full_like(y_log10d, np.mean(y_log10d))
            mask = np.ones(len(y_log10d), dtype=bool)
            # LOO for mean
            y_pred_loo = np.empty_like(y_log10d)
            groups_arr = np.array(groups)
            logo = LeaveOneGroupOut()
            for tr, te in logo.split(X, y_log10d, groups_arr):
                y_pred_loo[te] = np.mean(y_log10d[tr])
            mae = mean_absolute_error(y_log10d, y_pred_loo)
            rmse = np.sqrt(mean_squared_error(y_log10d, y_pred_loo))
            y_pred = y_pred_loo
        else:
            mae, rmse, y_pred = run_leave_one_out(X, y_log10d, groups, cls, kwargs)
            if mae is None:
                print(f"  {name:20s}: FAILED")
                continue

        print(f"  {name:20s}: MAE={mae:.4f}  RMSE={rmse:.4f}")
        cv_results.append({"model": name, "mae_log10D": f"{mae:.4f}", "rmse_log10D": f"{rmse:.4f}",
                           "n_structures": n_struct})
        if mae < best_mae:
            best_mae = mae
            best_preds = {"model": name, "y_true": y_log10d.copy(), "y_pred": y_pred.copy(),
                          "sids": sids[:]}

    # Write CV results
    cv_path = args.output_dir / "ml_cv_results.csv"
    with open(cv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "mae_log10D", "rmse_log10D", "n_structures"])
        w.writeheader()
        w.writerows(cv_results)
    print(f"\n[OK] {cv_path}")

    # ---- Feature importance (RandomForest) ----
    fi = get_feature_importance(X, y_log10d, feat_names,
                                RandomForestRegressor,
                                {"n_estimators": 100, "random_state": 42, "max_depth": 3})
    if fi:
        fi_sorted = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
        fi_path = args.output_dir / "ml_feature_importance.csv"
        with open(fi_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature", "importance"])
            for name, imp in fi_sorted:
                w.writerow([name, f"{imp:.6f}"])
        print(f"[OK] {fi_path}")

    # ---- Plots ----
    if HAS_MPL and best_preds:
        # Predicted vs Actual
        fig, ax = plt.subplots(figsize=(5, 5))
        y_t = best_preds["y_true"]
        y_p = best_preds["y_pred"]
        ax.scatter(y_t, y_p, c="#E53935", s=60, edgecolors="white", zorder=5)
        for i, sid in enumerate(best_preds["sids"]):
            label = sid.replace("gb_Sigma3_", "").replace("_Li_vac_", "+V")
            ax.annotate(label, (y_t[i], y_p[i]), fontsize=6,
                        textcoords="offset points", xytext=(5, 5))
        lims = [min(y_t.min(), y_p.min()) - 0.1, max(y_t.max(), y_p.max()) + 0.1]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual log₁₀(D_tracer) [cm²/s]", fontsize=11)
        ax.set_ylabel("Predicted log₁₀(D_tracer) [cm²/s]", fontsize=11)
        ax.set_title(f"ML Prediction ({best_preds['model']})\n"
                     f"Leave-one-structure-out, MAE={best_mae:.3f}",
                     fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(args.output_dir / "fig_ml_predicted_vs_actual.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] fig_ml_predicted_vs_actual.png")

        # Feature importance bar
        if fi:
            top_n = min(12, len(fi_sorted))
            fig, ax = plt.subplots(figsize=(6, 4))
            names = [x[0] for x in fi_sorted[:top_n]][::-1]
            vals = [x[1] for x in fi_sorted[:top_n]][::-1]
            ax.barh(range(top_n), vals, color="#2196F3", edgecolor="white")
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("Feature Importance", fontsize=11)
            ax.set_title("RandomForest Feature Importance", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(args.output_dir / "fig_ml_feature_importance.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] fig_ml_feature_importance.png")

    # ---- Summary markdown ----
    summary_lines = [
        "# ML Predictor Summary",
        "",
        f"- Training structures: **{n_struct}**",
        f"- Features: **{X.shape[1]}** structural descriptors",
        f"- Target: **log₁₀(D_tracer)** averaged across 500/600/700/800/900 K",
        f"- Validation: **Leave-one-structure-out**",
        "",
        "## Cross-Validation Results",
        "",
        "| Model | MAE (log₁₀D) | RMSE (log₁₀D) |",
        "| --- | ---: | ---: |",
    ]
    for r in cv_results:
        summary_lines.append(f"| {r['model']} | {r['mae_log10D']} | {r['rmse_log10D']} |")

    if fi:
        summary_lines.extend([
            "",
            "## Top Features (RandomForest)",
            "",
        ])
        for name, imp in fi_sorted[:8]:
            summary_lines.append(f"- `{name}`: {imp:.4f}")

    summary_lines.extend([
        "",
        "## Interpretation",
        "",
        "- This ML model predicts tracer diffusion coefficients from static structural features.",
        "- The model is trained entirely on UMA-MD simulation labels, not experimental data.",
        "- Feature importance reveals which structural descriptors most influence Li transport.",
        "- This constitutes the 'structure → transport' ML framework of the thesis.",
    ])

    (args.output_dir / "ml_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[OK] ml_summary.md")
    print("\n=== ML Predictor complete ===")


if __name__ == "__main__":
    main()
