from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


NUMERIC_FEATURES = [
    "natoms",
    "li_count",
    "volume_ang3",
    "mass_density_g_cm3",
    "li_fraction",
    "li_number_density_per_A3",
    "vacancy_count",
    "translation_index",
    "min_pair_distance_A",
    "li_s_nn_mean_A",
    "li_s_nn_std_A",
    "li_cl_nn_mean_A",
    "li_cl_nn_std_A",
    "li_li_nn_mean_A",
    "li_li_nn_std_A",
    "p_s_nn_mean_A",
    "p_s_nn_std_A",
    "li_s_coord_3p0_mean",
    "li_cl_coord_3p5_mean",
    "target_temperature_K",
]

CATEGORICAL_FEATURES = ["structure_class", "translation_state", "li_vac_region"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def filter_rows(rows: list[dict[str, str]], tier: str, steps: int, temperatures: set[int]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row["label_tier"] != tier:
            continue
        if int(row["target_md_steps"]) != steps:
            continue
        if int(row["target_temperature_K"]) not in temperatures:
            continue
        target = safe_float(row["log10_li_diffusion_last_half_cm2_per_s"])
        if math.isnan(target):
            continue
        selected.append(row)
    return selected


def build_category_maps(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {feature: sorted({row[feature] for row in rows}) for feature in CATEGORICAL_FEATURES}


def encode_rows(rows: list[dict[str, str]], category_maps: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    feature_count = len(NUMERIC_FEATURES) + sum(len(values) for values in category_maps.values())
    x = np.zeros((len(rows), feature_count), dtype=float)
    y = np.zeros(len(rows), dtype=float)

    for i, row in enumerate(rows):
        cursor = 0
        for feature in NUMERIC_FEATURES:
            value = safe_float(row.get(feature))
            x[i, cursor] = 0.0 if math.isnan(value) else value
            cursor += 1
        for feature in CATEGORICAL_FEATURES:
            current = row[feature]
            for value in category_maps[feature]:
                x[i, cursor] = 1.0 if current == value else 0.0
                cursor += 1
        y[i] = safe_float(row["log10_li_diffusion_last_half_cm2_per_s"])
    return x, y


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    std[std < 1.0e-12] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x_train, x_test = standardize(train_x, test_x)
    y_mean = float(np.mean(train_y))
    y_centered = train_y - y_mean
    gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
    coef = np.linalg.solve(gram, x_train.T @ y_centered)
    return x_test @ coef + y_mean


def knn_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int = 3) -> np.ndarray:
    x_train, x_test = standardize(train_x, test_x)
    k = max(1, min(k, x_train.shape[0]))
    predictions = np.zeros(test_x.shape[0], dtype=float)
    for i in range(test_x.shape[0]):
        distances = np.sqrt(np.sum((x_train - x_test[i]) ** 2, axis=1))
        nearest = np.argsort(distances)[:k]
        predictions[i] = float(np.mean(train_y[nearest]))
    return predictions


def mean_predict(train_y: np.ndarray, test_size: int) -> np.ndarray:
    return np.full(test_size, float(np.mean(train_y)), dtype=float)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def loocv_by_structure(task_name: str, rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    structure_ids = sorted({row["structure_id"] for row in rows})
    if len(structure_ids) < 5:
        return [], [
            {
                "task_name": task_name,
                "model_name": "insufficient_data",
                "row_count": len(rows),
                "structure_count": len(structure_ids),
                "mae_log10D": math.nan,
                "rmse_log10D": math.nan,
                "notes": "Need at least 5 unique structures for meaningful leave-one-structure-out validation.",
            }
        ]

    category_maps = build_category_maps(rows)
    x, y = encode_rows(rows, category_maps)
    groups = np.array([row["structure_id"] for row in rows], dtype=object)

    prediction_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    truth_lookup: dict[str, list[float]] = {"mean": [], "ridge": [], "knn": []}
    pred_lookup: dict[str, list[float]] = {"mean": [], "ridge": [], "knn": []}

    for held_out in structure_ids:
        test_mask = groups == held_out
        train_mask = ~test_mask
        train_x = x[train_mask]
        train_y = y[train_mask]
        test_x = x[test_mask]
        test_y = y[test_mask]

        model_predictions = {
            "mean": mean_predict(train_y, test_x.shape[0]),
            "ridge": ridge_predict(train_x, train_y, test_x, alpha=1.0),
            "knn": knn_predict(train_x, train_y, test_x, k=3),
        }

        held_out_rows = [row for row in rows if row["structure_id"] == held_out]
        for index, row in enumerate(held_out_rows):
            for model_name, predictions in model_predictions.items():
                prediction_rows.append(
                    {
                        "task_name": task_name,
                        "model_name": model_name,
                        "structure_id": row["structure_id"],
                        "run_name": row["run_name"],
                        "target_temperature_K": row["target_temperature_K"],
                        "target_md_steps": row["target_md_steps"],
                        "true_log10D": float(test_y[index]),
                        "predicted_log10D": float(predictions[index]),
                        "abs_error": float(abs(test_y[index] - predictions[index])),
                    }
                )
                truth_lookup[model_name].append(float(test_y[index]))
                pred_lookup[model_name].append(float(predictions[index]))

    for model_name in ["mean", "ridge", "knn"]:
        y_true = np.array(truth_lookup[model_name], dtype=float)
        y_pred = np.array(pred_lookup[model_name], dtype=float)
        metrics_rows.append(
            {
                "task_name": task_name,
                "model_name": model_name,
                "row_count": len(rows),
                "structure_count": len(structure_ids),
                "mae_log10D": mae(y_true, y_pred),
                "rmse_log10D": rmse(y_true, y_pred),
                "notes": "Leave-one-structure-out validation on available labeled rows.",
            }
        )

    return prediction_rows, metrics_rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# ML Baseline Summary",
        "",
        "## Interpretation",
        "",
        "- This baseline validates the data pipeline, not the final thesis claim.",
        "- The direct ML target is `log10(D_tracer)`, not experimental-equivalent room-temperature conductivity.",
        "- Formal conductivity remains a derived upper-bound quantity.",
        "",
        "## Results",
        "",
        "| Task | Model | Rows | Structures | MAE (log10D) | RMSE (log10D) | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        mae_value = row["mae_log10D"]
        rmse_value = row["rmse_log10D"]
        mae_text = "NA" if isinstance(mae_value, float) and math.isnan(mae_value) else f"{float(mae_value):.4f}"
        rmse_text = "NA" if isinstance(rmse_value, float) and math.isnan(rmse_value) else f"{float(rmse_value):.4f}"
        lines.append(
            f"| {row['task_name']} | {row['model_name']} | {int(row['row_count'])} | {int(row['structure_count'])} | "
            f"{mae_text} | {rmse_text} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Current recommendation",
            "",
            "- Screening-level MD labels are sufficient to validate the ML plumbing.",
            "- Formal `20000`-step labels are still too few for a serious conductivity predictor.",
            "- The next GPU batch should expand formal labels before any stronger ML claim is made.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "07_ml_pipeline" / "01_datasets"
    notes_dir = project_root / "07_ml_pipeline" / "00_notes"
    rows = read_csv_rows(dataset_dir / "ml_training_table.csv")

    tasks = [
        ("screening_1000steps_600_700K_log10D", "screening", 1000, {600, 700}),
        ("formal_20000steps_600_700_800K_log10D", "formal", 20000, {600, 700, 800}),
    ]

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for task_name, tier, steps, temperatures in tasks:
        selected = filter_rows(rows, tier=tier, steps=steps, temperatures=temperatures)
        predictions, metrics = loocv_by_structure(task_name, selected)
        prediction_rows.extend(predictions)
        metric_rows.extend(metrics)

    write_csv_rows(dataset_dir / "baseline_predictions.csv", prediction_rows)
    write_csv_rows(dataset_dir / "baseline_metrics.csv", metric_rows)
    write_markdown(notes_dir / "ml_baseline_summary.md", metric_rows)

    print(f"[baseline] predictions={dataset_dir / 'baseline_predictions.csv'}")
    print(f"[baseline] metrics={dataset_dir / 'baseline_metrics.csv'}")
    print(f"[baseline] summary={notes_dir / 'ml_baseline_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
