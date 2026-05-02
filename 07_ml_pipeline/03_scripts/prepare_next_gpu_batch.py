from __future__ import annotations

import csv
from pathlib import Path


PRIORITY_ORDER = [
    "bulk_Li_vac_c1_s1",
    "bulk_Li_vac_c2_s1",
    "gb_Sigma3_t1",
    "gb_Sigma3_t2",
    "gb_Sigma3_t1_Li_vac_c1_s1",
    "gb_Sigma3_t2_Li_vac_c1_s2",
    "gb_Sigma3_t3_Li_vac_c1_s2",
    "gb_Sigma3_t3_Li_vac_c2_s2",
]

TEMPERATURES = [600, 700, 800]


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


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "07_ml_pipeline" / "01_datasets"
    gpu_dir = project_root / "07_ml_pipeline" / "02_gpu_batch"

    structure_rows = {row["structure_id"]: row for row in read_csv_rows(dataset_dir / "structure_features.csv")}
    label_rows = {row["structure_id"]: row for row in read_csv_rows(dataset_dir / "structure_label_summary.csv")}

    output_rows: list[dict[str, object]] = []
    for priority, structure_id in enumerate(PRIORITY_ORDER, start=1):
        structure = structure_rows[structure_id]
        labels = label_rows.get(structure_id, {})
        if str(labels.get("has_formal_transport", "0")) == "1":
            continue
        for temperature in TEMPERATURES:
            output_rows.append(
                {
                    "priority": priority,
                    "structure_id": structure_id,
                    "structure_class": structure["structure_class"],
                    "translation_state": structure["translation_state"],
                    "vacancy_count": structure["vacancy_count"],
                    "target_temperature_K": temperature,
                    "target_md_steps": 20000,
                    "save_interval_steps": 100,
                    "replicate_id": "seed1",
                    "recommended_reason": "Expand formal MD labels for structure-to-transport ML.",
                }
            )

    write_csv_rows(gpu_dir / "next_md_labeling_manifest.csv", output_rows)
    print(f"[gpu-batch] manifest={gpu_dir / 'next_md_labeling_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
