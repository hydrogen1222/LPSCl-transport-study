from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.io import read


AVOGADRO = 6.02214076e23


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


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: str | None) -> int | None:
    parsed = safe_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def structure_class(row: dict[str, str]) -> str:
    return "gb" if row["gb_type"] != "NA" else "bulk"


def translation_index(state: str) -> int:
    if state in {"t1", "t2", "t3"}:
        return int(state[1])
    return 0


def pair_label(symbol_a: str, symbol_b: str) -> str:
    left, right = sorted((symbol_a, symbol_b))
    return f"{left}-{right}"


def summary_stats(values: Iterable[float]) -> tuple[float, float, float, float]:
    array = np.array(list(values), dtype=float)
    if array.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    return float(np.mean(array)), float(np.std(array)), float(np.min(array)), float(np.max(array))


def nearest_neighbor_values(
    distances: np.ndarray,
    symbols: list[str],
    center_symbol: str,
    neighbor_symbol: str,
) -> list[float]:
    center_indices = [idx for idx, symbol in enumerate(symbols) if symbol == center_symbol]
    neighbor_indices = [idx for idx, symbol in enumerate(symbols) if symbol == neighbor_symbol]
    if not center_indices or not neighbor_indices:
        return []

    values: list[float] = []
    for center in center_indices:
        current_neighbors = neighbor_indices
        if center_symbol == neighbor_symbol:
            current_neighbors = [idx for idx in neighbor_indices if idx != center]
        if not current_neighbors:
            continue
        nearest = float(np.min(distances[center, current_neighbors]))
        values.append(nearest)
    return values


def coordination_values(
    distances: np.ndarray,
    symbols: list[str],
    center_symbol: str,
    neighbor_symbol: str,
    cutoff: float,
) -> list[int]:
    center_indices = [idx for idx, symbol in enumerate(symbols) if symbol == center_symbol]
    neighbor_indices = [idx for idx, symbol in enumerate(symbols) if symbol == neighbor_symbol]
    if not center_indices or not neighbor_indices:
        return []

    values: list[int] = []
    for center in center_indices:
        current_neighbors = neighbor_indices
        if center_symbol == neighbor_symbol:
            current_neighbors = [idx for idx in neighbor_indices if idx != center]
        count = int(np.sum(distances[center, current_neighbors] <= cutoff))
        values.append(count)
    return values


def compute_pair_features(atoms) -> dict[str, object]:
    symbols = atoms.get_chemical_symbols()
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)

    flat_index = int(np.argmin(distances))
    atom_i, atom_j = np.unravel_index(flat_index, distances.shape)
    overall_min = float(distances[atom_i, atom_j])
    overall_label = pair_label(symbols[atom_i], symbols[atom_j])

    li_s = nearest_neighbor_values(distances, symbols, "Li", "S")
    li_cl = nearest_neighbor_values(distances, symbols, "Li", "Cl")
    li_li = nearest_neighbor_values(distances, symbols, "Li", "Li")
    p_s = nearest_neighbor_values(distances, symbols, "P", "S")
    li_s_coord = coordination_values(distances, symbols, "Li", "S", 3.0)
    li_cl_coord = coordination_values(distances, symbols, "Li", "Cl", 3.5)

    li_s_mean, li_s_std, li_s_min, li_s_max = summary_stats(li_s)
    li_cl_mean, li_cl_std, li_cl_min, li_cl_max = summary_stats(li_cl)
    li_li_mean, li_li_std, li_li_min, li_li_max = summary_stats(li_li)
    p_s_mean, p_s_std, p_s_min, p_s_max = summary_stats(p_s)
    li_s_coord_mean, li_s_coord_std, _, _ = summary_stats(li_s_coord)
    li_cl_coord_mean, li_cl_coord_std, _, _ = summary_stats(li_cl_coord)

    return {
        "min_pair_distance_A": overall_min,
        "min_pair_label": overall_label,
        "li_s_nn_mean_A": li_s_mean,
        "li_s_nn_std_A": li_s_std,
        "li_s_nn_min_A": li_s_min,
        "li_s_nn_max_A": li_s_max,
        "li_cl_nn_mean_A": li_cl_mean,
        "li_cl_nn_std_A": li_cl_std,
        "li_cl_nn_min_A": li_cl_min,
        "li_cl_nn_max_A": li_cl_max,
        "li_li_nn_mean_A": li_li_mean,
        "li_li_nn_std_A": li_li_std,
        "li_li_nn_min_A": li_li_min,
        "li_li_nn_max_A": li_li_max,
        "p_s_nn_mean_A": p_s_mean,
        "p_s_nn_std_A": p_s_std,
        "p_s_nn_min_A": p_s_min,
        "p_s_nn_max_A": p_s_max,
        "li_s_coord_3p0_mean": li_s_coord_mean,
        "li_s_coord_3p0_std": li_s_coord_std,
        "li_cl_coord_3p5_mean": li_cl_coord_mean,
        "li_cl_coord_3p5_std": li_cl_coord_std,
    }


def build_structure_features(project_root: Path) -> list[dict[str, object]]:
    manifest_path = project_root / "00_notes" / "structure_manifest.csv"
    rows = read_csv_rows(manifest_path)
    feature_rows: list[dict[str, object]] = []

    for row in rows:
        structure_id = row["structure_id"]
        poscar_path = project_root / row["path"]
        atoms = read(poscar_path)
        symbols = atoms.get_chemical_symbols()
        counts = defaultdict(int)
        for symbol in symbols:
            counts[symbol] += 1

        cell_lengths = atoms.cell.lengths()
        volume_ang3 = float(atoms.get_volume())
        molar_mass = float(sum(atomic_masses[atomic_numbers[symbol]] for symbol in symbols))
        density_g_cm3 = molar_mass / (AVOGADRO * volume_ang3 * 1.0e-24)
        pair_features = compute_pair_features(atoms)

        feature_rows.append(
            {
                "structure_id": structure_id,
                "source_path": str(poscar_path),
                "formula": atoms.get_chemical_formula(mode="reduce"),
                "structure_class": structure_class(row),
                "is_bulk": 1 if structure_class(row) == "bulk" else 0,
                "is_gb": 1 if structure_class(row) == "gb" else 0,
                "translation_state": row["translation_state"],
                "translation_index": translation_index(row["translation_state"]),
                "li_vac_region": row["li_vac_region"],
                "vacancy_count": safe_int(row["li_vac_count"]) or 0,
                "has_vacancy": 1 if (safe_int(row["li_vac_count"]) or 0) > 0 else 0,
                "natoms": len(symbols),
                "li_count": counts["Li"],
                "p_count": counts["P"],
                "s_count": counts["S"],
                "cl_count": counts["Cl"],
                "lattice_a_A": float(cell_lengths[0]),
                "lattice_b_A": float(cell_lengths[1]),
                "lattice_c_A": float(cell_lengths[2]),
                "volume_ang3": volume_ang3,
                "mass_density_g_cm3": density_g_cm3,
                "li_fraction": counts["Li"] / len(symbols),
                "li_number_density_per_A3": counts["Li"] / volume_ang3,
                "gb_delta_A": safe_float(row["delta_A"]) if row["delta_A"] != "NA" else math.nan,
                "gb_z0_A": safe_float(row["z0_A"]) if row["z0_A"] != "NA" else math.nan,
                **pair_features,
            }
        )

    return feature_rows


def load_cp2k_labels(project_root: Path) -> dict[str, dict[str, object]]:
    labels: dict[str, dict[str, object]] = {}

    neutral_path = project_root / "03_followup" / "00_notes" / "results_summary.csv"
    if neutral_path.exists():
        for row in read_csv_rows(neutral_path):
            labels[row["structure_id"]] = {
                "cp2k_energy_Ha": safe_float(row["cp2k_energy_Ha"]),
                "cp2k_max_force_ev_per_A": math.nan,
                "cp2k_warning_count": safe_int(row["cp2k_warning_count"]) or 0,
                "cp2k_scf_failures": safe_int(row["cp2k_scf_failures"]) or 0,
                "cp2k_label_source": "03_followup/results_summary.csv",
            }

    vacancy_path = project_root / "03_followup" / "00_notes" / "followup_final_results.csv"
    if vacancy_path.exists():
        for row in read_csv_rows(vacancy_path):
            labels[row["structure_id"]] = {
                "cp2k_energy_Ha": safe_float(row["cp2k_singlepoint_energy_Ha"]),
                "cp2k_max_force_ev_per_A": safe_float(row["cp2k_max_force_ev_per_A"]),
                "cp2k_warning_count": safe_int(row["cp2k_warning_count"]) or 0,
                "cp2k_scf_failures": safe_int(row["cp2k_scf_failures"]) or 0,
                "cp2k_label_source": "03_followup/followup_final_results.csv",
            }

    return labels


def load_transport_run_rows(project_root: Path) -> list[dict[str, object]]:
    summary_path = project_root / "06_cloud_vm_gpu_bundle" / "00_notes" / "md_run_summary.csv"
    rows = read_csv_rows(summary_path)
    processed: list[dict[str, object]] = []

    for row in rows:
        steps = safe_int(row["target_md_steps"]) or 0
        temperature = safe_int(row["target_temperature_K"]) or 0
        diffusion = safe_float(row["li_diffusion_last_half_cm2_per_s"])
        log10_diffusion = math.nan if diffusion is None or diffusion <= 0.0 else math.log10(diffusion)

        if steps >= 20000:
            label_tier = "formal"
        elif steps >= 1000:
            label_tier = "screening"
        else:
            label_tier = "pilot"

        processed.append(
            {
                "structure_id": row["structure_id"],
                "run_name": row["run_name"],
                "label_tier": label_tier,
                "target_temperature_K": temperature,
                "reported_temperature_K": safe_float(row["reported_temperature_K"]),
                "target_md_steps": steps,
                "reported_md_steps": safe_int(row["reported_md_steps"]),
                "trajectory_time_ps": safe_float(row["trajectory_time_ps"]),
                "li_diffusion_last_half_cm2_per_s": diffusion,
                "log10_li_diffusion_last_half_cm2_per_s": log10_diffusion,
                "li_msd_last_A2": safe_float(row["li_msd_last_A2"]),
                "li_rmsd_last_A": safe_float(row["li_rmsd_last_A"]),
                "frames_saved": safe_int(row["frames_saved"]),
                "save_interval_steps": safe_int(row["save_interval_steps"]),
                "energy_per_atom_eV": safe_float(row["energy_per_atom_eV"]),
                "fmax_eV_per_A": safe_float(row["fmax_eV_per_A"]),
                "fmean_eV_per_A": safe_float(row["fmean_eV_per_A"]),
                "frms_eV_per_A": safe_float(row["frms_eV_per_A"]),
                "run_min_pair_distance_A": safe_float(row["min_pair_distance_A"]),
                "run_min_pair_label": row["min_pair_label"],
                "model_path": row["model_path"],
                "device": row["device"],
            }
        )

    return processed


def build_structure_label_summary(
    project_root: Path,
    cp2k_labels: dict[str, dict[str, object]],
    transport_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    conductivity_path = project_root / "06_cloud_vm_gpu_bundle" / "00_notes" / "conductivity_production_summary.csv"
    conductivity_rows = {row["structure_id"]: row for row in read_csv_rows(conductivity_path)}
    grouped_transport: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in transport_rows:
        grouped_transport[str(row["structure_id"])].append(row)

    structure_ids = sorted(set(grouped_transport.keys()) | set(cp2k_labels.keys()) | set(conductivity_rows.keys()))
    output_rows: list[dict[str, object]] = []
    for structure_id in structure_ids:
        runs = grouped_transport.get(structure_id, [])
        screening_600 = next((r for r in runs if r["label_tier"] == "screening" and r["target_temperature_K"] == 600), None)
        screening_700 = next((r for r in runs if r["label_tier"] == "screening" and r["target_temperature_K"] == 700), None)
        formal_600 = next((r for r in runs if r["label_tier"] == "formal" and r["target_temperature_K"] == 600), None)
        formal_700 = next((r for r in runs if r["label_tier"] == "formal" and r["target_temperature_K"] == 700), None)
        formal_800 = next((r for r in runs if r["label_tier"] == "formal" and r["target_temperature_K"] == 800), None)
        cp2k = cp2k_labels.get(structure_id, {})
        conductivity = conductivity_rows.get(structure_id, {})

        output_rows.append(
            {
                "structure_id": structure_id,
                "has_cp2k_label": 1 if cp2k else 0,
                "cp2k_energy_Ha": cp2k.get("cp2k_energy_Ha", math.nan),
                "cp2k_max_force_ev_per_A": cp2k.get("cp2k_max_force_ev_per_A", math.nan),
                "cp2k_warning_count": cp2k.get("cp2k_warning_count", math.nan),
                "cp2k_scf_failures": cp2k.get("cp2k_scf_failures", math.nan),
                "cp2k_label_source": cp2k.get("cp2k_label_source", ""),
                "screening_600K_d_cm2_s": screening_600["li_diffusion_last_half_cm2_per_s"] if screening_600 else math.nan,
                "screening_600K_log10D": screening_600["log10_li_diffusion_last_half_cm2_per_s"] if screening_600 else math.nan,
                "screening_700K_d_cm2_s": screening_700["li_diffusion_last_half_cm2_per_s"] if screening_700 else math.nan,
                "screening_700K_log10D": screening_700["log10_li_diffusion_last_half_cm2_per_s"] if screening_700 else math.nan,
                "formal_600K_d_cm2_s": formal_600["li_diffusion_last_half_cm2_per_s"] if formal_600 else math.nan,
                "formal_700K_d_cm2_s": formal_700["li_diffusion_last_half_cm2_per_s"] if formal_700 else math.nan,
                "formal_800K_d_cm2_s": formal_800["li_diffusion_last_half_cm2_per_s"] if formal_800 else math.nan,
                "has_formal_transport": 1 if conductivity else 0,
                "formal_activation_energy_eV": safe_float(conductivity.get("activation_energy_eV")),
                "formal_d_tracer_300K_cm2_s": safe_float(conductivity.get("d_tracer_300K_cm2_s")),
                "formal_sigma_ne_upper_300K_mS_cm": safe_float(conductivity.get("sigma_ne_upper_300K_mS_cm")),
            }
        )

    return output_rows


def build_training_table(
    structure_features: list[dict[str, object]],
    transport_rows: list[dict[str, object]],
    structure_label_summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    feature_lookup = {row["structure_id"]: row for row in structure_features}
    summary_lookup = {row["structure_id"]: row for row in structure_label_summary}
    output_rows: list[dict[str, object]] = []

    for label_row in transport_rows:
        structure_id = str(label_row["structure_id"])
        merged = {}
        merged.update(feature_lookup[structure_id])
        merged.update(summary_lookup.get(structure_id, {}))
        merged.update(label_row)
        output_rows.append(merged)

    return output_rows


def build_task_manifest(training_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in training_rows:
        key = (str(row["label_tier"]), int(row["target_md_steps"]))
        grouped[key].add(str(row["structure_id"]))
        counts[key] += 1

    output_rows: list[dict[str, object]] = []
    for (tier, steps), structure_ids in sorted(grouped.items()):
        output_rows.append(
            {
                "task_id": f"{tier}_{steps}steps_log10D",
                "label_tier": tier,
                "target_md_steps": steps,
                "target_column": "log10_li_diffusion_last_half_cm2_per_s",
                "row_count": counts[(tier, steps)],
                "structure_count": len(structure_ids),
                "evaluation_recommendation": "leave-one-structure-out" if len(structure_ids) >= 5 else "fit-preview-only",
                "notes": "Use temperature as an explicit input feature; absolute conductivity is not the primary ML target at this stage.",
            }
        )
    return output_rows


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "07_ml_pipeline" / "01_datasets"

    structure_features = build_structure_features(project_root)
    cp2k_labels = load_cp2k_labels(project_root)
    transport_rows = load_transport_run_rows(project_root)
    structure_label_summary = build_structure_label_summary(project_root, cp2k_labels, transport_rows)
    training_table = build_training_table(structure_features, transport_rows, structure_label_summary)
    task_manifest = build_task_manifest(training_table)

    write_csv_rows(dataset_dir / "structure_features.csv", structure_features)
    write_csv_rows(dataset_dir / "transport_run_labels.csv", transport_rows)
    write_csv_rows(dataset_dir / "structure_label_summary.csv", structure_label_summary)
    write_csv_rows(dataset_dir / "ml_training_table.csv", training_table)
    write_csv_rows(dataset_dir / "task_manifest.csv", task_manifest)

    print(f"[ml] structure_features={dataset_dir / 'structure_features.csv'}")
    print(f"[ml] transport_run_labels={dataset_dir / 'transport_run_labels.csv'}")
    print(f"[ml] structure_label_summary={dataset_dir / 'structure_label_summary.csv'}")
    print(f"[ml] ml_training_table={dataset_dir / 'ml_training_table.csv'}")
    print(f"[ml] task_manifest={dataset_dir / 'task_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
