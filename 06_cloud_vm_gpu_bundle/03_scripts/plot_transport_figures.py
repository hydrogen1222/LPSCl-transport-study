from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def formal_structures() -> list[str]:
    return ["bulk_ordered", "gb_Sigma3_t3", "gb_Sigma3_t3_Li_vac_c1_s1"]


def structure_label(structure_id: str) -> str:
    labels = {
        "bulk_ordered": "Bulk ordered",
        "gb_Sigma3_t3": "GB Sigma3 t3",
        "gb_Sigma3_t3_Li_vac_c1_s1": "GB t3 + Li vac",
    }
    return labels.get(structure_id, structure_id)


def sigma_upper_value(row: dict[str, str]) -> float:
    if "sigma_ne_upper_300K_mS_cm" in row and row["sigma_ne_upper_300K_mS_cm"]:
        return float(row["sigma_ne_upper_300K_mS_cm"])
    return float(row["sigma_300K_mS_cm"])


def compute_msd_series(postprocess_module, xdatcar_path: Path, total_steps: int, timestep_fs: float = 1.0):
    cell, symbols, frames = postprocess_module.parse_xdatcar(xdatcar_path)
    li_indices = np.array([i for i, symbol in enumerate(symbols) if symbol == "Li"], dtype=int)
    frac = np.stack([frame[li_indices] for frame in frames], axis=0)
    unwrapped = np.empty_like(frac)
    unwrapped[0] = frac[0]
    for idx in range(1, len(frac)):
        delta = frac[idx] - frac[idx - 1]
        delta -= np.round(delta)
        unwrapped[idx] = unwrapped[idx - 1] + delta

    cart = np.einsum("tni,ij->tnj", unwrapped, cell)
    displacement = cart - cart[0]
    squared = np.sum(displacement**2, axis=2)
    msd = np.mean(squared, axis=1)

    save_interval = postprocess_module.infer_save_interval_steps(total_steps, len(frames))
    times_ps = np.arange(len(frames), dtype=float) * save_interval * timestep_fs / 1000.0
    return times_ps, msd


def plot_msd_700k(md_root: Path, postprocess_module, output_path: Path) -> None:
    plt.figure(figsize=(7.5, 5.2))
    for structure_id in formal_structures():
        run_name = "md_700K_20000steps"
        xdatcar_path = md_root / structure_id / run_name / run_name / "XDATCAR"
        times_ps, msd = compute_msd_series(postprocess_module, xdatcar_path, total_steps=20000)
        plt.plot(times_ps, msd, linewidth=2.0, label=structure_label(structure_id))

    plt.xlabel("Time (ps)")
    plt.ylabel(r"Li MSD ($\mathrm{\AA}^2$)")
    plt.title("Li MSD comparison at 700 K")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_arrhenius(conductivity_rows: list[dict[str, str]], output_path: Path) -> None:
    plt.figure(figsize=(7.5, 5.2))
    colors = {
        "bulk_ordered": "#1f77b4",
        "gb_Sigma3_t3": "#d62728",
        "gb_Sigma3_t3_Li_vac_c1_s1": "#2ca02c",
    }
    for row in conductivity_rows:
        temps = [float(token.strip()) for token in row["temperatures_K"].split("/")]
        diffs = [float(token.strip()) for token in row["diffusion_cm2_s"].split("/")]
        x = 1000.0 / np.array(temps)
        y = np.log(np.array(diffs))
        coeff = np.polyfit(x, y, 1)
        xfit = np.linspace(min(x), max(x), 100)
        yfit = coeff[0] * xfit + coeff[1]
        color = colors.get(row["structure_id"], None)
        label = structure_label(row["structure_id"])
        plt.scatter(x, y, s=32, color=color, label=label)
        plt.plot(xfit, yfit, linewidth=1.8, color=color)

    plt.xlabel(r"$1000 / T$ (K$^{-1}$)")
    plt.ylabel(r"$\ln D_{\mathrm{tracer}}$ (cm$^2$ s$^{-1}$)")
    plt.title("Arrhenius comparison from tracer diffusion")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_sigma_bar(conductivity_rows: list[dict[str, str]], output_path: Path) -> None:
    ordered = sorted(conductivity_rows, key=sigma_upper_value)
    labels = [structure_label(row["structure_id"]) for row in ordered]
    values = [sigma_upper_value(row) for row in ordered]

    plt.figure(figsize=(7.5, 5.0))
    bars = plt.bar(labels, values, color=["#1f77b4", "#d62728", "#2ca02c"])
    plt.ylabel(r"$\sigma_{\mathrm{NE,upper}}$ at 300 K (mS/cm)")
    plt.title("Room-temperature conductivity upper-bound estimate")
    plt.xticks(rotation=10)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate transport figures from formal production outputs.")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Bundle root directory",
    )
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    notes_dir = bundle_root / "00_notes"
    figures_dir = notes_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    conductivity_rows = read_csv(notes_dir / "conductivity_production_summary.csv")

    postprocess_module = load_module(bundle_root / "03_scripts" / "postprocess_md_runs.py", "postprocess_md_runs")
    md_root = bundle_root / "04_runs" / "md"

    plot_msd_700k(md_root, postprocess_module, figures_dir / "01_msd_700K_comparison.png")
    plot_arrhenius(conductivity_rows, figures_dir / "02_arrhenius_compare.png")
    plot_sigma_bar(conductivity_rows, figures_dir / "03_sigma_300K_compare.png")

    print(f"[figures] output_dir={figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
