"""
plot_all_figures.py — 一键生成论文核心图

用法:
    python 08_analysis/01_scripts/plot_all_figures.py \
        --results-dir 08_analysis/02_results \
        --output-dir 08_analysis/03_figures

生成:
    - fig_msd_comparison.png     MSD(t) 多温度多结构对比
    - fig_arrhenius.png          Arrhenius plot: ln(D) vs 1000/T
    - fig_sigma_bar.png          σ_NE 柱状图
    - fig_rdf_comparison.png     RDF 对比: bulk vs GB vs GB+vac
    - fig_ea_bar.png             活化能 Ea 柱状图
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

# Lazy import matplotlib for better error messages
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---- Style ----
COLORS = {
    "bulk_ordered": "#2196F3",
    "bulk_Li_vac_c1_s1": "#64B5F6",
    "bulk_Li_vac_c2_s1": "#90CAF9",
    "gb_Sigma3_t1": "#FF5722",
    "gb_Sigma3_t2": "#FF7043",
    "gb_Sigma3_t3": "#E53935",
    "gb_Sigma3_t1_Li_vac_c1_s1": "#AB47BC",
    "gb_Sigma3_t2_Li_vac_c1_s2": "#CE93D8",
    "gb_Sigma3_t3_Li_vac_c1_s1": "#7B1FA2",
    "gb_Sigma3_t3_Li_vac_c1_s2": "#9C27B0",
    "gb_Sigma3_t3_Li_vac_c2_s2": "#BA68C8",
}
TEMP_MARKERS = {600: "o", 700: "s", 800: "^"}

SHORT_LABELS = {
    "bulk_ordered": "Bulk",
    "bulk_Li_vac_c1_s1": "Bulk+V₁",
    "bulk_Li_vac_c2_s1": "Bulk+V₂",
    "gb_Sigma3_t1": "GB-t1",
    "gb_Sigma3_t2": "GB-t2",
    "gb_Sigma3_t3": "GB-t3",
    "gb_Sigma3_t1_Li_vac_c1_s1": "GB-t1+V₁",
    "gb_Sigma3_t2_Li_vac_c1_s2": "GB-t2+V₁",
    "gb_Sigma3_t3_Li_vac_c1_s1": "GB-t3+V₁",
    "gb_Sigma3_t3_Li_vac_c1_s2": "GB-t3+V₁b",
    "gb_Sigma3_t3_Li_vac_c2_s2": "GB-t3+V₂",
}


def get_color(sid):
    return COLORS.get(sid, "#666666")


def get_label(sid):
    return SHORT_LABELS.get(sid, sid)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_msd(results_dir: Path, output_dir: Path):
    """Plot MSD(t) curves for all structures grouped by temperature."""
    rows = read_csv_rows(results_dir / "msd_curves.csv")
    if not rows:
        print("  [SKIP] No msd_curves.csv")
        return

    # Group by (structure_id, temperature)
    data = defaultdict(lambda: ([], []))
    temps_available = set()
    for r in rows:
        key = (r["structure_id"], int(r["temperature_K"]))
        data[key][0].append(float(r["time_ps"]))
        data[key][1].append(float(r["msd_A2"]))
        temps_available.add(int(r["temperature_K"]))

    temps_sorted = sorted(temps_available)
    n_temps = len(temps_sorted)
    fig, axes = plt.subplots(1, n_temps, figsize=(5 * n_temps, 4.5), sharey=True)
    if n_temps == 1:
        axes = [axes]

    sids = sorted({r["structure_id"] for r in rows})

    for ax, temp in zip(axes, temps_sorted):
        for sid in sids:
            key = (sid, temp)
            if key not in data:
                continue
            t_arr, msd_arr = data[key]
            ax.plot(t_arr, msd_arr, color=get_color(sid), linewidth=1.5,
                    label=get_label(sid), alpha=0.9)
        ax.set_xlabel("Time (ps)", fontsize=12)
        ax.set_title(f"{temp} K", fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("Li MSD (Å²)", fontsize=12)
    # Legend in last panel
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8,
               bbox_to_anchor=(1.18, 0.5))
    fig.suptitle("Li Mean Square Displacement", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / "fig_msd_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_arrhenius(results_dir: Path, output_dir: Path):
    """Plot ln(D) vs 1000/T with linear fits."""
    diff_rows = read_csv_rows(results_dir / "diffusion_table.csv")
    arr_rows = read_csv_rows(results_dir / "arrhenius_fit.csv")
    if not diff_rows:
        print("  [SKIP] No diffusion_table.csv")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Group diffusion data
    grouped = defaultdict(lambda: ([], []))
    for r in diff_rows:
        sid = r["structure_id"]
        grouped[sid][0].append(float(r["temperature_K"]))
        grouped[sid][1].append(float(r["d_tracer_cm2_s"]))

    # Ea lookup
    ea_map = {}
    for r in arr_rows:
        ea_map[r["structure_id"]] = float(r["Ea_eV"])

    for sid in sorted(grouped.keys()):
        temps, ds = grouped[sid]
        inv_t = [1000.0 / t for t in temps]
        ln_d = [np.log(d) for d in ds]
        color = get_color(sid)
        label = get_label(sid)
        ea_str = f" (Ea={ea_map[sid]:.2f})" if sid in ea_map else ""

        ax.scatter(inv_t, ln_d, color=color, s=50, zorder=5, marker="o")

        if len(inv_t) >= 2:
            x_fit = np.linspace(min(inv_t) - 0.05, max(inv_t) + 0.05, 50)
            coeffs = np.polyfit(inv_t, ln_d, 1)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, linewidth=1.5,
                    linestyle="--", alpha=0.8, label=f"{label}{ea_str}")

    ax.set_xlabel("1000/T (K⁻¹)", fontsize=12)
    ax.set_ylabel("ln(D_tracer) [cm²/s]", fontsize=12)
    ax.set_title("Arrhenius Plot — Li Tracer Diffusion", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    out = output_dir / "fig_arrhenius.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_sigma_bar(results_dir: Path, output_dir: Path):
    """Bar chart of σ_NE(300K)."""
    rows = read_csv_rows(results_dir / "arrhenius_fit.csv")
    rows = [r for r in rows if r.get("sigma_NE_upper_300K_mS_cm")]
    if not rows:
        print("  [SKIP] No arrhenius_fit.csv with sigma")
        return

    rows.sort(key=lambda r: float(r["sigma_NE_upper_300K_mS_cm"]))
    sids = [r["structure_id"] for r in rows]
    sigmas = [float(r["sigma_NE_upper_300K_mS_cm"]) for r in rows]
    labels = [get_label(s) for s in sids]
    colors = [get_color(s) for s in sids]

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.8), 4.5))
    bars = ax.bar(range(len(sids)), sigmas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("σ_NE(300 K) [mS/cm]", fontsize=11)
    ax.set_title("Nernst-Einstein Conductivity Estimate", fontsize=13, fontweight="bold")

    # Add value labels on bars
    for bar, val in zip(bars, sigmas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = output_dir / "fig_sigma_bar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_ea_bar(results_dir: Path, output_dir: Path):
    """Bar chart of activation energies."""
    rows = read_csv_rows(results_dir / "arrhenius_fit.csv")
    if not rows:
        return

    rows.sort(key=lambda r: float(r["Ea_eV"]), reverse=True)
    sids = [r["structure_id"] for r in rows]
    eas = [float(r["Ea_eV"]) for r in rows]
    labels = [get_label(s) for s in sids]
    colors = [get_color(s) for s in sids]

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.8), 4.5))
    bars = ax.bar(range(len(sids)), eas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Activation Energy Ea (eV)", fontsize=11)
    ax.set_title("Migration Activation Energy", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, eas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = output_dir / "fig_ea_bar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_rdf(results_dir: Path, output_dir: Path):
    """RDF comparison: 2×2 panels for Li-S, Li-Cl, Li-Li, P-S,
    each showing 3 representative structures (bulk / GB / GB+vac)."""
    rows = read_csv_rows(results_dir / "rdf_data.csv")
    if not rows:
        print("  [SKIP] No rdf_data.csv")
        return

    # Representative structures for the comparison figure
    focus = [
        ("bulk_ordered", "Perfect Bulk", "#2196F3"),
        ("gb_Sigma3_t1", "Σ3 GB (t1)", "#FF5722"),
        ("gb_Sigma3_t1_Li_vac_c1_s1", "Σ3 GB + 1 Vac", "#AB47BC"),
    ]
    pairs_to_plot = [("Li", "S"), ("Li", "Cl"), ("Li", "Li"), ("P", "S")]
    pair_titles = {"Li-S": "Li–S RDF", "Li-Cl": "Li–Cl RDF",
                   "Li-Li": "Li–Li RDF", "P-S": "P–S RDF"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (ea, eb) in zip(axes_flat, pairs_to_plot):
        pair_label = f"{ea}-{eb}"
        for sid, label, color in focus:
            subset = [r for r in rows if r["structure_id"] == sid and r["pair"] == pair_label]
            if not subset:
                continue
            r_arr = [float(r["r_A"]) for r in subset]
            g_arr = [float(r["g_r"]) for r in subset]
            ax.plot(r_arr, g_arr, color=color, linewidth=1.5,
                    label=label, alpha=0.9)

        ax.set_xlabel("r (Å)", fontsize=11)
        ax.set_ylabel("g(r)", fontsize=11)
        ax.set_title(pair_titles.get(pair_label, pair_label),
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(1.5, 7.0)
        ax.tick_params(labelsize=10)

    fig.suptitle("Radial Distribution Functions — Li₆PS₅Cl",
                 fontsize=14, fontweight="bold")

    out = output_dir / "fig_rdf_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


def main():
    if not HAS_MPL:
        print("[ERROR] matplotlib not installed. Run: uv pip install matplotlib")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "figure.dpi": 150,
        "axes.linewidth": 1.2,
    })

    print("=== Generating thesis figures ===\n")
    plot_msd(args.results_dir, args.output_dir)
    plot_arrhenius(args.results_dir, args.output_dir)
    plot_sigma_bar(args.results_dir, args.output_dir)
    plot_ea_bar(args.results_dir, args.output_dir)
    plot_rdf(args.results_dir, args.output_dir)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
