"""
compute_li_density.py — Li 离子概率密度热力图

从 XDATCAR 轨迹中提取 Li 原子坐标，投影到 ab 平面，
生成 2D 概率密度热力图，对比 bulk / GB / GB+vacancy 的扩散通道。

用法:
    python 08_analysis/01_scripts/compute_li_density.py \
        --md-root 06_cloud_vm_gpu_bundle/04_runs/md \
        --output-dir 08_analysis/02_results \
        --figures-dir 08_analysis/03_figures

输出:
    - li_density_data.npz      密度矩阵（可复用）
    - fig_li_density_map.png   热力图（论文图）
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

RUN_PATTERN = re.compile(r"md_(?P<temp>\d+)K_(?P<steps>\d+)steps")
NBINS_2D = 500  # 2D histogram resolution


def parse_xdatcar(path: Path):
    """Parse XDATCAR → cell, symbols, list of fractional-coord frames."""
    lines = path.read_text(encoding="utf-8").splitlines()
    scale = float(lines[1].strip())
    cell = np.array([[float(v) for v in lines[i].split()] for i in range(2, 5)]) * scale
    species = lines[5].split()
    counts = [int(t) for t in lines[6].split()]
    natoms = sum(counts)
    symbols = [s for s, c in zip(species, counts) for _ in range(c)]

    payload = [l.strip() for l in lines[7:] if l.strip()]
    frames = []
    i = 0
    while i < len(payload):
        if payload[i].lower().startswith("direct"):
            i += 1
        if i >= len(payload):
            break
        block = payload[i:i + natoms]
        if len(block) != natoms:
            break
        frame = np.array([[float(v) for v in line.split()[:3]] for line in block])
        frames.append(frame)
        i += natoms
    return cell, symbols, frames


def compute_density_2d(cell, symbols, frames, projection="ab"):
    """Accumulate Li positions projected onto a 2D plane, return density grid."""
    li_idx = np.array([i for i, s in enumerate(symbols) if s == "Li"])
    if len(li_idx) == 0:
        return None, None, None

    # Collect all Li fractional coords across all frames
    all_frac = np.concatenate([f[li_idx] for f in frames], axis=0)
    # Wrap to [0, 1)
    all_frac = all_frac % 1.0
    # Convert to Cartesian
    all_cart = all_frac @ cell

    if projection == "ab":
        x, y = all_cart[:, 0], all_cart[:, 1]
        x_label, y_label = "a (Å)", "b (Å)"
        x_max, y_max = np.linalg.norm(cell[0]), np.linalg.norm(cell[1])
    elif projection == "ac":
        x, y = all_cart[:, 0], all_cart[:, 2]
        x_label, y_label = "a (Å)", "c (Å)"
        x_max, y_max = np.linalg.norm(cell[0]), np.linalg.norm(cell[2])
    else:
        x, y = all_cart[:, 1], all_cart[:, 2]
        x_label, y_label = "b (Å)", "c (Å)"
        x_max, y_max = np.linalg.norm(cell[1]), np.linalg.norm(cell[2])

    hist, xedges, yedges = np.histogram2d(x, y, bins=NBINS_2D,
                                           range=[[0, x_max], [0, y_max]])
    # Normalize to probability density
    area = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
    density = hist / (hist.sum() * area) if hist.sum() > 0 else hist

    return density, (xedges, yedges, x_label, y_label), {
        "n_frames": len(frames),
        "n_li": len(li_idx),
        "n_points": len(all_frac),
    }


def find_representative_xdatcar(md_root: Path, sid: str, temp: int = 600):
    """Find XDATCAR for a given structure at given temperature.
    Handles both flat and double-nested directory layouts."""
    sid_dir = md_root / sid
    if not sid_dir.exists():
        return None
    target = f"md_{temp}K_"
    for xdc in sorted(sid_dir.rglob("XDATCAR")):
        if target in str(xdc) and "20000steps" in str(xdc):
            return xdc
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--temp", type=int, default=600,
                        help="Temperature to use for density map (default 600 K)")
    parser.add_argument("--projection", default="ab",
                        choices=["ab", "ac", "bc"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.figures_dir:
        args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Find representative structures: one bulk, one GB, one GB+vac
    representatives = {
        "bulk_ordered": "Perfect Bulk",
        "gb_Sigma3_t1": "Σ3 GB (t1)",
        "gb_Sigma3_t1_Li_vac_c1_s1": "Σ3 GB + 1 Vacancy",
    }

    # Also process all available structures
    all_sids = sorted([d.name for d in args.md_root.iterdir()
                       if d.is_dir() and not d.name.startswith(".")])

    densities = {}
    for sid in all_sids:
        xdc = find_representative_xdatcar(args.md_root, sid, args.temp)
        if xdc is None:
            print(f"  [SKIP] {sid}: no XDATCAR at {args.temp} K")
            continue

        print(f"  Computing density for {sid} @ {args.temp} K ...")
        cell, symbols, frames = parse_xdatcar(xdc)
        density, edges, info = compute_density_2d(cell, symbols, frames, args.projection)
        if density is not None:
            densities[sid] = {
                "density": density,
                "edges": edges,
                "info": info,
            }
            print(f"    {info['n_frames']} frames × {info['n_li']} Li = {info['n_points']} points")

    # Save all density data
    save_dict = {}
    for sid, d in densities.items():
        save_dict[f"{sid}_density"] = d["density"]
        save_dict[f"{sid}_xedges"] = d["edges"][0]
        save_dict[f"{sid}_yedges"] = d["edges"][1]
    np.savez_compressed(args.output_dir / "li_density_data.npz", **save_dict)
    print(f"\n[OK] Saved density data for {len(densities)} structures")

    # Plot representative comparison figure
    fig_dir = args.figures_dir or args.output_dir
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # Select representatives that exist
        plot_sids = [(sid, label) for sid, label in representatives.items()
                     if sid in densities]
        if not plot_sids:
            # Fallback: use first 3 available
            plot_sids = [(sid, sid) for sid in list(densities.keys())[:3]]

        ncols = len(plot_sids)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5),
                                  constrained_layout=True)
        if ncols == 1:
            axes = [axes]

        vmax = max(d["density"].max() for d in densities.values())

        for ax, (sid, label) in zip(axes, plot_sids):
            d = densities[sid]
            xedges, yedges = d["edges"][0], d["edges"][1]
            im = ax.pcolormesh(xedges, yedges, d["density"].T,
                               cmap="hot", vmin=0, vmax=vmax * 0.8)
            ax.set_xlabel(d["edges"][2], fontsize=11)
            ax.set_ylabel(d["edges"][3], fontsize=11)
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.set_aspect("equal")

        cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="Li probability density")
        fig.suptitle(f"Li Probability Density ({args.temp} K, {args.projection} projection)",
                     fontsize=13, fontweight="bold")

        fig_path = fig_dir / "fig_li_density_map.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Figure saved: {fig_path}")

    except ImportError:
        print("[WARN] matplotlib not available, skipping figure generation")

    print("[DONE] Li density analysis complete.")


if __name__ == "__main__":
    main()
