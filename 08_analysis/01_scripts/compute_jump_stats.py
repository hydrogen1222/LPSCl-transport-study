"""
compute_jump_stats.py — Li 离子跳跃事件统计

定义跳跃: 连续帧之间 Li 原子的位移超过阈值 (默认 1.5 Å)。
统计每个结构/温度的跳跃频率、平均跳跃距离和跳跃距离分布。

用法:
    python 08_analysis/01_scripts/compute_jump_stats.py \
        --md-root 06_cloud_vm_gpu_bundle/04_runs/md \
        --output-dir 08_analysis/02_results \
        --figures-dir 08_analysis/03_figures

输出:
    - jump_statistics.csv         跳跃频率汇总
    - jump_distance_dist.csv      跳跃距离分布
    - fig_jump_frequency.png      跳跃频率柱状图
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

RUN_PATTERN = re.compile(r"md_(?P<temp>\d+)K_(?P<steps>\d+)steps")
TIMESTEP_FS = 1.0
JUMP_THRESHOLD_A = 1.5  # Å — displacement between consecutive saved frames


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


def compute_jumps(cell, symbols, frames, threshold=JUMP_THRESHOLD_A,
                  total_steps=20000, save_interval=100):
    """Detect jump events for Li atoms between consecutive frames."""
    li_idx = np.array([i for i, s in enumerate(symbols) if s == "Li"])
    if len(li_idx) == 0 or len(frames) < 2:
        return None

    n_li = len(li_idx)
    n_frames = len(frames)
    frame_dt_ps = save_interval * TIMESTEP_FS / 1000.0  # time between saved frames

    jump_events = []  # list of (li_atom_idx, frame_idx, displacement)
    jump_distances = []

    for t in range(1, n_frames):
        delta_frac = frames[t][li_idx] - frames[t - 1][li_idx]
        # Minimum image convention
        delta_frac -= np.round(delta_frac)
        delta_cart = delta_frac @ cell
        distances = np.linalg.norm(delta_cart, axis=1)

        for local_i, (global_i, dist) in enumerate(zip(li_idx, distances)):
            if dist >= threshold:
                jump_events.append((global_i, t, dist))
                jump_distances.append(dist)

    total_time_ps = (n_frames - 1) * frame_dt_ps
    n_jumps = len(jump_events)
    jump_freq = n_jumps / (n_li * total_time_ps) if total_time_ps > 0 else 0.0

    # Identify unique Li atoms that jumped at least once
    jumping_li = len(set(e[0] for e in jump_events))

    return {
        "n_jumps": n_jumps,
        "n_li": n_li,
        "n_frames": n_frames,
        "total_time_ps": total_time_ps,
        "jump_freq_per_li_per_ps": jump_freq,
        "jumping_li_fraction": jumping_li / n_li if n_li > 0 else 0.0,
        "mean_jump_distance_A": float(np.mean(jump_distances)) if jump_distances else 0.0,
        "std_jump_distance_A": float(np.std(jump_distances)) if jump_distances else 0.0,
        "jump_distances": jump_distances,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=JUMP_THRESHOLD_A,
                        help="Jump displacement threshold in Å (default 1.5)")
    parser.add_argument("--run-filter", type=str,
                        default=r"^md_\d+K_20000steps$",
                        help="Regex filter for run_name directories")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_filter_re = re.compile(args.run_filter)

    # Discover all runs
    runs = []
    for xdatcar in sorted(args.md_root.rglob("XDATCAR")):
        run_dir = xdatcar.parent
        m = RUN_PATTERN.search(run_dir.name)
        if not m:
            continue
        parts = xdatcar.relative_to(args.md_root).parts
        sid = parts[0] if len(parts) >= 2 else run_dir.parent.name
        run_name = parts[1] if len(parts) >= 3 else run_dir.name
        if not run_filter_re.match(run_name):
            continue
        runs.append({
            "structure_id": sid,
            "temperature_K": int(m.group("temp")),
            "target_steps": int(m.group("steps")),
            "run_name": run_name,
            "xdatcar_path": xdatcar,
        })

    print(f"[INFO] Found {len(runs)} runs (filter: {args.run_filter})")

    stats_rows = []
    dist_rows = []

    for run in runs:
        sid = run["structure_id"]
        temp = run["temperature_K"]
        print(f"  Processing {sid} / {temp} K ...")

        try:
            cell, symbols, frames = parse_xdatcar(run["xdatcar_path"])
        except Exception as e:
            print(f"    [WARN] Failed to parse: {e}")
            continue

        result = compute_jumps(cell, symbols, frames, threshold=args.threshold,
                               total_steps=run["target_steps"])
        if result is None:
            continue

        stats_rows.append({
            "structure_id": sid,
            "temperature_K": temp,
            "n_jumps": result["n_jumps"],
            "n_li": result["n_li"],
            "total_time_ps": f"{result['total_time_ps']:.1f}",
            "jump_freq_per_li_per_ps": f"{result['jump_freq_per_li_per_ps']:.6f}",
            "jumping_li_fraction": f"{result['jumping_li_fraction']:.4f}",
            "mean_jump_distance_A": f"{result['mean_jump_distance_A']:.4f}",
            "std_jump_distance_A": f"{result['std_jump_distance_A']:.4f}",
        })

        # Binned distance distribution
        if result["jump_distances"]:
            dists = np.array(result["jump_distances"])
            bins = np.arange(args.threshold, min(dists.max() + 0.5, 8.0), 0.25)
            if len(bins) > 1:
                hist, edges = np.histogram(dists, bins=bins)
                for k in range(len(hist)):
                    dist_rows.append({
                        "structure_id": sid,
                        "temperature_K": temp,
                        "distance_bin_A": f"{(edges[k]+edges[k+1])/2:.3f}",
                        "count": int(hist[k]),
                    })

        print(f"    {result['n_jumps']} jumps, freq={result['jump_freq_per_li_per_ps']:.4f} /Li/ps")

    # Write CSV
    stats_path = args.output_dir / "jump_statistics.csv"
    if stats_rows:
        with open(stats_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            w.writeheader()
            w.writerows(stats_rows)
        print(f"\n[OK] {stats_path} ({len(stats_rows)} rows)")

    dist_path = args.output_dir / "jump_distance_dist.csv"
    if dist_rows:
        with open(dist_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
            w.writeheader()
            w.writerows(dist_rows)
        print(f"[OK] {dist_path} ({len(dist_rows)} rows)")

    # Plot jump frequency bar chart
    fig_dir = args.figures_dir or args.output_dir
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use 600 K data for the main figure
        data_600 = [r for r in stats_rows if int(r["temperature_K"]) == 600]
        if data_600:
            data_600.sort(key=lambda r: float(r["jump_freq_per_li_per_ps"]))
            sids = [r["structure_id"].replace("gb_Sigma3_", "gb_") for r in data_600]
            freqs = [float(r["jump_freq_per_li_per_ps"]) for r in data_600]

            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            colors = []
            for r in data_600:
                sid = r["structure_id"]
                if "vac" in sid and "gb" in sid:
                    colors.append("#e74c3c")
                elif "gb" in sid:
                    colors.append("#3498db")
                else:
                    colors.append("#2ecc71")

            bars = ax.barh(range(len(sids)), freqs, color=colors, edgecolor="white", height=0.7)
            ax.set_yticks(range(len(sids)))
            ax.set_yticklabels(sids, fontsize=9)
            ax.set_xlabel("Jump frequency (jumps / Li / ps)", fontsize=11)
            ax.set_title(f"Li Jump Frequency at 600 K (threshold = {args.threshold} Å)",
                         fontsize=12, fontweight="bold")

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#2ecc71", label="Bulk"),
                Patch(facecolor="#3498db", label="GB"),
                Patch(facecolor="#e74c3c", label="GB + Vacancy"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

            fig_path = fig_dir / "fig_jump_frequency.png"
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Figure saved: {fig_path}")

    except ImportError:
        print("[WARN] matplotlib not available, skipping figure")

    print("[DONE] Jump statistics analysis complete.")


if __name__ == "__main__":
    main()
