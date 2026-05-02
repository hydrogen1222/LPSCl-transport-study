"""
compute_vdos.py — 振动态密度 (VDOS) 分析

从 XDATCAR 轨迹中通过数值差分得到速度，
计算 Li 离子速度自相关函数 (VACF) 并做 FFT 得到振动态密度谱。

用法:
    python 08_analysis/01_scripts/compute_vdos.py \
        --md-root 06_cloud_vm_gpu_bundle/04_runs/md \
        --output-dir 08_analysis/02_results \
        --figures-dir 08_analysis/03_figures

输出:
    - vdos_data.csv             频率-VDOS 数据
    - fig_vdos_comparison.png   VDOS 对比图
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

RUN_PATTERN = re.compile(r"md_(?P<temp>\d+)K_(?P<steps>\d+)steps")
TIMESTEP_FS = 1.0  # MD integration timestep


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


def compute_vdos(cell, symbols, frames, total_steps=20000, save_interval=100):
    """Compute Li VDOS from finite-difference velocities and VACF."""
    li_idx = np.array([i for i, s in enumerate(symbols) if s == "Li"])
    if len(li_idx) == 0 or len(frames) < 10:
        return None

    n_li = len(li_idx)
    n_frames = len(frames)
    dt_ps = save_interval * TIMESTEP_FS / 1000.0  # time between saved frames in ps

    # Unwrap PBC and convert to Cartesian
    frac = np.stack([f[li_idx] for f in frames])
    unwrapped = np.empty_like(frac)
    unwrapped[0] = frac[0]
    for t in range(1, n_frames):
        delta = frac[t] - frac[t - 1]
        delta -= np.round(delta)
        unwrapped[t] = unwrapped[t - 1] + delta
    cart = np.einsum("tni,ij->tnj", unwrapped, cell)

    # Finite-difference velocity: v(t) ≈ [r(t+1) - r(t-1)] / (2*dt)
    # At boundaries: forward/backward difference
    vel = np.zeros_like(cart)
    vel[0] = (cart[1] - cart[0]) / dt_ps
    vel[-1] = (cart[-1] - cart[-2]) / dt_ps
    for t in range(1, n_frames - 1):
        vel[t] = (cart[t + 1] - cart[t - 1]) / (2.0 * dt_ps)

    # Velocity autocorrelation function (VACF)
    # C(τ) = <v(t)·v(t+τ)> averaged over all Li atoms and time origins
    max_lag = n_frames // 2
    vacf = np.zeros(max_lag)
    for lag in range(max_lag):
        n_origins = n_frames - lag
        dot_products = np.sum(vel[:n_origins] * vel[lag:lag + n_origins], axis=2)
        vacf[lag] = np.mean(dot_products)

    # Normalize
    vacf_norm = vacf / vacf[0] if vacf[0] != 0 else vacf

    # FFT to get VDOS
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(vacf_norm))
    vacf_windowed = vacf_norm * window

    fft_result = np.fft.rfft(vacf_windowed)
    vdos = np.abs(fft_result)

    # Frequency axis (in THz)
    freq_thz = np.fft.rfftfreq(len(vacf_windowed), d=dt_ps)

    return {
        "freq_thz": freq_thz,
        "vdos": vdos,
        "vacf": vacf_norm,
        "dt_ps": dt_ps,
        "n_li": n_li,
        "n_frames": n_frames,
    }


def find_xdatcar(md_root: Path, sid: str, temp: int = 600):
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
                        help="Temperature to analyze (default 600 K)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_sids = sorted([d.name for d in args.md_root.iterdir()
                       if d.is_dir() and not d.name.startswith(".")])

    csv_rows = []
    vdos_data = {}

    for sid in all_sids:
        xdc = find_xdatcar(args.md_root, sid, args.temp)
        if xdc is None:
            print(f"  [SKIP] {sid}: no XDATCAR at {args.temp} K")
            continue

        print(f"  Computing VDOS for {sid} @ {args.temp} K ...")
        try:
            cell, symbols, frames = parse_xdatcar(xdc)
        except Exception as e:
            print(f"    [WARN] Failed to parse: {e}")
            continue

        result = compute_vdos(cell, symbols, frames)
        if result is None:
            print(f"    [WARN] Not enough data")
            continue

        vdos_data[sid] = result
        print(f"    {result['n_frames']} frames, {result['n_li']} Li atoms")

        # Only save frequencies up to 50 THz (above is noise)
        mask = result["freq_thz"] <= 50.0
        for freq, intensity in zip(result["freq_thz"][mask], result["vdos"][mask]):
            csv_rows.append({
                "structure_id": sid,
                "frequency_THz": f"{freq:.4f}",
                "vdos": f"{intensity:.6f}",
            })

    # Write CSV
    csv_path = args.output_dir / "vdos_data.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["structure_id", "frequency_THz", "vdos"])
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n[OK] {csv_path} ({len(csv_rows)} rows)")

    # Plot comparison figure
    fig_dir = args.figures_dir or args.output_dir
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        representatives = {
            "bulk_ordered": ("Perfect Bulk", "#2ecc71", "-"),
            "gb_Sigma3_t1": ("Σ3 GB (t1)", "#3498db", "-"),
            "gb_Sigma3_t1_Li_vac_c1_s1": ("Σ3 GB + 1 Vac", "#e74c3c", "-"),
            "gb_Sigma3_t3_Li_vac_c2_s2": ("Σ3 GB + 2 Vac (neg.)", "#9b59b6", "--"),
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

        # Left: VDOS
        for sid, (label, color, ls) in representatives.items():
            if sid not in vdos_data:
                continue
            r = vdos_data[sid]
            mask = r["freq_thz"] <= 40.0
            # Normalize each VDOS for shape comparison
            vdos_norm = r["vdos"][mask] / r["vdos"][mask].max() if r["vdos"][mask].max() > 0 else r["vdos"][mask]
            ax1.plot(r["freq_thz"][mask], vdos_norm, color=color, ls=ls,
                     lw=1.5, label=label, alpha=0.85)

        ax1.set_xlabel("Frequency (THz)", fontsize=11)
        ax1.set_ylabel("Normalized VDOS", fontsize=11)
        ax1.set_title(f"Li Vibrational Density of States ({args.temp} K)", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9, loc="upper right")
        ax1.set_xlim(0, 40)

        # Right: VACF
        for sid, (label, color, ls) in representatives.items():
            if sid not in vdos_data:
                continue
            r = vdos_data[sid]
            t_vacf = np.arange(len(r["vacf"])) * r["dt_ps"]
            ax2.plot(t_vacf, r["vacf"], color=color, ls=ls, lw=1.5, label=label, alpha=0.85)

        ax2.set_xlabel("Lag time (ps)", fontsize=11)
        ax2.set_ylabel("VACF C(τ)/C(0)", fontsize=11)
        ax2.set_title(f"Li Velocity Autocorrelation ({args.temp} K)", fontsize=12, fontweight="bold")
        ax2.axhline(0, color="gray", lw=0.5, ls=":")
        ax2.legend(fontsize=9, loc="upper right")

        fig_path = fig_dir / "fig_vdos_comparison.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Figure saved: {fig_path}")

    except ImportError:
        print("[WARN] matplotlib not available, skipping figure")

    print("[DONE] VDOS analysis complete.")


if __name__ == "__main__":
    main()
