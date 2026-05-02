"""
compute_msd_all.py — 统一计算所有 formal MD 轨迹的 MSD 和扩散系数

用法:
    python 08_analysis/01_scripts/compute_msd_all.py \
        --md-root 06_cloud_vm_gpu_bundle/04_runs/md \
        --output-dir 08_analysis/02_results

输出:
    - msd_curves.csv       每条轨迹的 MSD(t) 时间序列
    - diffusion_table.csv  每条轨迹的 D_tracer 汇总
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

RUN_PATTERN = re.compile(r"md_(?P<temp>\d+)K_(?P<steps>\d+)steps")
TIMESTEP_FS = 1.0  # fs


def parse_xdatcar(path: Path):
    """Parse XDATCAR and return cell, symbols, list of fractional-coordinate frames."""
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


def compute_li_msd(cell, symbols, frames, total_steps):
    """Compute Li MSD(t), D_tracer from last-half linear fit, and block-averaged error."""
    li_idx = np.array([i for i, s in enumerate(symbols) if s == "Li"])
    if len(li_idx) == 0 or len(frames) < 3:
        return None

    frac = np.stack([f[li_idx] for f in frames])
    # Unwrap PBC
    unwrapped = np.empty_like(frac)
    unwrapped[0] = frac[0]
    for t in range(1, len(frac)):
        delta = frac[t] - frac[t - 1]
        delta -= np.round(delta)
        unwrapped[t] = unwrapped[t - 1] + delta

    cart = np.einsum("tni,ij->tnj", unwrapped, cell)
    disp = cart - cart[0]
    sq = np.sum(disp ** 2, axis=2)  # (nframes, nLi)
    msd_mean = np.mean(sq, axis=1)  # (nframes,)

    nframes = len(frames)
    save_interval = max(1, total_steps // (nframes - 1)) if nframes > 1 else 1
    times_ps = np.arange(nframes) * save_interval * TIMESTEP_FS / 1000.0

    # D_tracer from last-half fit
    fit_start = nframes // 2
    if nframes - fit_start < 3:
        return None

    slope, intercept = np.polyfit(times_ps[fit_start:], msd_mean[fit_start:], 1)
    d_tracer = slope / 6.0 * 1e-4  # Å²/ps → cm²/s

    # Block averaging for error estimate (split last half into 4 blocks)
    last_half_msd = msd_mean[fit_start:]
    last_half_t = times_ps[fit_start:]
    n_blocks = 4
    block_size = len(last_half_msd) // n_blocks
    block_ds = []
    for b in range(n_blocks):
        s = b * block_size
        e = s + block_size
        if e > len(last_half_msd):
            break
        sl, _ = np.polyfit(last_half_t[s:e], last_half_msd[s:e], 1)
        block_ds.append(sl / 6.0 * 1e-4)
    d_err = np.std(block_ds) / np.sqrt(len(block_ds)) if len(block_ds) > 1 else 0.0

    return {
        "times_ps": times_ps,
        "msd_mean": msd_mean,
        "d_tracer_cm2_s": d_tracer,
        "d_tracer_err_cm2_s": d_err,
        "li_count": len(li_idx),
        "nframes": nframes,
        "save_interval": save_interval,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-steps", type=int, default=10000,
                        help="Only process runs with >= this many steps")
    parser.add_argument("--run-filter", type=str,
                        default=r"^md_\d+K_20000steps$",
                        help="Regex to match run_name (directory). "
                             "Default: only formal 20ps runs (excludes seed/50ps)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_filter_re = re.compile(args.run_filter)

    # Collect all runs
    runs = []
    for xdatcar in sorted(args.md_root.rglob("XDATCAR")):
        run_dir = xdatcar.parent
        m = RUN_PATTERN.search(run_dir.name)
        if not m:
            continue
        target_steps = int(m.group("steps"))
        if target_steps < args.min_steps:
            continue

        # Determine the run_name: first dir level under structure_id
        parts = xdatcar.relative_to(args.md_root).parts
        if len(parts) >= 3:
            sid = parts[0]
            run_name = parts[1]
        elif len(parts) >= 2:
            sid = parts[0]
            run_name = run_dir.name
        else:
            continue

        # Apply run name filter
        if not run_filter_re.match(run_name):
            continue

        runs.append({
            "structure_id": sid,
            "run_name": run_name,
            "target_temp_K": int(m.group("temp")),
            "target_steps": target_steps,
            "xdatcar_path": xdatcar,
        })

    print(f"[INFO] Found {len(runs)} runs (filter: {args.run_filter})")

    # Process each run
    msd_curves_file = args.output_dir / "msd_curves.csv"
    diff_table_file = args.output_dir / "diffusion_table.csv"

    diff_rows = []
    curve_rows = []

    for run in runs:
        print(f"  Processing {run['structure_id']} / {run['run_name']} ...")
        try:
            cell, symbols, frames = parse_xdatcar(run["xdatcar_path"])
        except Exception as e:
            print(f"    [WARN] Failed to parse XDATCAR: {e}")
            continue

        result = compute_li_msd(cell, symbols, frames, run["target_steps"])
        if result is None:
            print(f"    [WARN] Not enough data for MSD")
            continue

        # Diffusion table row
        diff_rows.append({
            "structure_id": run["structure_id"],
            "temperature_K": run["target_temp_K"],
            "run_name": run["run_name"],
            "li_count": result["li_count"],
            "nframes": result["nframes"],
            "trajectory_time_ps": float(result["times_ps"][-1]),
            "d_tracer_cm2_s": result["d_tracer_cm2_s"],
            "d_tracer_err_cm2_s": result["d_tracer_err_cm2_s"],
            "log10_d_tracer": float(np.log10(result["d_tracer_cm2_s"])),
            "msd_last_A2": float(result["msd_mean"][-1]),
        })

        # MSD curve rows (subsample to max 200 points for manageability)
        indices = np.linspace(0, len(result["times_ps"]) - 1,
                              min(200, len(result["times_ps"])), dtype=int)
        for idx in indices:
            curve_rows.append({
                "structure_id": run["structure_id"],
                "temperature_K": run["target_temp_K"],
                "time_ps": float(result["times_ps"][idx]),
                "msd_A2": float(result["msd_mean"][idx]),
            })

    # Write diffusion table
    if diff_rows:
        diff_rows.sort(key=lambda r: (r["structure_id"], r["temperature_K"]))
        with open(diff_table_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys()))
            w.writeheader()
            w.writerows(diff_rows)
        print(f"[OK] Diffusion table: {diff_table_file} ({len(diff_rows)} rows)")

    # Write MSD curves
    if curve_rows:
        with open(args.output_dir / "msd_curves.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
            w.writeheader()
            w.writerows(curve_rows)
        print(f"[OK] MSD curves: {msd_curves_file} ({len(curve_rows)} rows)")

    print(f"\n[DONE] {len(diff_rows)} diffusion coefficients extracted.")


if __name__ == "__main__":
    main()
