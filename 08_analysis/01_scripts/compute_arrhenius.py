"""
compute_arrhenius.py — Arrhenius 拟合 + 电导率估算

用法:
    python 08_analysis/01_scripts/compute_arrhenius.py \
        --diffusion-table 08_analysis/02_results/diffusion_table.csv \
        --volume-source 06_cloud_vm_gpu_bundle/04_runs/md \
        --output-dir 08_analysis/02_results

输出:
    - arrhenius_fit.csv    每个结构的 Ea, D0, D(300K), σ_NE
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

KB_EV = 8.617333262145e-5      # eV/K
KB_J = 1.380649e-23            # J/K
E_CHARGE = 1.602176634e-19     # C


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_volume(md_root: Path, sid: str) -> float | None:
    """Try to read cell volume from any CONTCAR under this structure's MD dir."""
    for contcar in sorted((md_root / sid).rglob("CONTCAR")):
        try:
            from ase.io import read as ase_read
            atoms = ase_read(contcar)
            return float(atoms.get_volume())
        except Exception:
            continue
    return None


def fit_arrhenius(temps_k, d_vals):
    """Fit ln(D) = ln(D0) - Ea/(kB*T). Returns Ea, D0, D(300K), R²."""
    inv_t = np.array([1.0 / t for t in temps_k])
    ln_d = np.log(np.array(d_vals))
    coeffs = np.polyfit(inv_t, ln_d, 1)
    slope, intercept = coeffs

    ea_ev = -slope * KB_EV
    d0 = math.exp(intercept)
    d_300 = d0 * math.exp(-ea_ev / (KB_EV * 300.0))

    # R²
    ln_d_pred = np.polyval(coeffs, inv_t)
    ss_res = np.sum((ln_d - ln_d_pred) ** 2)
    ss_tot = np.sum((ln_d - np.mean(ln_d)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ea_ev, d0, d_300, r2


def nernst_einstein(d_cm2_s, li_count, volume_a3, temp_k=300.0):
    """σ = n·e²·D / (kB·T), returned in S/m and mS/cm."""
    n_m3 = li_count / (volume_a3 * 1e-30)
    d_m2_s = d_cm2_s * 1e-4
    sigma_sm = n_m3 * E_CHARGE ** 2 * d_m2_s / (KB_J * temp_k)
    return sigma_sm, sigma_sm * 10.0  # S/m, mS/cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-table", type=Path, required=True)
    parser.add_argument("--volume-source", type=Path, default=None,
                        help="MD root dir to read CONTCAR for volume")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.diffusion_table)

    # Group by structure
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["structure_id"], []).append(r)

    results = []
    for sid, group in sorted(grouped.items()):
        temps = [float(r["temperature_K"]) for r in group]
        ds = [float(r["d_tracer_cm2_s"]) for r in group]
        li_count = int(group[0]["li_count"])

        if len(temps) < 2:
            print(f"  [SKIP] {sid}: only {len(temps)} temperature(s)")
            continue

        # Filter out any zero or negative D values
        valid = [(t, d) for t, d in zip(temps, ds) if d > 0]
        if len(valid) < 2:
            print(f"  [SKIP] {sid}: not enough valid D values")
            continue

        temps_v, ds_v = zip(*valid)
        ea, d0, d300, r2 = fit_arrhenius(temps_v, ds_v)

        # Volume for conductivity
        volume = None
        sigma_sm, sigma_mscm = None, None
        if args.volume_source and args.volume_source.exists():
            volume = get_volume(args.volume_source, sid)
        if volume and d300 > 0:
            sigma_sm, sigma_mscm = nernst_einstein(d300, li_count, volume)

        results.append({
            "structure_id": sid,
            "n_temperatures": len(temps_v),
            "temperatures_K": "/".join(str(int(t)) for t in temps_v),
            "Ea_eV": f"{ea:.4f}",
            "D0_cm2_s": f"{d0:.4e}",
            "D_tracer_300K_cm2_s": f"{d300:.4e}",
            "Arrhenius_R2": f"{r2:.4f}",
            "li_count": li_count,
            "cell_volume_A3": f"{volume:.2f}" if volume else "",
            # Keep legacy column names for downstream compatibility; values are σ_NE estimates.
            "sigma_NE_upper_300K_S_m": f"{sigma_sm:.4f}" if sigma_sm else "",
            "sigma_NE_upper_300K_mS_cm": f"{sigma_mscm:.4f}" if sigma_mscm else "",
        })

        if sigma_mscm:
            print(f"  {sid}: Ea={ea:.3f} eV, D(300K)={d300:.2e}, "
                  f"sigma_NE={sigma_mscm:.1f} mS/cm, R2={r2:.4f}")
        else:
            print(f"  {sid}: Ea={ea:.3f} eV, D(300K)={d300:.2e}, R2={r2:.4f}")

    # Write
    out_path = args.output_dir / "arrhenius_fit.csv"
    if results:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\n[OK] {out_path} ({len(results)} structures)")

    print(f"[DONE] Arrhenius fitting complete.")


if __name__ == "__main__":
    main()
