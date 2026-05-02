"""
compute_rdf.py — 径向分布函数分析 (从 UMA 优化后结构)

用法:
    python 08_analysis/01_scripts/compute_rdf.py \
        --structures-root 06_cloud_vm_gpu_bundle/04_runs/prepared \
        --fallback-root 06_cloud_vm_gpu_bundle/01_inputs \
        --output-dir 08_analysis/02_results

输出:
    - rdf_data.csv    各原子对的 g(r) 数据
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read


PAIRS = [("Li", "S"), ("Li", "Cl"), ("Li", "Li"), ("P", "S")]
R_MAX = 8.0   # Å
DR = 0.05     # Å bin width
NBINS = int(R_MAX / DR)


def compute_rdf_pair(atoms, elem_a, elem_b, r_max=R_MAX, dr=DR):
    """Compute RDF g(r) between two element types."""
    symbols = atoms.get_chemical_symbols()
    idx_a = [i for i, s in enumerate(symbols) if s == elem_a]
    idx_b = [i for i, s in enumerate(symbols) if s == elem_b]
    if not idx_a or not idx_b:
        return None, None

    cell = atoms.cell.array
    volume = atoms.get_volume()
    positions = atoms.get_positions()
    nbins = int(r_max / dr)
    hist = np.zeros(nbins)

    same_type = (elem_a == elem_b)

    for i in idx_a:
        for j in idx_b:
            if same_type and j <= i:
                continue
            # Minimum image convention
            delta = positions[j] - positions[i]
            frac = np.linalg.solve(cell.T, delta)
            frac -= np.round(frac)
            cart = cell.T @ frac
            dist = np.linalg.norm(cart)
            if dist < r_max:
                bin_idx = int(dist / dr)
                if bin_idx < nbins:
                    hist[bin_idx] += 1

    # Normalize to g(r)
    n_a = len(idx_a)
    n_b = len(idx_b)
    if same_type:
        n_pairs = n_a * (n_a - 1) / 2
        rho = n_a / volume  # number density of same type
    else:
        n_pairs = n_a * n_b
        rho = n_b / volume

    r_edges = np.arange(nbins + 1) * dr
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    shell_vol = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

    if same_type:
        g_r = hist / (n_a * rho * shell_vol) if rho > 0 else hist * 0
    else:
        g_r = hist / (n_a * rho * shell_vol) if rho > 0 else hist * 0

    return r_centers, g_r


def find_structure_file(sid, prepared_root, fallback_root):
    """Find the best available structure file for a given structure ID."""
    # Priority: prepared CONTCAR > prepared POSCAR.start > input POSCAR.start > input POSCAR.original
    candidates = [
        prepared_root / sid / "CONTCAR",
        prepared_root / sid / "POSCAR.start",
        fallback_root / sid / "POSCAR.start",
        fallback_root / sid / "POSCAR.original",
        fallback_root / sid / "POSCAR",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--fallback-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover structures
    structure_ids = set()
    for d in args.structures_root.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            structure_ids.add(d.name)
    for d in args.fallback_root.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            structure_ids.add(d.name)

    rows = []
    for sid in sorted(structure_ids):
        struct_path = find_structure_file(sid, args.structures_root, args.fallback_root)
        if struct_path is None:
            print(f"  [SKIP] {sid}: no structure file found")
            continue

        try:
            atoms = read(struct_path)
        except Exception as e:
            print(f"  [WARN] {sid}: failed to read {struct_path}: {e}")
            continue

        print(f"  Computing RDF for {sid} ({len(atoms)} atoms) ...")

        for elem_a, elem_b in PAIRS:
            r_centers, g_r = compute_rdf_pair(atoms, elem_a, elem_b)
            if r_centers is None:
                continue
            for r, g in zip(r_centers, g_r):
                rows.append({
                    "structure_id": sid,
                    "pair": f"{elem_a}-{elem_b}",
                    "r_A": f"{r:.3f}",
                    "g_r": f"{g:.6f}",
                })

    out_path = args.output_dir / "rdf_data.csv"
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["structure_id", "pair", "r_A", "g_r"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n[OK] {out_path} ({len(rows)} data points)")

    print("[DONE] RDF analysis complete.")


if __name__ == "__main__":
    main()
