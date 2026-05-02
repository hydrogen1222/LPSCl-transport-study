from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from ase.io import read


RUN_PATTERN = re.compile(r"md_(?P<temp>\d+)K_(?P<steps>\d+)steps")
TIMESTEP_FS = 1.0


def parse_xdatcar(path: Path) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"XDATCAR too short: {path}")

    scale = float(lines[1].strip())
    cell = np.array([[float(value) for value in lines[i].split()] for i in range(2, 5)], dtype=float) * scale
    species = lines[5].split()
    counts = [int(token) for token in lines[6].split()]
    natoms = sum(counts)
    symbols = [symbol for symbol, count in zip(species, counts) for _ in range(count)]

    payload = [line.strip() for line in lines[7:] if line.strip()]
    if payload and payload[0].lower() == "direct":
        payload = payload[1:]

    frames: list[np.ndarray] = []
    i = 0
    while i < len(payload):
        if payload[i].lower().startswith("direct configuration"):
            i += 1
        if i >= len(payload):
            break
        frame_lines = payload[i : i + natoms]
        if len(frame_lines) != natoms:
            raise ValueError(f"Incomplete XDATCAR frame in {path}")
        frame = np.array([[float(value) for value in line.split()[:3]] for line in frame_lines], dtype=float)
        frames.append(frame)
        i += natoms

    return cell, symbols, frames


def infer_save_interval_steps(total_steps: int, nframes: int) -> int | None:
    if total_steps <= 0 or nframes <= 0:
        return None
    candidates: list[int] = []
    for denom in (nframes, nframes - 1):
        if denom > 0 and total_steps % denom == 0:
            candidates.append(total_steps // denom)
    if candidates:
        return min(candidates)
    return max(1, round(total_steps / nframes))


def compute_li_msd_metrics(
    cell: np.ndarray,
    symbols: list[str],
    frames: list[np.ndarray],
    total_steps: int,
    timestep_fs: float = TIMESTEP_FS,
) -> dict[str, float | int | None]:
    li_indices = np.array([i for i, symbol in enumerate(symbols) if symbol == "Li"], dtype=int)
    if len(li_indices) == 0 or len(frames) < 2:
        return {
            "frames_saved": len(frames),
            "save_interval_steps": None,
            "trajectory_time_ps": None,
            "li_msd_last_A2": None,
            "li_rmsd_last_A": None,
            "li_msd_slope_last_half_A2_per_ps": None,
            "li_diffusion_last_half_cm2_per_s": None,
        }

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

    save_interval_steps = infer_save_interval_steps(total_steps, len(frames))
    if save_interval_steps is None:
        times_ps = np.arange(len(frames), dtype=float)
    else:
        times_ps = np.arange(len(frames), dtype=float) * save_interval_steps * timestep_fs / 1000.0

    slope = None
    diffusion = None
    fit_start = len(frames) // 2
    if len(frames) - fit_start >= 3:
        slope = float(np.polyfit(times_ps[fit_start:], msd[fit_start:], 1)[0])
        diffusion = slope / 6.0 * 1.0e-4

    return {
        "frames_saved": len(frames),
        "save_interval_steps": save_interval_steps,
        "trajectory_time_ps": float(times_ps[-1]) if len(times_ps) else None,
        "li_msd_last_A2": float(msd[-1]),
        "li_rmsd_last_A": float(math.sqrt(msd[-1])),
        "li_msd_slope_last_half_A2_per_ps": slope,
        "li_diffusion_last_half_cm2_per_s": diffusion,
    }


def compute_min_distance_metrics(path: Path) -> dict[str, float | str | None]:
    atoms = read(path)
    symbols = atoms.get_chemical_symbols()
    frac = atoms.get_scaled_positions()
    diff = frac[:, None, :] - frac[None, :, :]
    diff -= np.round(diff)
    cart = np.einsum("ijk,kl->ijl", diff, atoms.cell.array)
    dist = np.linalg.norm(cart, axis=2)
    np.fill_diagonal(dist, np.inf)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return {
        "min_pair_distance_A": float(dist[i, j]),
        "min_pair_label": f"{symbols[i]}-{symbols[j]}",
    }


def extract_row(run_dir: Path) -> dict[str, object]:
    structure_id = run_dir.parents[1].name
    run_name = run_dir.name
    match = RUN_PATTERN.fullmatch(run_name)
    target_temp = int(match.group("temp")) if match else None
    target_steps = int(match.group("steps")) if match else None

    json_path = run_dir / "uma_results.json"
    outcar_path = run_dir / "OUTCAR"
    contcar_path = run_dir / "CONTCAR"
    xdatcar_path = run_dir / "XDATCAR"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing uma_results.json: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    calc = data["calculation"]
    system = calc["system"]
    results = calc["results"]
    force_stats = results["force_statistics"]
    md = calc.get("md", {})
    timing = calc.get("timing", {})
    metadata = data.get("metadata", {})

    row: dict[str, object] = {
        "structure_id": structure_id,
        "run_name": run_name,
        "formula": system["formula"],
        "natoms": system["natoms"],
        "li_count": sum(1 for symbol in system["symbols"] if symbol == "Li"),
        "target_temperature_K": target_temp,
        "reported_temperature_K": md.get("temperature"),
        "target_md_steps": target_steps,
        "reported_md_steps": md.get("steps"),
        "ensemble": md.get("ensemble"),
        "calculation_time_s": timing.get("calculation_time_s"),
        "energy_eV": results.get("energy"),
        "energy_per_atom_eV": results.get("energy_per_atom"),
        "fmax_eV_per_A": force_stats.get("fmax"),
        "fmean_eV_per_A": force_stats.get("fmean"),
        "frms_eV_per_A": force_stats.get("frms"),
        "device": metadata.get("device"),
        "model_path": metadata.get("model_path"),
        "outcar_exists": outcar_path.exists(),
        "xdatcar_exists": xdatcar_path.exists(),
    }

    if contcar_path.exists():
        row.update(compute_min_distance_metrics(contcar_path))
    else:
        row["min_pair_distance_A"] = None
        row["min_pair_label"] = None

    if xdatcar_path.exists() and target_steps is not None:
        cell, symbols, frames = parse_xdatcar(xdatcar_path)
        row.update(compute_li_msd_metrics(cell, symbols, frames, target_steps))
    else:
        row.update(
            {
                "frames_saved": None,
                "save_interval_steps": None,
                "trajectory_time_ps": None,
                "li_msd_last_A2": None,
                "li_rmsd_last_A": None,
                "li_msd_slope_last_half_A2_per_ps": None,
                "li_diffusion_last_half_cm2_per_s": None,
            }
        )

    return row


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    total_runs = len(rows)
    structure_count = len({row["structure_id"] for row in rows})
    temps = sorted({int(row["target_temperature_K"]) for row in rows if row["target_temperature_K"] is not None})
    longest_ps = max((float(row["trajectory_time_ps"]) for row in rows if row["trajectory_time_ps"] is not None), default=0.0)

    provisional_rows = [
        row for row in rows if row["target_md_steps"] == 1000 and row["li_diffusion_last_half_cm2_per_s"] is not None
    ]
    provisional_rows.sort(key=lambda row: float(row["li_diffusion_last_half_cm2_per_s"]), reverse=True)

    lines = [
        "# 当前 MD Screening 总结",
        "",
        f"- 已完成 MD 运行数：`{total_runs}`",
        f"- 已覆盖结构数：`{structure_count}`",
        f"- 当前已覆盖温度：`{', '.join(f'{temp} K' for temp in temps)}`",
        f"- 当前最长轨迹长度：`{longest_ps:.2f} ps`",
        "",
        "## 当前结果如何理解",
        "",
        "- 当前 `1000` 步轨迹属于 screening，不是最终 conductivity 生产数据。",
        "- 在 `1.0 fs` 步长下，`1000` 步大约只有 `1.0 ps`；这足够做稳定性和短时间 Li 运动趋势判断，但不足以稳健给出室温电导率。",
        "- `600 K` 和 `700 K` 属于高温采样点，用来加速 Li 跳跃并为后续 Arrhenius 外推做准备；它们不是室温电导率本身。",
        "- 当前 workflow 仍然是预训练 UMA 推理 + MD，不是模型训练阶段。",
        "",
        "## 初步扩散排序",
        "",
        "| 结构 | 运行名 | 目标温度 (K) | 末帧 Li MSD (A^2) | 粗略 D (cm^2/s) | 最短原子对 (A) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in provisional_rows[:10]:
        lines.append(
            "| "
            f"{row['structure_id']} | {row['run_name']} | {int(row['target_temperature_K'])} | "
            f"{float(row['li_msd_last_A2']):.4f} | {float(row['li_diffusion_last_half_cm2_per_s']):.3e} | "
            f"{float(row['min_pair_distance_A']):.3f} ({row['min_pair_label']}) |"
        )

    lines.extend(
        [
            "",
            "- 当前 `1 ps` screening 中，`t1` 的短时间 Li 运动看起来比 `t3` 更强，但这仍然只是 screening 级别的观察。",
            "- 正式 production 仍然优先保留 `t3`，因为在已有静态验证里，`t3` 仍然是结构和能量上更稳妥的 GB 家族。",
            "",
            "## 下一步正式 production 建议",
            "",
            "正式 conductivity 生产，建议先保留这三个结构：",
            "",
            "- `bulk_ordered`",
            "- `gb_Sigma3_t3`",
            "- `gb_Sigma3_t3_Li_vac_c1_s1`",
            "",
            "推荐的第一轮 production 矩阵：",
            "",
            "- 温度：`600 K`, `700 K`, `800 K`",
            "- 单条轨迹长度：至少 `20000` 步，也就是 `20 ps`",
            "- 保存间隔：`100` 步",
            "- 晶胞保持固定，继续复用当前已准备好的起始结构",
            "",
            "之后再对 `ln(D)` 与 `1/T` 做 Arrhenius 拟合，外推到 `300 K`，并在明确说明 Nernst-Einstein 假设的前提下换算为电导率。",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess UMA MD runs and generate a screening summary.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "04_runs" / "md",
        help="Root directory containing MD run folders",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "00_notes" / "md_run_summary.csv",
        help="Output CSV summary path",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "00_notes" / "md_screening_summary.md",
        help="Output Markdown summary path",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for run_dir in sorted(args.root.rglob("md_*steps")):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "uma_results.json").exists():
            continue
        rows.append(extract_row(run_dir))

    rows.sort(key=lambda row: (str(row["structure_id"]), int(row["target_temperature_K"] or 0), int(row["target_md_steps"] or 0)))

    fieldnames = list(rows[0].keys()) if rows else []
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, args.md_out)

    print(f"[summary] csv={args.csv_out}")
    print(f"[summary] markdown={args.md_out}")
    print(f"[summary] runs={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
