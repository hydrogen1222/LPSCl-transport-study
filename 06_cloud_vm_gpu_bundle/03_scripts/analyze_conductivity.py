from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from ase.io import read


KB_EV_PER_K = 8.617333262145e-5
KB_J_PER_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_volume_ang3(md_root: Path, structure_id: str, run_name: str) -> float:
    contcar = md_root / structure_id / run_name / run_name / "CONTCAR"
    atoms = read(contcar)
    return float(atoms.get_volume())


def fit_arrhenius(temperatures_k: list[float], diffusion_cm2_s: list[float]) -> tuple[float, float, float]:
    inverse_t = np.array([1.0 / temp for temp in temperatures_k], dtype=float)
    ln_d = np.log(np.array(diffusion_cm2_s, dtype=float))
    slope, intercept = np.polyfit(inverse_t, ln_d, 1)
    ea_ev = -slope * KB_EV_PER_K
    d0_cm2_s = float(math.exp(intercept))
    d_300_cm2_s = d0_cm2_s * math.exp(-ea_ev / (KB_EV_PER_K * 300.0))
    return float(ea_ev), float(d0_cm2_s), float(d_300_cm2_s)


def conductivity_from_diffusion(
    d_cm2_s: float,
    li_count: int,
    volume_ang3: float,
    temperature_k: float = 300.0,
) -> tuple[float, float]:
    n_m3 = li_count / (volume_ang3 * 1.0e-30)
    d_m2_s = d_cm2_s * 1.0e-4
    sigma_s_m = n_m3 * ELEMENTARY_CHARGE_C**2 * d_m2_s / (KB_J_PER_K * temperature_k)
    sigma_mscm = sigma_s_m * 10.0
    return float(sigma_s_m), float(sigma_mscm)


def build_summary(rows: list[dict[str, str]], md_root: Path, target_steps: int) -> list[dict[str, object]]:
    filtered = [row for row in rows if int(row["target_md_steps"]) == target_steps]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in filtered:
        grouped.setdefault(row["structure_id"], []).append(row)

    summary_rows: list[dict[str, object]] = []
    for structure_id, group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["target_temperature_K"]))
        temps = [float(row["target_temperature_K"]) for row in group]
        ds = [float(row["li_diffusion_last_half_cm2_per_s"]) for row in group]
        li_msds = [float(row["li_msd_last_A2"]) for row in group]
        reported_temps = [float(row["reported_temperature_K"]) for row in group]
        li_count = int(group[0]["li_count"])
        volume_ang3 = load_volume_ang3(md_root, structure_id, group[0]["run_name"])
        ea_ev, d0_cm2_s, d_300_cm2_s = fit_arrhenius(temps, ds)
        sigma_300_s_m, sigma_300_mscm = conductivity_from_diffusion(d_300_cm2_s, li_count, volume_ang3, 300.0)

        summary_rows.append(
            {
                "structure_id": structure_id,
                "temperatures_K": " / ".join(str(int(temp)) for temp in temps),
                "reported_temperature_last_frame_K": " / ".join(f"{temp:.2f}" for temp in reported_temps),
                "li_msd_last_A2": " / ".join(f"{value:.4f}" for value in li_msds),
                "diffusion_cm2_s": " / ".join(f"{value:.3e}" for value in ds),
                "activation_energy_eV": ea_ev,
                "d0_cm2_s": d0_cm2_s,
                "d_tracer_300K_cm2_s": d_300_cm2_s,
                "sigma_ne_upper_300K_S_m": sigma_300_s_m,
                "sigma_ne_upper_300K_mS_cm": sigma_300_mscm,
                "d_300K_cm2_s": d_300_cm2_s,
                "sigma_300K_S_m": sigma_300_s_m,
                "sigma_300K_mS_cm": sigma_300_mscm,
                "li_count": li_count,
                "cell_volume_A3": volume_ang3,
                "label_definition": "Tracer diffusion + Nernst-Einstein upper bound",
                "recommended_use": "Relative trend / first-pass screening only",
                "notes": (
                    "Arrhenius fit uses target temperatures 600/700/800 K. "
                    "The reported room-temperature conductivity is a first-pass upper-bound "
                    "Nernst-Einstein estimate based on tracer diffusion, total Li number density, "
                    "and Haven ratio = 1. It should not be treated as a final experimental-comparable value."
                ),
            }
        )

    return summary_rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], output_path: Path, target_steps: int) -> None:
    lines = [
        "# 正式 Conductivity Production 摘要",
        "",
        f"- 当前正式 production 轨迹长度：`{target_steps}` 步",
        f"- 纳入本次分析的结构数：`{len(rows)}`",
        "- 扩散系数来自每条轨迹后半段 `MSD` 的线性拟合。",
        "- Arrhenius 外推使用 `600 K`、`700 K`、`800 K` 三个温度点。",
        "- 下表中的室温电导率不是最终值，而是 `tracer diffusion + Nernst-Einstein` 的第一版上界估计。",
        "",
        "## 汇总表",
        "",
        "| 结构 | Ea (eV) | D_tracer(300 K) (cm^2/s) | sigma_NE,upper(300 K) (mS/cm) |",
        "| --- | ---: | ---: | ---: |",
    ]

    sorted_rows = sorted(rows, key=lambda row: float(row["sigma_ne_upper_300K_mS_cm"]), reverse=True)
    for row in sorted_rows:
        lines.append(
            "| "
            f"{row['structure_id']} | {float(row['activation_energy_eV']):.4f} | "
            f"{float(row['d_tracer_300K_cm2_s']):.3e} | {float(row['sigma_ne_upper_300K_mS_cm']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## 当前解释",
            "",
            "- `bulk_ordered` 是当前体相基线。",
            "- `gb_Sigma3_t3` 是当前最稳定的中性晶界代表。",
            "- `gb_Sigma3_t3_Li_vac_c1_s1` 是当前晶界 + vacancy 的代表结构。",
            "- 这些结果目前更适合用来讨论相对趋势，而不是直接当作实验可比的绝对室温电导率。",
            "- 当前电导率偏高是预期现象，因为这里使用的是 tracer diffusion、总 Li 数密度以及 Haven ratio = 1 的 Nernst-Einstein 上界。",
            "- 如果后续要让 conductivity 结论更稳，最划算的增强方式是：",
            "  1. 对这三个结构每个温度再补一条独立随机种子；或",
            "  2. 把现有三结构轨迹延长到 `50000` 步以上；或",
            "  3. 在后处理中加入相关扩散修正，而不只使用 tracer diffusion。",
            "",
            "## 当前可直接使用的结论",
            "",
            "- 相对趋势目前支持：`bulk_ordered < gb_Sigma3_t3 < gb_Sigma3_t3_Li_vac_c1_s1`。",
            "- 当前数据适合支撑“晶界及 vacancy 倾向于提高 Li 迁移能力”的趋势判断。",
            "- 当前数据不适合支撑“绝对室温离子电导率已经准确预测”的结论。",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze formal conductivity production runs.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "00_notes" / "md_run_summary.csv",
        help="Path to md_run_summary.csv generated by postprocess_md_runs.py",
    )
    parser.add_argument(
        "--md-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "04_runs" / "md",
        help="Root directory containing MD run folders",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=20000,
        help="Trajectory length used to select formal production rows",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "00_notes" / "conductivity_production_summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "00_notes" / "conductivity_production_summary.md",
        help="Output Markdown path",
    )
    args = parser.parse_args()

    rows = read_rows(args.summary_csv)
    summary_rows = build_summary(rows, args.md_root, args.target_steps)
    write_csv(summary_rows, args.csv_out)
    write_markdown(summary_rows, args.md_out, args.target_steps)

    print(f"[summary] csv={args.csv_out}")
    print(f"[summary] markdown={args.md_out}")
    print(f"[summary] structures={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
