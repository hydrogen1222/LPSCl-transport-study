from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = WORKFLOW_ROOT.parent / "00_notes"
MANIFEST_PATH = NOTES_DIR / "workflow_manifest.csv"
ORDER_PATH = NOTES_DIR / "cp2k_job_order.txt"
LOG_PATH = WORKFLOW_ROOT / "job_scripts" / "serial_submit_cp2k.log"
STATE_PATH = WORKFLOW_ROOT / "job_scripts" / "serial_submit_cp2k_state.json"

PREOPT_STAGE = "00_uma_relax"
CP2K_SINGLEPOINT_STAGE = "01_cp2k_singlepoint"
UMA_MD_STAGE = "02_uma_md_nvt"

PREOPT_BYPASS_MARKER = "SKIP_PREOPT"
PREOPT_BYPASS_MARKER_ALIASES = (PREOPT_BYPASS_MARKER, "SKIP_XTB")

DEFAULT_CP2K_SUBMIT_SCRIPT = "job_scripts/cp2k_singlepoint_large_mem.sh"
DEFAULT_PREOPT_SUBMIT_SCRIPT = "job_scripts/uma_preopt_cpu.sh"
DEFAULT_UMA_MD_SUBMIT_SCRIPT = "job_scripts/uma_md_cpu.sh"

PREOPT_CONTCAR_NAME = "CONTCAR"
PREOPT_POSCAR_NAME = "POSCAR"
CP2K_SINGLEPOINT_BASE_NAME = "1.inp.base"
UMA_MD_DONE_MARKER = "MD_DONE"

END_MARKER = "PROGRAM ENDED AT"
FAILURE_MARKERS = (
    "SCF run NOT converged",
    "exceeded requested execution time",
    "ABORT",
    "abort",
    "Out Of Memory",
    "oom-kill",
    "Segmentation fault",
    "SIGSEGV",
    "error",
)
UMA_HARD_FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "Segmentation fault",
    "SIGSEGV",
    "Out Of Memory",
    "oom-kill",
    "Killed",
)

MIN_DISTANCE_GLOBAL = 1.20
MIN_DISTANCE_THRESHOLDS = {
    ("Cl", "Cl"): 1.80,
    ("Cl", "Li"): 1.50,
    ("Cl", "P"): 1.70,
    ("Cl", "S"): 1.70,
    ("Li", "Li"): 1.20,
    ("Li", "P"): 1.50,
    ("Li", "S"): 1.35,
    ("P", "P"): 1.80,
    ("P", "S"): 1.70,
    ("S", "S"): 1.60,
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_text_tail(path: Path, max_bytes: int = 262144) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="ignore")


def read_latest_glob_text(stage_dir: Path, pattern: str) -> str:
    matches = sorted(stage_dir.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        return ""
    return read_text_tail(matches[-1])


def parse_job_id(stdout: str) -> str | None:
    cleaned = stdout.strip()
    if not cleaned:
        return None
    return cleaned.splitlines()[-1].split(";")[0].strip()


def load_stage_manifest() -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if ORDER_PATH.exists():
        structure_order = [
            line.strip()
            for line in ORDER_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        structure_order = []
        seen = set()
        for row in rows:
            structure_id = row["structure_id"]
            if structure_id not in seen:
                seen.add(structure_id)
                structure_order.append(structure_id)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        row.setdefault("enabled", "1")
        grouped[row["structure_id"]].append(row)
    for stages in grouped.values():
        stages.sort(key=lambda item: int(item["stage_order"]))
    return structure_order, grouped


def load_state(structure_order: list[str], grouped: dict[str, list[dict[str, str]]]) -> dict:
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except JSONDecodeError:
            state = {"structures": {}}
    else:
        state = {"structures": {}}

    state.setdefault("workflow_root", str(WORKFLOW_ROOT))
    state.setdefault("structures", {})
    state.setdefault("current_structure", None)
    state.setdefault("current_stage", None)
    state.setdefault("updated_at", now())

    for structure_id in structure_order:
        entry = state["structures"].setdefault(structure_id, {"stages": {}})
        entry.setdefault("stages", {})
        for row in grouped[structure_id]:
            stage_name = row["stage_name"]
            entry["stages"].setdefault(
                stage_name,
                {
                    "status": "pending",
                    "path": row["path"],
                    "run_type": row["run_type"],
                    "enabled": row.get("enabled", "1"),
                    "job_id": None,
                    "submitted_at": None,
                    "completed_at": None,
                    "stdout_file": None,
                    "stderr_file": None,
                    "artifact_file": None,
                },
            )
    return state


def save_state(state: dict) -> None:
    state["updated_at"] = now()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def output_has_failure_marker(output_text: str) -> bool:
    lowered = output_text.lower()
    for marker in FAILURE_MARKERS:
        marker_lower = marker.lower()
        if marker_lower == "error":
            if "error" in lowered and END_MARKER.lower() not in lowered:
                return True
        elif marker_lower in lowered:
            return True
    return False


def parse_cp2k_cell(text: str) -> list[tuple[float, float, float]] | None:
    match = re.search(r"&CELL(.*?)&END CELL", text, re.S)
    if not match:
        return None
    block = match.group(1)
    vectors = []
    for axis in ("A", "B", "C"):
        axis_match = re.search(
            rf"\b{axis}\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
            block,
        )
        if not axis_match:
            return None
        vectors.append(tuple(float(value) for value in axis_match.groups()))
    return vectors


def parse_cp2k_coords(text: str) -> list[tuple[str, tuple[float, float, float]]] | None:
    match = re.search(r"&COORD(.*?)&END COORD", text, re.S)
    if not match:
        return None
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for raw_line in match.group(1).splitlines():
        parts = raw_line.split()
        if len(parts) != 4:
            continue
        try:
            coords = tuple(float(value) for value in parts[1:4])
        except ValueError:
            continue
        atoms.append((parts[0], coords))
    return atoms or None


def read_poscar_structure(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[str, tuple[float, float, float]]]] | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if len(lines) < 8:
        return None
    try:
        scale = float(lines[1].split()[0])
        cell = []
        for idx in range(2, 5):
            vector = tuple(scale * float(value) for value in lines[idx].split()[:3])
            if len(vector) != 3:
                return None
            cell.append(vector)
    except (IndexError, ValueError):
        return None

    cursor = 5
    element_tokens = lines[cursor].split()
    if not element_tokens:
        return None
    cursor += 1

    try:
        counts = [int(token) for token in lines[cursor].split()]
    except ValueError:
        return None
    cursor += 1

    if cursor < len(lines) and lines[cursor].lower().startswith("selective"):
        cursor += 1
    if cursor >= len(lines):
        return None
    coord_mode = lines[cursor].lower()
    cursor += 1

    natoms = sum(counts)
    symbols: list[str] = []
    for symbol, count in zip(element_tokens, counts):
        symbols.extend([symbol] * count)
    if len(symbols) != natoms or len(lines) < cursor + natoms:
        return None

    def frac_to_cart(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            cell[0][0] * vector[0] + cell[1][0] * vector[1] + cell[2][0] * vector[2],
            cell[0][1] * vector[0] + cell[1][1] * vector[1] + cell[2][1] * vector[2],
            cell[0][2] * vector[0] + cell[1][2] * vector[1] + cell[2][2] * vector[2],
        )

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for index in range(natoms):
        parts = lines[cursor + index].split()
        if len(parts) < 3:
            return None
        try:
            values = tuple(float(value) for value in parts[:3])
        except ValueError:
            return None
        if coord_mode.startswith("d"):
            coords = frac_to_cart(values)
        else:
            coords = tuple(scale * value for value in values)
        atoms.append((symbols[index], coords))
    return cell, atoms


def invert_3x3(matrix: list[tuple[float, float, float]]) -> list[list[float]] | None:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1.0e-12:
        return None
    inv_det = 1.0 / det
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def minimum_distance_check(
    cell: list[tuple[float, float, float]],
    atoms: list[tuple[str, tuple[float, float, float]]],
    source_name: str,
    method_label: str,
) -> tuple[bool, str]:
    if not atoms or len(atoms) < 2:
        return False, f"failed to parse final geometry from {source_name}"
    inv_cell = invert_3x3(cell)
    if inv_cell is None:
        return False, "cell matrix is singular"

    def cart_to_frac(vector: tuple[float, float, float]) -> list[float]:
        return [
            inv_cell[0][0] * vector[0] + inv_cell[0][1] * vector[1] + inv_cell[0][2] * vector[2],
            inv_cell[1][0] * vector[0] + inv_cell[1][1] * vector[1] + inv_cell[1][2] * vector[2],
            inv_cell[2][0] * vector[0] + inv_cell[2][1] * vector[1] + inv_cell[2][2] * vector[2],
        ]

    def frac_to_cart(vector: list[float]) -> tuple[float, float, float]:
        return (
            cell[0][0] * vector[0] + cell[1][0] * vector[1] + cell[2][0] * vector[2],
            cell[0][1] * vector[0] + cell[1][1] * vector[1] + cell[2][1] * vector[2],
            cell[0][2] * vector[0] + cell[1][2] * vector[1] + cell[2][2] * vector[2],
        )

    minimum: tuple[float, int, int, str, str] | None = None
    for i in range(len(atoms) - 1):
        frac_i = cart_to_frac(atoms[i][1])
        for j in range(i + 1, len(atoms)):
            frac_j = cart_to_frac(atoms[j][1])
            delta_frac = [frac_j[k] - frac_i[k] for k in range(3)]
            delta_frac = [value - round(value) for value in delta_frac]
            dx, dy, dz = frac_to_cart(delta_frac)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            pair = tuple(sorted((atoms[i][0], atoms[j][0])))
            threshold = max(MIN_DISTANCE_GLOBAL, MIN_DISTANCE_THRESHOLDS.get(pair, MIN_DISTANCE_GLOBAL))
            if distance < threshold:
                return (
                    False,
                    f"unphysical final geometry in {source_name}: min distance {distance:.3f} A for "
                    f"{atoms[i][0]}-{atoms[j][0]} (atoms {i + 1}-{j + 1}) below threshold {threshold:.3f} A",
                )
            if minimum is None or distance < minimum[0]:
                minimum = (distance, i + 1, j + 1, atoms[i][0], atoms[j][0])

    if minimum is None:
        return False, "could not evaluate pair distances"
    return (
        True,
        f"{method_label} geometry sanity check passed: minimum distance {minimum[0]:.3f} A "
        f"for {minimum[3]}-{minimum[4]} (atoms {minimum[1]}-{minimum[2]})",
    )


def structure_dir_from_stage(stage_dir: Path) -> Path:
    return stage_dir.parent


def preopt_bypass_reason(stage_dir: Path) -> str:
    structure_dir = structure_dir_from_stage(stage_dir)
    for marker_name in PREOPT_BYPASS_MARKER_ALIASES:
        if (structure_dir / marker_name).exists():
            return f"{marker_name} marker is present"
    return "no bypass requested"


def preopt_bypass_requested(stage_dir: Path) -> bool:
    return preopt_bypass_reason(stage_dir) != "no bypass requested"


def activate_preopt_fallback(stage_dir: Path, reason: str) -> str:
    structure_dir = structure_dir_from_stage(stage_dir)
    for marker_name in PREOPT_BYPASS_MARKER_ALIASES:
        marker_path = structure_dir / marker_name
        if marker_path.exists() and marker_name != PREOPT_BYPASS_MARKER:
            marker_path.unlink()
    marker_path = structure_dir / PREOPT_BYPASS_MARKER
    marker_path.write_text(
        f"Auto-generated after pre-optimization fallback at {now()}\nReason: {reason}\n",
        encoding="utf-8",
    )
    return f"Activated fallback marker: {marker_path.name}"


def preopt_min_distance_check(stage_dir: Path, source_name: str = PREOPT_CONTCAR_NAME) -> tuple[bool, str]:
    source_path = stage_dir / source_name
    structure = read_poscar_structure(source_path)
    if structure is None:
        return False, f"missing or unreadable pre-optimized geometry: {source_path}"
    cell, atoms = structure
    return minimum_distance_check(cell, atoms, source_path.name, "UMA")


def resolve_submit_script(workflow_dir: Path, script_arg: str) -> Path:
    candidate = Path(script_arg)
    if candidate.is_absolute():
        return candidate
    return (workflow_dir / script_arg).resolve()


def build_submit_command(args: argparse.Namespace, stage_name: str, structure_id: str, stage_dir: Path) -> list[str]:
    if stage_name == PREOPT_STAGE:
        submit_script = resolve_submit_script(WORKFLOW_ROOT, args.preopt_submit_script)
    elif stage_name == UMA_MD_STAGE:
        submit_script = resolve_submit_script(WORKFLOW_ROOT, args.uma_md_submit_script)
    else:
        submit_script = resolve_submit_script(WORKFLOW_ROOT, args.submit_script)
    if not submit_script.exists():
        raise FileNotFoundError(f"Submit script not found: {submit_script}")
    return [
        "sbatch",
        "--parsable",
        "--wait",
        "--job-name",
        f"{structure_id}_{stage_name}",
        "--output",
        "slurm-%j.out",
        "--error",
        "slurm-%j.err",
        str(submit_script),
    ]


def replace_named_block(text: str, block_name: str, new_block: str) -> tuple[str, bool]:
    pattern = re.compile(rf"&{block_name}\b.*?&END {block_name}", re.S)
    new_text, count = pattern.subn(new_block, text, count=1)
    return new_text, count > 0


def render_cp2k_cell_block(cell: list[tuple[float, float, float]]) -> str:
    return (
        "&CELL\n"
        f"  A {cell[0][0]:.10f} {cell[0][1]:.10f} {cell[0][2]:.10f}\n"
        f"  B {cell[1][0]:.10f} {cell[1][1]:.10f} {cell[1][2]:.10f}\n"
        f"  C {cell[2][0]:.10f} {cell[2][1]:.10f} {cell[2][2]:.10f}\n"
        "&END CELL"
    )


def render_cp2k_coord_block(atoms: list[tuple[str, tuple[float, float, float]]]) -> str:
    lines = ["&COORD"]
    for symbol, coords in atoms:
        lines.append(f"  {symbol:<2} {coords[0]: .10f} {coords[1]: .10f} {coords[2]: .10f}")
    lines.append("&END COORD")
    return "\n".join(lines)


def set_scf_guess(text: str, value: str) -> tuple[str, bool]:
    new_text, count = re.subn(r"(\bSCF_GUESS\s+)\S+", rf"\1{value}", text, count=1)
    return new_text, count > 0


def load_cp2k_singlepoint_base_text(stage_dir: Path) -> tuple[str, Path]:
    base_path = stage_dir / CP2K_SINGLEPOINT_BASE_NAME
    if base_path.exists():
        return base_path.read_text(encoding="utf-8", errors="ignore"), base_path
    input_path = stage_dir / "1.inp"
    return input_path.read_text(encoding="utf-8", errors="ignore"), input_path


def prepare_cp2k_singlepoint_from_poscar(stage_dir: Path, source_path: Path) -> tuple[bool, str]:
    input_path = stage_dir / "1.inp"
    if not input_path.exists():
        return False, f"missing CP2K single-point input: {input_path}"
    structure = read_poscar_structure(source_path)
    if structure is None:
        return False, f"missing or unreadable geometry source: {source_path}"
    cell, atoms = structure

    base_text, base_path = load_cp2k_singlepoint_base_text(stage_dir)
    new_text, ok = set_scf_guess(base_text, "ATOMIC")
    if not ok:
        return False, f"failed to set SCF_GUESS ATOMIC in {base_path.name}"

    new_text, cell_replaced = replace_named_block(new_text, "CELL", render_cp2k_cell_block(cell))
    if not cell_replaced:
        return False, f"failed to replace CELL block in {base_path.name}"
    new_text, coord_replaced = replace_named_block(new_text, "COORD", render_cp2k_coord_block(atoms))
    if not coord_replaced:
        return False, f"failed to replace COORD block in {base_path.name}"

    current_text = input_path.read_text(encoding="utf-8", errors="ignore")
    if new_text != current_text:
        input_path.write_text(new_text, encoding="utf-8")
    return True, f"Prepared {input_path.name} from {source_path.parent.name}/{source_path.name}"


def prepare_uma_md_input(stage_dir: Path, source_path: Path) -> tuple[bool, str]:
    target_path = stage_dir / PREOPT_POSCAR_NAME
    if not source_path.exists():
        return False, f"missing MD geometry source: {source_path}"
    source_text = source_path.read_text(encoding="utf-8", errors="ignore")
    current_text = target_path.read_text(encoding="utf-8", errors="ignore") if target_path.exists() else None
    if source_text != current_text:
        target_path.write_text(source_text, encoding="utf-8")
    return True, f"Prepared {target_path.name} from {source_path.parent.name}/{source_path.name}"


def stage_success(stage_dir: Path, run_type: str, stage_name: str) -> tuple[bool, str]:
    output_path = stage_dir / "1.out"
    output_text = read_text_tail(output_path)
    stdout_text = read_latest_glob_text(stage_dir, "slurm-*.out")
    stderr_text = read_latest_glob_text(stage_dir, "slurm-*.err")
    combined_text = "\n".join(part for part in [output_text, stdout_text, stderr_text] if part)

    if stage_name == PREOPT_STAGE:
        contcar_path = stage_dir / PREOPT_CONTCAR_NAME
        if not contcar_path.exists():
            return False, f"missing UMA CONTCAR: {contcar_path}"
        lowered = combined_text.lower()
        for marker in UMA_HARD_FAILURE_MARKERS:
            if marker.lower() in lowered:
                return False, f"UMA pre-optimization hit a hard failure: {marker}"
        sane, sanity_reason = preopt_min_distance_check(stage_dir)
        if not sane:
            return False, sanity_reason
        if not combined_text:
            return False, "UMA pre-optimization produced no readable output"
        return True, f"UMA pre-optimization produced CONTCAR and passed geometry sanity check. {sanity_reason}"

    if stage_name == UMA_MD_STAGE:
        done_marker = stage_dir / UMA_MD_DONE_MARKER
        if not done_marker.exists():
            return False, f"missing UMA MD completion marker: {done_marker}"
        lowered = combined_text.lower()
        for marker in UMA_HARD_FAILURE_MARKERS:
            if marker.lower() in lowered:
                return False, f"UMA MD hit a hard failure: {marker}"
        return True, "UMA MD completed and produced a completion marker"

    if not output_text:
        return False, f"missing output: {output_path}"
    if END_MARKER not in output_text:
        return False, "missing CP2K end marker"
    if output_has_failure_marker(output_text):
        return False, "failure marker detected in 1.out"
    if stderr_text and output_has_failure_marker(stderr_text):
        return False, "failure marker detected in stderr"
    return True, "CP2K single-point completed"


def stage_artifact(stage_dir: Path, stage_name: str) -> str | None:
    if stage_name == PREOPT_STAGE:
        contcar = stage_dir / PREOPT_CONTCAR_NAME
        return str(contcar) if contcar.exists() else None
    if stage_name == UMA_MD_STAGE:
        marker = stage_dir / UMA_MD_DONE_MARKER
        return str(marker) if marker.exists() else None
    output = stage_dir / "1.out"
    return str(output) if output.exists() else None


def stage_has_existing_artifacts(stage_dir: Path, stage_name: str) -> bool:
    if stage_name == PREOPT_STAGE:
        return (
            (stage_dir / PREOPT_CONTCAR_NAME).exists()
            or any(stage_dir.glob("slurm-*.out"))
            or any(stage_dir.glob("slurm-*.err"))
            or (stage_dir / "uma_preopt").exists()
        )
    if stage_name == UMA_MD_STAGE:
        return (
            (stage_dir / UMA_MD_DONE_MARKER).exists()
            or any(stage_dir.glob("slurm-*.out"))
            or any(stage_dir.glob("slurm-*.err"))
            or (stage_dir / "uma_md").exists()
        )
    return (stage_dir / "1.out").exists() or any(stage_dir.glob("slurm-*.out")) or any(stage_dir.glob("slurm-*.err"))


def mark_completed_from_existing_output(
    state: dict,
    structure_id: str,
    stage_name: str,
    stage_dir: Path,
    run_type: str,
) -> bool | None:
    ok, reason = stage_success(stage_dir, run_type, stage_name)
    stage_state = state["structures"][structure_id]["stages"][stage_name]
    if ok:
        if stage_state["status"] != "completed":
            stage_state["status"] = "completed"
            stage_state["completed_at"] = now()
            stage_state["artifact_file"] = stage_artifact(stage_dir, stage_name)
            save_state(state)
        log(f"Skip completed stage: {structure_id} / {stage_name}")
        return True

    if stage_state["status"] == "failed":
        log(f"Encountered previously failed stage: {structure_id} / {stage_name}")
        log("Reason kept in state; fix it manually before resuming.")
        return False

    if stage_has_existing_artifacts(stage_dir, stage_name):
        if stage_name == PREOPT_STAGE:
            stage_state["status"] = "completed"
            stage_state["completed_at"] = now()
            stage_state["artifact_file"] = None
            save_state(state)
            log(f"Pre-optimization stage did not pass success checks for {structure_id}; using fallback path.")
            log(activate_preopt_fallback(stage_dir, reason))
            return True
        stage_state["status"] = "failed"
        stage_state["completed_at"] = now()
        save_state(state)
        log(f"Stage output exists but is not successful: {structure_id} / {stage_name}")
        log(f"Reason: {reason}")
        return False

    return None


def stage_enabled(stage_row: dict[str, str], args: argparse.Namespace) -> bool:
    enabled = stage_row.get("enabled", "1") == "1"
    if enabled:
        return True
    if args.include_optional:
        return True
    if args.include_md and stage_row["stage_name"] == UMA_MD_STAGE:
        return True
    return False


def run_stage(args: argparse.Namespace, state: dict, structure_id: str, stage_row: dict[str, str]) -> bool:
    stage_name = stage_row["stage_name"]
    run_type = stage_row["run_type"]
    stage_dir = WORKFLOW_ROOT / stage_row["path"]
    stage_state = state["structures"][structure_id]["stages"][stage_name]

    if not stage_dir.exists():
        stage_state["status"] = "failed"
        stage_state["completed_at"] = now()
        save_state(state)
        log(f"Stage directory does not exist: {stage_dir}")
        return False

    if args.dry_run:
        if stage_state["status"] == "completed":
            log(f"Skip completed stage: {structure_id} / {stage_name}")
            return True
        if stage_state["status"] == "failed":
            log(f"Encountered previously failed stage during dry-run: {structure_id} / {stage_name}")
            return False
        if stage_name == PREOPT_STAGE and preopt_bypass_requested(stage_dir):
            log(f"Would bypass pre-optimization stage for {structure_id} because {preopt_bypass_reason(stage_dir)}")
            return True
        command = build_submit_command(args, stage_name, structure_id, stage_dir)
        log(f"Submitting stage: {structure_id} / {stage_name}")
        log(f"Command: {' '.join(command)}")
        return True

    if stage_name == PREOPT_STAGE and preopt_bypass_requested(stage_dir):
        stage_state["status"] = "completed"
        stage_state["completed_at"] = now()
        stage_state["artifact_file"] = None
        state["current_structure"] = None
        state["current_stage"] = None
        save_state(state)
        log(f"Bypassing pre-optimization stage for {structure_id} because {preopt_bypass_reason(stage_dir)}")
        return True

    preopt_dir = stage_dir.parent / PREOPT_STAGE
    relaxed_geometry = preopt_dir / PREOPT_CONTCAR_NAME
    original_geometry = preopt_dir / PREOPT_POSCAR_NAME

    if stage_name == CP2K_SINGLEPOINT_STAGE:
        if preopt_bypass_requested(stage_dir):
            prepared, reason = prepare_cp2k_singlepoint_from_poscar(stage_dir, original_geometry)
            log(reason)
            if not prepared:
                stage_state["status"] = "failed"
                stage_state["completed_at"] = now()
                save_state(state)
                return False
        else:
            prepared, reason = prepare_cp2k_singlepoint_from_poscar(stage_dir, relaxed_geometry)
            log(reason)
            if not prepared:
                log(activate_preopt_fallback(stage_dir, reason))
                fallback_prepared, fallback_reason = prepare_cp2k_singlepoint_from_poscar(stage_dir, original_geometry)
                log(fallback_reason)
                if not fallback_prepared:
                    stage_state["status"] = "failed"
                    stage_state["completed_at"] = now()
                    save_state(state)
                    return False

    if stage_name == UMA_MD_STAGE:
        if preopt_bypass_requested(stage_dir):
            prepared, reason = prepare_uma_md_input(stage_dir, original_geometry)
            log(reason)
            if not prepared:
                stage_state["status"] = "failed"
                stage_state["completed_at"] = now()
                save_state(state)
                return False
        else:
            prepared, reason = prepare_uma_md_input(stage_dir, relaxed_geometry)
            log(reason)
            if not prepared:
                log(activate_preopt_fallback(stage_dir, reason))
                fallback_prepared, fallback_reason = prepare_uma_md_input(stage_dir, original_geometry)
                log(fallback_reason)
                if not fallback_prepared:
                    stage_state["status"] = "failed"
                    stage_state["completed_at"] = now()
                    save_state(state)
                    return False

    existing = mark_completed_from_existing_output(state, structure_id, stage_name, stage_dir, run_type)
    if existing is not None:
        return existing

    command = build_submit_command(args, stage_name, structure_id, stage_dir)
    log(f"Submitting stage: {structure_id} / {stage_name}")
    log(f"Command: {' '.join(command)}")

    state["current_structure"] = structure_id
    state["current_stage"] = stage_name
    stage_state["status"] = "running"
    stage_state["submitted_at"] = now()
    save_state(state)

    result = subprocess.run(command, capture_output=True, text=True, cwd=stage_dir)
    job_id = parse_job_id(result.stdout)
    stage_state["job_id"] = job_id
    stage_state["stdout_file"] = str(stage_dir / f"slurm-{job_id}.out") if job_id else None
    stage_state["stderr_file"] = str(stage_dir / f"slurm-{job_id}.err") if job_id else None

    ok, reason = stage_success(stage_dir, run_type, stage_name)
    if not ok:
        if stage_name == PREOPT_STAGE:
            stage_state["status"] = "completed"
            stage_state["completed_at"] = now()
            stage_state["artifact_file"] = None
            state["current_structure"] = None
            state["current_stage"] = None
            save_state(state)
            log(f"Pre-optimization stage failed for {structure_id}; falling back to direct CP2K single point.")
            log(f"Reason: {reason}")
            log(activate_preopt_fallback(stage_dir, reason))
            if result.stdout.strip():
                log(f"sbatch stdout: {result.stdout.strip()}")
            if result.stderr.strip():
                log(f"sbatch stderr: {result.stderr.strip()}")
            return True

        stage_state["status"] = "failed"
        stage_state["completed_at"] = now()
        state["current_structure"] = None
        state["current_stage"] = None
        save_state(state)
        if result.returncode != 0:
            log(f"Stage failed during submission or execution: {structure_id} / {stage_name}")
        else:
            log(f"Stage ended but did not pass success checks: {structure_id} / {stage_name}")
        log(f"Reason: {reason}")
        if result.stdout.strip():
            log(f"sbatch stdout: {result.stdout.strip()}")
        if result.stderr.strip():
            log(f"sbatch stderr: {result.stderr.strip()}")
        return False

    stage_state["status"] = "completed"
    stage_state["completed_at"] = now()
    stage_state["artifact_file"] = stage_artifact(stage_dir, stage_name)
    state["current_structure"] = None
    state["current_stage"] = None
    save_state(state)
    if result.returncode != 0:
        log(f"Stage returned non-zero but produced acceptable output: {structure_id} / {stage_name}")
        log(f"Reason accepted: {reason}")
    log(f"Completed stage: {structure_id} / {stage_name}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit the transport workflow sequentially and skip completed stages."
    )
    parser.add_argument("--submit-script", "--cp2k-submit-script", dest="submit_script", default=DEFAULT_CP2K_SUBMIT_SCRIPT)
    parser.add_argument("--preopt-submit-script", default=DEFAULT_PREOPT_SUBMIT_SCRIPT)
    parser.add_argument("--uma-md-submit-script", default=DEFAULT_UMA_MD_SUBMIT_SCRIPT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pause-seconds", type=int, default=10)
    parser.add_argument("--from-structure", default=None)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--include-md", action="store_true")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    structure_order, grouped = load_stage_manifest()
    if args.from_structure:
        if args.from_structure not in grouped:
            raise SystemExit(f"Unknown structure id: {args.from_structure}")
        start_index = structure_order.index(args.from_structure)
        structure_order = structure_order[start_index:]
    state = load_state(structure_order, grouped)

    log(f"Workflow root: {WORKFLOW_ROOT}")
    log(f"Selected {len(structure_order)} structure(s)")

    completed_structures = []
    failed_structures = []

    try:
        for structure_index, structure_id in enumerate(structure_order, start=1):
            log(f"Starting structure {structure_index}/{len(structure_order)}: {structure_id}")
            structure_state = state["structures"][structure_id]
            failed_stages = [name for name, info in structure_state["stages"].items() if info["status"] == "failed"]
            if failed_stages:
                log(
                    f"Rechecking structure with previously failed stage(s) for {structure_id}: "
                    + ", ".join(failed_stages)
                )

            structure_failed = False
            any_enabled_stage = False
            for stage_row in grouped[structure_id]:
                if not stage_enabled(stage_row, args):
                    log(f"Skip optional stage by default: {structure_id} / {stage_row['stage_name']}")
                    continue
                any_enabled_stage = True
                ok = run_stage(args, state, structure_id, stage_row)
                if not ok:
                    structure_failed = True
                    failed_structures.append(structure_id)
                    log(f"Skipping remaining stages for failed structure: {structure_id}")
                    break
                if args.pause_seconds > 0 and not args.dry_run:
                    time.sleep(args.pause_seconds)

            if not structure_failed and any_enabled_stage:
                completed_structures.append(structure_id)
    except KeyboardInterrupt:
        save_state(state)
        log("Interrupted by user.")
        return 130

    if failed_structures:
        log(f"Completed structures: {', '.join(completed_structures) if completed_structures else 'none'}")
        log(f"Failed or skipped structures: {', '.join(failed_structures)}")
        log("Workflow finished with partial failures. Failed structures were not resubmitted.")
        return 1

    log("All selected workflow stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
