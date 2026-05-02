from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aimsgb import Grain, GrainBoundary
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT.parent
SOURCE_CIF = PROJECT_DIR / "done.cif"
NOTES_DIR = PROJECT_DIR / "00_notes"
STRUCTURES_DIR = PROJECT_DIR / "01_structures"
LEGACY_DFT_WORKFLOWS_DIR = PROJECT_DIR / "_legacy_dft_reference_not_included"
WORKFLOWS_DIR = PROJECT_DIR / "02_uma_workflows"
JOB_SCRIPTS_DIR = WORKFLOWS_DIR / "job_scripts"

ELEMENT_ORDER = {"Li": 0, "P": 1, "S": 2, "Cl": 3}
PREOPT_STAGE = "00_uma_relax"
CP2K_SINGLEPOINT_STAGE = "01_cp2k_singlepoint"
UMA_MD_STAGE = "02_uma_md_nvt"
STAGE_DEFS = [
    (PREOPT_STAGE, "UMA_OPT", True),
    (CP2K_SINGLEPOINT_STAGE, "ENERGY_FORCE", True),
    (UMA_MD_STAGE, "UMA_MD", False),
]

# done.cif is already the optimized 2x2x2 reference supercell.
# Bulk structures should therefore be built directly from it.
BULK_SUPERCELL = [1, 1, 1]
GB_AXIS = [1, 1, 1]
GB_SIGMA = 3
GB_PLANE = [1, -2, 1]
GB_GAP_A = 0.8
GB_DELTA_A = 10.0
GB_C_REPEATS = 1
GB_SHIFTS = {
    "t1": (1.0 / 3.0, 0.0),
    "t2": (1.0 / 3.0, 0.5),
    "t3": (1.0 / 6.0, 0.25),
}

PREOPT_BYPASS_MARKER = "SKIP_PREOPT"
XTB_BYPASS_MARKER = PREOPT_BYPASS_MARKER
DEFAULT_GEOM_OPTIMIZER = "BFGS"
GB_GEOM_OPTIMIZER = "LBFGS"
CELL_OPT_MAX_ITER = 1000
# Slightly looser than CP2K defaults to make 400+ atom cell optimizations less brittle.
CELL_OPT_MAX_DR = "5.0E-3"
CELL_OPT_RMS_DR = "2.5E-3"
CELL_OPT_MAX_FORCE = "8.0E-4"
CELL_OPT_RMS_FORCE = "5.0E-4"
GB_DFT_MAX_DR = "4.0E-2"
GB_DFT_RMS_DR = "4.0E-3"

XTB_INNER_MAX_SCF = 50
XTB_OUTER_MAX_SCF = 20
XTB_EPS_SCF = "1.0E-5"
XTB_IGNORE_SCF_FAILURE = "TRUE"
XTB_OT_PRECONDITIONER = "FULL_SINGLE_INVERSE"
XTB_OT_ENERGY_GAP = "0.08"
XTB_BFGS_TRUST_RADIUS = "0.10"
GB_UMA_OPTIMIZER = "FIRE"
GB_UMA_FMAX = "0.08"
GB_UMA_MAX_STEPS = 200
GB_UMA_THREADS = 8
GB_UMA_MD_THREADS = 8
GB_UMA_MD_TEMP_K = "600"
GB_UMA_MD_TIMESTEP_FS = "1.0"
GB_UMA_MD_STEPS = 5000
GB_UMA_MD_SAVE_INTERVAL = 20

DFT_INNER_MAX_SCF = 25
DFT_OUTER_MAX_SCF = 20
GB_DFT_INNER_MAX_SCF = 30
GB_DFT_OUTER_MAX_SCF = 15
GB_DFT_EPS_SCF = "1.0E-5"
DFT_OT_MINIMIZER = "CG"
DFT_OT_LINESEARCH = "2PNT"
DFT_OT_ALGORITHM = "IRAC"
DFT_OT_PRECONDITIONER = "FULL_SINGLE_INVERSE"
DFT_OT_ENERGY_GAP = "0.05"
GB_DFT_OT_PRECONDITIONER = "FULL_KINETIC"
GB_DFT_OT_ENERGY_GAP = "0.10"
SP_INNER_MAX_SCF = 50
SP_OUTER_MAX_SCF = 20
SP_EPS_SCF = "1.0E-6"
SP_OT_PRECONDITIONER = "FULL_KINETIC"
SP_OT_ENERGY_GAP = "0.10"

KIND_LINES_DZVP = {
    "Li": ("DZVP-MOLOPT-SR-GTH-q3", "GTH-PBE-q3"),
    "P": ("DZVP-MOLOPT-SR-GTH-q5", "GTH-PBE-q5"),
    "S": ("DZVP-MOLOPT-SR-GTH-q6", "GTH-PBE-q6"),
    "Cl": ("DZVP-MOLOPT-SR-GTH-q7", "GTH-PBE-q7"),
}

KIND_LINES_TZVP = {
    "Li": ("TZVP-MOLOPT-PBE-GTH-q3", "GTH-PBE-q3"),
    "P": ("TZVP-MOLOPT-PBE-GTH-q5", "GTH-PBE-q5"),
    "S": ("TZVP-MOLOPT-PBE-GTH-q6", "GTH-PBE-q6"),
    "Cl": ("TZVP-MOLOPT-PBE-GTH-q7", "GTH-PBE-q7"),
}

VALENCE_ELECTRONS = {
    "Li": 3,
    "P": 5,
    "S": 6,
    "Cl": 7,
}

BASIS_PATH = "/home/ctan/cp2k/cp2k-2026.1/data/BASIS_MOLOPT"
POTENTIAL_PATH = "/home/ctan/cp2k/cp2k-2026.1/data/POTENTIAL"
CP2K_SUBMIT_SCRIPT = "job_scripts/cp2k_singlepoint_large_mem.sh"
GB_PREOPT_SUBMIT_SCRIPT = "job_scripts/uma_preopt_cpu.sh"
UMA_MD_SUBMIT_SCRIPT = "job_scripts/uma_md_cpu.sh"
SERIAL_SUBMIT_TEMPLATE = ROOT.parent / "serial_submit_cp2k_template.py"

STRUCTURE_MANIFEST_COLUMNS = [
    "structure_id",
    "path",
    "n_atoms",
    "lattice_a",
    "lattice_b",
    "lattice_c",
    "gb_type",
    "gb_normal",
    "z0_A",
    "delta_A",
    "translation_state",
    "li_vac_count",
    "li_vac_conc",
    "li_vac_region",
    "deleted_li_indices",
    "notes",
]

WORKFLOW_MANIFEST_COLUMNS = [
    "structure_id",
    "stage_order",
    "stage_name",
    "run_type",
    "path",
    "restart_from",
    "enabled",
]


SERIAL_SUBMIT_SCRIPT = inspect.cleandoc(
    """\
    from __future__ import annotations

    import argparse
    import csv
    import json
    import subprocess
    import sys
    import time
    from collections import defaultdict
    from datetime import datetime
    from pathlib import Path


    WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
    NOTES_DIR = WORKFLOW_ROOT.parent / "00_notes"
    MANIFEST_PATH = NOTES_DIR / "workflow_manifest.csv"
    ORDER_PATH = NOTES_DIR / "cp2k_job_order.txt"
    LOG_PATH = WORKFLOW_ROOT / "job_scripts" / "serial_submit_cp2k.log"
    STATE_PATH = WORKFLOW_ROOT / "job_scripts" / "serial_submit_cp2k_state.json"
    DEFAULT_SUBMIT_SCRIPT = "/home/ctan/cp2k/cp2k-2026.1/cp2k.sh"
    XTB_PREOPT_STAGE = "00_xtb_gfn1"

    END_MARKER = "PROGRAM ENDED AT"
    OPT_SUCCESS_MARKERS = (
        "OPTIMIZATION COMPLETED",
        "CELL OPTIMIZATION COMPLETED",
        "GEOMETRY OPTIMIZATION COMPLETED",
    )
    XTB_HARD_FAILURE_MARKERS = (
        "Use the LSD option for an odd number of electrons",
        "Out Of Memory",
        "oom-kill",
        "Segmentation fault",
        "SIGSEGV",
        "exceeded requested execution time",
    )
    FAILURE_MARKERS = (
        "SCF run NOT converged",
        "MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED",
        "exceeded requested execution time",
        "ABORT",
        "abort",
        "error",
    )


    def now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")


    def log(message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\\n")


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
            grouped[row["structure_id"]].append(row)
        for stages in grouped.values():
            stages.sort(key=lambda item: int(item["stage_order"]))
        return structure_order, grouped


    def load_state(structure_order: list[str], grouped: dict[str, list[dict[str, str]]]) -> dict:
        if STATE_PATH.exists():
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
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
                        "job_id": None,
                        "submitted_at": None,
                        "completed_at": None,
                        "stdout_file": None,
                        "stderr_file": None,
                        "restart_file": None,
                    },
                )
        return state


    def save_state(state: dict) -> None:
        state["updated_at"] = now()
        STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")


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


    def stage_success(stage_dir: Path, run_type: str, stage_name: str) -> tuple[bool, str]:
        output_path = stage_dir / "1.out"
        output_text = read_text_tail(output_path)
        stdout_text = read_latest_glob_text(stage_dir, "slurm-*.out")
        stderr_text = read_latest_glob_text(stage_dir, "slurm-*.err")
        combined_text = "\\n".join(part for part in [output_text, stdout_text, stderr_text] if part)
        if not output_text and stage_name != XTB_PREOPT_STAGE:
            return False, f"missing output: {output_path}"

        if run_type == "CELL_OPT":
            restart_path = stage_dir / "final.restart"
            if not restart_path.exists():
                return False, f"missing restart file: {restart_path}"
            if stage_name == XTB_PREOPT_STAGE:
                for marker in XTB_HARD_FAILURE_MARKERS:
                    if marker.lower() in combined_text.lower():
                        return False, f"xTB pre-optimization hit a hard failure: {marker}"
                if output_text:
                    return True, "xTB pre-optimization produced final.restart; the latest structure will be used for the DFT cell optimization regardless of convergence"
                if END_MARKER not in output_text and not combined_text:
                    return False, "xTB pre-optimization produced no readable output"
                if "exceeded requested execution time" in combined_text.lower():
                    return False, "xTB pre-optimization ended by walltime"
                return True, "xTB pre-optimization produced final.restart; continue with the DFT cell optimization"
            if END_MARKER not in output_text:
                return False, "missing CP2K end marker"
            if output_has_failure_marker(output_text):
                return False, "failure marker detected in 1.out"
            if not any(marker in output_text for marker in OPT_SUCCESS_MARKERS):
                return False, "optimization completion marker not found"
            return True, "cell optimization completed"

        if END_MARKER not in output_text:
            return False, "missing CP2K end marker"
        if output_has_failure_marker(output_text):
            return False, "failure marker detected in 1.out"
        if "Total FORCE_EVAL" not in output_text and "Total energy" not in output_text:
            return False, "static energy marker not found"
        return True, "static calculation completed"


    def build_submit_command(submit_script: str, stage_name: str, structure_id: str) -> list[str]:
        return [
            "sbatch",
            "--parsable",
            "--wait",
            f"--job-name={structure_id}_{stage_name}",
            "--output=slurm-%j.out",
            "--error=slurm-%j.err",
            submit_script,
        ]


    def mark_completed_from_existing_output(state: dict, structure_id: str, stage_name: str, stage_dir: Path, run_type: str) -> bool:
        ok, reason = stage_success(stage_dir, run_type, stage_name)
        stage_state = state["structures"][structure_id]["stages"][stage_name]
        if ok:
            if stage_state["status"] != "completed":
                stage_state["status"] = "completed"
                stage_state["completed_at"] = now()
                stage_state["restart_file"] = str(stage_dir / "final.restart") if run_type == "CELL_OPT" else None
                save_state(state)
            log(f"Skip completed stage: {structure_id} / {stage_name}")
            return True
        if stage_state["status"] == "failed":
            log(f"Encountered previously failed stage: {structure_id} / {stage_name}")
            log(f"Reason kept in state; fix it manually before resuming.")
            return False
        if (stage_dir / "1.out").exists():
            stage_state["status"] = "failed"
            stage_state["completed_at"] = now()
            save_state(state)
            log(f"Stage output exists but is not successful: {structure_id} / {stage_name}")
            log(f"Reason: {reason}")
            return False
        return None


    def run_stage(args: argparse.Namespace, state: dict, structure_id: str, stage_row: dict[str, str]) -> bool:
        stage_name = stage_row["stage_name"]
        run_type = stage_row["run_type"]
        stage_dir = WORKFLOW_ROOT / stage_row["path"]
        stage_state = state["structures"][structure_id]["stages"][stage_name]

        existing = mark_completed_from_existing_output(state, structure_id, stage_name, stage_dir, run_type)
        if existing is not None:
            return existing

        command = build_submit_command(args.submit_script, stage_name, structure_id)
        log(f"Submitting stage: {structure_id} / {stage_name}")
        log(f"Command: {' '.join(command)}")

        state["current_structure"] = structure_id
        state["current_stage"] = stage_name
        stage_state["status"] = "running"
        stage_state["submitted_at"] = now()
        save_state(state)

        if args.dry_run:
            stage_state["status"] = "pending"
            state["current_structure"] = None
            state["current_stage"] = None
            save_state(state)
            return True

        result = subprocess.run(command, capture_output=True, text=True, cwd=stage_dir)
        job_id = parse_job_id(result.stdout)
        stage_state["job_id"] = job_id
        stage_state["stdout_file"] = str(stage_dir / f"slurm-{job_id}.out") if job_id else None
        stage_state["stderr_file"] = str(stage_dir / f"slurm-{job_id}.err") if job_id else None

        ok, reason = stage_success(stage_dir, run_type, stage_name)
        if not ok:
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
        stage_state["restart_file"] = str(stage_dir / "final.restart") if run_type == "CELL_OPT" else None
        state["current_structure"] = None
        state["current_stage"] = None
        save_state(state)
        if result.returncode != 0:
            log(f"Stage returned non-zero but produced acceptable output: {structure_id} / {stage_name}")
            log(f"Reason accepted: {reason}")
        log(f"Completed stage: {structure_id} / {stage_name}")
        return True


    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Submit the CP2K workflow sequentially and skip failed structures without resubmitting them.")
        parser.add_argument("--submit-script", default=DEFAULT_SUBMIT_SCRIPT)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--pause-seconds", type=int, default=10)
        parser.add_argument("--from-structure", default=None)
        return parser


    def main() -> int:
        args = build_parser().parse_args()
        structure_order, grouped = load_stage_manifest()
        state = load_state(structure_order, grouped)
        save_state(state)

        if args.from_structure:
            if args.from_structure not in grouped:
                log(f"Unknown structure_id: {args.from_structure}")
                return 2
            start_index = structure_order.index(args.from_structure)
            structure_order = structure_order[start_index:]

        failed_structures = []
        completed_structures = []

        log(f"Workflow root: {WORKFLOW_ROOT}")
        log(f"Selected {len(structure_order)} structure(s)")

        try:
            for structure_index, structure_id in enumerate(structure_order, start=1):
                log(f"Starting structure {structure_index}/{len(structure_order)}: {structure_id}")
                structure_state = state["structures"][structure_id]
                failed_stages = [name for name, info in structure_state["stages"].items() if info["status"] == "failed"]
                if failed_stages:
                    log(f"Skipping structure because failed stage(s) are already recorded for {structure_id}: {', '.join(failed_stages)}")
                    failed_structures.append(structure_id)
                    continue

                structure_failed = False
                for stage_row in grouped[structure_id]:
                    ok = run_stage(args, state, structure_id, stage_row)
                    if not ok:
                        structure_failed = True
                        failed_structures.append(structure_id)
                        log(f"Skipping remaining stages for failed structure: {structure_id}")
                        break
                    if args.pause_seconds > 0 and not args.dry_run:
                        time.sleep(args.pause_seconds)
                if not structure_failed:
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

        log("All workflow stages completed successfully.")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """
).replace("\n    ", "\n") + "\n"

RUN_SERIAL_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -euo pipefail

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    WORKFLOW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

    cd "${WORKFLOW_DIR}"
    python3 job_scripts/serial_submit_cp2k.py "$@"
    """
)

CP2K_LARGE_MEM_SCRIPT = textwrap.dedent(
    """\
    #!/bin/bash
    #SBATCH --job-name=cp2k_sp
    #SBATCH --partition=workq
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=40
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=120G
    #SBATCH --time=72:00:00

    set -euo pipefail

    INPUT_FILE="1.inp"
    OUTPUT_FILE="1.out"
    JOB_BASENAME="$(basename "${INPUT_FILE%.*}")"

    if [[ ! -f "${INPUT_FILE}" ]]; then
      echo "Input file not found: ${INPUT_FILE}"
      exit 1
    fi

    ulimit -d unlimited
    ulimit -s unlimited
    ulimit -t unlimited
    ulimit -v unlimited

    export SLURM_EXPORT_ENV=ALL

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    export LD_RUN_PATH="${LD_RUN_PATH:-}"
    export LIBRARY_PATH="${LIBRARY_PATH:-}"
    export CPATH="${CPATH:-}"
    export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
    export CP_DFLAGS="${CP_DFLAGS:-}"
    export CP_CFLAGS="${CP_CFLAGS:-}"
    export CP_LDFLAGS="${CP_LDFLAGS:-}"
    export CP_LIBS="${CP_LIBS:-}"

    set +u
    source /home/ctan/cp2k/cp2k-2026.1/tools/toolchain/install/setup
    set -u

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export ELPA_DEFAULT_OMP_THREADS=1
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    EXE="/home/ctan/cp2k/cp2k-2026.1/build/bin/cp2k.psmp"
    MPI_TYPE="${CP2K_MPI_TYPE:-pmix}"
    LOG_FILE="${JOB_BASENAME}.${SLURM_JOB_ID}.log"

    cd "${SLURM_SUBMIT_DIR}"

    echo "Starting CP2K single point on $(hostname) at $(date)" | tee "${LOG_FILE}"
    echo "Input: ${INPUT_FILE}" | tee -a "${LOG_FILE}"
    echo "Output: ${OUTPUT_FILE}" | tee -a "${LOG_FILE}"
    echo "MPI ranks: ${SLURM_NTASKS}" | tee -a "${LOG_FILE}"
    echo "OMP threads per rank: ${OMP_NUM_THREADS}" | tee -a "${LOG_FILE}"
    echo "Launcher: srun --mpi=${MPI_TYPE}" | tee -a "${LOG_FILE}"

    srun --mpi="${MPI_TYPE}" --cpu-bind=cores "${EXE}" -i "${INPUT_FILE}" -o "${OUTPUT_FILE}"

    echo "Finished CP2K single point on $(hostname) at $(date)" | tee -a "${LOG_FILE}"
    """
)

CP2K_XTB_GFN1_OMP_SCRIPT = textwrap.dedent(
    """\
    #!/bin/bash
    #SBATCH --job-name=cp2k_xtb
    #SBATCH --partition=workq
    #SBATCH --nodelist=baifq-hpc141
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=48G
    #SBATCH --time=48:00:00

    set -euo pipefail

    INPUT_FILE="${INPUT_FILE:-1.inp}"
    OUTPUT_FILE="${OUTPUT_FILE:-1.out}"
    JOB_BASENAME="$(basename "${INPUT_FILE%.*}")"
    OMP_THREADS="${CP2K_XTB_OMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

    if [[ ! -f "${INPUT_FILE}" ]]; then
      echo "Input file not found: ${INPUT_FILE}"
      exit 1
    fi

    ulimit -d unlimited
    ulimit -s unlimited
    ulimit -t unlimited
    ulimit -v unlimited

    export SLURM_EXPORT_ENV=ALL

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    export LD_RUN_PATH="${LD_RUN_PATH:-}"
    export LIBRARY_PATH="${LIBRARY_PATH:-}"
    export CPATH="${CPATH:-}"
    export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
    export CP_DFLAGS="${CP_DFLAGS:-}"
    export CP_CFLAGS="${CP_CFLAGS:-}"
    export CP_LDFLAGS="${CP_LDFLAGS:-}"
    export CP_LIBS="${CP_LIBS:-}"

    set +u
    source /home/ctan/cp2k/cp2k-2026.1/tools/toolchain/install/setup
    set -u

    export OMP_NUM_THREADS="${OMP_THREADS}"
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export ELPA_DEFAULT_OMP_THREADS=1
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    export OMP_STACKSIZE=256M

    EXE="/home/ctan/cp2k/cp2k-2026.1/build/bin/cp2k.psmp"
    MPI_TYPE="${CP2K_MPI_TYPE:-pmix}"
    LOG_FILE="${JOB_BASENAME}.${SLURM_JOB_ID}.log"

    cd "${SLURM_SUBMIT_DIR}"

    echo "Starting CP2K GFN1-xTB on $(hostname) at $(date)" | tee "${LOG_FILE}"
    echo "Input: ${INPUT_FILE}" | tee -a "${LOG_FILE}"
    echo "Output: ${OUTPUT_FILE}" | tee -a "${LOG_FILE}"
    echo "MPI ranks: 1" | tee -a "${LOG_FILE}"
    echo "OMP threads: ${OMP_NUM_THREADS}" | tee -a "${LOG_FILE}"
    echo "Launcher: srun --mpi=${MPI_TYPE} -n 1" | tee -a "${LOG_FILE}"

    srun --mpi="${MPI_TYPE}" -n 1 --cpu-bind=cores "${EXE}" -i "${INPUT_FILE}" -o "${OUTPUT_FILE}"

    echo "Finished CP2K GFN1-xTB on $(hostname) at $(date)" | tee -a "${LOG_FILE}"
    """
)

UMA_PREOPT_CPU_SCRIPT = textwrap.dedent(
    f"""\
    #!/bin/bash
    #SBATCH --job-name=uma_preopt
    #SBATCH --partition=workq
    #SBATCH --nodelist=baifq-hpc141
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={GB_UMA_THREADS}
    #SBATCH --mem=32G
    #SBATCH --time=24:00:00

    set -euo pipefail

    STRUCTURE_FILE="${{UMA_STRUCTURE_FILE:-POSCAR}}"
    MODEL_PATH="${{UMA_MODEL_PATH:-/home/ctan/uma-m-1p1.pt}}"
    ENV_PREFIX="${{UMA_ENV_PREFIX:-/home/ctan/uma-offline-env}}"
    TASK_NAME="${{UMA_TASK_NAME:-omat}}"
    OPTIMIZER="${{UMA_OPTIMIZER:-{GB_UMA_OPTIMIZER}}}"
    FMAX="${{UMA_FMAX:-{GB_UMA_FMAX}}}"
    MAX_STEPS="${{UMA_MAX_STEPS:-{GB_UMA_MAX_STEPS}}}"
    RUN_NAME="${{UMA_RUN_NAME:-uma_preopt}}"
    OUTPUT_ROOT="${{UMA_OUTPUT_ROOT:-${{SLURM_SUBMIT_DIR}}}}"

    if [[ ! -f "${{STRUCTURE_FILE}}" ]]; then
      echo "Input structure not found: ${{STRUCTURE_FILE}}" >&2
      exit 1
    fi

    if [[ ! -x "${{ENV_PREFIX}}/bin/python" ]]; then
      echo "UMA offline environment not found: ${{ENV_PREFIX}}" >&2
      exit 1
    fi

    if [[ ! -f "${{MODEL_PATH}}" ]]; then
      echo "UMA model not found: ${{MODEL_PATH}}" >&2
      exit 1
    fi

    source "${{ENV_PREFIX}}/bin/activate"

    export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{GB_UMA_THREADS}}}"
    export MKL_NUM_THREADS="${{OMP_NUM_THREADS}}"
    export OPENBLAS_NUM_THREADS="${{OMP_NUM_THREADS}}"
    export PYTHONNOUSERSITE=1

    cd "${{SLURM_SUBMIT_DIR}}"

    uma_calc opt "${{STRUCTURE_FILE}}" \\
      --model "${{MODEL_PATH}}" \\
      --task "${{TASK_NAME}}" \\
      --device cpu \\
      --optimizer "${{OPTIMIZER}}" \\
      --fmax "${{FMAX}}" \\
      --max-steps "${{MAX_STEPS}}" \\
      --output "${{OUTPUT_ROOT}}" \\
      --name "${{RUN_NAME}}"

    RESULT_DIR="${{OUTPUT_ROOT}}/${{RUN_NAME}}"
    for artifact in CONTCAR OUTCAR OSZICAR optimization.log uma_results.json; do
      if [[ -f "${{RESULT_DIR}}/${{artifact}}" ]]; then
        cp -f "${{RESULT_DIR}}/${{artifact}}" "${{artifact}}"
      fi
    done

    if [[ ! -f "CONTCAR" ]]; then
      echo "UMA optimization did not produce CONTCAR" >&2
      exit 1
    fi
    """
)

UMA_MD_CPU_SCRIPT = textwrap.dedent(
    f"""\
    #!/bin/bash
    #SBATCH --job-name=uma_md
    #SBATCH --partition=workq
    #SBATCH --nodelist=baifq-hpc141
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={GB_UMA_MD_THREADS}
    #SBATCH --mem=32G
    #SBATCH --time=72:00:00

    set -euo pipefail

    STRUCTURE_FILE="${{UMA_STRUCTURE_FILE:-POSCAR}}"
    MODEL_PATH="${{UMA_MODEL_PATH:-/home/ctan/uma-m-1p1.pt}}"
    ENV_PREFIX="${{UMA_ENV_PREFIX:-/home/ctan/uma-offline-env}}"
    TASK_NAME="${{UMA_TASK_NAME:-omat}}"
    ENSEMBLE="${{UMA_MD_ENSEMBLE:-NVT}}"
    TEMP_K="${{UMA_MD_TEMP:-{GB_UMA_MD_TEMP_K}}}"
    TIMESTEP_FS="${{UMA_MD_TIMESTEP:-{GB_UMA_MD_TIMESTEP_FS}}}"
    STEPS="${{UMA_MD_STEPS:-{GB_UMA_MD_STEPS}}}"
    SAVE_INTERVAL="${{UMA_MD_SAVE_INTERVAL:-{GB_UMA_MD_SAVE_INTERVAL}}}"
    RUN_NAME="${{UMA_RUN_NAME:-uma_md}}"
    OUTPUT_ROOT="${{UMA_OUTPUT_ROOT:-${{SLURM_SUBMIT_DIR}}}}"

    if [[ ! -f "${{STRUCTURE_FILE}}" ]]; then
      echo "Input structure not found: ${{STRUCTURE_FILE}}" >&2
      exit 1
    fi

    if [[ ! -x "${{ENV_PREFIX}}/bin/python" ]]; then
      echo "UMA offline environment not found: ${{ENV_PREFIX}}" >&2
      exit 1
    fi

    if [[ ! -f "${{MODEL_PATH}}" ]]; then
      echo "UMA model not found: ${{MODEL_PATH}}" >&2
      exit 1
    fi

    source "${{ENV_PREFIX}}/bin/activate"

    export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{GB_UMA_MD_THREADS}}}"
    export MKL_NUM_THREADS="${{OMP_NUM_THREADS}}"
    export OPENBLAS_NUM_THREADS="${{OMP_NUM_THREADS}}"
    export PYTHONNOUSERSITE=1

    cd "${{SLURM_SUBMIT_DIR}}"

    uma_calc md "${{STRUCTURE_FILE}}" \\
      --model "${{MODEL_PATH}}" \\
      --task "${{TASK_NAME}}" \\
      --device cpu \\
      --ensemble "${{ENSEMBLE}}" \\
      --temp "${{TEMP_K}}" \\
      --timestep "${{TIMESTEP_FS}}" \\
      --steps "${{STEPS}}" \\
      --save-interval "${{SAVE_INTERVAL}}" \\
      --output "${{OUTPUT_ROOT}}" \\
      --name "${{RUN_NAME}}"

    RESULT_DIR="${{OUTPUT_ROOT}}/${{RUN_NAME}}"
    if [[ -d "${{RESULT_DIR}}" ]]; then
      cp -Rf "${{RESULT_DIR}}/." .
    fi

    touch MD_DONE
    """
)

XTB_GFNFF_SCRIPT = textwrap.dedent(
    """\
    #!/bin/bash
    #SBATCH --job-name=xtb_gfnff
    #SBATCH --partition=workq
    #SBATCH --nodelist=baifq-hpc141
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=16G
    #SBATCH --time=24:00:00

    set -euo pipefail

    GEOMETRY_FILE="${XTB_GEOMETRY_FILE:-POSCAR}"
    OPT_LEVEL="${XTB_OPT_LEVEL:-loose}"
    XTB_BIN="${XTB_BIN:-/home/ctan/xtb-dist/bin/xtb}"
    LOG_FILE="${XTB_LOG_FILE:-xtb.out}"
    ERR_FILE="${XTB_ERR_FILE:-xtb.err}"
    THREADS="${SLURM_CPUS_PER_TASK:-1}"

    if [[ ! -f "${GEOMETRY_FILE}" ]]; then
      echo "Input geometry not found: ${GEOMETRY_FILE}" >&2
      exit 1
    fi

    if [[ ! -x "${XTB_BIN}" ]]; then
      echo "xTB executable not found or not executable: ${XTB_BIN}" >&2
      exit 1
    fi

    export OMP_NUM_THREADS="${THREADS}"
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    cd "${SLURM_SUBMIT_DIR}"

    echo "Starting xTB GFN-FF on $(hostname) at $(date)" > "${LOG_FILE}"
    echo "Geometry: ${GEOMETRY_FILE}" >> "${LOG_FILE}"
    echo "Threads: ${THREADS}" >> "${LOG_FILE}"
    echo "Optimization level: ${OPT_LEVEL}" >> "${LOG_FILE}"

    "${XTB_BIN}" "${GEOMETRY_FILE}" --gfnff --opt "${OPT_LEVEL}" -P "${THREADS}" >> "${LOG_FILE}" 2>> "${ERR_FILE}"

    echo "Finished xTB GFN-FF at $(date)" >> "${LOG_FILE}"
    """
)

XTB_GFNFF_NOCELLOPT_SCRIPT = textwrap.dedent(
    """\
    #!/bin/bash
    #SBATCH --job-name=xtb_gfnff_nocell
    #SBATCH --partition=workq
    #SBATCH --nodelist=baifq-hpc141
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=16G
    #SBATCH --time=24:00:00

    set -euo pipefail

    GEOMETRY_FILE="${XTB_GEOMETRY_FILE:-POSCAR}"
    OPT_LEVEL="${XTB_OPT_LEVEL:-loose}"
    XTB_BIN="${XTB_BIN:-/home/ctan/xtb-dist/bin/xtb}"
    LOG_FILE="${XTB_LOG_FILE:-xtb.out}"
    ERR_FILE="${XTB_ERR_FILE:-xtb.err}"
    THREADS="${SLURM_CPUS_PER_TASK:-1}"

    if [[ ! -f "${GEOMETRY_FILE}" ]]; then
      echo "Input geometry not found: ${GEOMETRY_FILE}" >&2
      exit 1
    fi

    if [[ ! -x "${XTB_BIN}" ]]; then
      echo "xTB executable not found or not executable: ${XTB_BIN}" >&2
      exit 1
    fi

    export OMP_NUM_THREADS="${THREADS}"
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    cd "${SLURM_SUBMIT_DIR}"

    echo "Starting xTB GFN-FF (--nocellopt) on $(hostname) at $(date)" > "${LOG_FILE}"
    echo "Geometry: ${GEOMETRY_FILE}" >> "${LOG_FILE}"
    echo "Threads: ${THREADS}" >> "${LOG_FILE}"
    echo "Optimization level: ${OPT_LEVEL}" >> "${LOG_FILE}"

    "${XTB_BIN}" "${GEOMETRY_FILE}" --gfnff --opt "${OPT_LEVEL}" --nocellopt -P "${THREADS}" >> "${LOG_FILE}" 2>> "${ERR_FILE}"

    echo "Finished xTB GFN-FF (--nocellopt) at $(date)" >> "${LOG_FILE}"
    """
)


@dataclass
class StructureRecord:
    structure_id: str
    rel_path: str
    structure: Structure
    gb_type: str
    gb_normal: str
    z0_A: str
    delta_A: str
    translation_state: str
    li_vac_count: int
    li_vac_conc: str
    li_vac_region: str
    deleted_li_indices: str
    notes: str

    def to_manifest_row(self) -> dict[str, str]:
        lattice = self.structure.lattice
        return {
            "structure_id": self.structure_id,
            "path": self.rel_path,
            "n_atoms": str(len(self.structure)),
            "lattice_a": f"{lattice.a:.6f}",
            "lattice_b": f"{lattice.b:.6f}",
            "lattice_c": f"{lattice.c:.6f}",
            "gb_type": self.gb_type,
            "gb_normal": self.gb_normal,
            "z0_A": self.z0_A,
            "delta_A": self.delta_A,
            "translation_state": self.translation_state,
            "li_vac_count": str(self.li_vac_count),
            "li_vac_conc": self.li_vac_conc,
            "li_vac_region": self.li_vac_region,
            "deleted_li_indices": self.deleted_li_indices,
            "notes": self.notes,
        }


def sort_structure(structure: Structure) -> Structure:
    return structure.get_sorted_structure(key=lambda site: (ELEMENT_ORDER[site.specie.symbol],))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized)


def write_poscar(structure: Structure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Poscar(structure).write_file(destination)


def clear_gb_directories() -> None:
    for root in [STRUCTURES_DIR, WORKFLOWS_DIR]:
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("gb_"):
                shutil.rmtree(path)


def reset_project_tree(gb_only: bool = False) -> None:
    if gb_only:
        clear_gb_directories()
    else:
        for path in [STRUCTURES_DIR, WORKFLOWS_DIR]:
            if path.exists():
                shutil.rmtree(path)
    for path in [NOTES_DIR / "source_inputs", STRUCTURES_DIR, JOB_SCRIPTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def li_indices(structure: Structure) -> list[int]:
    return [index for index, site in enumerate(structure) if site.specie.symbol == "Li"]


def make_vacancy(structure: Structure, indices_to_delete: list[int]) -> Structure:
    defect_structure = structure.copy()
    defect_structure.remove_sites(sorted(indices_to_delete, reverse=True))
    return sort_structure(defect_structure)


def farthest_disjoint_pairs(structure: Structure, candidate_indices: list[int]) -> list[tuple[int, int]]:
    distance_matrix = structure.distance_matrix[np.ix_(candidate_indices, candidate_indices)]
    pairs: list[tuple[float, int, int]] = []
    for i, j in itertools.combinations(range(len(candidate_indices)), 2):
        pairs.append((float(distance_matrix[i, j]), candidate_indices[i], candidate_indices[j]))
    pairs.sort(reverse=True)

    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for _, first, second in pairs:
        if first in used or second in used:
            continue
        selected.append(tuple(sorted((first, second))))
        used.update({first, second})
        if len(selected) == 2:
            break
    if len(selected) < 2:
        raise RuntimeError("Could not find two disjoint vacancy pairs.")
    return selected


def select_bulk_vacancies(structure: Structure) -> dict[str, list[int]]:
    pairs = farthest_disjoint_pairs(structure, li_indices(structure))
    return {
        "bulk_Li_vac_c1_s1": [pairs[0][0]],
        "bulk_Li_vac_c1_s2": [pairs[0][1]],
        "bulk_Li_vac_c2_s1": list(pairs[0]),
        "bulk_Li_vac_c2_s2": list(pairs[1]),
    }


def build_bulk_ordered() -> Structure:
    structure = Structure.from_file(SOURCE_CIF)
    structure.make_supercell(BULK_SUPERCELL)
    return sort_structure(structure)


def build_sigma3_grain_boundary(shift_xy: tuple[float, float]) -> Structure:
    grain = Grain.from_file(SOURCE_CIF)
    gb = GrainBoundary(GB_AXIS, GB_SIGMA, GB_PLANE, grain, uc_a=1, uc_b=1)

    grain_a = gb.grain_a.copy()
    grain_b = gb.grain_b.copy()
    grain_a.make_supercell([1, 1, GB_C_REPEATS])
    grain_b.make_supercell([1, 1, GB_C_REPEATS])
    grain_b.translate_sites(
        list(range(len(grain_b))),
        [shift_xy[0], shift_xy[1], 0.0],
        frac_coords=True,
        to_unit_cell=True,
    )

    structure = Grain.stack_grains(
        grain_a,
        grain_b,
        direction=2,
        gap=GB_GAP_A,
        vacuum=0.0,
        to_primitive=False,
    )
    structure = Structure.from_sites(structure.sites)

    central_interface = grain_a.lattice.c + GB_GAP_A / 2.0
    shift_cart = structure.lattice.c / 2.0 - central_interface
    structure.translate_sites(
        list(range(len(structure))),
        [0.0, 0.0, shift_cart],
        frac_coords=False,
        to_unit_cell=True,
    )
    return sort_structure(structure)


def select_gb_vacancies(structure: Structure) -> dict[str, list[int]]:
    z0 = structure.lattice.c / 2.0
    candidates = [
        index
        for index in li_indices(structure)
        if abs(structure.cart_coords[index][2] - z0) <= GB_DELTA_A
    ]
    pairs = farthest_disjoint_pairs(structure, candidates)
    return {
        "c1_s1": [pairs[0][0]],
        "c1_s2": [pairs[0][1]],
        "c2_s1": list(pairs[0]),
        "c2_s2": list(pairs[1]),
    }


def li_vacancy_concentration(parent_structure: Structure, vacancy_count: int) -> str:
    parent_li_count = len(li_indices(parent_structure))
    return f"{vacancy_count / parent_li_count:.6f}"


def create_structure_records() -> list[StructureRecord]:
    records: list[StructureRecord] = []

    bulk_ordered = build_bulk_ordered()
    bulk_rel_path = "01_structures/bulk_ordered/POSCAR"
    write_poscar(bulk_ordered, PROJECT_DIR / bulk_rel_path)
    records.append(
        StructureRecord(
            structure_id="bulk_ordered",
            rel_path=bulk_rel_path,
            structure=bulk_ordered,
            gb_type="NA",
            gb_normal="NA",
            z0_A="NA",
            delta_A="NA",
            translation_state="NA",
                li_vac_count=0,
                li_vac_conc="0.000000",
                li_vac_region="bulk",
                deleted_li_indices="",
                notes="source=done.cif; optimized_2x2x2_reference; bulk_supercell=1x1x1_from_source",
            )
        )

    bulk_vacancies = select_bulk_vacancies(bulk_ordered)
    for structure_id, deleted_indices in bulk_vacancies.items():
        defect_structure = make_vacancy(bulk_ordered, deleted_indices)
        rel_path = f"01_structures/{structure_id}/POSCAR"
        write_poscar(defect_structure, PROJECT_DIR / rel_path)
        records.append(
            StructureRecord(
                structure_id=structure_id,
                rel_path=rel_path,
                structure=defect_structure,
                gb_type="NA",
                gb_normal="NA",
                z0_A="NA",
                delta_A="NA",
                translation_state="NA",
                li_vac_count=len(deleted_indices),
                li_vac_conc=li_vacancy_concentration(bulk_ordered, len(deleted_indices)),
                li_vac_region="bulk",
                deleted_li_indices=",".join(str(index) for index in deleted_indices),
                notes=(
                    f"source=bulk_ordered; deleted_sorted_Li_indices={deleted_indices}; "
                    "selection=maximally_separated_pairs"
                ),
            )
        )

    for translation_state, shift_xy in GB_SHIFTS.items():
        gb_structure = build_sigma3_grain_boundary(shift_xy)
        base_id = f"gb_Sigma3_{translation_state}"
        rel_path = f"01_structures/{base_id}/POSCAR"
        write_poscar(gb_structure, PROJECT_DIR / rel_path)
        z0 = gb_structure.lattice.c / 2.0
        common_notes = (
            f"source=Li6PS5Cl.cif; sigma=3; axis={GB_AXIS}; plane={GB_PLANE}; "
            f"gb_gap_A={GB_GAP_A:.3f}; gb_c_repeats={GB_C_REPEATS}; shift_xy={shift_xy}"
        )
        records.append(
            StructureRecord(
                structure_id=base_id,
                rel_path=rel_path,
                structure=gb_structure,
                gb_type="Sigma3_[111]_(1-21)",
                gb_normal="z",
                z0_A=f"{z0:.6f}",
                delta_A=f"{GB_DELTA_A:.6f}",
                translation_state=translation_state,
                li_vac_count=0,
                li_vac_conc="0.000000",
                li_vac_region="within_delta",
                deleted_li_indices="",
                notes=common_notes,
            )
        )

        gb_vacancies = select_gb_vacancies(gb_structure)
        for vac_label, deleted_indices in gb_vacancies.items():
            vac_id = f"{base_id}_Li_vac_{vac_label}"
            vac_structure = make_vacancy(gb_structure, deleted_indices)
            vac_rel_path = f"01_structures/{vac_id}/POSCAR"
            write_poscar(vac_structure, PROJECT_DIR / vac_rel_path)
            records.append(
                StructureRecord(
                    structure_id=vac_id,
                    rel_path=vac_rel_path,
                    structure=vac_structure,
                    gb_type="Sigma3_[111]_(1-21)",
                    gb_normal="z",
                    z0_A=f"{z0:.6f}",
                    delta_A=f"{GB_DELTA_A:.6f}",
                    translation_state=translation_state,
                    li_vac_count=len(deleted_indices),
                    li_vac_conc=li_vacancy_concentration(gb_structure, len(deleted_indices)),
                    li_vac_region="within_delta",
                    deleted_li_indices=",".join(str(index) for index in deleted_indices),
                    notes=(
                        f"{common_notes}; deleted_sorted_Li_indices={deleted_indices}; "
                        "selection=within_delta_maximally_separated_pairs"
                    ),
                )
            )

    return records


def format_cell(structure: Structure) -> str:
    matrix = structure.lattice.matrix
    return textwrap.dedent(
        f"""\
        &CELL
          A {matrix[0][0]:.10f} {matrix[0][1]:.10f} {matrix[0][2]:.10f}
          B {matrix[1][0]:.10f} {matrix[1][1]:.10f} {matrix[1][2]:.10f}
          C {matrix[2][0]:.10f} {matrix[2][1]:.10f} {matrix[2][2]:.10f}
          PERIODIC XYZ
        &END CELL
        """
    ).rstrip()


def format_coord(structure: Structure) -> str:
    lines = ["&COORD"]
    for site in structure:
        x, y, z = site.coords
        lines.append(f"  {site.specie.symbol:<2} {x: .10f} {y: .10f} {z: .10f}")
    lines.append("&END COORD")
    return "\n".join(lines)


def format_kinds(basis_set_type: str = "DZVP") -> str:
    lines: list[str] = []
    kind_dict = KIND_LINES_DZVP if basis_set_type == "DZVP" else KIND_LINES_TZVP
    for symbol in ("Li", "P", "S", "Cl"):
        basis, potential = kind_dict[symbol]
        lines.extend(
            [
                f"&KIND {symbol}",
                f"  BASIS_SET {basis}",
                f"  POTENTIAL {potential}",
                "&END KIND",
            ]
        )
    return "\n".join(lines)


def format_subsys(structure: Structure, basis_set_type: str = "DZVP") -> str:
    blocks = ["&SUBSYS", textwrap.indent(format_cell(structure), "  "), textwrap.indent(format_coord(structure), "  ")]
    if basis_set_type is not None:
        blocks.append(textwrap.indent(format_kinds(basis_set_type), "  "))
    blocks.append("&END SUBSYS")
    return "\n".join(blocks)


def render_ext_restart(relative_restart: str) -> str:
    return textwrap.dedent(
        f"""\
        &EXT_RESTART
          RESTART_FILE_NAME {relative_restart}
          RESTART_DEFAULT F
          RESTART_POS T
          RESTART_CELL T
        &END EXT_RESTART
        """
    ).rstrip()


def defect_charge(record: StructureRecord) -> int:
    # Li+ vacancy: removing a Li+ positive ion 閳?structure gains net negative charge
    # -1 per removed Li+; bulk and GB without vacancy are neutral (charge 0)
    return -record.li_vac_count


def render_spin_settings(record: StructureRecord) -> str:
    # All structures (bulk, GB, Li+ vacancies) are closed-shell singlet systems.
    # Spin multiplicity is always 1; no unpaired electrons in any case.
    charge = defect_charge(record)
    return f"CHARGE {charge}\nMULTIPLICITY 1"


def is_gb_record(record: StructureRecord) -> bool:
    return record.gb_type != "NA" or record.structure_id.startswith("gb_")


def geo_opt_optimizer(record: StructureRecord) -> str:
    return GB_GEOM_OPTIMIZER if is_gb_record(record) else DEFAULT_GEOM_OPTIMIZER


def dft_inner_max_scf(record: StructureRecord) -> int:
    return GB_DFT_INNER_MAX_SCF if is_gb_record(record) else DFT_INNER_MAX_SCF


def dft_outer_max_scf(record: StructureRecord) -> int:
    return GB_DFT_OUTER_MAX_SCF if is_gb_record(record) else DFT_OUTER_MAX_SCF


def dft_geoopt_eps_scf(record: StructureRecord) -> str:
    return GB_DFT_EPS_SCF if is_gb_record(record) else "1.0E-6"


def dft_geoopt_max_dr(record: StructureRecord) -> str:
    return GB_DFT_MAX_DR if is_gb_record(record) else CELL_OPT_MAX_DR


def dft_geoopt_rms_dr(record: StructureRecord) -> str:
    return GB_DFT_RMS_DR if is_gb_record(record) else CELL_OPT_RMS_DR


def dft_geoopt_scf_guess(record: StructureRecord) -> str:
    return "ATOMIC" if is_gb_record(record) else "RESTART"


def dft_geoopt_ot_preconditioner(record: StructureRecord) -> str:
    return GB_DFT_OT_PRECONDITIONER if is_gb_record(record) else DFT_OT_PRECONDITIONER


def dft_geoopt_ot_energy_gap(record: StructureRecord) -> str:
    return GB_DFT_OT_ENERGY_GAP if is_gb_record(record) else DFT_OT_ENERGY_GAP


def xtb_inner_max_scf(record: StructureRecord) -> int:
    return XTB_INNER_MAX_SCF


def xtb_outer_max_scf(record: StructureRecord) -> int:
    return XTB_OUTER_MAX_SCF


def xtb_eps_scf(record: StructureRecord) -> str:
    return XTB_EPS_SCF


def xtb_ignore_scf_failure(record: StructureRecord) -> str:
    return XTB_IGNORE_SCF_FAILURE


def xtb_max_iter(record: StructureRecord) -> int:
    return CELL_OPT_MAX_ITER


def xtb_check_atomic_charges(record: StructureRecord) -> str:
    return "F"


def render_xtb_input(record: StructureRecord) -> str:
    """Render GFN1-xTB pre-optimization input using OT DIIS method."""
    project = f"{record.structure_id}_xtb_gfn1"
    optimizer = geo_opt_optimizer(record)
    bfgs_block = ""
    if optimizer == "BFGS":
        bfgs_block = textwrap.dedent(
            f"""\
            &BFGS
              TRUST_RADIUS {XTB_BFGS_TRUST_RADIUS}
            &END BFGS
            """
        ).rstrip()
    return textwrap.dedent(
        f"""\
        &GLOBAL
          PROJECT {project}
          PRINT_LEVEL MEDIUM
          RUN_TYPE GEO_OPT
        &END GLOBAL

        &FORCE_EVAL
          METHOD QS
          &DFT
            &QS
              METHOD XTB
              &XTB
                CHECK_ATOMIC_CHARGES {xtb_check_atomic_charges(record)}
                DO_EWALD  T
                USE_HALOGEN_CORRECTION T
              &END XTB
            &END QS
            &SCF
              SCF_GUESS ATOMIC
              MAX_SCF {xtb_inner_max_scf(record)}
              EPS_SCF {xtb_eps_scf(record)}
              &OT ON
                PRECONDITIONER {XTB_OT_PRECONDITIONER}
                MINIMIZER DIIS
                ENERGY_GAP {XTB_OT_ENERGY_GAP}
              &END OT
              &OUTER_SCF
                MAX_SCF {xtb_outer_max_scf(record)}
                EPS_SCF {xtb_eps_scf(record)}
              &END OUTER_SCF
              IGNORE_CONVERGENCE_FAILURE {xtb_ignore_scf_failure(record)}
              &PRINT
                &RESTART ON
                  BACKUP_COPIES 0
                &END RESTART
              &END PRINT
            &END SCF
          &END DFT
        {textwrap.indent(format_subsys(record.structure, basis_set_type=None), "  ")}
        &END FORCE_EVAL

          &MOTION
            &GEO_OPT
              OPTIMIZER {optimizer}
              TYPE MINIMIZATION
              MAX_ITER {xtb_max_iter(record)}
              MAX_DR {CELL_OPT_MAX_DR}
              RMS_DR {CELL_OPT_RMS_DR}
              MAX_FORCE {CELL_OPT_MAX_FORCE}
              RMS_FORCE {CELL_OPT_RMS_FORCE}
        {textwrap.indent(bfgs_block, "      ") if bfgs_block else ""}
            &END GEO_OPT
            &PRINT
              &RESTART ON
                FILENAME =final.restart
              BACKUP_COPIES 0
            &END RESTART
          &END PRINT
        &END MOTION
        """
    ).strip() + "\n"


def render_dft_geoopt_input(record: StructureRecord) -> str:
    """Render DFT PBE geometry optimization input (fixed cell, OT method)."""
    project = f"{record.structure_id}_dft_geoopt"
    ext_restart = "" if is_gb_record(record) else render_ext_restart("../00_xtb_gfn1/final.restart")
    scf_guess = dft_geoopt_scf_guess(record)
    spin_settings = render_spin_settings(record)
    optimizer = geo_opt_optimizer(record)
    return textwrap.dedent(
        f"""\
        &GLOBAL
          PROJECT {project}
          PRINT_LEVEL MEDIUM
          RUN_TYPE GEO_OPT
        &END GLOBAL

        {ext_restart}

        &FORCE_EVAL
          METHOD QS
          &DFT
            BASIS_SET_FILE_NAME {BASIS_PATH}
            POTENTIAL_FILE_NAME {POTENTIAL_PATH}
        {textwrap.indent(spin_settings, "    ")}
            &XC
              &XC_FUNCTIONAL PBE
              &END XC_FUNCTIONAL
            &END XC
            &MGRID
              CUTOFF 400
              REL_CUTOFF 40
            &END MGRID
            &SCF
              SCF_GUESS {scf_guess}
              MAX_SCF {dft_inner_max_scf(record)}
              EPS_SCF {dft_geoopt_eps_scf(record)}
              &OT ON
                PRECONDITIONER {dft_geoopt_ot_preconditioner(record)}
                ENERGY_GAP {dft_geoopt_ot_energy_gap(record)}
                ALGORITHM {DFT_OT_ALGORITHM}
                MINIMIZER {DFT_OT_MINIMIZER}
                LINESEARCH {DFT_OT_LINESEARCH}
              &END OT
              &OUTER_SCF
                MAX_SCF {dft_outer_max_scf(record)}
                EPS_SCF {dft_geoopt_eps_scf(record)}
              &END OUTER_SCF
              &PRINT
                &RESTART ON
                  BACKUP_COPIES 0
                &END RESTART
              &END PRINT
            &END SCF
          &END DFT
        {textwrap.indent(format_subsys(record.structure, basis_set_type="DZVP"), "  ")}
        &END FORCE_EVAL

        &MOTION
          &GEO_OPT
            OPTIMIZER {optimizer}
            TYPE MINIMIZATION
            MAX_ITER {CELL_OPT_MAX_ITER}
            MAX_DR {dft_geoopt_max_dr(record)}
            RMS_DR {dft_geoopt_rms_dr(record)}
            MAX_FORCE {CELL_OPT_MAX_FORCE}
            RMS_FORCE {CELL_OPT_RMS_FORCE}
          &END GEO_OPT
          &PRINT
            &RESTART ON
              FILENAME =final.restart
              BACKUP_COPIES 0
            &END RESTART
          &END PRINT
        &END MOTION
        """
    ).strip() + "\n"


def render_static_input(record: StructureRecord) -> str:
    project = f"{record.structure_id}_static"
    ext_restart = render_ext_restart("../01_dft_geoopt/final.restart")
    wfn_restart = f"../01_dft_geoopt/{record.structure_id}_dft_geoopt-RESTART.wfn"
    spin_settings = render_spin_settings(record)
    return textwrap.dedent(
        f"""\
        &GLOBAL
          PROJECT {project}
          PRINT_LEVEL MEDIUM
          RUN_TYPE ENERGY_FORCE
        &END GLOBAL

        {ext_restart}

        &FORCE_EVAL
          METHOD QS
          &DFT
            BASIS_SET_FILE_NAME {BASIS_PATH}
            POTENTIAL_FILE_NAME {POTENTIAL_PATH}
            WFN_RESTART_FILE_NAME {wfn_restart}
        {textwrap.indent(spin_settings, "    ")}
            &QS
              EPS_DEFAULT 1.0E-12
            &END QS
            &POISSON
              PERIODIC XYZ
              PSOLVER PERIODIC
            &END POISSON
            &XC
              &XC_FUNCTIONAL PBE
              &END XC_FUNCTIONAL
            &END XC
            &MGRID
              CUTOFF 400
              REL_CUTOFF 40
            &END MGRID
            &SCF
              SCF_GUESS RESTART
              MAX_SCF {dft_inner_max_scf(record)}
              EPS_SCF 1.0E-6
              &OT ON
                PRECONDITIONER {DFT_OT_PRECONDITIONER}
                ENERGY_GAP {DFT_OT_ENERGY_GAP}
                ALGORITHM {DFT_OT_ALGORITHM}
                MINIMIZER {DFT_OT_MINIMIZER}
                LINESEARCH {DFT_OT_LINESEARCH}
              &END OT
              &OUTER_SCF
                MAX_SCF {dft_outer_max_scf(record)}
                EPS_SCF 1.0E-6
              &END OUTER_SCF
            &END SCF
          &END DFT
        {textwrap.indent(format_subsys(record.structure, basis_set_type="TZVP"), "  ")}
        &END FORCE_EVAL
        """
    ).strip() + "\n"


def render_cp2k_singlepoint_input(record: StructureRecord) -> str:
    """Render the transport-workflow CP2K single-point input."""
    project = f"{record.structure_id}_cp2k_singlepoint"
    spin_settings = render_spin_settings(record)
    return textwrap.dedent(
        f"""\
        &GLOBAL
          PROJECT {project}
          PRINT_LEVEL MEDIUM
          RUN_TYPE ENERGY_FORCE
        &END GLOBAL

        &FORCE_EVAL
          METHOD QS
          &DFT
            BASIS_SET_FILE_NAME {BASIS_PATH}
            POTENTIAL_FILE_NAME {POTENTIAL_PATH}
        {textwrap.indent(spin_settings, "    ")}
            &QS
              EPS_DEFAULT 1.0E-12
            &END QS
            &POISSON
              PERIODIC XYZ
              PSOLVER PERIODIC
            &END POISSON
            &XC
              &XC_FUNCTIONAL PBE
              &END XC_FUNCTIONAL
            &END XC
            &MGRID
              CUTOFF 400
              REL_CUTOFF 40
            &END MGRID
            &SCF
              SCF_GUESS ATOMIC
              MAX_SCF {SP_INNER_MAX_SCF}
              EPS_SCF {SP_EPS_SCF}
              &OT ON
                PRECONDITIONER {SP_OT_PRECONDITIONER}
                ENERGY_GAP {SP_OT_ENERGY_GAP}
                ALGORITHM {DFT_OT_ALGORITHM}
                MINIMIZER {DFT_OT_MINIMIZER}
                LINESEARCH {DFT_OT_LINESEARCH}
              &END OT
              &OUTER_SCF
                MAX_SCF {SP_OUTER_MAX_SCF}
                EPS_SCF {SP_EPS_SCF}
              &END OUTER_SCF
            &END SCF
          &END DFT
        {textwrap.indent(format_subsys(record.structure, basis_set_type="TZVP"), "  ")}
        &END FORCE_EVAL
        """
    ).strip() + "\n"


def write_structure_manifest(records: list[StructureRecord]) -> None:
    manifest_path = NOTES_DIR / "structure_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURE_MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_manifest_row())


def structure_order(records: list[StructureRecord]) -> list[str]:
    preferred_prefix = [
        "bulk_ordered",
        "bulk_Li_vac_c1_s1",
        "bulk_Li_vac_c1_s2",
        "bulk_Li_vac_c2_s1",
        "bulk_Li_vac_c2_s2",
        "gb_Sigma3_t1",
        "gb_Sigma3_t1_Li_vac_c1_s1",
        "gb_Sigma3_t1_Li_vac_c1_s2",
        "gb_Sigma3_t1_Li_vac_c2_s1",
        "gb_Sigma3_t1_Li_vac_c2_s2",
        "gb_Sigma3_t2",
        "gb_Sigma3_t2_Li_vac_c1_s1",
        "gb_Sigma3_t2_Li_vac_c1_s2",
        "gb_Sigma3_t2_Li_vac_c2_s1",
        "gb_Sigma3_t2_Li_vac_c2_s2",
        "gb_Sigma3_t3",
        "gb_Sigma3_t3_Li_vac_c1_s1",
        "gb_Sigma3_t3_Li_vac_c1_s2",
        "gb_Sigma3_t3_Li_vac_c2_s1",
        "gb_Sigma3_t3_Li_vac_c2_s2",
    ]
    available = {record.structure_id for record in records}
    return [item for item in preferred_prefix if item in available]


def write_cp2k_job_order(records: list[StructureRecord]) -> None:
    order = [structure_id for structure_id in structure_order(records) if structure_id.startswith("gb_")]
    write_text(NOTES_DIR / "cp2k_job_order.txt", "\n".join(order) + "\n")


def write_workflow_manifest(records: list[StructureRecord]) -> None:
    manifest_path = NOTES_DIR / "workflow_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORKFLOW_MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            if not is_gb_record(record):
                continue
            for stage_order, (stage_name, run_type, enabled) in enumerate(STAGE_DEFS):
                restart_from = ""
                if stage_name in (CP2K_SINGLEPOINT_STAGE, UMA_MD_STAGE):
                    restart_from = f"../{PREOPT_STAGE}/CONTCAR"
                writer.writerow(
                    {
                        "structure_id": record.structure_id,
                        "stage_order": str(stage_order),
                        "stage_name": stage_name,
                        "run_type": run_type,
                        "path": f"{record.structure_id}/{stage_name}",
                        "restart_from": restart_from,
                        "enabled": "1" if enabled else "0",
                    }
                )


def write_cp2k_inputs(records: list[StructureRecord]) -> None:
    for record in records:
        if not is_gb_record(record):
            continue
        base_dir = WORKFLOWS_DIR / record.structure_id
        preopt_dir = base_dir / PREOPT_STAGE
        singlepoint_dir = base_dir / CP2K_SINGLEPOINT_STAGE
        md_dir = base_dir / UMA_MD_STAGE

        poscar_text = Poscar(record.structure).get_str()
        write_text(preopt_dir / "POSCAR", poscar_text)
        write_text(md_dir / "POSCAR", poscar_text)

        singlepoint_input = render_cp2k_singlepoint_input(record)
        write_text(singlepoint_dir / "1.inp", singlepoint_input)
        write_text(singlepoint_dir / "1.inp.base", singlepoint_input)

        for marker_name in (PREOPT_BYPASS_MARKER, "SKIP_XTB"):
            marker_path = base_dir / marker_name
            if marker_path.exists():
                marker_path.unlink()


def write_job_scripts() -> None:
    write_text(JOB_SCRIPTS_DIR / "serial_submit_cp2k.py", SERIAL_SUBMIT_TEMPLATE.read_text(encoding="utf-8"))
    write_text(JOB_SCRIPTS_DIR / "run_serial_cp2k.sh", RUN_SERIAL_SCRIPT)
    write_text(JOB_SCRIPTS_DIR / "cp2k_singlepoint_large_mem.sh", CP2K_LARGE_MEM_SCRIPT)
    write_text(JOB_SCRIPTS_DIR / "uma_preopt_cpu.sh", UMA_PREOPT_CPU_SCRIPT)
    write_text(JOB_SCRIPTS_DIR / "uma_md_cpu.sh", UMA_MD_CPU_SCRIPT)
    (JOB_SCRIPTS_DIR / "run_serial_cp2k.sh").chmod(0o755)
    (JOB_SCRIPTS_DIR / "cp2k_singlepoint_large_mem.sh").chmod(0o755)
    (JOB_SCRIPTS_DIR / "uma_preopt_cpu.sh").chmod(0o755)
    (JOB_SCRIPTS_DIR / "uma_md_cpu.sh").chmod(0o755)


def write_notes(records: list[StructureRecord]) -> None:
    n_bulk = sum(1 for record in records if record.gb_type == "NA")
    n_gb = len(records) - n_bulk
    gb_base_atoms = next((record.structure.num_sites for record in records if record.structure_id == "gb_Sigma3_t1"), "NA")
    readme = textwrap.dedent(
        f"""\
        # LPSCl Transport Workflow

        `done.cif` remains the single reference parent structure. The production route is now ML-first and aligned with the thesis goal: use machine learning to explore how defects and grain boundaries affect transport-related behavior in `Li6PS5Cl`.

        ## Workflow roots
        - Clean transport production workflow: `02_uma_workflows`
        - Legacy DFT benchmark data are kept outside this clean project

        ## Production stages in `02_uma_workflows`
        1. `{PREOPT_STAGE}`
           - UMA CPU geometry pre-optimization from `POSCAR`
           - default optimizer: `{GB_UMA_OPTIMIZER}`
           - `fmax = {GB_UMA_FMAX}`
           - `max_steps = {GB_UMA_MAX_STEPS}`
           - CPU layout: `1 MPI + {GB_UMA_THREADS} OpenMP`
           - successful runs produce `CONTCAR`
        2. `{CP2K_SINGLEPOINT_STAGE}`
           - fixed-cell CP2K high-accuracy single-point calculation
           - `RUN_TYPE ENERGY_FORCE`
           - `PBE + TZVP-MOLOPT-PBE-GTH + GTH-PBE`
           - coordinates are injected from `../{PREOPT_STAGE}/CONTCAR`
           - if UMA fails or the geometry sanity check fails, the workflow falls back to the original `POSCAR`
        3. `{UMA_MD_STAGE}` (optional, disabled by default)
           - UMA NVT MD production stage for later transport analysis
           - default temperature: `{GB_UMA_MD_TEMP_K} K`
           - default timestep: `{GB_UMA_MD_TIMESTEP_FS} fs`
           - default steps: `{GB_UMA_MD_STEPS}`
           - enabled only when the serial script is called with `--include-md`

        ## Research split
        - bulk structures are no longer part of the default serial production workflow
        - bulk DFT benchmark data are managed outside this clean project
        - the new serial workflow only covers `gb_*` structures

        ## Shared settings
        - reduced GB model uses `GB_C_REPEATS = {GB_C_REPEATS}`
        - representative reduced GB size: `{gb_base_atoms}` atoms for neutral `gb_Sigma3_t1`
        - default pre-optimization submit script: `{GB_PREOPT_SUBMIT_SCRIPT}`
        - default CP2K single-point submit script: `{CP2K_SUBMIT_SCRIPT}`
        - default UMA MD submit script: `{UMA_MD_SUBMIT_SCRIPT}`
        - default UMA offline env: `/home/ctan/uma-offline-env`
        - default UMA model: `/home/ctan/uma-m-1p1.pt`
        - `BASIS_MOLOPT`: `{BASIS_PATH}`
        - `POTENTIAL`: `{POTENTIAL_PATH}`

        ## Structure set
        - bulk structures in manifest: `{n_bulk}`
        - GB structures in manifest: `{n_gb}`
        - total structures in manifest: `{len(records)}`
        - workflow-managed GB structures: `{n_gb}`

        ## Index files
        - `00_notes/structure_manifest.csv`
        - `00_notes/workflow_manifest.csv`
        - `00_notes/cp2k_job_order.txt`

        ## Sequential submission
        ```bash
        cd LPSCl_UMA_transport_project/02_uma_workflows
        screen -S lpscl_transport
        python3 job_scripts/serial_submit_cp2k.py
        ```

        or

        ```bash
        cd LPSCl_UMA_transport_project/02_uma_workflows
        ./job_scripts/run_serial_cp2k.sh
        ```

        To include the optional UMA MD stage:

        ```bash
        python3 job_scripts/serial_submit_cp2k.py --include-md
        ```

        ## Failure policy
        - completed stages are skipped automatically
        - if `{PREOPT_STAGE}` fails or produces an unphysical geometry, the workflow writes `{PREOPT_BYPASS_MARKER}`
        - `{CP2K_SINGLEPOINT_STAGE}` then falls back to the original structure instead of stopping at the pre-optimization stage
        - CP2K single-point failure stops the remaining stages for that structure
        - failed stages are not resubmitted unless the user resets outputs or the state file

        ## Modeling note
        - Li vacancies are modeled as charged defects with `CHARGE = -li_vac_count`
        - all structures use `MULTIPLICITY 1`
        - the production route is now `UMA -> CP2K single point -> optional UMA MD`, which matches the ML-first transport-study goal
        """
    )
    write_text(NOTES_DIR / "cluster_cp2k_workflow.md", readme)


def copy_source_inputs() -> None:
    targets = [
        (SOURCE_CIF, NOTES_DIR / "source_inputs" / "done.cif"),
        (Path(__file__), NOTES_DIR / "rebuild_cp2k_project.py"),
    ]
    for source, target in targets:
        if source.resolve() == target.resolve():
            continue
        shutil.copy2(source, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the ML-first transport workflow project from done.cif.")
    parser.add_argument(
        "--gb-only",
        action="store_true",
        help="Regenerate only gb_* structures and transport workflows while preserving existing bulk outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not SOURCE_CIF.exists():
        raise FileNotFoundError(f"Missing source CIF: {SOURCE_CIF}")

    reset_project_tree(gb_only=args.gb_only)
    records = create_structure_records()
    write_structure_manifest(records)
    write_cp2k_job_order(records)
    write_workflow_manifest(records)
    write_cp2k_inputs(records)
    write_job_scripts()
    write_notes(records)
    copy_source_inputs()
    print(f"Built transport workflow project at {PROJECT_DIR}")
    if args.gb_only:
        print("Regenerated GB structures and transport workflows while preserving existing bulk outputs.")
    print(f"Generated {len(records)} structures.")


if __name__ == "__main__":
    main()
