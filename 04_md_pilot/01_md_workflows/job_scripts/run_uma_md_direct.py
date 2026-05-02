from __future__ import annotations

import os
import sys
from pathlib import Path

from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from umakit.calculator import UMACalculator
from umakit.runners.md import MDRunner


def getenv_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def getenv_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def getenv_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def getenv_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    structure_file = Path(getenv_str("UMA_STRUCTURE_FILE", "POSCAR"))
    model_path = Path(getenv_str("UMA_MODEL_PATH", "/home/ctan/uma-m-1p1.pt"))
    task_name = getenv_str("UMA_TASK_NAME", "omat")
    device = getenv_str("UMA_DEVICE", "cpu")
    run_name = getenv_str("UMA_RUN_NAME", "uma_md")
    output_root = Path(getenv_str("UMA_OUTPUT_ROOT", str(Path.cwd())))

    ensemble = getenv_str("UMA_MD_ENSEMBLE", "NVT")
    temperature = getenv_float("UMA_MD_TEMP", 600.0)
    timestep = getenv_float("UMA_MD_TIMESTEP", 1.0)
    steps = getenv_int("UMA_MD_STEPS", 5000)
    friction = getenv_float("UMA_MD_FRICTION", 0.001)
    save_interval = getenv_int("UMA_MD_SAVE_INTERVAL", 20)
    inference_mode = getenv_str("UMA_INFERENCE_MODE", "default")
    pre_relax = getenv_bool("UMA_MD_PRE_RELAX", False)
    pre_relax_steps = getenv_int("UMA_MD_PRE_RELAX_STEPS", 50)
    pre_relax_fmax = getenv_float("UMA_MD_PRE_RELAX_FMAX", 0.1)

    print("=" * 80, flush=True)
    print(" UMA MD DIRECT DRIVER", flush=True)
    print("=" * 80, flush=True)
    print(f"structure = {structure_file}", flush=True)
    print(f"model = {model_path}", flush=True)
    print(f"task = {task_name}", flush=True)
    print(f"device = {device}", flush=True)
    print(f"ensemble = {ensemble}", flush=True)
    print(f"temperature_K = {temperature}", flush=True)
    print(f"timestep_fs = {timestep}", flush=True)
    print(f"steps = {steps}", flush=True)
    print(f"save_interval = {save_interval}", flush=True)
    print(f"pre_relax = {pre_relax}", flush=True)
    print(f"output_root = {output_root}", flush=True)
    print(f"run_name = {run_name}", flush=True)
    print("=" * 80, flush=True)

    if not structure_file.exists():
        print(f"Missing structure file: {structure_file}", file=sys.stderr, flush=True)
        return 1
    if not model_path.exists():
        print(f"Missing model file: {model_path}", file=sys.stderr, flush=True)
        return 1

    atoms = read(structure_file)
    if ensemble.strip().upper() == "NVT":
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        print(f"initialized Maxwell-Boltzmann velocities at {temperature} K", flush=True)
    calculator = UMACalculator(
        model_path=model_path,
        task=task_name,
        device=device,
        inference_mode=inference_mode,
    )
    runner = MDRunner(
        calculator,
        ensemble=ensemble,
        temperature=temperature,
        timestep=timestep,
        steps=steps,
        friction=friction,
        save_interval=save_interval,
        output_dir=output_root,
        verbose=True,
        job_name=run_name,
        pre_relax=pre_relax,
        pre_relax_steps=pre_relax_steps,
        pre_relax_fmax=pre_relax_fmax,
    )
    runner.run(atoms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
