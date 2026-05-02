from __future__ import annotations

import argparse
import time
from pathlib import Path

from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from umakit.calculator import UMACalculator
from umakit.runners.md import MDRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UMA MD directly with explicit GPU settings.")
    parser.add_argument("--structure", required=True, help="Input structure file")
    parser.add_argument("--model", required=True, help="Path to UMA model checkpoint")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--task", default="omat", help="UMA task name")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--inference-mode", default="default", choices=["default", "turbo"], help="UMA inference mode")
    parser.add_argument("--ensemble", default="NVT", choices=["NVT", "NVE"], help="MD ensemble")
    parser.add_argument("--temp", type=float, default=600.0, help="Temperature in K")
    parser.add_argument("--timestep", type=float, default=1.0, help="Time step in fs")
    parser.add_argument("--steps", type=int, default=1000, help="Number of MD steps")
    parser.add_argument("--friction", type=float, default=0.001, help="Langevin friction")
    parser.add_argument("--save-interval", type=int, default=20, help="Trajectory save interval")
    parser.add_argument("--name", default="uma_md", help="Run name")
    parser.add_argument("--pre-relax", action="store_true", help="Enable pre-relaxation before MD")
    parser.add_argument("--pre-relax-steps", type=int, default=50, help="Max steps for pre-relax")
    parser.add_argument("--pre-relax-fmax", type=float, default=0.1, help="Force threshold for pre-relax")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    structure_file = Path(args.structure).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(" UMA GPU MD DIRECT DRIVER", flush=True)
    print("=" * 80, flush=True)
    print(f"structure       = {structure_file}", flush=True)
    print(f"model           = {model_path}", flush=True)
    print(f"output          = {output_dir}", flush=True)
    print(f"device          = {args.device}", flush=True)
    print(f"inference_mode  = {args.inference_mode}", flush=True)
    print(f"ensemble        = {args.ensemble}", flush=True)
    print(f"temperature_K   = {args.temp}", flush=True)
    print(f"timestep_fs     = {args.timestep}", flush=True)
    print(f"steps           = {args.steps}", flush=True)
    print(f"save_interval   = {args.save_interval}", flush=True)
    print(f"pre_relax       = {args.pre_relax}", flush=True)
    print("=" * 80, flush=True)

    if not structure_file.exists():
        raise FileNotFoundError(f"Missing structure file: {structure_file}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    atoms = read(structure_file)
    if args.ensemble.upper() == "NVT":
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp)
        print(f"[INFO] Initialized Maxwell-Boltzmann velocities at {args.temp} K", flush=True)

    calculator = UMACalculator(
        model_path=model_path,
        task=args.task,
        device=args.device,
        inference_mode=args.inference_mode,
    )

    runner = MDRunner(
        calculator,
        ensemble=args.ensemble,
        temperature=args.temp,
        timestep=args.timestep,
        steps=args.steps,
        friction=args.friction,
        save_interval=args.save_interval,
        output_dir=output_dir,
        verbose=True,
        job_name=args.name,
        pre_relax=args.pre_relax,
        pre_relax_steps=args.pre_relax_steps,
        pre_relax_fmax=args.pre_relax_fmax,
    )

    start = time.time()
    runner.run(atoms)
    elapsed = time.time() - start
    print(f"[INFO] MD finished in {elapsed:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
