# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Command-line interface for UMA Calculator.

Provides subcommands for different calculation types:
- run: Run from INCAR configuration file
- sp: Single point calculation
- opt: Geometry optimization
- md: Molecular dynamics
- batch: Batch processing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io import read

from umakit.calculator import UMACalculator
from umakit.config import IncarConfig, get_default_config
from umakit.runners.batch import BatchRunner
from umakit.runners.md import MDRunner
from umakit.runners.optimization import OptimizationRunner
from umakit.runners.singlepoint import SinglePointRunner

if TYPE_CHECKING:
    from typing import Any


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="uma_calc",
        description="UMA Calculator - VASP-like interface for FAIRChem UMA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from INCAR file
  uma_calc run

  # Single point calculation
  uma_calc sp structure.cif --model uma-s-1.pt --task omat

  # Geometry optimization with cell relaxation
  uma_calc opt structure.cif --cell-opt --fmax 0.02

  # Molecular dynamics (NVT)
  uma_calc md structure.cif --ensemble NVT --temp 300 --steps 10000

  # Batch processing
  uma_calc batch structures/ --pattern "*.cif" --output results/

  # Generate template INCAR
  uma_calc template sp
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run calculation from INCAR file",
        description="Read configuration from INCAR.uma and run calculation",
    )
    run_parser.add_argument(
        "-i", "--incar",
        type=str,
        default="INCAR.uma",
        help="Path to INCAR configuration file (default: INCAR.uma)",
    )
    run_parser.add_argument(
        "-s", "--structure",
        type=str,
        default=None,
        help="Structure file (default: POSCAR, CONTCAR, or from INCAR)",
    )
    run_parser.add_argument(
        "-o", "--output",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )

    # sp command
    sp_parser = subparsers.add_parser(
        "sp",
        help="Single point calculation",
        description="Calculate energy, forces, and stress",
    )
    sp_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file (CIF, XYZ, POSCAR, etc.)",
    )
    sp_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to UMA model checkpoint (.pt file)",
    )
    sp_parser.add_argument(
        "--task",
        type=str,
        default="omat",
        choices=["omat", "omol", "oc20", "oc25", "odac", "omc"],
        help="Task type (default: omat)",
    )
    sp_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cpu)",
    )
    sp_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".",
        help="Output directory",
    )
    sp_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )

    # opt command
    opt_parser = subparsers.add_parser(
        "opt",
        help="Geometry optimization",
        description="Optimize atomic positions and optionally cell parameters",
    )
    opt_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file",
    )
    opt_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to UMA model checkpoint",
    )
    opt_parser.add_argument(
        "--task",
        type=str,
        default="omat",
        choices=["omat", "omol", "oc20", "oc25", "odac", "omc"],
        help="Task type (default: omat)",
    )
    opt_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cpu)",
    )
    opt_parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence threshold in eV/Å (default: 0.05)",
    )
    opt_parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum optimization steps (default: 500)",
    )
    opt_parser.add_argument(
        "--optimizer",
        type=str,
        default="FIRE",
        choices=["FIRE", "BFGS", "LBFGS"],
        help="Optimization algorithm (default: FIRE)",
    )
    opt_parser.add_argument(
        "--cell-opt",
        action="store_true",
        help="Optimize cell parameters (requires stress support)",
    )
    opt_parser.add_argument(
        "--fix-symmetry",
        action="store_true",
        help="Preserve crystal symmetry during optimization",
    )
    opt_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".",
        help="Output directory",
    )
    opt_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )

    # md command
    md_parser = subparsers.add_parser(
        "md",
        help="Molecular dynamics",
        description="Run MD simulation (NVT or NVE ensemble)",
    )
    md_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file",
    )
    md_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to UMA model checkpoint",
    )
    md_parser.add_argument(
        "--task",
        type=str,
        default="omat",
        choices=["omat", "omol", "oc20", "oc25", "odac", "omc"],
        help="Task type (default: omat)",
    )
    md_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cpu)",
    )
    md_parser.add_argument(
        "--ensemble",
        type=str,
        default="NVT",
        choices=["NVT", "NVE"],
        help="MD ensemble (default: NVT)",
    )
    md_parser.add_argument(
        "--temp",
        type=float,
        default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    md_parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Time step in femtoseconds (default: 1.0)",
    )
    md_parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of MD steps (default: 1000)",
    )
    md_parser.add_argument(
        "--friction",
        type=float,
        default=0.001,
        help="Friction coefficient for NVT (default: 0.001)",
    )
    md_parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Interval for saving trajectory frames (default: 10)",
    )
    md_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".",
        help="Output directory",
    )
    md_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )

    # batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch processing",
        description="Process multiple structures in batch mode",
    )
    batch_parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing structure files",
    )
    batch_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to UMA model checkpoint",
    )
    batch_parser.add_argument(
        "--task",
        type=str,
        default="omat",
        choices=["omat", "omol", "oc20", "oc25", "odac", "omc"],
        help="Task type (default: omat)",
    )
    batch_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cpu)",
    )
    batch_parser.add_argument(
        "--calc-type",
        type=str,
        default="sp",
        choices=["sp", "opt"],
        help="Type of calculation (default: sp)",
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default="*.cif",
        help="File pattern to match (default: *.cif)",
    )
    batch_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="batch_results",
        help="Output directory (default: batch_results)",
    )
    batch_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )

    # template command
    template_parser = subparsers.add_parser(
        "template",
        help="Generate template INCAR files",
        description="Generate template configuration files",
    )
    template_parser.add_argument(
        "type",
        choices=["sp", "opt", "md"],
        help="Type of template to generate",
    )
    template_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file name (default: INCAR.{type})",
    )

    # tui command
    tui_parser = subparsers.add_parser(
        "tui",
        help="Launch interactive TUI mode",
        description="Launch interactive terminal UI for visual configuration",
    )

    return parser


def print_header():
    """Print UMA Calc header."""
    print("=" * 80)
    print(" UMA CALCULATOR".center(80))
    print(" (Universal Material Application - FAIRChem)".center(80))
    print("=" * 80)
    print()


def cmd_run(args: argparse.Namespace) -> int:
    """Execute 'run' command."""
    # Load configuration
    incar_path = Path(args.incar)
    if not incar_path.exists():
        print(f"Error: INCAR file not found: {incar_path}")
        return 1

    config = IncarConfig.from_file(incar_path)

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    # Determine structure file
    structure_file = args.structure
    if structure_file is None:
        # Try common defaults
        for default in ["POSCAR", "CONTCAR", "structure.cif", "structure.xyz"]:
            if Path(default).exists():
                structure_file = default
                break

    if structure_file is None:
        print("Error: No structure file specified and no default found")
        return 1

    structure_path = Path(structure_file)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    # Read structure
    print(f"Reading structure from: {structure_path}")
    try:
        atoms = read(structure_path)
    except Exception as e:
        print(f"Error reading structure: {e}")
        return 1

    print(f"System: {atoms.get_chemical_formula()}")
    print(f"Atoms: {len(atoms)}")
    print()

    # Setup calculator
    model_path = config.get_str("MODEL_PATH", "uma-s-1.pt")
    task = config.get_str("TASK", "omat")
    device = config.get_str("DEVICE", "cpu")
    inference_mode = config.get_str("INFERENCE_MODE", "default")

    print(f"Loading model: {model_path}")
    try:
        calculator = UMACalculator(
            model_path=model_path,
            task=task,
            device=device,
            inference_mode=inference_mode,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Determine calculation type
    calc_type = config.get_str("CALC_TYPE", "sp").lower()

    # Run calculation
    output_dir = Path(args.output)

    # Get job name from config if available
    job_name = config.get_str("JOB_NAME", None)

    if calc_type == "sp":
        runner = SinglePointRunner(
            calculator,
            output_dir=output_dir,
            verbose=True,
            job_name=job_name,
        )
        runner.run(atoms)
    elif calc_type == "opt":
        runner = OptimizationRunner(
            calculator,
            fmax=config.get_float("FMAX", 0.05),
            max_steps=config.get_int("MAX_STEPS", 500),
            optimizer=config.get_str("OPT_ALGO", "FIRE"),
            cell_opt=config.get_bool("CELL_OPT", False),
            fix_symmetry=config.get_bool("FIX_SYMMETRY", False),
            output_dir=output_dir,
            verbose=True,
            job_name=job_name,
        )
        runner.run(atoms)
    elif calc_type == "md":
        runner = MDRunner(
            calculator,
            ensemble=config.get_str("MD_ENSEMBLE", "NVT"),
            temperature=config.get_float("TEMPERATURE", 300.0),
            timestep=config.get_float("TIMESTEP", 1.0),
            steps=config.get_int("STEPS", 1000),
            friction=config.get_float("FRICTION", 0.001),
            save_interval=config.get_int("SAVE_INTERVAL", 10),
            output_dir=output_dir,
            verbose=True,
            job_name=job_name,
        )
        runner.run(atoms)
    else:
        print(f"Error: Unknown calculation type: {calc_type}")
        return 1

    return 0


def cmd_sp(args: argparse.Namespace) -> int:
    """Execute 'sp' command."""
    # Read structure
    structure_path = Path(args.structure)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    print(f"Reading structure from: {structure_path}")
    try:
        atoms = read(structure_path)
    except Exception as e:
        print(f"Error reading structure: {e}")
        return 1

    print(f"System: {atoms.get_chemical_formula()}")
    print(f"Atoms: {len(atoms)}")
    print()

    # Setup calculator
    print(f"Loading model: {args.model}")
    try:
        calculator = UMACalculator(
            model_path=args.model,
            task=args.task,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Run calculation
    output_dir = Path(args.output)
    runner = SinglePointRunner(
        calculator,
        output_dir=output_dir,
        verbose=True,
        job_name=args.name,
    )
    try:
        runner.run(atoms)
    except (RuntimeError, ValueError) as e:
        print(f"\n{e}")
        return 1
    except Exception as e:
        print(f"\nError during calculation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_opt(args: argparse.Namespace) -> int:
    """Execute 'opt' command."""
    # Read structure
    structure_path = Path(args.structure)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    print(f"Reading structure from: {structure_path}")
    try:
        atoms = read(structure_path)
    except Exception as e:
        print(f"Error reading structure: {e}")
        return 1

    print(f"System: {atoms.get_chemical_formula()}")
    print(f"Atoms: {len(atoms)}")
    print()

    # Setup calculator
    print(f"Loading model: {args.model}")
    try:
        calculator = UMACalculator(
            model_path=args.model,
            task=args.task,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Run calculation
    output_dir = Path(args.output)
    runner = OptimizationRunner(
        calculator,
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
        cell_opt=args.cell_opt,
        fix_symmetry=args.fix_symmetry,
        output_dir=output_dir,
        verbose=True,
        job_name=args.name,
    )
    runner.run(atoms)

    return 0


def cmd_md(args: argparse.Namespace) -> int:
    """Execute 'md' command."""
    # Read structure
    structure_path = Path(args.structure)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    print(f"Reading structure from: {structure_path}")
    try:
        atoms = read(structure_path)
    except Exception as e:
        print(f"Error reading structure: {e}")
        return 1

    print(f"System: {atoms.get_chemical_formula()}")
    print(f"Atoms: {len(atoms)}")
    print()

    # Setup calculator with turbo mode for MD
    print(f"Loading model: {args.model}")
    try:
        calculator = UMACalculator(
            model_path=args.model,
            task=args.task,
            device=args.device,
            inference_mode="turbo" if args.device != "cpu" else "default",
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Run calculation
    output_dir = Path(args.output)
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
    )
    runner.run(atoms)

    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Execute 'batch' command."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Setup calculator
    print(f"Loading model: {args.model}")
    try:
        calculator = UMACalculator(
            model_path=args.model,
            task=args.task,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Run batch
    output_dir = Path(args.output)
    runner = BatchRunner(
        calculator,
        calc_type=args.calc_type,
        output_dir=output_dir,
        verbose=True,
        job_name=args.name,
    )
    summary = runner.run_from_directory(input_dir, pattern=args.pattern)

    if summary["failed"] > 0:
        return 1
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    """Execute 'template' command."""
    config = get_default_config(args.type)

    output_file = args.output
    if output_file is None:
        output_file = f"INCAR.{args.type}"

    config.write(output_file)
    print(f"Template written to: {output_file}")

    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    """Launch interactive TUI mode."""
    try:
        from umakit.tui import UmaCalcApp
    except ImportError as e:
        print("Error: TUI mode requires textual.")
        print(f"Install with: pip install textual")
        print(f"Import error: {e}")
        return 1

    app = UmaCalcApp()
    return app.run()


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    # Check if running in TUI mode (no command, or explicit 'tui' command)
    if argv is None:
        argv = sys.argv[1:]

    # If no arguments provided, launch TUI by default
    if len(argv) == 0:
        try:
            from umakit.tui import UmaCalcApp
            print("Launching interactive TUI mode...")
            print("(Use --help for command-line interface)")
            import time
            time.sleep(0.5)
            app = UmaCalcApp()
            return app.run()
        except ImportError:
            # Fall back to CLI help if textual not installed
            pass

    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle TUI command
    if args.command == "tui":
        return cmd_tui(args)

    if args.command is None:
        parser.print_help()
        return 1

    print_header()

    # Dispatch to appropriate command handler
    commands = {
        "run": cmd_run,
        "sp": cmd_sp,
        "opt": cmd_opt,
        "md": cmd_md,
        "batch": cmd_batch,
        "template": cmd_template,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Error: Unknown command: {args.command}")
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
