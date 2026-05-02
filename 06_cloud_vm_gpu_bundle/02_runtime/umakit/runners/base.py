from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Base runner class for UMA calculations.

Provides common functionality for all calculation runners,
including output directory management and logging.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms

    from umakit.calculator import UMACalculator


class BaseRunner(ABC):
    """Base class for all calculation runners."""

    def __init__(
        self,
        calculator: UMACalculator,
        output_dir: Path | str = ".",
        verbose: bool = True,
        job_name: str | None = None,
    ):
        self.calculator = calculator
        self.job_name = job_name
        self.verbose = verbose

        base_dir = Path(output_dir)
        self.output_dir = base_dir / job_name if job_name else base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "info") -> None:
        """Print a log message when verbose mode is enabled."""
        if not self.verbose:
            return

        prefix = {
            "info": "  ",
            "warning": "! ",
            "error": "ERROR: ",
        }.get(level, "  ")

        print(f"{prefix}{message}")

    def print_header(self, title: str) -> None:
        """Print a section header."""
        if self.verbose:
            print()
            print("-" * 80)
            print(f" {title}")
            print("-" * 80)

    @abstractmethod
    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run a calculation and return the results."""

    def _prepare_atoms(self, atoms: Atoms) -> Atoms:
        """Prepare atoms for calculation."""
        task = self.calculator.task

        if task in ("omat", "oc20", "oc25", "odac", "omc"):
            if not atoms.pbc.any():
                self.log("Setting PBC=True for periodic system", level="warning")
                atoms.pbc = True
            cell = atoms.cell
            if cell.volume <= 0:
                raise ValueError("Invalid cell: zero volume. Check input structure.")
            self.log(
                f"Cell: {cell.lengths()[0]:.4f} x {cell.lengths()[1]:.4f} x {cell.lengths()[2]:.4f} A"
            )
        elif task == "omol":
            atoms.pbc = False
            if "charge" not in atoms.info:
                self.log("Setting default charge=0 for omol task", level="warning")
                atoms.info["charge"] = 0
            if "spin" not in atoms.info:
                self.log("Setting default spin=1 for omol task", level="warning")
                atoms.info["spin"] = 1

        return atoms

    def _get_calculator(self) -> "Calculator":
        """Get the ASE calculator instance."""
        return self.calculator.get_calculator()

    def _write_summary(self, results: dict[str, Any], atoms: Atoms) -> None:
        """Write a short text summary to stdout."""
        if not self.verbose:
            return

        energy = results.get("energy")
        forces = results.get("forces")

        print()
        print("=" * 80)
        print(f" SUMMARY - {self.job_name}" if self.job_name else " SUMMARY")
        print("=" * 80)

        if energy is not None:
            print(f"Total energy:     {energy:16.8f} eV")
            print(f"Energy per atom:  {energy / len(atoms):16.8f} eV/atom")

        if forces is not None:
            import numpy as np

            force_mags = np.linalg.norm(forces, axis=1)
            print(f"Max force:        {np.max(force_mags):16.8f} eV/A")
            print(f"RMS force:        {np.sqrt(np.mean(force_mags**2)):16.8f} eV/A")

        if results.get("stress") is not None:
            stress = results["stress"]
            pressure = -(stress[0] + stress[1] + stress[2]) / 3.0 * 160.2177
            print(f"Pressure:         {pressure:16.8f} GPa")

        calc_time = results.get("time")
        if calc_time is not None:
            print(f"Calculation time: {calc_time:16.2f} s")

        print("=" * 80)
