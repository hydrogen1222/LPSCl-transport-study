from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Single point calculation runner."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

from umakit.runners.base import BaseRunner
from umakit.writers.contcar import ContcarWriter
from umakit.writers.json_writer import JsonWriter
from umakit.writers.outcar import OutcarWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class SinglePointRunner(BaseRunner):
    """Run single-point calculations."""

    def __init__(
        self,
        calculator,
        output_dir: Path | str = ".",
        write_outcar: bool = True,
        write_json: bool = True,
        write_contcar: bool = True,
        verbose: bool = True,
        job_name: str | None = None,
    ):
        super().__init__(calculator, output_dir, verbose, job_name)
        self.write_outcar = write_outcar
        self.write_json = write_json
        self.write_contcar = write_contcar

    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run a single-point calculation."""
        self.print_header("SINGLE POINT CALCULATION")

        atoms = self._prepare_atoms(atoms)

        self.log(f"Cell: {' x '.join([f'{x:.4f}' for x in atoms.cell.lengths()])} A")
        self.log(f"PBC: {atoms.pbc}")
        self.log(f"Volume: {atoms.cell.volume:.2f} A^3")

        if atoms.cell.volume <= 0:
            raise ValueError("Invalid cell: zero or negative volume. Check input structure.")

        atoms.calc = self._get_calculator()

        self.log("Calculating energy and forces...")
        start_time = time.time()

        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        except ValueError as e:
            error_msg = str(e)
            if "No edges found" in error_msg:
                raise RuntimeError(
                    "\n"
                    + "=" * 70
                    + "\nCALCULATION FAILED: No edges found in structure\n"
                    + "=" * 70
                    + "\n\n"
                    + "The model could not build a neighbor list for your structure.\n\n"
                    + "Common causes:\n"
                    + "  1. Atoms are too far apart (>6 A cutoff)\n"
                    + "  2. Cell is too large or has wrong PBC settings\n"
                    + "  3. Structure is not periodic but should be (or vice versa)\n\n"
                    + "Debug information:\n"
                    + f"  Cell lengths: {atoms.cell.lengths()}\n"
                    + f"  Cell volume: {atoms.cell.volume:.2f} A^3\n"
                    + f"  PBC: {atoms.pbc}\n"
                    + f"  Number of atoms: {len(atoms)}\n\n"
                    + "Suggestions:\n"
                    + "  - Check that the input structure file is correct\n"
                    + "  - For bulk materials, ensure cell is not too large\n"
                    + "  - Try the original POSCAR format instead of CIF\n"
                    + "=" * 70
                ) from e
            raise

        stress = None
        if self.calculator.has_stress:
            self.log("Calculating stress...")
            stress = atoms.get_stress()

        calc_time = time.time() - start_time

        self.log(f"Energy: {energy:.6f} eV")
        self.log(f"Calculation completed in {calc_time:.2f} s")

        results = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "time": calc_time,
        }

        self._write_outputs(atoms, results)
        self._write_summary(results, atoms)
        return results

    def _write_outputs(self, atoms: Atoms, results: dict[str, Any]) -> None:
        """Write standard output files."""
        metadata = self.calculator.info()

        if self.write_outcar:
            outcar_path = self.output_dir / "OUTCAR"
            writer = OutcarWriter()
            writer.write(
                atoms,
                results,
                outcar_path,
                mode="single_point",
                task_name=self.calculator.task,
                metadata=metadata,
            )
            self.log(f"OUTCAR written to: {outcar_path}")

        if self.write_json:
            json_path = self.output_dir / "uma_results.json"
            writer = JsonWriter()
            json_metadata = metadata.copy() if metadata else {}
            if self.job_name:
                json_metadata["job_name"] = self.job_name
            writer.write(
                atoms,
                results,
                json_path,
                mode="single_point",
                metadata=json_metadata,
            )
            self.log(f"JSON results written to: {json_path}")

        if self.write_contcar:
            contcar_path = self.output_dir / "CONTCAR"
            writer = ContcarWriter()
            writer.write_with_energy(
                atoms,
                contcar_path,
                energy=results["energy"],
                forces=results["forces"],
            )
            self.log(f"CONTCAR written to: {contcar_path}")
