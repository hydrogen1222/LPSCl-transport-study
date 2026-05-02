from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Geometry optimization runner."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS

from umakit.runners.base import BaseRunner
from umakit.writers.contcar import ContcarWriter
from umakit.writers.json_writer import JsonWriter
from umakit.writers.oszicar import OszicarWriter
from umakit.writers.outcar import OutcarWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class OptimizationRunner(BaseRunner):
    """Run geometry optimizations."""

    OPTIMIZERS = {
        "fire": FIRE,
        "bfgs": BFGS,
        "lbfgs": LBFGS,
    }

    def __init__(
        self,
        calculator,
        fmax: float = 0.05,
        max_steps: int = 500,
        optimizer: str = "FIRE",
        cell_opt: bool = False,
        fix_symmetry: bool = False,
        output_dir: Path | str = ".",
        write_outcar: bool = True,
        write_oszicar: bool = True,
        write_json: bool = True,
        trajectory_interval: int = 1,
        verbose: bool = True,
        job_name: str | None = None,
    ):
        super().__init__(calculator, output_dir, verbose, job_name)
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer_name = optimizer.lower()
        self.cell_opt = cell_opt
        self.fix_symmetry = fix_symmetry
        self.write_outcar = write_outcar
        self.write_oszicar = write_oszicar
        self.write_json = write_json
        self.trajectory_interval = trajectory_interval

        if self.optimizer_name not in self.OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer: {optimizer}. "
                f"Use one of: {', '.join(self.OPTIMIZERS.keys())}"
            )

        if cell_opt and not calculator.has_stress:
            self.log(
                "Warning: Stress not supported for this task, cell optimization disabled",
                level="warning",
            )
            self.cell_opt = False

    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run geometry optimization."""
        self.print_header("GEOMETRY OPTIMIZATION")

        self.log(f"Optimizer:        {self.optimizer_name.upper()}")
        self.log(f"Convergence:      fmax = {self.fmax} eV/A")
        self.log(f"Max steps:        {self.max_steps}")
        self.log(f"Cell optimization: {'Yes' if self.cell_opt else 'No'}")
        self.log(f"Fix symmetry:     {'Yes' if self.fix_symmetry else 'No'}")

        atoms = self._prepare_atoms(atoms)

        if self.fix_symmetry:
            self.log("Applying symmetry constraint")
            atoms.set_constraint(FixSymmetry(atoms))

        atoms.calc = self._get_calculator()

        optimizer_class = self.OPTIMIZERS[self.optimizer_name]
        opt_atoms = FrechetCellFilter(atoms) if self.cell_opt else atoms
        logfile = str(self.output_dir / "optimization.log")
        opt = optimizer_class(opt_atoms, logfile=logfile)

        trajectory = []

        def trajectory_callback() -> None:
            step = opt.nsteps
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            if step == 1 or step % self.trajectory_interval == 0:
                trajectory.append(
                    {
                        "step": step,
                        "energy": energy,
                        "forces": forces.copy(),
                        "natoms": len(atoms),
                    }
                )

            if step % 10 == 0 or step == 1:
                force_mags = np.linalg.norm(forces, axis=1)
                fmax_current = np.max(force_mags)
                self.log(
                    f"Step {step:4d}: E = {energy:12.6f} eV, fmax = {fmax_current:.6f} eV/A"
                )

        opt.attach(trajectory_callback)

        self.log("\nStarting optimization...")
        start_time = time.time()

        try:
            opt.run(fmax=self.fmax, steps=self.max_steps)
            converged = opt.converged()
        except Exception as e:
            self.log(f"Optimization failed: {e}", level="error")
            converged = False

        opt_time = time.time() - start_time

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress() if self.calculator.has_stress else None

        self.log(f"\nOptimization finished in {opt.nsteps} steps")
        self.log(f"Converged: {'Yes' if converged else 'No'}")

        results = {
            "energy": energy,
            "forces": forces,
            "stress": stress,
            "nsteps": opt.nsteps,
            "converged": converged,
            "fmax": self.fmax,
            "time": opt_time,
            "trajectory": trajectory,
        }

        self._write_outputs(atoms, results, trajectory)
        self._write_summary(results, atoms)
        return results

    def _write_outputs(
        self,
        atoms: Atoms,
        results: dict[str, Any],
        trajectory: list,
    ) -> None:
        """Write optimization outputs."""
        metadata = self.calculator.info()

        if self.write_outcar:
            outcar_path = self.output_dir / "OUTCAR"
            writer = OutcarWriter()
            writer.write(
                atoms,
                results,
                outcar_path,
                mode="optimization",
                task_name=self.calculator.task,
                metadata=metadata,
            )
            self.log(f"OUTCAR written to: {outcar_path}")

        if self.write_oszicar and trajectory:
            oszicar_path = self.output_dir / "OSZICAR"
            writer = OszicarWriter()
            writer.write(
                oszicar_path,
                trajectory,
                fmax_target=self.fmax,
            )
            self.log(f"OSZICAR written to: {oszicar_path}")

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
                mode="optimization",
                metadata=json_metadata,
            )
            self.log(f"JSON results written to: {json_path}")

        contcar_path = self.output_dir / "CONTCAR"
        writer = ContcarWriter()
        writer.write_with_energy(
            atoms,
            contcar_path,
            energy=results["energy"],
            forces=results["forces"],
        )
        self.log(f"CONTCAR written to: {contcar_path}")
