from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Run screen for executing calculations with live output."""

import sys
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io import read
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Log, ProgressBar, Static

if TYPE_CHECKING:
    from typing import Any


class RunScreen(Screen):
    """Screen for running calculations with live output."""

    def compose(self) -> ComposeResult:
        """Compose the run screen."""
        calc_type = self.app.get_config("calc_type", "sp")
        structure = self.app.get_config("structure_file", "Not set")

        yield Container(
            Static(f"Running: {calc_type.upper()}", id="title"),
            Static(f"Structure: {structure}", id="subtitle"),

            # Progress bar
            Static("Progress:"),
            ProgressBar(total=100, id="progress-bar"),
            Static("Initializing...", id="status-text"),

            # Log output
            Static("Log Output:", classes="section-header"),
            Log(id="run-log"),

            # Action buttons
            Horizontal(
                Button("◀ Cancel", variant="error", id="cancel-btn"),
                Button("🔄 Back", id="back-btn", disabled=True),
                id="action-buttons"
            ),
            id="run-container"
        )

    def on_mount(self) -> None:
        """Start calculation when screen mounts."""
        self.log_widget = self.query_one("#run-log", Log)
        self.progress = self.query_one("#progress-bar", ProgressBar)
        self.status = self.query_one("#status-text", Static)

        # Start calculation in background thread
        self.calculation_thread = threading.Thread(target=self._run_calculation)
        self.calculation_thread.daemon = True
        self.calculation_thread.start()

    def _log(self, message: str) -> None:
        """Add message to log."""
        self.app.call_from_thread(self.log_widget.write_line, message)

    def _update_progress(self, value: float, status: str = "") -> None:
        """Update progress bar."""
        def update():
            self.progress.update(progress=value)
            if status:
                self.status.update(status)
        self.app.call_from_thread(update)

    def _run_calculation(self) -> None:
        """Run the calculation in background thread."""
        try:
            calc_type = self.app.get_config("calc_type")

            if calc_type == "sp":
                self._run_sp()
            elif calc_type == "opt":
                self._run_opt()
            elif calc_type == "md":
                self._run_md()
            elif calc_type == "batch":
                self._run_batch()

        except Exception as e:
            self._log(f"\n❌ ERROR: {e}")
            self._log(traceback.format_exc())
            self._update_progress(0, "Failed")

        finally:
            # Enable back button
            def enable_back():
                back_btn = self.query_one("#back-btn", Button)
                back_btn.disabled = False
            self.app.call_from_thread(enable_back)

    def _run_sp(self) -> None:
        """Run single point calculation."""
        from umakit.calculator import UMACalculator
        from umakit.runners.singlepoint import SinglePointRunner

        self._log("Loading structure...")
        atoms = read(self.app.get_config("structure_file"))
        self._log(f"Loaded: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

        self._update_progress(10, "Loading model...")
        calc = UMACalculator(
            model_path=self.app.get_config("model_file"),
            task=self.app.get_config("task"),
            device=self.app.get_config("device"),
        )

        self._update_progress(30, "Running calculation...")
        runner = SinglePointRunner(
            calc,
            output_dir=self.app.get_config("output_dir"),
            verbose=False,
            job_name=self.app.get_config("job_name"),
        )

        results = runner.run(atoms)

        self._update_progress(100, "Complete")
        self._log(f"\n✅ Calculation complete!")
        self._log(f"Energy: {results['energy']:.6f} eV")
        self._log(f"Output: {self.app.get_config('output_dir')}")

    def _run_opt(self) -> None:
        """Run geometry optimization."""
        from umakit.calculator import UMACalculator
        from umakit.runners.optimization import OptimizationRunner

        self._log("Loading structure...")
        atoms = read(self.app.get_config("structure_file"))
        self._log(f"Loaded: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

        self._update_progress(10, "Loading model...")
        calc = UMACalculator(
            model_path=self.app.get_config("model_file"),
            task=self.app.get_config("task"),
            device=self.app.get_config("device"),
        )

        self._update_progress(20, "Running optimization...")
        runner = OptimizationRunner(
            calc,
            fmax=self.app.get_config("fmax"),
            max_steps=self.app.get_config("max_steps"),
            optimizer=self.app.get_config("optimizer"),
            cell_opt=self.app.get_config("cell_opt"),
            fix_symmetry=self.app.get_config("fix_symmetry"),
            output_dir=self.app.get_config("output_dir"),
            verbose=False,
            job_name=self.app.get_config("job_name"),
        )

        # Custom progress callback
        original_callback = None

        results = runner.run(atoms)

        self._update_progress(100, "Complete")
        self._log(f"\n✅ Optimization complete!")
        self._log(f"Converged: {results['converged']}")
        self._log(f"Steps: {results['nsteps']}")
        self._log(f"Final energy: {results['energy']:.6f} eV")

    def _run_md(self) -> None:
        """Run molecular dynamics."""
        from umakit.calculator import UMACalculator
        from umakit.runners.md import MDRunner

        self._log("Loading structure...")
        atoms = read(self.app.get_config("structure_file"))
        self._log(f"Loaded: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

        self._update_progress(10, "Loading model...")
        calc = UMACalculator(
            model_path=self.app.get_config("model_file"),
            task=self.app.get_config("task"),
            device=self.app.get_config("device"),
        )

        self._update_progress(20, "Setting up MD...")
        runner = MDRunner(
            calc,
            ensemble=self.app.get_config("ensemble"),
            temperature=self.app.get_config("temperature"),
            timestep=self.app.get_config("timestep"),
            steps=self.app.get_config("md_steps"),
            save_interval=self.app.get_config("save_interval"),
            output_dir=self.app.get_config("output_dir"),
            verbose=False,
            job_name=self.app.get_config("job_name"),
            # NEW: Pre-relaxation options
            pre_relax=self.app.get_config("pre_relax", True),
            pre_relax_steps=self.app.get_config("pre_relax_steps", 50),
            pre_relax_fmax=self.app.get_config("pre_relax_fmax", 0.1),
        )

        self._log(f"Ensemble: {self.app.get_config('ensemble')}")
        self._log(f"Temperature: {self.app.get_config('temperature')} K")
        self._log(f"Steps: {self.app.get_config('md_steps')}")

        if self.app.get_config("pre_relax", True):
            self._log("Pre-relaxation: Enabled")

        results = runner.run(atoms)

        self._update_progress(100, "Complete")
        self._log(f"\n✅ MD simulation complete!")
        self._log(f"Final temperature: {results['temperature']:.1f} K")
        self._log(f"Steps completed: {results['md_steps']}")

    def _run_batch(self) -> None:
        """Run batch processing."""
        self._log("Batch processing not yet implemented in TUI")
        self._log("Please use command line: uma_calc batch ...")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "cancel-btn":
            self._log("\n⚠️ Cancel requested (finishing current step)...")

        elif button_id == "back-btn":
            self.app.pop_screen()


from textual.app import ComposeResult
