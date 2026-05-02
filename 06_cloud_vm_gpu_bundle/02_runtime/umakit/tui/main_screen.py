from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Main screen for UMA Calculator TUI."""

from pathlib import Path
from typing import TYPE_CHECKING

from textual.containers import Container, Grid, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    DirectoryTree,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    Switch,
)

if TYPE_CHECKING:
    from typing import Any


class MainScreen(Screen):
    """Main menu screen for selecting calculation type."""

    def compose(self) -> ComposeResult:
        """Compose the main screen."""
        yield Container(
            Static("Select Calculation Type", id="title"),
            Static("Choose the type of calculation to run", id="subtitle"),
            ListView(
                ListItem(
                    Static("📊 Single Point (SP)\n   Calculate energy, forces, and stress"),
                    id="sp"
                ),
                ListItem(
                    Static("🔧 Geometry Optimization (OPT)\n   Optimize atomic positions"),
                    id="opt"
                ),
                ListItem(
                    Static("🌡️  Molecular Dynamics (MD)\n   Run NVT/NVE simulations"),
                    id="md"
                ),
                ListItem(
                    Static("📁 Batch Processing\n   Process multiple structures"),
                    id="batch"
                ),
                ListItem(
                    Static("📝 Generate Template\n   Create INCAR template file"),
                    id="template"
                ),
                ListItem(
                    Static("❌ Exit\n   Quit the application"),
                    id="exit"
                ),
                id="calc-type-list"
            ),
            id="main-container"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection from list view."""
        item_id = event.item.id

        if item_id == "exit":
            self.app.exit()
            return

        if item_id == "template":
            self.app.push_screen("template")
            return

        # Update config and go to config screen
        self.app.update_config("calc_type", item_id)
        self.app.push_screen("config")


class ConfigScreen(Screen):
    """Configuration screen for setting up calculation parameters."""

    def compose(self) -> ComposeResult:
        """Compose the configuration screen."""
        calc_type = self.app.get_config("calc_type", "sp")

        yield Container(
            Static(f"Configuration: {calc_type.upper()}", id="title"),

            # File selection section
            Static("📁 File Selection", classes="section-header"),
            Grid(
                Label("Structure File:"),
                Input(
                    placeholder="Path to structure file (CIF, POSCAR, XYZ)",
                    id="structure-input"
                ),
                Label("Model File:"),
                Input(
                    placeholder="Path to model (.pt file)",
                    value=str(self.app.get_config("model_file", "")),
                    id="model-input"
                ),
                Label("Output Directory:"),
                Input(
                    value=self.app.get_config("output_dir", "./results"),
                    id="output-input"
                ),
                classes="config-grid"
            ),

            # Task and device section
            Static("⚙️  Calculation Settings", classes="section-header"),
            Grid(
                Label("Task:"),
                self._create_task_selector(),
                Label("Device:"),
                self._create_device_selector(),
                classes="config-grid"
            ),

            # Calculation-specific options
            Static("🔧 Calculation Options", classes="section-header"),
            self._create_calc_options(calc_type),

            # Action buttons
            Horizontal(
                Button("◀ Back", id="back-btn"),
                Button("🚀 Run Calculation", variant="success", id="run-btn"),
                id="action-buttons"
            ),
            id="config-panel"
        )

    def _create_task_selector(self) -> ListView:
        """Create task type selector."""
        current_task = self.app.get_config("task", "omat")
        tasks = [
            ("omat", "Inorganic Materials"),
            ("omol", "Molecules"),
            ("oc20", "Catalysis (OC20)"),
            ("oc25", "Catalysis (OC25)"),
            ("odac", "MOFs"),
            ("omc", "Molecular Crystals"),
        ]

        items = []
        for task_id, task_name in tasks:
            marker = "● " if task_id == current_task else "○ "
            items.append(ListItem(Static(f"{marker}{task_name}"), id=task_id))

        return ListView(*items, id="task-selector")

    def _create_device_selector(self) -> ListView:
        """Create device selector."""
        current_device = self.app.get_config("device", "cpu")
        devices = [
            ("cpu", "CPU"),
            ("cuda", "CUDA (GPU)"),
        ]

        items = []
        for dev_id, dev_name in devices:
            marker = "● " if dev_id == current_device else "○ "
            items.append(ListItem(Static(f"{marker}{dev_name}"), id=dev_id))

        return ListView(*items, id="device-selector")

    def _create_calc_options(self, calc_type: str) -> Container:
        """Create calculation-specific options."""
        if calc_type == "sp":
            return Container(
                Horizontal(
                    Label("Write Forces:"),
                    Switch(value=self.app.get_config("write_forces", True), id="write-forces"),
                ),
                Horizontal(
                    Label("Write Stress:"),
                    Switch(value=self.app.get_config("write_stress", True), id="write-stress"),
                ),
            )

        elif calc_type == "opt":
            return Container(
                Grid(
                    Label("Fmax (eV/Å):"),
                    Input(value=str(self.app.get_config("fmax", 0.05)), id="fmax-input"),
                    Label("Max Steps:"),
                    Input(value=str(self.app.get_config("max_steps", 500)), id="max-steps-input"),
                    Label("Optimizer:"),
                    self._create_optimizer_selector(),
                    classes="config-grid"
                ),
                Horizontal(
                    Label("Cell Optimization:"),
                    Switch(value=self.app.get_config("cell_opt", False), id="cell-opt"),
                    Label("Fix Symmetry:"),
                    Switch(value=self.app.get_config("fix_symmetry", False), id="fix-symmetry"),
                ),
            )

        elif calc_type == "md":
            return Container(
                Grid(
                    Label("Ensemble:"),
                    self._create_ensemble_selector(),
                    Label("Temperature (K):"),
                    Input(value=str(self.app.get_config("temperature", 300.0)), id="temp-input"),
                    Label("Timestep (fs):"),
                    Input(value=str(self.app.get_config("timestep", 1.0)), id="timestep-input"),
                    Label("Steps:"),
                    Input(value=str(self.app.get_config("md_steps", 1000)), id="md-steps-input"),
                    Label("Save Interval:"),
                    Input(value=str(self.app.get_config("save_interval", 10)), id="save-interval-input"),
                    classes="config-grid"
                ),
                Horizontal(
                    Label("Pre-relaxation:"),
                    Switch(value=self.app.get_config("pre_relax", True), id="pre-relax"),
                    Label("Pre-relax Steps:"),
                    Input(value=str(self.app.get_config("pre_relax_steps", 50)), id="pre-relax-steps"),
                ),
            )

        elif calc_type == "batch":
            return Container(
                Grid(
                    Label("Pattern:"),
                    Input(value=self.app.get_config("pattern", "*.cif"), id="pattern-input"),
                    Label("Calc Type:"),
                    self._create_batch_calc_selector(),
                    classes="config-grid"
                ),
            )

        return Container(Static("No additional options"))

    def _create_optimizer_selector(self) -> ListView:
        """Create optimizer selector."""
        current = self.app.get_config("optimizer", "FIRE")
        optimizers = ["FIRE", "BFGS", "LBFGS"]

        items = []
        for opt in optimizers:
            marker = "● " if opt == current else "○ "
            items.append(ListItem(Static(f"{marker}{opt}"), id=opt.lower()))

        return ListView(*items, id="optimizer-selector")

    def _create_ensemble_selector(self) -> ListView:
        """Create MD ensemble selector."""
        current = self.app.get_config("ensemble", "NVT")
        ensembles = ["NVT", "NVE"]

        items = []
        for ens in ensembles:
            marker = "● " if ens == current else "○ "
            items.append(ListItem(Static(f"{marker}{ens}"), id=ens.lower()))

        return ListView(*items, id="ensemble-selector")

    def _create_batch_calc_selector(self) -> ListView:
        """Create batch calculation type selector."""
        calcs = [("sp", "Single Point"), ("opt", "Geometry Optimization")]
        items = [ListItem(Static(f"○ {name}"), id=cid) for cid, name in calcs]
        return ListView(*items, id="batch-calc-selector")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "back-btn":
            self.app.pop_screen()

        elif button_id == "run-btn":
            self._save_config()
            self.app.push_screen("run")

    def _save_config(self) -> None:
        """Save configuration from UI inputs."""
        # Save file paths
        structure = self.query_one("#structure-input", Input).value
        if structure:
            self.app.update_config("structure_file", structure)

        model = self.query_one("#model-input", Input).value
        if model:
            self.app.update_config("model_file", model)

        output = self.query_one("#output-input", Input).value
        if output:
            self.app.update_config("output_dir", output)

        # Save calculation-specific options
        calc_type = self.app.get_config("calc_type")

        if calc_type == "opt":
            try:
                fmax = float(self.query_one("#fmax-input", Input).value)
                self.app.update_config("fmax", fmax)
            except ValueError:
                pass

            try:
                max_steps = int(self.query_one("#max-steps-input", Input).value)
                self.app.update_config("max_steps", max_steps)
            except ValueError:
                pass

        elif calc_type == "md":
            try:
                temp = float(self.query_one("#temp-input", Input).value)
                self.app.update_config("temperature", temp)
            except ValueError:
                pass

            try:
                steps = int(self.query_one("#md-steps-input", Input).value)
                self.app.update_config("md_steps", steps)
            except ValueError:
                pass


class TemplateScreen(Screen):
    """Template generation screen."""

    def compose(self) -> ComposeResult:
        """Compose the template screen."""
        yield Container(
            Static("Generate INCAR Template", id="title"),
            Static("Select template type:", id="subtitle"),
            ListView(
                ListItem(Static("Single Point (SP)"), id="sp"),
                ListItem(Static("Geometry Optimization (OPT)"), id="opt"),
                ListItem(Static("Molecular Dynamics (MD)"), id="md"),
                id="template-list"
            ),
            Horizontal(
                Button("◀ Back", id="back-btn"),
                Button("Generate", variant="success", id="generate-btn"),
            ),
            id="main-container"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle template type selection."""
        self.selected_type = event.item.id

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "back-btn":
            self.app.pop_screen()

        elif button_id == "generate-btn":
            self._generate_template()

    def _generate_template(self) -> None:
        """Generate INCAR template file."""
        template_type = getattr(self, "selected_type", "sp")

        templates = {
            "sp": self._sp_template(),
            "opt": self._opt_template(),
            "md": self._md_template(),
        }

        content = templates.get(template_type, self._sp_template())
        filename = f"INCAR.{template_type}.template"

        try:
            with open(filename, "w") as f:
                f.write(content)
            self.app.notify(f"Template saved to {filename}", title="Success")
        except Exception as e:
            self.app.notify(f"Error: {e}", title="Error", severity="error")

    def _sp_template(self) -> str:
        return """# Single Point Calculation Template
CALC_TYPE = SP
TASK = omat
MODEL_PATH = uma-s-1.pt
DEVICE = cpu

# Output Control
WRITE_FORCES = .TRUE.
WRITE_STRESS = .TRUE.
"""

    def _opt_template(self) -> str:
        return """# Geometry Optimization Template
CALC_TYPE = OPT
TASK = omat
MODEL_PATH = uma-s-1.pt
DEVICE = cpu

# Optimization Settings
OPT_ALGO = FIRE
FMAX = 0.05
MAX_STEPS = 500
CELL_OPT = .FALSE.
FIX_SYMMETRY = .FALSE.
"""

    def _md_template(self) -> str:
        return """# Molecular Dynamics Template
CALC_TYPE = MD
TASK = omat
MODEL_PATH = uma-s-1.pt
DEVICE = cpu

# MD Settings
ENSEMBLE = NVT
TEMPERATURE = 300.0
TIMESTEP = 1.0
STEPS = 1000
SAVE_INTERVAL = 10

# Pre-relaxation (recommended)
PRE_RELAX = .TRUE.
PRE_RELAX_STEPS = 50
PRE_RELAX_FMAX = 0.1
"""


from textual.reactive import reactive