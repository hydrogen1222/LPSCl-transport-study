from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""Configuration screen for UMA Calculator TUI."""

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
)

if TYPE_CHECKING:
    from typing import Any


class ConfigScreen(Screen):
    """Configuration screen for setting up calculation parameters."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("up", "scroll_up", "Scroll Up"),
        ("down", "scroll_down", "Scroll Down"),
        ("pageup", "page_up", "Page Up"),
        ("pagedown", "page_down", "Page Down"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the configuration screen."""
        calc_type = self.app.get_config("calc_type", "sp")

        # Main container fills the screen
        with Container(id="config-main"):
            # Header at top
            yield Container(
                Static(f"Configuration: {calc_type.upper()}", id="title"),
                id="header"
            )

            # Scrollable content area
            with VerticalScroll(id="config-scroll"):
                # File paths section
                yield Static("📁 File Paths", classes="section-title")

                yield Label("Structure File:")
                yield Input(
                    placeholder="e.g., structure.cif or POSCAR",
                    id="structure-input"
                )

                yield Label("Model File:")
                yield Input(
                    placeholder="e.g., uma-s-1.pt",
                    value=str(self.app.get_config("model_file", "")),
                    id="model-input"
                )

                yield Label("Output Directory:")
                yield Input(
                    value=self.app.get_config("output_dir", "./results"),
                    id="output-input"
                )

                yield Label("Job Name (optional):")
                yield Input(
                    placeholder="e.g., structure_01",
                    value=str(self.app.get_config("job_name", "")),
                    id="job-name-input"
                )

                # Task selection
                yield Static("⚙️  Task & Device", classes="section-title")

                yield Label("Task Type:")
                yield Select(
                    options=[
                        ("Inorganic Materials (omat)", "omat"),
                        ("Molecules (omol)", "omol"),
                        ("Catalysis OC20 (oc20)", "oc20"),
                        ("Catalysis OC25 (oc25)", "oc25"),
                        ("MOFs (odac)", "odac"),
                        ("Molecular Crystals (omc)", "omc"),
                    ],
                    value=self.app.get_config("task", "omat"),
                    id="task-select"
                )

                yield Label("Device:")
                yield RadioSet(
                    RadioButton("CPU", id="cpu", value=self.app.get_config("device") == "cpu"),
                    RadioButton("CUDA (GPU)", id="cuda", value=self.app.get_config("device") == "cuda"),
                    id="device-radio"
                )

                # Calculation-specific options
                yield Static("🔧 Calculation Options", classes="section-title")
                yield from self._calc_options(calc_type)

                # Add some bottom padding
                yield Static("")

            # Action buttons fixed at bottom
            with Horizontal(id="button-bar"):
                yield Button("◀ Back", id="back-btn")
                yield Button("🚀 Run", variant="success", id="run-btn")

    def _calc_options(self, calc_type: str):
        """Generate calculation-specific options."""
        if calc_type == "sp":
            yield Horizontal(
                Label("Write Forces:"),
                Switch(value=True, id="write-forces"),
            )
            yield Horizontal(
                Label("Write Stress:"),
                Switch(value=True, id="write-stress"),
            )

        elif calc_type == "opt":
            yield Label("Force Threshold (eV/Å):")
            yield Input(value="0.05", id="fmax-input")

            yield Label("Max Steps:")
            yield Input(value="500", id="max-steps-input")

            yield Label("Optimizer:")
            yield Select(
                options=[("FIRE", "FIRE"), ("BFGS", "BFGS"), ("LBFGS", "LBFGS")],
                value="FIRE",
                id="optimizer-select"
            )

            yield Horizontal(
                Label("Cell Optimization:"),
                Switch(value=False, id="cell-opt"),
            )

        elif calc_type == "md":
            yield Label("Ensemble:")
            yield RadioSet(
                RadioButton("NVT", id="nvt", value=True),
                RadioButton("NVE", id="nve"),
                id="ensemble-radio"
            )

            yield Label("Temperature (K):")
            yield Input(value="300", id="temp-input")

            yield Label("Time Step (fs):")
            yield Input(value="1.0", id="timestep-input")

            yield Label("Steps:")
            yield Input(value="1000", id="steps-input")

            yield Horizontal(
                Label("Pre-relaxation (recommended):"),
                Switch(value=True, id="pre-relax"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "back-btn":
            self.app.pop_screen()

        elif button_id == "run-btn":
            self._save_and_run()

    def _save_and_run(self) -> None:
        """Save configuration and run calculation."""
        # Get file paths
        structure = self.query_one("#structure-input", Input).value
        model = self.query_one("#model-input", Input).value
        output = self.query_one("#output-input", Input).value

        if not structure:
            self.notify("Please specify a structure file", severity="error")
            return

        if not model:
            self.notify("Please specify a model file", severity="error")
            return

        # Save to config
        self.app.update_config("structure_file", structure)
        self.app.update_config("model_file", model)
        self.app.update_config("output_dir", output)

        # Get job name
        job_name = self.query_one("#job-name-input", Input).value
        if job_name:
            self.app.update_config("job_name", job_name)
        else:
            self.app.update_config("job_name", None)

        # Get task and device
        task_select = self.query_one("#task-select", Select)
        if task_select.value:
            self.app.update_config("task", task_select.value)

        # Get device from radio buttons
        device_radio = self.query_one("#device-radio", RadioSet)
        selected_device = device_radio.pressed_button
        if selected_device:
            self.app.update_config("device", selected_device.id)

        # Get calculation-specific options
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

            optimizer_select = self.query_one("#optimizer-select", Select)
            if optimizer_select.value:
                self.app.update_config("optimizer", optimizer_select.value)

            cell_opt = self.query_one("#cell-opt", Switch)
            self.app.update_config("cell_opt", cell_opt.value)

        elif calc_type == "md":
            try:
                temp = float(self.query_one("#temp-input", Input).value)
                self.app.update_config("temperature", temp)
            except ValueError:
                pass

            try:
                steps = int(self.query_one("#steps-input", Input).value)
                self.app.update_config("md_steps", steps)
            except ValueError:
                pass

            pre_relax = self.query_one("#pre-relax", Switch)
            self.app.update_config("pre_relax", pre_relax.value)

        # Go to run screen
        self.app.push_screen("run")

    def action_back(self) -> None:
        """Go back to main screen."""
        self.app.pop_screen()

    def action_scroll_up(self) -> None:
        """Scroll up in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_up()

    def action_scroll_down(self) -> None:
        """Scroll down in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_down()

    def action_page_up(self) -> None:
        """Page up in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_page_up()

    def action_page_down(self) -> None:
        """Page down in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_page_down()
