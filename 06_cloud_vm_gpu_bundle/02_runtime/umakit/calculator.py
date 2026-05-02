# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Calculator wrapper for UMA models.

Provides a simplified interface for loading and using FAIRChem calculators
with local model files.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

if TYPE_CHECKING:
    from typing import Literal


class UMACalculator:
    """Wrapper for FAIRChem UMA calculators.

    Simplifies model loading and calculator creation from local checkpoint files.

    Example:
        >>> calc_wrapper = UMACalculator("uma-s-1.pt", task="omat", device="cuda")
        >>> calculator = calc_wrapper.get_calculator()
        >>> atoms.calc = calculator
        >>> energy = atoms.get_potential_energy()
    """

    VALID_TASKS = {"omat", "omol", "oc20", "oc25", "odac", "omc"}
    VALID_DEVICES = {"cpu", "cuda", "gpu"}
    VALID_INFERENCE_MODES = {"default", "turbo"}

    def __init__(
        self,
        model_path: str | Path,
        task: str = "omat",
        device: Literal["cpu", "cuda"] | None = None,
        inference_mode: str = "default",
    ):
        """Initialize UMA calculator wrapper.

        Args:
            model_path: Path to model checkpoint (.pt file)
            task: Task type (omat, omol, oc20, odac, omc)
            device: Device for calculation (cpu or cuda)
            inference_mode: Inference mode (default or turbo)

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If invalid task or inference mode
        """
        self.model_path = Path(model_path)
        self.task = task.lower()
        self.device = device or "cpu"
        self.inference_mode = inference_mode.lower()

        # Validate inputs
        self._validate()

        # Load predictor
        self._predictor = None
        self._calculator = None

    def _validate(self) -> None:
        """Validate initialization parameters."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.task not in self.VALID_TASKS:
            raise ValueError(
                f"Invalid task '{self.task}'. "
                f"Must be one of: {', '.join(self.VALID_TASKS)}"
            )

        if self.inference_mode not in self.VALID_INFERENCE_MODES:
            raise ValueError(
                f"Invalid inference mode '{self.inference_mode}'. "
                f"Must be one of: {', '.join(self.VALID_INFERENCE_MODES)}"
            )

        if self.device not in self.VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{self.device}'. "
                f"Must be one of: {', '.join(self.VALID_DEVICES)}"
            )

    def load_predictor(self) -> "MLIPPredictUnit":
        """Load the prediction unit from checkpoint.

        Returns:
            Loaded MLIPPredictUnit
        """
        if self._predictor is None:
            # For default mode, don't pass inference_settings to avoid issues
            # For turbo mode, use the settings
            if self.inference_mode == "turbo":
                is_cpu = self.device == "cpu"
                can_compile = (not is_cpu) and platform.system() != "Windows"
                settings = InferenceSettings(
                    tf32=not is_cpu,
                    activation_checkpointing=False,
                    merge_mole=True,
                    compile=can_compile,
                )
                self._predictor = load_predict_unit(
                    path=self.model_path,
                    device=self.device,
                    inference_settings=settings,
                )
            else:
                # Default mode - simpler approach like test_simple.py
                self._predictor = load_predict_unit(
                    path=self.model_path,
                    device=self.device,
                )

        return self._predictor

    def get_calculator(self) -> FAIRChemCalculator:
        """Get ASE calculator instance.

        Returns:
            FAIRChemCalculator instance
        """
        if self._calculator is None:
            predictor = self.load_predictor()
            self._calculator = FAIRChemCalculator(
                predict_unit=predictor,
                task_name=self.task,
            )

        return self._calculator

    @property
    def implemented_properties(self) -> list[str]:
        """Get list of implemented properties.

        Returns:
            List of property names
        """
        calc = self.get_calculator()
        return list(calc.implemented_properties)

    @property
    def has_stress(self) -> bool:
        """Check if stress calculation is supported.

        Returns:
            True if stress is supported
        """
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Get calculator information.

        Returns:
            Dictionary with calculator details
        """
        return {
            "model_path": str(self.model_path),
            "task": self.task,
            "device": self.device,
            "inference_mode": self.inference_mode,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
