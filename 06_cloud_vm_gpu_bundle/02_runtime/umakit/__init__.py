# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
UMAKit - A VASP-like interface for FAIRChem UMA models.

This package provides a comprehensive interface for running MLIP calculations
using FAIRChem's UMA models with VASP-style input/output conventions.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "FAIRChem User"

from umakit.config import IncarConfig
from umakit.calculator import UMACalculator
from umakit.runners.singlepoint import SinglePointRunner
from umakit.runners.optimization import OptimizationRunner
from umakit.runners.md import MDRunner
from umakit.runners.batch import BatchRunner

# API functions for programmatic usage
try:
    from umakit.api import (
        run_single_point,
        run_optimization,
        run_md,
        calculate_energy,
        calculate_adsorption_energy,
    )
    _api_available = True
except ImportError:
    _api_available = False

__all__ = [
    "IncarConfig",
    "UMACalculator",
    "SinglePointRunner",
    "OptimizationRunner",
    "MDRunner",
    "BatchRunner",
]

# Add API functions if available
if _api_available:
    __all__.extend([
        "run_single_point",
        "run_optimization",
        "run_md",
        "calculate_energy",
        "calculate_adsorption_energy",
    ])
