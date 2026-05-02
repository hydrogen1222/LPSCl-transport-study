# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
UMA Calc - VASP-like interface for FAIRChem UMA models.

A comprehensive command-line tool for running MLIP calculations using
FAIRChem's Universal Material Application (UMA) models.

Usage:
    uma_calc run                    # Run from INCAR.uma
    uma_calc sp structure.cif       # Single point calculation
    uma_calc opt structure.cif      # Geometry optimization
    uma_calc md structure.cif       # Molecular dynamics
    uma_calc batch structures/      # Batch processing
    uma_calc template sp            # Generate template INCAR

For detailed help on subcommands:
    uma_calc <command> -h
"""

from __future__ import annotations

import sys

from umakit.cli import main

if __name__ == "__main__":
    sys.exit(main())
