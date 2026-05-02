# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Setup script for UMA Calculator package.
"""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="umakit",
    version="1.0.0",
    description="VASP-like interface for FAIRChem UMA models",
    author="FAIRChem User",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "uma_calc=umakit.cli:main",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "fairchem-core",
        "ase>=3.26.0",
        "numpy",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
