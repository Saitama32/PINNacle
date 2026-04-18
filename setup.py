#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List

import setuptools


HERE = Path(__file__).parent.resolve()

NAME = "pinnacle-benchmark"
VERSION = "0.1.0"
AUTHOR = "PINNacle contributors"
AUTHOR_EMAIL = ""
SHORT_DESCRIPTION = (
    "PINNacle benchmark codebase for physics-informed neural networks and PDE experiments."
)
README = Path(HERE, "README.md").read_text(encoding="utf-8")
URL = "https://github.com/i207M/PINNacle"
REQUIRES_PYTHON = ">=3.9"
LICENSE = "MIT License"


def _readlines(*names: str, **kwargs) -> List[str]:
    encodings = [kwargs.get("encoding", "utf-8"), "utf-8-sig", "utf-16"]
    path = HERE.joinpath(*names)
    last_error = None
    for encoding in encodings:
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            return list(map(str.strip, lines))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def _extract_requirements(file_name: str) -> List[str]:
    return [line for line in _readlines(file_name) if line and not line.startswith("#")]


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=SHORT_DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    python_requires=REQUIRES_PYTHON,
    license=LICENSE,
    packages=setuptools.find_packages(
        include=[
            "RL",
            "RL.*",
            "deepxde",
            "deepxde.*",
            "landscape_visualization",
            "landscape_visualization.*",
            "src",
            "src.*",
            "vpinn",
            "vpinn.*",
        ],
        exclude=[
            "data",
            "data.*",
            "experiments",
            "experiments.*",
            "resources",
            "resources.*",
            "runs",
            "runs.*",
            "runs_single",
            "runs_single.*",
            "test",
            "test.*",
            "transitions",
            "transitions.*",
            "venv",
            "venv.*",
        ],
    ),
    include_package_data=True,
    install_requires=_extract_requirements("requirements.txt"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
