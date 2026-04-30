from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="pinnacle-benchmark",
    version="0.1.0",
    description="PINNacle benchmark codebase for physics-informed neural networks and PDE experiments.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(
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
    install_requires=[
        "matplotlib",
        "pandas",
        "scipy==1.13.1",
        "scikit-learn",
        "numpy==1.26.4",
        "gym",
        "seaborn",
        "torch>=2.0",
        "autodocsumm",
        "SALib",
        "comet_ml",
        "python-dotenv",
    ],
)
