import argparse
import os
import sys

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from muon_example_common import add_muon_example_args, run_muon_example
from src.pde.burgers import Burgers1D


def main():
    parser = argparse.ArgumentParser(description="Burgers1D PINN with Muon.")
    add_muon_example_args(parser)
    parser.set_defaults(name="burgers1d_muon")
    args = parser.parse_args()
    run_muon_example(Burgers1D, args)


if __name__ == "__main__":
    main()
