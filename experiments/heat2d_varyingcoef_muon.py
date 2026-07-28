import argparse
import os
import sys

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from muon_example_common import add_muon_example_args, run_muon_example
from src.pde.heat import Heat2D_VaryingCoef


def main():
    parser = argparse.ArgumentParser(description="Heat2D VaryingCoef PINN with Muon.")
    add_muon_example_args(parser)
    parser.set_defaults(name="heat2d_varyingcoef_muon")
    args = parser.parse_args()
    run_muon_example(Heat2D_VaryingCoef, args)


if __name__ == "__main__":
    main()
