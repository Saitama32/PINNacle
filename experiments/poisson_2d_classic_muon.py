import argparse
import os
import sys

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from muon_example_common import add_muon_example_args, run_muon_example
from src.pde.poisson import Poisson2D_Classic


def main():
    parser = argparse.ArgumentParser(description="Poisson2D Classic PINN with Muon.")
    add_muon_example_args(parser)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.set_defaults(name="poisson2d_classic_muon")
    args = parser.parse_args()
    run_muon_example(Poisson2D_Classic, args, pde_kwargs={"scale": args.scale})


if __name__ == "__main__":
    main()
