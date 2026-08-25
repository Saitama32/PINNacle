"""Run a chain of sampled local Gauss--Newton refinements for the KS RWF MLP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.Chaotic.run_data_ks_local_basin import (
    build_parser as build_local_parser,
    run as run_local_basin,
)


# Edit this dictionary to configure a launch from an IDE without CLI arguments.
# Command-line arguments still have priority over these values.
CHAIN_DEFAULTS = {
    # Initial checkpoint and common settings.
    "model": r"C:\Users\Рустам\Documents\GitHub\PINNacle\runs_data_ks_local_basin_chain\08.25-05.17.20-ks-local-basin-chain\steps\08.25-06.06.30-ks-local-basin-float64\weights_local_best.pt",
    "data": None,
    # run_chain replaces this with <chain_out>/.../steps for every chain step.
    "out": str(PROJECT_ROOT / "runs_data_ks_local_basin"),
    "precision": "float64",
    "device": "auto",
    "seed": 2367,

    # Points used to construct the local linear problem.
    "jacobian_domain_points": 3072,
    "jacobian_data_points": 3072,
    "validation_domain_points": 10000,
    "validation_batch_size": 256,
    "eval_batch_size": 16384,

    # Optional soft initial-condition block (no boundary-condition block).
    "include_ic": False,
    "ic_weight": 100.0,
    "jacobian_ic_points": 512,
    "validation_ic_points": 2048,

    # Gauss--Newton/Levenberg--Marquardt parameter sweep at every chain step.
    "data_weights": [3000.0, 30000.0, 300000.0, 3000000.0],
    "step_scales": [0.2, 0.25, 0.3, 0.35, 0.4],
    "damping": 1e-5,
    "dampings": [1e-5, 1e-4, 5e-4],
    "max_l2_growth": 0.0,

    # Chain settings. Each step starts from the previous step's best checkpoint.
    "chain_steps": 5,
    "chain_seed_stride": 100000,
    "chain_stop_pde": 0.0,
    "chain_out": str(PROJECT_ROOT / "runs_data_ks_local_basin_chain"),
}


def write_chain_artifacts(chain_dir: Path, configuration: dict, history: list[dict]):
    with (chain_dir / "chain_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(configuration, file_obj, indent=2, sort_keys=True)
    with (chain_dir / "chain_history.json").open("w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2, sort_keys=True)
    if history:
        with (chain_dir / "chain_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def run_chain(args) -> Path:
    if args.chain_steps <= 0:
        raise ValueError("chain_steps must be positive")
    if args.chain_seed_stride <= 0:
        raise ValueError("chain_seed_stride must be positive")
    if args.chain_stop_pde < 0 or not math.isfinite(args.chain_stop_pde):
        raise ValueError("chain_stop_pde must be finite and non-negative")

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    chain_dir = Path(args.chain_out).expanduser().resolve() / (
        f"{timestamp}-ks-local-basin-chain"
    )
    steps_dir = chain_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=False)
    configuration = vars(args).copy()
    configuration["initial_model"] = str(Path(args.model).expanduser().resolve())
    configuration["chain_dir"] = str(chain_dir)
    history = []
    write_chain_artifacts(chain_dir, configuration, history)

    current_model = args.model
    for step_index in range(args.chain_steps):
        stage_args = argparse.Namespace(**vars(args))
        stage_args.model = str(current_model)
        stage_args.seed = args.seed + step_index * args.chain_seed_stride
        stage_args.out = str(steps_dir)
        print(
            f"Starting chain step {step_index + 1}/{args.chain_steps}; "
            f"seed={stage_args.seed}; model={stage_args.model}"
        )
        stage_dir = run_local_basin(stage_args)
        stage_checkpoint = stage_dir / "weights_local_best.pt"
        if not stage_checkpoint.is_file():
            raise RuntimeError(
                f"Chain step {step_index + 1} did not produce a feasible checkpoint: "
                f"{stage_dir}"
            )
        with (stage_dir / "metrics.json").open("r", encoding="utf-8") as file_obj:
            metrics = json.load(file_obj)
        initial = metrics["initial"]
        best = metrics["best_feasible"]
        initial_pde = float(initial["validation_pde_mse"])
        final_pde = float(best["validation_pde_mse"])
        initial_objective = float(initial["validation_objective"])
        final_objective = float(best["validation_objective"])
        initial_l2 = float(initial["data"]["relative_l2"])
        final_l2 = float(best["full_data_relative_l2"])
        row = {
            "step": step_index + 1,
            "seed": stage_args.seed,
            "source_model": str(current_model),
            "stage_dir": str(stage_dir),
            "checkpoint": str(stage_checkpoint),
            "damping": best["damping"],
            "data_weight": best["data_weight"],
            "ic_weight": best["ic_weight"],
            "step_scale": best["step_scale"],
            "relative_parameter_step": best["relative_parameter_step"],
            "initial_validation_pde_mse": initial_pde,
            "final_validation_pde_mse": final_pde,
            "relative_pde_improvement": (
                (initial_pde - final_pde) / initial_pde if initial_pde > 0 else 0.0
            ),
            "initial_validation_ic_mse": initial["validation_ic_mse"],
            "final_validation_ic_mse": best["validation_ic_mse"],
            "initial_validation_objective": initial_objective,
            "final_validation_objective": final_objective,
            "relative_objective_improvement": (
                (initial_objective - final_objective) / initial_objective
                if initial_objective > 0
                else 0.0
            ),
            "initial_data_relative_l2": initial_l2,
            "final_data_relative_l2": final_l2,
            "relative_l2_improvement": (
                (initial_l2 - final_l2) / initial_l2 if initial_l2 > 0 else 0.0
            ),
        }
        history.append(row)
        current_model = stage_checkpoint
        configuration["final_checkpoint"] = str(stage_checkpoint)
        write_chain_artifacts(chain_dir, configuration, history)
        (chain_dir / "latest_checkpoint.txt").write_text(
            str(stage_checkpoint), encoding="utf-8"
        )
        print(
            f"Finished chain step {step_index + 1}: "
            f"PDE {initial_pde:.6e} -> {final_pde:.6e}; "
            f"IC {float(initial['validation_ic_mse']):.6e} -> "
            f"{float(best['validation_ic_mse']):.6e}; "
            f"L2 {initial_l2:.6e} -> {final_l2:.6e}"
        )
        if args.chain_stop_pde > 0 and final_pde <= args.chain_stop_pde:
            print(
                f"Stopping chain: validation PDE {final_pde:.6e} reached "
                f"target {args.chain_stop_pde:.6e}."
            )
            break

    print(
        f"Finished local-basin chain with {len(history)} steps; "
        f"final checkpoint={current_model}; artifacts={chain_dir}"
    )
    return chain_dir


def build_parser():
    parser = build_local_parser()
    parser.description = __doc__
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "--chain-steps",
        type=int,
        help="Number of consecutive local-refinement steps.",
    )
    parser.add_argument(
        "--chain-seed-stride",
        type=int,
        help="Seed increment between consecutive chain steps.",
    )
    parser.add_argument(
        "--chain-stop-pde",
        type=float,
        help="Stop after reaching this validation PDE MSE; zero disables stopping.",
    )
    parser.add_argument(
        "--chain-out",
        help="Root directory for chain artifacts.",
    )
    parser.set_defaults(**CHAIN_DEFAULTS)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    run_chain(parse_args())
