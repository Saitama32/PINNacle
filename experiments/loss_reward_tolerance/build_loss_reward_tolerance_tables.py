import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comet_ml import Experiment

from RL.rl_utils.load_buffer.load_loss_reward_tolerance import (
    OUTPUT_COLUMNS,
    collect_loss_reward_tolerance_dataframe,
)
from RL.rl_utils.load_buffer.load_exps_from_comet import WORKSPACE

PROJECT_NAME = "rlpinn_loss_reward_tolerances"
PROJECT_NAMES = [
    "rlpinn-burgers1d-tolerance",
    "rlpinn-burgers2d-tolerance",
    "rlpinn-poisson2d-classic-tolerance",
    "rlpinn-poisson-boltzmann2d-tolerance",
    "rlpinn-poisson3d-complexgeometry-tolerance",
    "rlpinn-poisson2d-manyarea-tolerance",
    "rlpinn-heat-2d-vc-tolerance",
    "rlpinn-heat2d-multiscale-tolerance-corrected",
    "rlpinn-heat-2d-cg-tolerance",
    "rlpinn-heat2d-longtime-tolerance",
    "rlpinn-ns2d-backstep-tolerance",
    "rlpinn-ns2d-liddriven-tolerance",
    "rlpinn-ns2d-longtime-tolerance",
    "rlpinn_wave1d_tolerance",
    "rlpinn-wave2d-heterogeneous-tolerance",
    "rlpinn-grayscott-tolerance",
    "rlpinn-kuramoto-sivashinsky-tolerance",
    "rlpinn-poissonnd-tolerance",
    "rlpinn-heatnd-tolerance",
    "rlpinn-poissoninv-tolerance",
    "rlpinn-heatinv-tolerance",
]


def log_csv_to_comet(exp, project_name, csv_path, row_count):
    try:
        exp.log_metric(f"{project_name}_rows", row_count)
        exp.log_asset(
            file_data=str(csv_path),
            file_name=csv_path.name,
            metadata={
                "kind": "loss_reward_tolerance_csv",
                "source_project_name": project_name,
            },
        )
        print(f"Logged CSV to Comet: {csv_path.name}")
    except Exception as exc:
        print(f"Failed to log CSV to Comet for {project_name}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build loss-reward tolerance CSV tables from Comet transitions."
    )
    parser.add_argument("--max-exps-last", type=int, default=200)
    parser.add_argument("--duration-grater-hours", type=float, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/loss_reward_tolerance"),
    )
    parser.add_argument(
        "--project",
        action="append",
        choices=PROJECT_NAMES,
        help="Project to process. Can be passed multiple times. Defaults to all projects.",
    )
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--prev-tol", type=float, default=0.0)
    parser.add_argument(
        "--no-use-tol",
        action="store_true",
        help="Disable filtering experiments by tolerance parameter.",default=False
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        workspace=WORKSPACE,
        project_name=PROJECT_NAME,
        auto_output_logging="simple",
    )
    exp.set_name("loss_reward_tolerance_csv_tables")
    exp.log_parameter("max_exps_last", args.max_exps_last)
    exp.log_parameter("duration_grater_hours", args.duration_grater_hours)
    exp.log_parameter("num_workers", args.num_workers)
    exp.log_parameter("tolerance", args.tolerance)
    exp.log_parameter("prev_tol", args.prev_tol)
    exp.log_parameter("use_tol", not args.no_use_tol)

    projects = args.project if args.project else PROJECT_NAMES
    exp.log_parameter("projects_count", len(projects))
    try:
        for project_name in projects:
            print(f"\n=== {project_name} ===")
            df = collect_loss_reward_tolerance_dataframe(
                proj_name=project_name,
                max_exps_last=args.max_exps_last,
                duration_grater_hours=args.duration_grater_hours,
                tolerance=args.tolerance,
                prev_tol=args.prev_tol,
                use_tol=not args.no_use_tol,
                num_workers=args.num_workers,
            )

            output_path = args.output_dir / f"{project_name}.csv"
            df.to_csv(output_path, index=False, encoding="utf-8", columns=OUTPUT_COLUMNS)
            log_csv_to_comet(exp, project_name, output_path, len(df))
            print(f"Saved {len(df)} rows to {output_path}")
    finally:
        exp.end()


if __name__ == "__main__":
    main()
