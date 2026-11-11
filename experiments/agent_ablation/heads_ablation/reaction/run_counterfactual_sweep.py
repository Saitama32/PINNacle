"""Select, execute and summarize the reaction parameter counterfactual sweep."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import time

import torch

from .comet_io import create_api, download_asset, get_experiment, list_assets, torch_load
from .config import ACTION_SPACES, EXPERIMENTS, FARM_PROJECT, WORKSPACE
from .counterfactual_sweep import (
    _last_state_dict,
    _payload_entries,
    _stable_seed,
    build_selection_manifest,
    evaluate_parameter_grid,
    summarize_groups,
    summarize_outcomes,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_CACHE = SCRIPT_DIR / "cache"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "counterfactual"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("select", "sweep", "summarize", "merge", "status", "all")
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states-per-source-phase", type=int, default=2)
    parser.add_argument("--max-states", type=int)
    parser.add_argument(
        "--optimizer",
        dest="optimizers",
        action="append",
        choices=("Adam", "LBFGS"),
        help="Run only states routed to this optimizer; may be repeated.",
    )
    parser.add_argument(
        "--selection-id",
        dest="selection_ids",
        action="append",
        type=int,
        help="Run only this manifest selection id; may be repeated.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=1729)
    parser.add_argument(
        "--suffix-horizon",
        type=int,
        default=3,
        help="Number of recorded actions applied after each counterfactual action.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def _load_or_select(api, args):
    manifest_path = _mode_root(args) / "selection_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("suffix_horizon", 0)) != args.suffix_horizon:
            raise RuntimeError("Manifest suffix horizon does not match CLI configuration")
        return manifest
    return build_selection_manifest(
        api,
        args.cache_dir,
        manifest_path,
        args.states_per_source_phase,
        torch.device(args.device),
        suffix_horizon=args.suffix_horizon,
    )


def _read_results(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {
        "selection_id": int,
        "repeat": int,
        "epochs_index": int,
        "lr_index": int,
        "epochs": int,
        "lr": float,
        "initial_loss": float,
        "final_loss": float,
        "score": float,
        "immediate_score": float,
        "suffix_final_loss": float,
        "suffix_reconstructed_immediate_loss": float,
        "suffix_reconstruction_abs_log_error": float,
        "horizon_best_loss": float,
        "horizon_score": float,
        "stability_gap": float,
        "segment_seconds": float,
        "suffix_seconds": float,
        "branch_seconds": float,
    }
    for row in rows:
        for key, cast in numeric.items():
            if row.get(key, "") != "":
                row[key] = cast(row[key])
    return rows


def _append_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _mode_root(args):
    name = "one_step" if args.suffix_horizon == 0 else f"suffix_h{args.suffix_horizon}"
    return args.output_dir / name


def _result_dir(args):
    if args.num_shards == 1:
        return _mode_root(args)
    return _mode_root(args) / "shards" / f"shard_{args.shard_index:02d}"


def _format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    return str(datetime.timedelta(seconds=max(0, int(seconds))))


class SweepProgress:
    def __init__(
        self, result_dir: Path, total: int, completed: int, durations, optimizer_label: str
    ):
        self.result_dir = result_dir
        self.total = total
        self.completed = completed
        self.started = time.perf_counter()
        self.durations = [
            float(value) for value in durations if math.isfinite(float(value)) and float(value) > 0
        ][-30:]
        mean_seconds = sum(self.durations) / len(self.durations) if self.durations else None
        eta_seconds = mean_seconds * max(0, total - completed) if mean_seconds else None
        self.result_dir.mkdir(parents=True, exist_ok=True)
        (self.result_dir / "progress.json").write_text(
            json.dumps(
                {
                    "completed_actions": completed,
                    "total_actions": total,
                    "percent": 100.0 * completed / max(1, total),
                    "elapsed_seconds_this_run": 0.0,
                    "mean_seconds_per_action": mean_seconds,
                    "eta_seconds": eta_seconds,
                    "eta_utc": None,
                    "last_selection_id": None,
                    "last_optimizer": optimizer_label,
                    "last_epochs_index": None,
                    "last_lr_index": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def advance(self, duration, state, outcome):
        self.completed += 1
        if math.isfinite(float(duration)) and float(duration) > 0:
            self.durations.append(float(duration))
            self.durations = self.durations[-30:]
        mean_seconds = sum(self.durations) / len(self.durations) if self.durations else None
        remaining = max(0, self.total - self.completed)
        eta_seconds = mean_seconds * remaining if mean_seconds is not None else None
        percent = 100.0 * self.completed / max(1, self.total)
        eta_at = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=eta_seconds)
            if eta_seconds is not None
            else None
        )
        snapshot = {
            "completed_actions": self.completed,
            "total_actions": self.total,
            "percent": percent,
            "elapsed_seconds_this_run": time.perf_counter() - self.started,
            "mean_seconds_per_action": mean_seconds,
            "eta_seconds": eta_seconds,
            "eta_utc": eta_at.isoformat() if eta_at is not None else None,
            "last_selection_id": state["selection_id"],
            "last_optimizer": state["g_optimizer"],
            "last_epochs_index": outcome["epochs_index"],
            "last_lr_index": outcome["lr_index"],
        }
        self.result_dir.mkdir(parents=True, exist_ok=True)
        (self.result_dir / "progress.json").write_text(
            json.dumps(snapshot, indent=2), encoding="utf-8"
        )
        width = 30
        filled = min(width, int(round(width * self.completed / max(1, self.total))))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"[{bar}] {self.completed}/{self.total} ({percent:5.1f}%) "
            f"last={_format_duration(duration)} ETA={_format_duration(eta_seconds)}",
            flush=True,
        )


def run_sweep(api, args, manifest):
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    results_path = _result_dir(args) / "sweep_results.csv"
    existing = _read_results(results_path)
    completed = {
        (row["selection_id"], row["repeat"], row["epochs_index"], row["lr_index"])
        for row in existing
    }
    states = manifest["states"]
    if args.optimizers:
        states = [state for state in states if state["g_optimizer"] in args.optimizers]
    if args.selection_ids:
        selected = set(args.selection_ids)
        states = [state for state in states if int(state["selection_id"]) in selected]
    if args.num_shards > 1:
        states = [
            state
            for state in states
            if int(state["selection_id"]) % args.num_shards == args.shard_index
        ]
    if args.max_states is not None:
        states = states[: args.max_states]
    optimizer_dict = ACTION_SPACES[EXPERIMENTS["G"]["action_space"]]
    experiments = {}
    asset_maps = {}
    total = len(states) * args.repeats
    expected_keys = {
        (int(state["selection_id"]), repeat, epoch_index, lr_index)
        for state in states
        for repeat in range(args.repeats)
        for epoch_index in range(3)
        for lr_index in range(3)
    }
    print(
        f"This worker: states={len(states)}, branches={len(expected_keys)}, "
        f"estimated_training_chunks={len(expected_keys) * (1 + args.suffix_horizon)}"
    )
    completed_relevant = completed & expected_keys
    existing_durations = [
        row.get("segment_seconds", float("nan"))
        for row in existing
        if (
            int(row["selection_id"]), int(row["repeat"]),
            int(row["epochs_index"]), int(row["lr_index"]),
        ) in completed_relevant
    ]
    progress = SweepProgress(
        _result_dir(args),
        len(expected_keys),
        len(completed_relevant),
        existing_durations,
        ",".join(args.optimizers or ["all"]),
    )
    finished = 0
    for state in states:
        key = state["source_key"]
        if key not in experiments:
            experiment = get_experiment(api, WORKSPACE, FARM_PROJECT, key)
            experiments[key] = experiment
            asset_maps[key] = {
                asset.step: asset
                for asset in list_assets(experiment)
                if asset.kind == "transition"
            }
        experiment = experiments[key]
        asset = asset_maps[key][int(state["previous_asset_step"])]
        raw_path = download_asset(
            experiment,
            asset,
            args.cache_dir / "counterfactual_raw" / key,
        )
        entries = _payload_entries(torch_load(raw_path))
        previous_entry = entries[int(state["previous_asset_local_index"])]
        state_dict = _last_state_dict(previous_entry.get("solver_models"))
        if state_dict is None:
            raise RuntimeError(
                f"No solver state in {key} step={state['previous_asset_step']}"
            )
        for repeat in range(args.repeats):
            expected = {
                (state["selection_id"], repeat, epoch_index, lr_index)
                for epoch_index in range(3)
                for lr_index in range(3)
            }
            if expected <= completed:
                finished += 1
                continue
            seed = _stable_seed(
                args.base_seed, key, int(state["asset_step"]), repeat
            )
            started = time.perf_counter()
            def persist_outcome(outcome):
                task_key = (
                    state["selection_id"], repeat,
                    outcome["epochs_index"], outcome["lr_index"],
                )
                if task_key in completed:
                    return
                row = {
                    "selection_id": state["selection_id"],
                    "repeat": repeat,
                    "source_key": key,
                    "phase": state["phase"],
                    "chain_step": state["chain_step"],
                    "asset_step": state["asset_step"],
                    "seed": seed,
                    "suffix_horizon": args.suffix_horizon,
                    "optimizer": state["g_optimizer"],
                    "g_epochs_index": state["g_parameter_indices"]["epochs"],
                    "g_lr_index": state["g_parameter_indices"]["lr"],
                    "b_epochs_index": state["b_parameter_indices"]["epochs"],
                    "b_lr_index": state["b_parameter_indices"]["lr"],
                    **outcome,
                }
                _append_rows(results_path, [row])
                completed.add(task_key)
                progress.advance(outcome["segment_seconds"], state, outcome)

            evaluate_parameter_grid(
                state_dict,
                state["g_optimizer"],
                optimizer_dict,
                seed,
                outcome_callback=persist_outcome,
                historical_suffix=state.get("historical_suffix", []),
                skip_actions={
                    (epoch_index, lr_index)
                    for selection_id, completed_repeat, epoch_index, lr_index in completed
                    if selection_id == state["selection_id"]
                    and completed_repeat == repeat
                },
            )
            finished += 1
            print(
                f"Sweep state {finished}/{total}: id={state['selection_id']} "
                f"phase={state['phase']} optimizer={state['g_optimizer']} "
                f"seconds={time.perf_counter() - started:.1f}"
            )
    return _read_results(results_path)


def run_summary(args, result_dir=None):
    result_dir = result_dir or _result_dir(args)
    rows = _read_results(result_dir / "sweep_results.csv")
    summaries, aggregate = summarize_outcomes(rows)
    write_csv(result_dir / "state_summary.csv", summaries)
    write_csv(result_dir / "group_summary.csv", summarize_groups(summaries))
    (result_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


def run_merge(args):
    mode_root = _mode_root(args)
    paths = []
    base_path = mode_root / "sweep_results.csv"
    if base_path.exists():
        paths.append(base_path)
    paths.extend(sorted((mode_root / "shards").glob("shard_*/sweep_results.csv")))
    merged = {}
    for path in paths:
        for row in _read_results(path):
            key = (
                int(row["selection_id"]),
                int(row["repeat"]),
                int(row["epochs_index"]),
                int(row["lr_index"]),
            )
            merged[key] = row
    rows = [merged[key] for key in sorted(merged)]
    if not rows:
        raise RuntimeError(f"No sweep results found under {mode_root}")
    write_csv(base_path, rows)
    print(f"Merged {len(rows)} action outcomes from {len(paths)} files")
    run_summary(args, mode_root)


def run_status(args):
    mode_root = _mode_root(args)
    paths = sorted((mode_root / "shards").glob("shard_*/progress.json"))
    if not paths and (mode_root / "progress.json").exists():
        paths = [mode_root / "progress.json"]
    snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not snapshots:
        print(f"No progress snapshots found under {mode_root}")
        return
    completed = sum(int(item["completed_actions"]) for item in snapshots)
    total = sum(int(item["total_actions"]) for item in snapshots)
    eta_values = [item.get("eta_seconds") for item in snapshots]
    eta_values = [float(value) for value in eta_values if value is not None]
    eta_seconds = max(eta_values) if eta_values else None
    percent = 100.0 * completed / max(1, total)
    width = 40
    filled = min(width, int(round(width * completed / max(1, total))))
    bar = "#" * filled + "-" * (width - filled)
    optimizers = sorted({item.get("last_optimizer", "unknown") for item in snapshots})
    print(
        f"[{bar}] {completed}/{total} ({percent:.1f}%) "
        f"optimizer={','.join(optimizers)} ETA={_format_duration(eta_seconds)}"
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        run_status(args)
        return
    if args.command == "merge":
        run_merge(args)
        return
    if args.command == "summarize":
        run_summary(args)
        return
    api = create_api(PROJECT_ROOT)
    manifest = _load_or_select(api, args)
    print(f"Selected {manifest['state_count']} counterfactual states")
    branches = int(manifest["state_count"]) * args.repeats * 9
    chunks = branches * (1 + args.suffix_horizon)
    print(
        f"Planned scale: {branches} counterfactual branches, "
        f"approximately {chunks} optimizer training chunks "
        f"(suffix_horizon={args.suffix_horizon})"
    )
    if args.command in ("sweep", "all"):
        run_sweep(api, args, manifest)
        run_summary(args)


if __name__ == "__main__":
    main()
