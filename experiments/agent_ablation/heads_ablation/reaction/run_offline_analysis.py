"""CLI for cached offline reconstruction of optimizer/parameter heads."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import torch

from .checkpoint_model import build_checkpoint_heads
from .comet_io import (
    create_api,
    experiment_key,
    experiment_parameters,
    get_experiment,
    inventory,
    list_assets,
    download_asset,
    load_checkpoint_pair,
    load_lightweight_transitions,
    pair_checkpoints,
    save_inventory,
    select_latest_farm_experiments,
)
from .config import (
    ACTION_SPACES,
    DEFAULT_FARM_EXPERIMENTS,
    DEFAULT_GAMMA,
    EXPERIMENTS,
    EXPERIMENT_COHORTS,
    FARM_PROJECT,
    WORKSPACE,
)
from .dataset import build_records, coverage_rows, finite_records
from .diagnostics import (
    analyze_checkpoint,
    calibration_rows,
    optimizer_calibration_rows,
    write_csv,
)
from .hybrid_comparison import compare_hybrid_policies
from .return_transfer import analyze_return_transfer, transfer_gap_rows


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_CACHE = SCRIPT_DIR / "cache"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "inventory",
            "coverage",
            "calibrate",
            "compare",
            "analyze",
            "return-matrix",
        ),
    )
    parser.add_argument(
        "--cohort",
        choices=tuple(EXPERIMENT_COHORTS),
        default="gb",
        help="Named reproducible set of trained agents.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional subset of labels from the selected cohort.",
    )
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-farm", action="store_true")
    parser.add_argument("--farm-project", default=FARM_PROJECT)
    parser.add_argument("--farm-experiments", type=int, default=DEFAULT_FARM_EXPERIMENTS)
    parser.add_argument("--farm-min-hours", type=float, default=0.07)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=5,
        help="Parallel Comet downloads (same default as the training loader).",
    )
    parser.add_argument("--checkpoint-stride", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refresh-transitions", action="store_true")
    parser.add_argument(
        "--refresh-return-matrix",
        action="store_true",
        help="Recompute completed per-checkpoint return-matrix parts.",
    )
    return parser.parse_args()


def _selected_configs(labels, cohort="gb"):
    experiments = EXPERIMENT_COHORTS[cohort]
    labels = labels or list(experiments)
    missing = [label for label in labels if label not in experiments]
    if missing:
        raise ValueError(
            f"Unknown experiment labels for cohort {cohort!r}: {missing}"
        )
    return [(label, experiments[label]) for label in labels]


def _inventory_models(api, args, selected):
    combined = {}

    def load_one(label, config):
        experiment = get_experiment(
            api, args.workspace, config["project"], config["key"]
        )
        data = inventory(experiment, config["key"], config["project"])
        data["label"] = label
        data["action_space"] = config["action_space"]
        return label, data

    worker_count = max(1, min(args.download_workers, len(selected)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(load_one, label, config): label
            for label, config in selected
        }
        for future in as_completed(futures):
            label, data = future.result()
            combined[label] = data
            save_inventory(data, args.output_dir / f"inventory_{label}.json")
            summary = data["asset_summary"]
            print(
                f"{label}: checkpoints={len(data['checkpoint_steps'])}, "
                f"transitions={summary['transition']['count']}, "
                f"known_bytes={sum(item['size_bytes'] for item in summary.values())}"
            )
    (args.output_dir / "inventory_all.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_blocks(api, args, selected):
    blocks = []

    worker_count = max(1, min(args.download_workers, len(selected)))
    per_experiment_asset_workers = max(
        1,
        args.download_workers // worker_count,
    )

    def load_model_experiment(label, config):
        experiment = get_experiment(
            api, args.workspace, config["project"], config["key"]
        )
        transitions = load_lightweight_transitions(
            experiment,
            config["key"],
            config["project"],
            args.cache_dir / "experiments" / config["key"],
            refresh=args.refresh_transitions,
            asset_workers=per_experiment_asset_workers,
        )
        return label, config["key"], transitions

    model_results = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(load_model_experiment, label, config): label
            for label, config in selected
        }
        for future in as_completed(futures):
            model_results.append(future.result())
    order = {label: index for index, (label, _) in enumerate(selected)}
    model_results.sort(key=lambda item: order[item[0]])
    for label, key, transitions in model_results:
        blocks.append(("online", key, transitions))
        print(f"{label}: loaded {len(transitions)} online transitions")

    if args.include_farm:
        experiments = select_latest_farm_experiments(
            api,
            args.workspace,
            args.farm_project,
            args.farm_experiments,
            minimum_duration_hours=args.farm_min_hours,
        )
        print(f"Farm: selected {len(experiments)} experiments")
        def load_farm_experiment(experiment):
            key = experiment_key(experiment)
            transitions = load_lightweight_transitions(
                experiment,
                key,
                args.farm_project,
                args.cache_dir / "farm" / key,
                refresh=args.refresh_transitions,
                # Farm experiments themselves are loaded concurrently, so a
                # nested asset pool would multiply the configured limit.
                asset_workers=1,
            )
            return key, transitions

        farm_results = [None] * len(experiments)
        worker_count = max(1, min(args.download_workers, len(experiments)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(load_farm_experiment, experiment): index
                for index, experiment in enumerate(experiments)
            }
            completed = 0
            for future in as_completed(futures):
                farm_results[futures[future]] = future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(experiments):
                    print(
                        f"Farm: loaded {completed}/{len(experiments)} experiments"
                    )
        for key, transitions in farm_results:
            blocks.append(("farm", key, transitions))
    return blocks


def _select_pairs(experiment, stride: int, maximum: int | None):
    pairs = pair_checkpoints(list_assets(experiment))[:: max(1, stride)]
    if maximum is not None and len(pairs) > maximum:
        # Keep an evenly spaced history including the final snapshot.
        indices = torch.linspace(0, len(pairs) - 1, maximum).round().long().tolist()
        pairs = [pairs[index] for index in dict.fromkeys(indices)]
    return pairs


def _prefetch_checkpoint_pairs(experiment, pairs, cache_dir, workers):
    """Download unique snapshot assets concurrently before GPU inference."""
    assets = []
    seen = set()
    for pair in pairs:
        for asset in (pair.optimizer, pair.parameters):
            if asset.asset_id not in seen:
                seen.add(asset.asset_id)
                assets.append(asset)
    worker_count = max(1, min(workers, len(assets)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(download_asset, experiment, asset, cache_dir / "models")
            for asset in assets
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 10 == 0 or completed == len(futures):
                print(f"Checkpoint assets: downloaded {completed}/{len(futures)}")


def _chronological_records(blocks, experiment_key_value: str, step: int, gamma: float):
    """Rebuild returns without leaking transitions collected after a snapshot.

    The fixed farm data is always available.  Online data is restricted to the
    experiment that produced the checkpoint and to assets at or before its
    global agent step.  Rebuilding records after filtering is important: MC
    returns must not contain rewards from the excluded future tail.
    """
    filtered_blocks = []
    for source_kind, source_key, entries in blocks:
        if source_kind == "farm":
            filtered_blocks.append((source_kind, source_key, entries))
            continue
        if source_key != experiment_key_value:
            continue
        available = []
        for entry in entries:
            source = entry.get("_source", {})
            asset_step = source.get("asset_step")
            if asset_step is not None and int(asset_step) <= step:
                available.append(entry)
        if available:
            filtered_blocks.append((source_kind, source_key, available))
    return finite_records(build_records(filtered_blocks, gamma=gamma))


def run_analysis(api, args, selected):
    blocks = _load_blocks(api, args, selected)
    device = torch.device(args.device)
    optimizer_rows = []
    parameter_rows = []
    factual_rows = []

    # G and B share an action space.  If future labels use another action
    # space, analyze them in a separate invocation to avoid semantic mixing.
    spaces = {config["action_space"] for _, config in selected}
    if len(spaces) != 1:
        raise ValueError(
            "Selected experiments use different action spaces; run them "
            "separately so transition actions remain unambiguous"
        )
    optimizer_dict = ACTION_SPACES[spaces.pop()]
    records = finite_records(build_records(blocks, gamma=args.gamma))
    print(f"Diagnostic dataset: {len(records)} finite transitions")
    write_csv(args.output_dir / "coverage.csv", coverage_rows(records, optimizer_dict))

    for label, config in selected:
        experiment = get_experiment(
            api, args.workspace, config["project"], config["key"]
        )
        parameters = experiment_parameters(experiment)
        pairs = _select_pairs(
            experiment, args.checkpoint_stride, args.max_checkpoints
        )
        if not pairs:
            raise RuntimeError(f"No paired checkpoints found for {label}")
        print(f"{label}: analyzing {len(pairs)} paired checkpoints")
        experiment_cache = args.cache_dir / "experiments" / config["key"]
        _prefetch_checkpoint_pairs(
            experiment,
            pairs,
            experiment_cache,
            args.download_workers,
        )
        for pair_index, pair in enumerate(pairs, 1):
            optim_state, params_state = load_checkpoint_pair(
                experiment,
                pair,
                experiment_cache,
            )
            model = build_checkpoint_heads(
                optimizer_dict,
                optim_state,
                params_state,
                experiment_parameters=parameters,
                device=device,
            )
            evaluation_sets = (
                ("fixed_all_online", records),
                (
                    "chronological",
                    _chronological_records(
                        blocks,
                        experiment_key_value=config["key"],
                        step=pair.step,
                        gamma=args.gamma,
                    ),
                ),
            )
            for evaluation_scope, evaluation_records in evaluation_sets:
                if not evaluation_records:
                    continue
                optim_row, param_part, factual_part = analyze_checkpoint(
                    model,
                    evaluation_records,
                    optimizer_dict,
                    checkpoint_step=pair.step,
                    experiment_label=label,
                    batch_size=args.batch_size,
                    device=device,
                    evaluation_scope=evaluation_scope,
                )
                optimizer_rows.append(optim_row)
                parameter_rows.extend(param_part)
                factual_rows.extend(factual_part)
            print(f"{label}: checkpoint {pair_index}/{len(pairs)} step={pair.step}")
            del model, optim_state, params_state
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(args.output_dir / "optimizer_head_history.csv", optimizer_rows)
    write_csv(args.output_dir / "parameter_head_history.csv", parameter_rows)
    write_csv(args.output_dir / "factual_q_history.csv", factual_rows)
    write_csv(
        args.output_dir / "factual_calibration_history.csv",
        calibration_rows(factual_rows),
    )
    write_csv(
        args.output_dir / "optimizer_factual_calibration_history.csv",
        optimizer_calibration_rows(factual_rows),
    )
    print(f"Saved diagnostics to {args.output_dir.resolve()}")


def run_coverage(api, args, selected):
    """Rebuild coverage from cached transitions without loading DQN models."""
    spaces = {config["action_space"] for _, config in selected}
    if len(spaces) != 1:
        raise ValueError("Coverage requires labels with the same action space")
    optimizer_dict = ACTION_SPACES[spaces.pop()]
    blocks = _load_blocks(api, args, selected)
    records = finite_records(build_records(blocks, gamma=args.gamma))
    write_csv(args.output_dir / "coverage.csv", coverage_rows(records, optimizer_dict))
    print(f"Saved coverage for {len(records)} transitions")


def run_calibration(args):
    """Regenerate calibration tables from factual rows without inference."""
    factual_path = args.output_dir / "factual_q_history.csv"
    if not factual_path.exists():
        raise FileNotFoundError(factual_path)
    integer_fields = {
        "checkpoint_step",
        "record_id",
        "chain_step",
        "done",
        "optimizer_rank",
        "observed_parameter_index",
        "parameter_rank",
        "parameter_argmax_index",
    }
    float_fields = {
        "mc_return",
        "reward",
        "observed_optimizer_q",
        "optimizer_max_gap",
        "observed_parameter_q",
        "parameter_max_gap",
    }
    rows = []
    with factual_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for field in integer_fields:
                if row.get(field, "") != "":
                    row[field] = int(float(row[field]))
            for field in float_fields:
                if row.get(field, "") != "":
                    row[field] = float(row[field])
            rows.append(row)
    write_csv(
        args.output_dir / "factual_calibration_history.csv",
        calibration_rows(rows),
    )
    write_csv(
        args.output_dir / "optimizer_factual_calibration_history.csv",
        optimizer_calibration_rows(rows),
    )
    print(f"Regenerated calibration from {len(rows)} factual rows")


def run_hybrid_comparison(api, args, selected):
    if len(selected) != 2:
        raise ValueError("Hybrid comparison currently requires exactly two labels")
    spaces = {config["action_space"] for _, config in selected}
    if len(spaces) != 1:
        raise ValueError("Hybrid comparison requires a shared action space")
    optimizer_dict = ACTION_SPACES[spaces.pop()]
    blocks = _load_blocks(api, args, selected)
    records = finite_records(build_records(blocks, gamma=args.gamma))
    device = torch.device(args.device)
    models = {}
    for label, config in selected:
        experiment = get_experiment(api, args.workspace, config["project"], config["key"])
        pairs = pair_checkpoints(list_assets(experiment))
        if not pairs:
            raise RuntimeError(f"No paired checkpoints found for {label}")
        pair = pairs[-1]
        experiment_cache = args.cache_dir / "experiments" / config["key"]
        optim_state, params_state = load_checkpoint_pair(
            experiment, pair, experiment_cache
        )
        models[label] = build_checkpoint_heads(
            optimizer_dict,
            optim_state,
            params_state,
            experiment_parameters=experiment_parameters(experiment),
            device=device,
        )
        print(f"{label}: loaded final checkpoint step={pair.step}")
    actions, agreements, summary = compare_hybrid_policies(
        models,
        records,
        optimizer_dict,
        batch_size=args.batch_size,
        device=device,
    )
    write_csv(args.output_dir / "hybrid_policy_actions.csv", actions)
    write_csv(args.output_dir / "head_choice_agreement.csv", agreements)
    write_csv(args.output_dir / "hybrid_factual_match_summary.csv", summary)
    print(f"Saved hybrid comparison for {len(records)} states")


def run_return_matrix(api, args, selected):
    """Evaluate return calibration across own, farm and other-agent archives."""
    spaces = {config["action_space"] for _, config in selected}
    if len(spaces) != 1:
        raise ValueError("Return matrix requires a shared action space")
    optimizer_dict = ACTION_SPACES[spaces.pop()]

    # Farm is part of the experiment definition, not an optional display
    # source: without it own-vs-farm specialization cannot be measured.
    args.include_farm = True
    blocks = _load_blocks(api, args, selected)
    records = finite_records(build_records(blocks, gamma=args.gamma))
    relation_counts = {}
    for label, config in selected:
        relation_counts[label] = {
            relation: sum(
                1
                for record in records
                if (
                    "farm"
                    if record.source_kind == "farm"
                    else "own"
                    if record.source_key == config["key"]
                    else "other"
                )
                == relation
            )
            for relation in ("own", "farm", "other")
        }
    print(f"Return-matrix dataset: {len(records)} transitions")

    device = torch.device(args.device)
    rows = []
    signature_payload = {
        "keys": [config["key"] for _, config in selected],
        "gamma": args.gamma,
        "farm_project": args.farm_project,
        "farm_experiments": args.farm_experiments,
        "farm_min_hours": args.farm_min_hours,
    }
    dataset_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    parts_dir = args.output_dir / "return_matrix_parts" / dataset_signature
    parts_dir.mkdir(parents=True, exist_ok=True)
    for label, config in selected:
        experiment = get_experiment(
            api, args.workspace, config["project"], config["key"]
        )
        parameters = experiment_parameters(experiment)
        pairs = _select_pairs(
            experiment, args.checkpoint_stride, args.max_checkpoints
        )
        if not pairs:
            raise RuntimeError(f"No paired checkpoints found for {label}")
        pair_parts = {
            pair.step: parts_dir / f"{label}_{pair.step:08d}.json"
            for pair in pairs
        }
        pending_pairs = [
            pair
            for pair in pairs
            if args.refresh_return_matrix or not pair_parts[pair.step].exists()
        ]
        print(
            f"{label}: checkpoints={len(pairs)}, "
            f"pending={len(pending_pairs)}, sources={relation_counts[label]}"
        )
        experiment_cache = args.cache_dir / "experiments" / config["key"]
        if pending_pairs:
            _prefetch_checkpoint_pairs(
                experiment,
                pending_pairs,
                experiment_cache,
                args.download_workers,
            )
        metadata_fields = {
            "agent_parameter_target_mode": parameters.get(
                "parameter_target_mode", "unknown"
            ),
            "agent_normalize_parameter_contribution": parameters.get(
                "normalize_parameter_contribution", "unknown"
            ),
            "agent_network_normalization": parameters.get(
                "network_normalization", "unknown"
            ),
            "agent_parameter_representation": parameters.get(
                "parameter_representation", "unknown"
            ),
            "agent_offline_pretrain_steps": parameters.get(
                "offline_pretrain_steps", "unknown"
            ),
            "agent_train_all_suffixes": parameters.get(
                "train_all_suffixes", "unknown"
            ),
        }
        for pair_index, pair in enumerate(pairs, 1):
            part_path = pair_parts[pair.step]
            if part_path.exists() and not args.refresh_return_matrix:
                rows.extend(json.loads(part_path.read_text(encoding="utf-8")))
                print(
                    f"{label}: return matrix {pair_index}/{len(pairs)} "
                    f"step={pair.step} [cached]"
                )
                continue
            optim_state, params_state = load_checkpoint_pair(
                experiment, pair, experiment_cache
            )
            model = build_checkpoint_heads(
                optimizer_dict,
                optim_state,
                params_state,
                experiment_parameters=parameters,
                device=device,
            )
            checkpoint_rows = analyze_return_transfer(
                model,
                records,
                optimizer_dict,
                checkpoint_step=pair.step,
                checkpoint_label=label,
                checkpoint_key=config["key"],
                batch_size=args.batch_size,
                device=device,
                checkpoint_metadata=metadata_fields,
            )
            rows.extend(checkpoint_rows)
            temporary = part_path.with_suffix(".json.part")
            temporary.write_text(
                json.dumps(checkpoint_rows, ensure_ascii=False, allow_nan=True),
                encoding="utf-8",
            )
            temporary.replace(part_path)
            print(
                f"{label}: return matrix {pair_index}/{len(pairs)} "
                f"step={pair.step}"
            )
            del model, optim_state, params_state
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(args.output_dir / "return_transfer_matrix.csv", rows)
    write_csv(
        args.output_dir / "return_transfer_gaps.csv",
        transfer_gap_rows(rows),
    )
    metadata = {
        "gamma": args.gamma,
        "cohort": args.cohort,
        "labels": [label for label, _ in selected],
        "farm_project": args.farm_project,
        "farm_experiments": args.farm_experiments,
        "dataset_signature": dataset_signature,
        "dataset_signature_payload": signature_payload,
        "evaluation_scope": "fixed_full_archives",
        "return_targets": {
            "one_step": "r_t",
            "mc": "sum_{k=0}^{terminal-t} gamma^k r_{t+k}",
            "best_future": "log((L_t + eps) / (min_{k>=t} L_k + eps))",
        },
    }
    (args.output_dir / "return_transfer_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved return transfer diagnostics to {args.output_dir.resolve()}")


def main():
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_configs(args.labels, cohort=args.cohort)
    if args.command == "calibrate":
        run_calibration(args)
        return
    api = create_api(PROJECT_ROOT)
    if args.command == "inventory":
        _inventory_models(api, args, selected)
    elif args.command == "coverage":
        run_coverage(api, args, selected)
    elif args.command == "compare":
        run_hybrid_comparison(api, args, selected)
    elif args.command == "return-matrix":
        run_return_matrix(api, args, selected)
    else:
        run_analysis(api, args, selected)


if __name__ == "__main__":
    main()
