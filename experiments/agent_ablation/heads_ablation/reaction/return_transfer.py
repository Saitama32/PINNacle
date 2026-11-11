"""Checkpoint/source/return-target transfer diagnostics for both DQN heads."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch

from .dataset import DiagnosticRecord, decode_action, stack_state
from .diagnostics import _correlation


RETURN_TARGETS = ("one_step", "mc", "best_future")


def return_target(record: DiagnosticRecord, name: str) -> float:
    if name == "one_step":
        return record.reward
    if name == "mc":
        return record.mc_return
    if name == "best_future":
        return record.best_future_progress
    raise ValueError(f"Unknown return target: {name}")


def source_relation(record: DiagnosticRecord, checkpoint_key: str) -> str:
    if record.source_kind == "farm":
        return "farm"
    if record.source_key == checkpoint_key:
        return "own"
    return "other"


def _rank_desc(values: torch.Tensor, index: int) -> int:
    return 1 + int((values > values[index]).sum().item())


def _summary(values: list[dict], identity: dict) -> dict:
    q = np.asarray([item["q"] for item in values], dtype=float)
    target = np.asarray([item["target"] for item in values], dtype=float)
    ranks = np.asarray([item["rank"] for item in values], dtype=float)
    gaps = np.asarray([item["max_gap"] for item in values], dtype=float)
    cutoff = np.quantile(q, 0.75) if len(q) else float("nan")
    top = target[q >= cutoff]
    target_std = float(np.std(target))
    lift = float(np.mean(top) - np.mean(target)) if len(top) else float("nan")
    return {
        **identity,
        "count": len(values),
        "source_experiment_count": len({item["source_key"] for item in values}),
        "q_mean": float(np.mean(q)),
        "q_std": float(np.std(q)),
        "target_mean": float(np.mean(target)),
        "target_std": target_std,
        "pearson_q_target": _correlation(q, target),
        "spearman_q_target": _correlation(q, target, rank=True),
        "top_q_quartile_target_lift": lift,
        "top_q_quartile_standardized_lift": (
            lift / target_std if target_std > 0 else float("nan")
        ),
        "observed_argmax_fraction": float(np.mean(ranks == 1)),
        "mean_observed_rank": float(np.mean(ranks)),
        "mean_max_gap": float(np.mean(gaps)),
        "best_future_horizon_mean": float(
            np.mean([item["best_future_horizon"] for item in values])
        ),
    }


@torch.inference_mode()
def analyze_return_transfer(
    model,
    records: list[DiagnosticRecord],
    optimizer_dict,
    checkpoint_step: int,
    checkpoint_label: str,
    checkpoint_key: str,
    batch_size: int,
    device: torch.device,
    checkpoint_metadata: dict | None = None,
) -> list[dict]:
    """Compare factual Q with three return definitions on each source domain."""
    groups = defaultdict(list)
    checkpoint_metadata = checkpoint_metadata or {}
    optimizer_names = tuple(optimizer_dict)

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        states = torch.stack([stack_state(record.state) for record in batch]).to(device)
        optimizer_q, parameter_q = model.forward_all(states)
        optimizer_q = optimizer_q.detach().cpu()
        parameter_q = {
            optimizer: {
                parameter: q.detach().cpu()
                for parameter, q in by_parameter.items()
            }
            for optimizer, by_parameter in parameter_q.items()
        }

        for batch_index, record in enumerate(batch):
            try:
                action = decode_action(record.action, optimizer_dict)
            except (ValueError, TypeError, KeyError, IndexError):
                continue
            relation = source_relation(record, checkpoint_key)
            optimizer_index = action["optimizer_index"]
            optimizer = action["optimizer"]
            optimizer_values = optimizer_q[batch_index]
            common = {
                "source_key": record.source_key,
                "best_future_horizon": record.best_future_horizon,
            }
            for target_name in RETURN_TARGETS:
                target = return_target(record, target_name)
                if not math.isfinite(target):
                    continue
                optimizer_observation = {
                    **common,
                    "q": float(optimizer_values[optimizer_index]),
                    "target": target,
                    "rank": _rank_desc(optimizer_values, optimizer_index),
                    "max_gap": float(
                        optimizer_values.max() - optimizer_values[optimizer_index]
                    ),
                }
                for source_scope, source_group in (
                    ("relation", relation),
                    ("experiment", record.source_key),
                ):
                    group_key = (
                        "optimizer",
                        relation,
                        source_scope,
                        source_group,
                        target_name,
                        "all",
                        "all",
                    )
                    groups[group_key].append(optimizer_observation)

                for parameter, parameter_index in action["parameter_indices"].items():
                    if parameter not in parameter_q.get(optimizer, {}):
                        continue
                    parameter_values = parameter_q[optimizer][parameter][batch_index]
                    parameter_observation = {
                        **common,
                        "q": float(parameter_values[parameter_index]),
                        "target": target,
                        "rank": _rank_desc(parameter_values, parameter_index),
                        "max_gap": float(
                            parameter_values.max()
                            - parameter_values[parameter_index]
                        ),
                    }
                    for source_scope, source_group in (
                        ("relation", relation),
                        ("experiment", record.source_key),
                    ):
                        parameter_key = (
                            "parameter",
                            relation,
                            source_scope,
                            source_group,
                            target_name,
                            optimizer,
                            parameter,
                        )
                        groups[parameter_key].append(parameter_observation)

    rows = []
    for key, values in sorted(groups.items()):
        (
            head,
            relation,
            source_scope,
            source_group,
            target_name,
            optimizer,
            parameter,
        ) = key
        rows.append(
            _summary(
                values,
                {
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_key": checkpoint_key,
                    "checkpoint_step": checkpoint_step,
                    **checkpoint_metadata,
                    "head": head,
                    "optimizer": optimizer,
                    "parameter": parameter,
                    "source_relation": relation,
                    "source_scope": source_scope,
                    "source_group": source_group,
                    "return_target": target_name,
                },
            )
        )
    return rows


def transfer_gap_rows(rows: list[dict]) -> list[dict]:
    """Subtract farm/other calibration from own calibration at each checkpoint."""
    grouped = defaultdict(dict)
    identity_fields = (
        "checkpoint_label",
        "checkpoint_key",
        "checkpoint_step",
        "head",
        "optimizer",
        "parameter",
        "return_target",
    )
    for row in rows:
        if row.get("source_scope", "relation") != "relation":
            continue
        identity = tuple(row[field] for field in identity_fields)
        grouped[identity][row["source_relation"]] = row

    output = []
    for identity, by_relation in sorted(grouped.items()):
        own = by_relation.get("own")
        if own is None:
            continue
        base = dict(zip(identity_fields, identity))
        for relation in ("farm", "other"):
            external = by_relation.get(relation)
            if external is None:
                continue
            for metric in (
                "pearson_q_target",
                "spearman_q_target",
                "top_q_quartile_standardized_lift",
                "observed_argmax_fraction",
            ):
                left = own[metric]
                right = external[metric]
                output.append(
                    {
                        **base,
                        "external_relation": relation,
                        "metric": metric,
                        "own_value": left,
                        "external_value": right,
                        "own_minus_external": (
                            left - right
                            if math.isfinite(left) and math.isfinite(right)
                            else float("nan")
                        ),
                    }
                )
    return output
