"""Head discrimination, routing and factual-calibration diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .dataset import DiagnosticRecord, decode_action, stack_state


def _entropy(q: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(q.float(), dim=-1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


def _margin(q: torch.Tensor) -> torch.Tensor:
    if q.shape[-1] < 2:
        return torch.zeros(q.shape[:-1], device=q.device)
    top = q.topk(2, dim=-1).values
    return top[..., 0] - top[..., 1]


def _rank_desc(q: torch.Tensor, index: int) -> int:
    return 1 + int((q > q[index]).sum().item())


def _mean(total: float, count: int) -> float:
    return total / count if count else float("nan")


def analyze_checkpoint(
    model,
    records: list[DiagnosticRecord],
    optimizer_dict,
    checkpoint_step: int,
    experiment_label: str,
    batch_size: int,
    device: torch.device,
    evaluation_scope: str = "fixed_all_online",
):
    optimizer_stats = {
        "count": 0,
        "range": 0.0,
        "margin": 0.0,
        "entropy": 0.0,
        "argmax": Counter(),
    }
    parameter_stats = defaultdict(
        lambda: {
            "count": 0,
            "range": 0.0,
            "margin": 0.0,
            "entropy": 0.0,
            "argmax": Counter(),
        }
    )
    factual_rows = []
    optimizer_names = tuple(optimizer_dict)

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        state_batch = torch.stack([stack_state(record.state) for record in batch_records])
        state_batch = state_batch.to(device)
        optimizer_q, parameter_q = model.forward_all(state_batch)
        optimizer_q = optimizer_q.detach().cpu()
        parameter_q = {
            optimizer: {
                parameter: values.detach().cpu()
                for parameter, values in by_parameter.items()
            }
            for optimizer, by_parameter in parameter_q.items()
        }

        optimizer_stats["count"] += len(batch_records)
        optimizer_stats["range"] += float(
            (optimizer_q.max(dim=1).values - optimizer_q.min(dim=1).values).sum()
        )
        optimizer_stats["margin"] += float(_margin(optimizer_q).sum())
        optimizer_stats["entropy"] += float(_entropy(optimizer_q).sum())
        optimizer_stats["argmax"].update(
            optimizer_names[index]
            for index in optimizer_q.argmax(dim=1).tolist()
        )

        for optimizer, by_parameter in parameter_q.items():
            for parameter, q_values in by_parameter.items():
                stats = parameter_stats[(optimizer, parameter)]
                stats["count"] += len(batch_records)
                stats["range"] += float(
                    (q_values.max(dim=1).values - q_values.min(dim=1).values).sum()
                )
                stats["margin"] += float(_margin(q_values).sum())
                stats["entropy"] += float(_entropy(q_values).sum())
                stats["argmax"].update(q_values.argmax(dim=1).tolist())

        for batch_index, record in enumerate(batch_records):
            try:
                action = decode_action(record.action, optimizer_dict)
            except (ValueError, TypeError, KeyError, IndexError):
                continue
            optimizer_index = action["optimizer_index"]
            selected_optimizer = action["optimizer"]
            selected_optimizer_q = optimizer_q[batch_index]
            base = {
                "experiment": experiment_label,
                "checkpoint_step": checkpoint_step,
                "evaluation_scope": evaluation_scope,
                "record_id": record.record_id,
                "source_kind": record.source_kind,
                "source_key": record.source_key,
                "chain_id": record.chain_id,
                "chain_step": record.chain_step,
                "mc_return": record.mc_return,
                "reward": record.reward,
                "done": record.done,
                "observed_optimizer": selected_optimizer,
                "observed_optimizer_q": float(selected_optimizer_q[optimizer_index]),
                "optimizer_rank": _rank_desc(selected_optimizer_q, optimizer_index),
                "optimizer_max_gap": float(
                    selected_optimizer_q.max() - selected_optimizer_q[optimizer_index]
                ),
            }
            for parameter, parameter_index in action["parameter_indices"].items():
                if parameter not in parameter_q[selected_optimizer]:
                    continue
                q_values = parameter_q[selected_optimizer][parameter][batch_index]
                factual_rows.append(
                    {
                        **base,
                        "parameter": parameter,
                        "observed_parameter_index": parameter_index,
                        "observed_parameter_value": action["parameter_values"][parameter],
                        "observed_parameter_q": float(q_values[parameter_index]),
                        "parameter_rank": _rank_desc(q_values, parameter_index),
                        "parameter_max_gap": float(
                            q_values.max() - q_values[parameter_index]
                        ),
                        "parameter_argmax_index": int(q_values.argmax()),
                    }
                )

    count = optimizer_stats["count"]
    optimizer_row = {
        "experiment": experiment_label,
        "checkpoint_step": checkpoint_step,
        "evaluation_scope": evaluation_scope,
        "state_count": count,
        "q_range_mean": _mean(optimizer_stats["range"], count),
        "margin_mean": _mean(optimizer_stats["margin"], count),
        "entropy_mean": _mean(optimizer_stats["entropy"], count),
    }
    for optimizer in optimizer_names:
        optimizer_row[f"argmax_fraction_{optimizer}"] = _mean(
            optimizer_stats["argmax"][optimizer], count
        )

    parameter_rows = []
    for (optimizer, parameter), stats in sorted(parameter_stats.items()):
        row = {
            "experiment": experiment_label,
            "checkpoint_step": checkpoint_step,
            "evaluation_scope": evaluation_scope,
            "optimizer": optimizer,
            "parameter": parameter,
            "state_count": stats["count"],
            "q_range_mean": _mean(stats["range"], stats["count"]),
            "margin_mean": _mean(stats["margin"], stats["count"]),
            "entropy_mean": _mean(stats["entropy"], stats["count"]),
        }
        for index, value in enumerate(optimizer_dict[optimizer][parameter]):
            row[f"argmax_fraction_{index}_{value}"] = _mean(
                stats["argmax"][index], stats["count"]
            )
        parameter_rows.append(row)
    return optimizer_row, parameter_rows, factual_rows


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left, right, rank=False) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    if rank:
        left, right = _average_ranks(left), _average_ranks(right)
    return float(np.corrcoef(left, right)[0, 1])


def calibration_rows(factual_rows: Iterable[dict]):
    groups = defaultdict(list)
    for row in factual_rows:
        key = (
            row["experiment"],
            row["checkpoint_step"],
            row["evaluation_scope"],
            row["source_kind"],
            row["source_key"],
            row["observed_optimizer"],
            row["parameter"],
        )
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        experiment, step, scope, source, source_key, optimizer, parameter = key
        q = [row["observed_parameter_q"] for row in rows]
        returns = [row["mc_return"] for row in rows]
        output.append(
            {
                "experiment": experiment,
                "checkpoint_step": step,
                "evaluation_scope": scope,
                "source_kind": source,
                "source_key": source_key,
                "optimizer": optimizer,
                "parameter": parameter,
                "count": len(rows),
                "pearson_q_mc": _correlation(q, returns),
                "spearman_q_mc": _correlation(q, returns, rank=True),
                "observed_argmax_fraction": sum(
                    row["parameter_rank"] == 1 for row in rows
                ) / len(rows),
                "mean_max_gap": float(
                    np.mean([row["parameter_max_gap"] for row in rows])
                ),
            }
        )
    return output


def optimizer_calibration_rows(factual_rows: Iterable[dict]):
    """Aggregate optimizer-head calibration once per physical transition."""
    unique = {}
    for row in factual_rows:
        identity = (
            row["experiment"],
            row["checkpoint_step"],
            row["evaluation_scope"],
            row["record_id"],
        )
        unique.setdefault(identity, row)

    groups = defaultdict(list)
    for row in unique.values():
        key = (
            row["experiment"],
            row["checkpoint_step"],
            row["evaluation_scope"],
            row["source_kind"],
            row["source_key"],
        )
        groups[key].append(row)

    output = []
    for key, rows in sorted(groups.items()):
        experiment, step, scope, source, source_key = key
        q = [row["observed_optimizer_q"] for row in rows]
        returns = [row["mc_return"] for row in rows]
        output.append(
            {
                "experiment": experiment,
                "checkpoint_step": step,
                "evaluation_scope": scope,
                "source_kind": source,
                "source_key": source_key,
                "count": len(rows),
                "pearson_q_mc": _correlation(q, returns),
                "spearman_q_mc": _correlation(q, returns, rank=True),
                "observed_argmax_fraction": sum(
                    row["optimizer_rank"] == 1 for row in rows
                ) / len(rows),
                "mean_observed_rank": float(
                    np.mean([row["optimizer_rank"] for row in rows])
                ),
                "mean_max_gap": float(
                    np.mean([row["optimizer_max_gap"] for row in rows])
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
