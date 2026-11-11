"""Compare optimizer/parameter checkpoint hybrids on identical states."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

from .dataset import DiagnosticRecord, decode_action, stack_state


def _margin_at(q_values: torch.Tensor, selected: int) -> float:
    selected_q = q_values[selected]
    alternatives = torch.cat((q_values[:selected], q_values[selected + 1 :]))
    if alternatives.numel() == 0:
        return 0.0
    return float(selected_q - alternatives.max())


def compare_hybrid_policies(
    models: dict[str, Any],
    records: list[DiagnosticRecord],
    optimizer_dict,
    batch_size: int,
    device: torch.device,
):
    """Evaluate GG/GB/BG/BB decisions without changing logged transitions."""
    labels = tuple(models)
    optimizer_names = tuple(optimizer_dict)
    action_rows = []
    optimizer_agreement = Counter()
    parameter_agreement = Counter()
    parameter_totals = Counter()

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        state_batch = torch.stack(
            [stack_state(record.state) for record in batch_records]
        ).to(device)
        predictions = {}
        for label, model in models.items():
            optimizer_q, parameter_q = model.forward_all(state_batch)
            predictions[label] = (
                optimizer_q.detach().cpu(),
                {
                    optimizer: {
                        parameter: values.detach().cpu()
                        for parameter, values in by_parameter.items()
                    }
                    for optimizer, by_parameter in parameter_q.items()
                },
            )

        if len(labels) == 2:
            left, right = labels
            left_optim = predictions[left][0].argmax(dim=1)
            right_optim = predictions[right][0].argmax(dim=1)
            optimizer_agreement["same"] += int((left_optim == right_optim).sum())
            optimizer_agreement["total"] += len(batch_records)
            for optimizer in optimizer_names:
                for parameter in optimizer_dict[optimizer]:
                    left_choice = predictions[left][1][optimizer][parameter].argmax(1)
                    right_choice = predictions[right][1][optimizer][parameter].argmax(1)
                    key = (optimizer, parameter)
                    parameter_agreement[key] += int(
                        (left_choice == right_choice).sum()
                    )
                    parameter_totals[key] += len(batch_records)

        for batch_index, record in enumerate(batch_records):
            try:
                factual = decode_action(record.action, optimizer_dict)
            except (ValueError, TypeError, KeyError, IndexError):
                factual = None

            for optimizer_source in labels:
                optimizer_q = predictions[optimizer_source][0][batch_index]
                optimizer_index = int(optimizer_q.argmax())
                optimizer = optimizer_names[optimizer_index]
                for parameter_source in labels:
                    parameter_predictions = predictions[parameter_source][1][optimizer]
                    indices = {
                        parameter: int(values[batch_index].argmax())
                        for parameter, values in parameter_predictions.items()
                    }
                    values = {
                        parameter: optimizer_dict[optimizer][parameter][index]
                        for parameter, index in indices.items()
                    }
                    optimizer_match = bool(
                        factual is not None
                        and factual["optimizer_index"] == optimizer_index
                    )
                    parameters_match = bool(
                        optimizer_match
                        and all(
                            factual["parameter_indices"].get(parameter) == index
                            for parameter, index in indices.items()
                        )
                    )
                    row = {
                        "hybrid": f"{optimizer_source}{parameter_source}",
                        "optimizer_source": optimizer_source,
                        "parameter_source": parameter_source,
                        "record_id": record.record_id,
                        "source_kind": record.source_kind,
                        "source_key": record.source_key,
                        "chain_id": record.chain_id,
                        "chain_step": record.chain_step,
                        "optimizer": optimizer,
                        "optimizer_index": optimizer_index,
                        "optimizer_selected_margin": _margin_at(
                            optimizer_q, optimizer_index
                        ),
                        "optimizer_match": int(optimizer_match),
                        "full_action_match": int(parameters_match),
                        "reward": record.reward,
                        "mc_return": record.mc_return,
                    }
                    for parameter, index in indices.items():
                        q_values = parameter_predictions[parameter][batch_index]
                        row[f"{parameter}_index"] = index
                        row[f"{parameter}_value"] = values[parameter]
                        row[f"{parameter}_selected_margin"] = _margin_at(
                            q_values, index
                        )
                    action_rows.append(row)

    agreement_rows = []
    if optimizer_agreement["total"]:
        agreement_rows.append(
            {
                "head": "optimizer",
                "optimizer": "*",
                "parameter": "*",
                "agreement_fraction": (
                    optimizer_agreement["same"] / optimizer_agreement["total"]
                ),
                "state_count": optimizer_agreement["total"],
            }
        )
    for (optimizer, parameter), same in sorted(parameter_agreement.items()):
        total = parameter_totals[(optimizer, parameter)]
        agreement_rows.append(
            {
                "head": "parameter",
                "optimizer": optimizer,
                "parameter": parameter,
                "agreement_fraction": same / total,
                "state_count": total,
            }
        )

    return action_rows, agreement_rows, _hybrid_summary(action_rows)


def _hybrid_summary(action_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in action_rows:
        groups[(row["hybrid"], row["source_kind"], row["source_key"])].append(row)
    output = []
    for (hybrid, source_kind, source_key), rows in sorted(groups.items()):
        optimizer_matches = [row for row in rows if row["optimizer_match"]]
        full_matches = [row for row in rows if row["full_action_match"]]
        output.append(
            {
                "hybrid": hybrid,
                "source_kind": source_kind,
                "source_key": source_key,
                "state_count": len(rows),
                "optimizer_match_fraction": len(optimizer_matches) / len(rows),
                "full_action_match_fraction": len(full_matches) / len(rows),
                "full_action_match_count": len(full_matches),
                "matched_reward_mean": (
                    float(np.mean([row["reward"] for row in full_matches]))
                    if full_matches
                    else float("nan")
                ),
                "matched_mc_return_mean": (
                    float(np.mean([row["mc_return"] for row in full_matches]))
                    if full_matches
                    else float("nan")
                ),
            }
        )
    return output

