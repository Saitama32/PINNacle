"""Convert archived entries into chronological diagnostic records."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from typing import Any, Iterable

import torch


@dataclass
class DiagnosticRecord:
    record_id: int
    source_kind: str
    source_key: str
    chain_id: str
    chain_step: int
    state: dict
    action: Any
    reward: float
    done: int
    mc_return: float = 0.0
    asset_step: int | None = None
    asset_file: str | None = None
    local_index: int | None = None


def scalar(value, default=0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return default
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stack_state(state: dict) -> torch.Tensor:
    total = torch.as_tensor(state["loss_total"]).float()
    oper = torch.as_tensor(state["loss_oper"]).float()
    boundary = torch.as_tensor(state["loss_bnd"]).float()
    delta = torch.as_tensor(state.get("delta", torch.zeros_like(total))).float()
    return torch.stack((total, oper, boundary, delta), dim=0)


def transition_action(entry: dict):
    return entry.get("action_raw", entry.get("action"))


def transition_reward(entry: dict) -> float:
    for key in ("reward_model", "model_reward", "reward_model_raw", "reward"):
        if key in entry:
            return scalar(entry[key])
    return 0.0


def build_records(
    blocks: Iterable[tuple[str, str, list[dict]]],
    gamma: float,
) -> list[DiagnosticRecord]:
    """Build trajectories; every experiment block is a hard chain boundary."""
    records = []
    record_id = 0
    for source_kind, source_key, entries in blocks:
        chain_number = 0
        current = []
        for entry in entries:
            done = int(scalar(entry.get("done", 0)))
            source = entry.get("_source", {})
            record = DiagnosticRecord(
                record_id=record_id,
                source_kind=source_kind,
                source_key=source_key,
                chain_id=f"{source_kind}:{source_key}:{chain_number}",
                chain_step=len(current),
                state=entry["state"],
                action=transition_action(entry),
                reward=transition_reward(entry),
                done=done,
                asset_step=source.get("asset_step"),
                asset_file=source.get("asset_file"),
                local_index=source.get("local_index"),
            )
            records.append(record)
            current.append(record)
            record_id += 1
            if done != 0:
                _assign_mc_returns(current, gamma)
                current = []
                chain_number += 1
        if current:
            _assign_mc_returns(current, gamma)
    return records


def _assign_mc_returns(chain: list[DiagnosticRecord], gamma: float) -> None:
    value = 0.0
    for record in reversed(chain):
        value = record.reward + gamma * value
        record.mc_return = value


def _as_int(value) -> int:
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    return int(value)


def decode_action(action, optimizer_dict) -> dict[str, Any]:
    """Normalize both historical raw-action tuple layouts."""
    if action is None:
        raise ValueError("Transition has no action")
    optimizer_index = _as_int(action[0])
    optimizer_names = tuple(optimizer_dict)
    if not 0 <= optimizer_index < len(optimizer_names):
        raise ValueError(f"Optimizer index {optimizer_index} is outside action space")
    optimizer = optimizer_names[optimizer_index]

    indices = {}
    if len(action) >= 3:
        indices["epochs"] = _as_int(action[1])
        indices.update({name: _as_int(value) for name, value in action[2].items()})
    elif len(action) == 2 and isinstance(action[1], dict):
        indices.update({name: _as_int(value) for name, value in action[1].items()})
    else:
        raise ValueError(f"Unsupported raw action layout: {action!r}")

    decoded = {
        "optimizer_index": optimizer_index,
        "optimizer": optimizer,
        "parameter_indices": indices,
        "parameter_values": {},
    }
    for name, index in indices.items():
        values = optimizer_dict[optimizer][name]
        if not 0 <= index < len(values):
            raise ValueError(f"{optimizer}.{name} index {index} is outside action space")
        decoded["parameter_values"][name] = values[index]
    return decoded


def coverage_rows(records: Iterable[DiagnosticRecord], optimizer_dict):
    counts = Counter()
    chains = defaultdict(set)
    for record in records:
        try:
            action = decode_action(record.action, optimizer_dict)
        except (ValueError, TypeError, KeyError, IndexError):
            continue
        signature = (
            record.source_kind,
            record.source_key,
            action["optimizer"],
            json.dumps(action["parameter_indices"], sort_keys=True),
        )
        counts[signature] += 1
        chains[signature].add(record.chain_id)
    rows = []
    for signature, count in sorted(counts.items()):
        source_kind, source_key, optimizer, parameters = signature
        rows.append(
            {
                "source_kind": source_kind,
                "source_key": source_key,
                "optimizer": optimizer,
                "parameter_indices": parameters,
                "transition_count": count,
                "unique_chain_count": len(chains[signature]),
            }
        )
    return rows


def finite_records(records: Iterable[DiagnosticRecord]) -> list[DiagnosticRecord]:
    valid = []
    for record in records:
        try:
            state = stack_state(record.state)
        except (KeyError, TypeError, ValueError):
            continue
        if torch.isfinite(state).all() and math.isfinite(record.mc_return):
            valid.append(record)
    return valid
