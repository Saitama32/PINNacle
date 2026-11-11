"""Counterfactual parameter sweep from archived PINN solver checkpoints."""

from __future__ import annotations

from collections import defaultdict
import copy
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

os.environ.setdefault("DDEBACKEND", "pytorch")

import numpy as np
import torch

from .checkpoint_model import build_checkpoint_heads
from .comet_io import (
    download_asset,
    experiment_key,
    experiment_parameters,
    get_experiment,
    list_assets,
    load_checkpoint_pair,
    load_lightweight_transitions,
    pair_checkpoints,
    select_latest_farm_experiments,
    torch_load,
)
from .config import ACTION_SPACES, EXPERIMENTS, FARM_PROJECT, WORKSPACE
from .dataset import build_records, decode_action, finite_records, stack_state


def build_reaction_diffusion_model(hidden_layers="100*5", nu=0.01, rho=20.0):
    import deepxde as dde
    from src.pde.reaction import ReactionDiffusion1D
    from src.utils.args import parse_hidden_layers

    pde = ReactionDiffusion1D(nu=nu, rho=rho)
    layer_args = type("LayerArgs", (), {"hidden_layers": hidden_layers})()
    layers = [pde.input_dim, *parse_hidden_layers(layer_args), pde.output_dim]
    net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
    loss_weights = np.ones(pde.num_loss, dtype=float)
    for index, config in enumerate(pde.loss_config):
        loss_weights[index] = (
            100.0
            if config.get("type", "") in ("boundary", "initial", "ic")
            else 1.0
        )
    return pde.create_model(net), loss_weights


def _seed_everything(seed: int) -> None:
    import deepxde as dde

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dde.config.set_random_seed(seed)


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _last_state_dict(solver_models) -> dict[str, torch.Tensor] | None:
    for model in reversed(solver_models or []):
        if model is None:
            continue
        state = model.get("state_dict") if isinstance(model, dict) else model.state_dict()
        if state:
            return {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in state.items()
            }
    return None


def _payload_entries(payload) -> list[dict]:
    if isinstance(payload, dict) and "memory" in payload:
        return list(payload["memory"])
    if isinstance(payload, list):
        return payload
    return [payload] if isinstance(payload, dict) else []


def _stable_seed(base_seed: int, source_key: str, asset_step: int, repeat: int) -> int:
    digest = hashlib.sha256(
        f"{source_key}:{asset_step}:{repeat}".encode("utf-8")
    ).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**31 - 1))


def _phase(step: int, maximum: int) -> str:
    ratio = step / max(1, maximum)
    if ratio <= 1 / 3:
        return "early"
    if ratio <= 2 / 3:
        return "middle"
    return "late"


def select_stratified_records(
    records, per_source_phase: int = 2, suffix_horizon: int = 0
):
    """Select states with a preceding checkpoint and a complete future suffix."""
    chains = defaultdict(list)
    for record in records:
        if record.source_kind == "farm":
            chains[record.chain_id].append(record)

    buckets = defaultdict(list)
    for chain in chains.values():
        chain.sort(key=lambda record: record.chain_step)
        maximum = max(record.chain_step for record in chain)
        for index, record in enumerate(chain):
            if index == 0 or record.chain_step == 0 or record.asset_step is None:
                continue
            previous = chain[index - 1]
            if previous.asset_step is None:
                continue
            suffix = chain[index + 1:index + 1 + suffix_horizon]
            if len(suffix) != suffix_horizon:
                continue
            loss_total = torch.as_tensor(record.state["loss_total"]).float()
            representative_loss = float(loss_total.reshape(-1)[-1])
            buckets[(record.source_key, _phase(record.chain_step, maximum))].append(
                (representative_loss, record, previous, suffix)
            )

    selected = []
    for (source_key, phase), candidates in sorted(buckets.items()):
        candidates.sort(key=lambda item: item[0])
        count = min(per_source_phase, len(candidates))
        indexes = np.linspace(0, len(candidates) - 1, count + 2)[1:-1]
        for position in indexes.round().astype(int):
            _, record, previous, suffix = candidates[position]
            selected.append((phase, record, previous, suffix))
    return selected


def attach_head_predictions(selected, models, optimizer_dict, device):
    optimizer_names = tuple(optimizer_dict)
    output = []
    for phase, record, previous, suffix in selected:
        state = stack_state(record.state).unsqueeze(0).to(device)
        predictions = {}
        for label, model in models.items():
            optimizer_q, parameter_q = model.forward_all(state)
            predictions[label] = (optimizer_q.detach().cpu(), parameter_q)
        optimizer_index = int(predictions["G"][0][0].argmax())
        optimizer = optimizer_names[optimizer_index]
        parameter_choices = {}
        for label in models:
            parameter_choices[label] = {
                parameter: int(values[0].argmax().item())
                for parameter, values in predictions[label][1][optimizer].items()
            }
        try:
            factual = decode_action(record.action, optimizer_dict)
        except Exception:
            factual = None
        output.append(
            {
                "selection_id": len(output),
                "phase": phase,
                "source_key": record.source_key,
                "chain_id": record.chain_id,
                "chain_step": record.chain_step,
                "asset_step": int(record.asset_step),
                "asset_local_index": int(record.local_index or 0),
                "previous_asset_step": int(previous.asset_step),
                "previous_asset_local_index": int(previous.local_index or 0),
                "starting_loss_from_state": float(
                    torch.as_tensor(record.state["loss_total"]).reshape(-1)[-1]
                ),
                "g_optimizer_index": optimizer_index,
                "g_optimizer": optimizer,
                "g_parameter_indices": parameter_choices["G"],
                "b_parameter_indices": parameter_choices["B"],
                "factual_action": factual,
                "historical_suffix": [
                    {
                        "chain_step": suffix_record.chain_step,
                        "asset_step": int(suffix_record.asset_step),
                        "asset_local_index": int(suffix_record.local_index or 0),
                        "action": decode_action(suffix_record.action, optimizer_dict),
                    }
                    for suffix_record in suffix
                ],
            }
        )
    return output


def load_final_head_models(api, cache_dir: Path, device: torch.device):
    optimizer_dict = ACTION_SPACES[EXPERIMENTS["G"]["action_space"]]
    models = {}
    for label, config in EXPERIMENTS.items():
        experiment = get_experiment(api, WORKSPACE, config["project"], config["key"])
        pair = pair_checkpoints(list_assets(experiment))[-1]
        optim_state, params_state = load_checkpoint_pair(
            experiment, pair, cache_dir / "experiments" / config["key"]
        )
        models[label] = build_checkpoint_heads(
            optimizer_dict,
            optim_state,
            params_state,
            experiment_parameters=experiment_parameters(experiment),
            device=device,
        )
    return models, optimizer_dict


def build_selection_manifest(
    api,
    cache_dir: Path,
    output_path: Path,
    per_source_phase: int,
    device: torch.device,
    suffix_horizon: int = 0,
):
    experiments = select_latest_farm_experiments(
        api, WORKSPACE, FARM_PROJECT, 8, minimum_duration_hours=0.07
    )
    blocks = []
    for experiment in experiments:
        key = experiment_key(experiment)
        entries = load_lightweight_transitions(
            experiment,
            key,
            FARM_PROJECT,
            cache_dir / "farm" / key,
        )
        blocks.append(("farm", key, entries))
    records = finite_records(build_records(blocks, gamma=0.9))
    selected = select_stratified_records(
        records, per_source_phase, suffix_horizon=suffix_horizon
    )
    models, optimizer_dict = load_final_head_models(api, cache_dir, device)
    manifest = {
        "farm_project": FARM_PROJECT,
        "per_source_phase": per_source_phase,
        "suffix_horizon": suffix_horizon,
        "state_count": len(selected),
        "states": attach_head_predictions(
            selected, models, optimizer_dict, device
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _evaluate_loss(loss_evaluator) -> float:
    value = float(loss_evaluator.evaluate(save_graph=False)["loss_total"])
    return value if math.isfinite(value) else float("nan")


def _decoded_to_train_action(decoded: dict) -> dict:
    values = dict(decoded["parameter_values"])
    epochs = int(values.pop("epochs"))
    return {
        "type": decoded["optimizer"],
        "epochs": epochs,
        "params": values,
    }


def _quiet_train(model, iterations: int) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(
            iterations=int(iterations),
            display_every=max(int(iterations) + 1, 2),
        )


def _historical_suffix_rollout(
    state_dict,
    historical_suffix: list[dict],
    seed: int,
    starting_loss: float,
    immediate_loss: float,
    hidden_layers: str,
    nu: float,
    rho: float,
) -> dict:
    """Continue one counterfactual checkpoint with fixed recorded actions."""
    from landscape_visualization._aux.PINN_loss_data import PINNLossData
    from rl_trainer import _build_torch_optimizer

    losses = [float(immediate_loss)]
    reconstructed_immediate_loss = float(immediate_loss)
    error = ""
    if historical_suffix:
        _seed_everything(seed)
        model, loss_weights = build_reaction_diffusion_model(
            hidden_layers=hidden_layers, nu=nu, rho=rho
        )
        model.net.load_state_dict(copy.deepcopy(state_dict), strict=True)
        loss_evaluator = PINNLossData(model, cache_points=True, use_train=True)
        reconstructed_immediate_loss = _evaluate_loss(loss_evaluator)
        losses[0] = reconstructed_immediate_loss
        for suffix_entry in historical_suffix:
            action = _decoded_to_train_action(suffix_entry["action"])
            try:
                torch_optimizer = _build_torch_optimizer(
                    action["type"], model.net.parameters(), action, model=model.net
                )
                model.compile(torch_optimizer, loss_weights=loss_weights)
                model.optimizer = torch_optimizer
                _quiet_train(model, action["epochs"])
                losses.append(_evaluate_loss(loss_evaluator))
            except Exception as exc:
                error = f"suffix step {suffix_entry['chain_step']}: {type(exc).__name__}: {exc}"
                break
        del model, loss_evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finite_path = [float(starting_loss)] + [loss for loss in losses if math.isfinite(loss)]
    best_loss = min(finite_path) if finite_path else float("nan")
    full_path = [float(starting_loss), *losses]
    best_step = (
        min(
            (index for index, loss in enumerate(full_path) if math.isfinite(loss)),
            key=lambda index: full_path[index],
        )
        if finite_path
        else -1
    )
    final_loss = losses[-1] if losses else float("nan")
    horizon_score = (
        math.log((starting_loss + 1e-12) / (best_loss + 1e-12))
        if math.isfinite(starting_loss) and math.isfinite(best_loss)
        else float("nan")
    )
    stability_gap = (
        math.log((final_loss + 1e-12) / (best_loss + 1e-12))
        if math.isfinite(final_loss) and math.isfinite(best_loss)
        else float("nan")
    )
    return {
        "suffix_steps_requested": len(historical_suffix),
        "suffix_steps_completed": max(0, len(losses) - 1),
        "suffix_final_loss": final_loss,
        "suffix_reconstructed_immediate_loss": reconstructed_immediate_loss,
        "suffix_reconstruction_abs_log_error": (
            abs(
                math.log(
                    (reconstructed_immediate_loss + 1e-12)
                    / (immediate_loss + 1e-12)
                )
            )
            if math.isfinite(reconstructed_immediate_loss)
            and math.isfinite(immediate_loss)
            else float("nan")
        ),
        "horizon_best_loss": best_loss,
        "horizon_best_step": best_step,
        "horizon_score": horizon_score,
        "stability_gap": stability_gap,
        "loss_path": json.dumps(full_path),
        "suffix_error": error,
    }


def evaluate_parameter_grid(
    state_dict,
    optimizer: str,
    optimizer_dict,
    seed: int,
    hidden_layers: str = "100*5",
    nu: float = 0.01,
    rho: float = 20.0,
    outcome_callback: Callable[[dict], None] | None = None,
    skip_actions: set[tuple[int, int]] | None = None,
    historical_suffix: list[dict] | None = None,
):
    """Train three deterministic lr branches and observe all epoch thresholds."""
    from landscape_visualization._aux.PINN_loss_data import PINNLossData
    from rl_trainer import _build_torch_optimizer

    epochs_values = sorted(set(optimizer_dict[optimizer]["epochs"]))
    historical_suffix = list(historical_suffix or [])
    outcomes = []
    for lr_index, lr in enumerate(optimizer_dict[optimizer]["lr"]):
        branch_actions = {
            (optimizer_dict[optimizer]["epochs"].index(epochs), lr_index)
            for epochs in epochs_values
        }
        if skip_actions and branch_actions <= skip_actions:
            continue
        _seed_everything(seed)
        model, loss_weights = build_reaction_diffusion_model(
            hidden_layers=hidden_layers, nu=nu, rho=rho
        )
        model.net.load_state_dict(copy.deepcopy(state_dict), strict=True)
        action = {"type": optimizer, "epochs": max(epochs_values), "params": {"lr": lr}}
        torch_optimizer = _build_torch_optimizer(
            optimizer, model.net.parameters(), action, model=model.net
        )
        model.compile(torch_optimizer, loss_weights=loss_weights)
        model.optimizer = torch_optimizer
        loss_evaluator = PINNLossData(model, cache_points=True, use_train=True)
        initial_loss = _evaluate_loss(loss_evaluator)
        previous_epochs = 0
        branch_start = time.perf_counter()
        for epochs in epochs_values:
            started = time.perf_counter()
            error = ""
            try:
                # Nine counterfactual actions per state otherwise produce nine
                # full DeepXDE tables. Keep the long sweep output compact;
                # exceptions are still captured and persisted below.
                _quiet_train(model, epochs - previous_epochs)
                final_loss = _evaluate_loss(loss_evaluator)
            except Exception as exc:
                final_loss = float("nan")
                error = f"{type(exc).__name__}: {exc}"
            epochs_index = optimizer_dict[optimizer]["epochs"].index(epochs)
            score = (
                math.log((initial_loss + 1e-12) / (final_loss + 1e-12))
                if math.isfinite(initial_loss) and math.isfinite(final_loss)
                else float("nan")
            )
            suffix_started = time.perf_counter()
            rng_state = _capture_rng_state()
            try:
                suffix_metrics = _historical_suffix_rollout(
                    model.net.state_dict(),
                    historical_suffix,
                    seed,
                    initial_loss,
                    final_loss,
                    hidden_layers,
                    nu,
                    rho,
                )
            finally:
                _restore_rng_state(rng_state)
            objective_loss = (
                suffix_metrics["horizon_best_loss"]
                if not error
                and not suffix_metrics["suffix_error"]
                and suffix_metrics["suffix_steps_completed"]
                == suffix_metrics["suffix_steps_requested"]
                else float("nan")
            )
            outcome = {
                    "optimizer": optimizer,
                    "epochs_index": epochs_index,
                    "epochs": epochs,
                    "lr_index": lr_index,
                    "lr": lr,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "score": score,
                    "immediate_score": score,
                    "objective_loss": objective_loss,
                    **suffix_metrics,
                    "segment_seconds": time.perf_counter() - started,
                    "suffix_seconds": time.perf_counter() - suffix_started,
                    "branch_seconds": time.perf_counter() - branch_start,
                    "error": error,
                }
            outcomes.append(outcome)
            if outcome_callback is not None:
                outcome_callback(outcome)
            previous_epochs = epochs
            if error:
                break
        del model, loss_evaluator, torch_optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outcomes


def summarize_outcomes(rows: list[dict]) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["selection_id"]), int(row["repeat"]))].append(row)
    summaries = []
    incomplete_states = 0
    expected_actions = {(epochs, lr) for epochs in range(3) for lr in range(3)}
    for (selection_id, repeat), group in sorted(groups.items()):
        observed_actions = {
            (int(row["epochs_index"]), int(row["lr_index"])) for row in group
        }
        if observed_actions != expected_actions:
            incomplete_states += 1
            continue
        valid = [row for row in group if math.isfinite(float(row["objective_loss"]))]
        if not valid:
            continue
        ranked = sorted(valid, key=lambda row: float(row["objective_loss"]))
        oracle = ranked[0]
        by_action = {
            (int(row["epochs_index"]), int(row["lr_index"])): row for row in valid
        }
        base = group[0]
        summary = {
            "selection_id": selection_id,
            "repeat": repeat,
            "source_key": base["source_key"],
            "phase": base["phase"],
            "optimizer": base["optimizer"],
            "oracle_epochs_index": int(oracle["epochs_index"]),
            "oracle_lr_index": int(oracle["lr_index"]),
            "oracle_objective_loss": float(oracle["objective_loss"]),
            "oracle_horizon_score": float(oracle["horizon_score"]),
        }
        for label in ("g", "b"):
            action = (
                int(base[f"{label}_epochs_index"]),
                int(base[f"{label}_lr_index"]),
            )
            selected = by_action.get(action)
            summary[f"{label}_exact_oracle"] = int(action == (
                int(oracle["epochs_index"]), int(oracle["lr_index"])
            ))
            summary[f"{label}_rank"] = (
                1 + next(index for index, row in enumerate(ranked) if row is selected)
                if selected is not None
                else len(ranked) + 1
            )
            summary[f"{label}_objective_loss"] = (
                float(selected["objective_loss"]) if selected is not None else float("nan")
            )
            summary[f"{label}_horizon_log_regret"] = (
                math.log((float(selected["objective_loss"]) + 1e-12) /
                         (float(oracle["objective_loss"]) + 1e-12))
                if selected is not None else float("nan")
            )
            summary[f"{label}_good_action"] = int(
                selected is not None
                and summary[f"{label}_horizon_log_regret"] <= 0.05
            )
            # Compatibility aliases for existing report consumers.
            summary[f"{label}_final_loss"] = summary[f"{label}_objective_loss"]
            summary[f"{label}_log_regret"] = summary[f"{label}_horizon_log_regret"]

        matrix = np.full((3, 3), np.nan)
        for row in valid:
            matrix[int(row["epochs_index"]), int(row["lr_index"])] = float(
                row["horizon_score"]
            )
        if np.isfinite(matrix).all():
            interaction = (
                matrix
                - matrix.mean(axis=1, keepdims=True)
                - matrix.mean(axis=0, keepdims=True)
                + matrix.mean()
            )
            summary["interaction_rms"] = float(np.sqrt(np.mean(interaction**2)))
        else:
            summary["interaction_rms"] = float("nan")
        immediate = np.asarray([float(row["immediate_score"]) for row in valid])
        horizon = np.asarray([float(row["horizon_score"]) for row in valid])
        finite_scores = np.isfinite(immediate) & np.isfinite(horizon)
        if finite_scores.sum() >= 2 and np.std(immediate[finite_scores]) > 0 and np.std(horizon[finite_scores]) > 0:
            summary["immediate_horizon_correlation"] = float(
                np.corrcoef(immediate[finite_scores], horizon[finite_scores])[0, 1]
            )
        else:
            summary["immediate_horizon_correlation"] = float("nan")
        summaries.append(summary)

    aggregate = {
        "evaluated_states": len(summaries),
        "incomplete_states": incomplete_states,
    }
    if summaries:
        for label in ("g", "b"):
            aggregate[f"{label}_oracle_accuracy"] = float(
                np.mean([row[f"{label}_exact_oracle"] for row in summaries])
            )
            aggregate[f"{label}_mean_rank"] = float(
                np.mean([row[f"{label}_rank"] for row in summaries])
            )
            aggregate[f"{label}_mean_log_regret"] = float(
                np.mean([row[f"{label}_log_regret"] for row in summaries])
            )
            aggregate[f"{label}_good_action_fraction"] = float(
                np.mean([row[f"{label}_good_action"] for row in summaries])
            )
        aggregate["interaction_rms_mean"] = float(
            np.mean([row["interaction_rms"] for row in summaries])
        )
        aggregate["immediate_horizon_correlation_mean"] = float(
            np.nanmean([row["immediate_horizon_correlation"] for row in summaries])
        )
    return summaries, aggregate


def summarize_groups(summaries: list[dict]) -> list[dict]:
    """Aggregate head quality overall and by optimizer/trajectory phase."""
    groups = defaultdict(list)
    for row in summaries:
        groups[("all", "all")].append(row)
        groups[(row["optimizer"], "all")].append(row)
        groups[("all", row["phase"])].append(row)
        groups[(row["optimizer"], row["phase"])].append(row)
    output = []
    for (optimizer, phase), group in sorted(groups.items()):
        row = {"optimizer": optimizer, "phase": phase, "states": len(group)}
        for label in ("g", "b"):
            ranks = [float(item[f"{label}_rank"]) for item in group]
            regrets = [float(item[f"{label}_log_regret"]) for item in group]
            row[f"{label}_oracle_accuracy"] = float(
                np.mean([item[f"{label}_exact_oracle"] for item in group])
            )
            row[f"{label}_mean_rank"] = float(np.mean(ranks))
            row[f"{label}_median_rank"] = float(np.median(ranks))
            row[f"{label}_mean_log_regret"] = float(np.nanmean(regrets))
            row[f"{label}_median_log_regret"] = float(np.nanmedian(regrets))
            row[f"{label}_good_action_fraction"] = float(
                np.mean([item[f"{label}_good_action"] for item in group])
            )
        g_losses = np.asarray([item["g_final_loss"] for item in group], dtype=float)
        b_losses = np.asarray([item["b_final_loss"] for item in group], dtype=float)
        finite = np.isfinite(g_losses) & np.isfinite(b_losses)
        row["g_b_comparable_states"] = int(finite.sum())
        row["g_better_fraction"] = (
            float(np.mean(g_losses[finite] < b_losses[finite])) if finite.any() else float("nan")
        )
        row["b_better_fraction"] = (
            float(np.mean(b_losses[finite] < g_losses[finite])) if finite.any() else float("nan")
        )
        interactions = np.asarray([item["interaction_rms"] for item in group], dtype=float)
        row["interaction_rms_mean"] = float(np.nanmean(interactions))
        row["interaction_rms_median"] = float(np.nanmedian(interactions))
        correlations = np.asarray(
            [item["immediate_horizon_correlation"] for item in group], dtype=float
        )
        row["immediate_horizon_correlation_mean"] = float(np.nanmean(correlations))
        row["immediate_horizon_correlation_median"] = float(np.nanmedian(correlations))
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
