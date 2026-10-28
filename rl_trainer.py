import os
import sys
import time
import json
import dill
import random
import itertools
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import datetime

dill.settings["recurse"] = True
import torch
import deepxde as dde
from RL.rl_environment import EnvRLOptimizer
from RL.rl_algorithms import DQNAgent
from RL.rl_utils.per_buffer import PrioritizedReplayBuffer
from RL.rl_utils.per_offline import recalc_all_priorities_batched
from src.utils.callbacks import ModelSaverCallback  
from deepxde.optimizers.config import set_LBFGS_options, set_MUON_options, set_PSO_options, LBFGS_options, PSO_options
from deepxde.optimizers.pytorch.optimizers import get as get_pytorch_optimizer
from deepxde.optimizers.pytorch.pcgrad import PCGrad
from deepxde.optimizers.pytorch.soap import SOAP
from deepxde.optimizers.pytorch.ssbroyden import SSBroyden
from typing import Any, Dict
from RL.rl_utils.load_buffer.load_exps_from_comet import collect_all_comet_transitions
from RL.rl_utils.load_buffer.load_transitions_into_buffer_pickle import to_cpu

# Enforce single-precision defaults before any model/layer creation.
dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_dir = os.path.join('.', 'transitions')

os.makedirs(output_dir, exist_ok=True)

# --- утилита: реинициализация torch модулей (для "новой траектории") ---
def reinit_torch_weights(module):
    import torch

    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def get_state_shape(loss_surface_params):
    if loss_surface_params.get("state_type") == "raw_loss":
        return (int(loss_surface_params.get("raw_loss_state_len", 10)),)

    min_x, max_x, xnum = loss_surface_params["x_range"]
    step_size = (max_x - min_x) / xnum
    latent_dim = int(loss_surface_params.get("latent_dim", 2))
    if latent_dim < 1:
        raise ValueError(f"latent_dim must be at least 1, got {latent_dim}.")

    coords = torch.arange(min_x, max_x + step_size, step_size)

    return (len(coords),) * latent_dim


def _serialize_solver_models(solver_models):
    if solver_models is None:
        return None

    serialized_models = []
    for solver_model in solver_models:
        if solver_model is None:
            serialized_models.append(None)
            continue
        serialized_models.append({
            "class_name": type(solver_model).__name__,
            "state_dict": {
                key: value.detach().to("cpu").clone()
                for key, value in solver_model.state_dict().items()
            },
        })
    return serialized_models


def _update_adaptive_tolerance(archive_transitions, current_tolerance, config):
    valid_losses = sorted(
        float(transition["current_loss"])
        for transition in archive_transitions
        if np.isfinite(transition.get("current_loss", np.nan))
        and float(transition["current_loss"]) >= 0.0
    )

    warmup_transitions = int(config.get("warmup_transitions", 64))
    elite_fraction = float(config.get("elite_fraction", 0.10))
    upward_factor = float(config.get("upward_factor", 1.30))
    min_improvement = float(
        config.get("downward_min_relative_improvement", 0.05)
    )

    if warmup_transitions < 1:
        raise ValueError(
            f"warmup_transitions must be at least 1, got {warmup_transitions}."
        )
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError(f"elite_fraction must be in (0, 1], got {elite_fraction}.")
    if upward_factor < 1.0:
        raise ValueError(f"upward_factor must be at least 1, got {upward_factor}.")
    if not 0.0 <= min_improvement < 1.0:
        raise ValueError(
            "downward_min_relative_improvement must be in [0, 1), "
            f"got {min_improvement}."
        )

    stats = {
        "valid_count": len(valid_losses),
        "target_count": 0,
        "good_count": 0,
        "target_tolerance": None,
        "previous_tolerance": current_tolerance,
        "new_tolerance": current_tolerance,
        "changed": False,
        "reason": "warmup",
    }
    if len(valid_losses) < warmup_transitions:
        return current_tolerance, stats

    target_count = max(1, int(len(valid_losses) * elite_fraction))
    target_count = min(target_count, len(valid_losses))
    target_tolerance = valid_losses[target_count - 1]
    stats["target_count"] = target_count
    stats["target_tolerance"] = target_tolerance

    if current_tolerance is None:
        new_tolerance = target_tolerance
        stats["reason"] = "initialized"
    else:
        current_tolerance = float(current_tolerance)
        good_count = sum(loss <= current_tolerance for loss in valid_losses)
        stats["good_count"] = good_count

        if good_count < target_count and target_tolerance > current_tolerance:
            if current_tolerance > 0.0:
                new_tolerance = min(
                    target_tolerance,
                    upward_factor * current_tolerance,
                )
            else:
                new_tolerance = target_tolerance
            stats["reason"] = "elite_shortage"
        elif target_tolerance <= current_tolerance * (1.0 - min_improvement):
            new_tolerance = target_tolerance
            stats["reason"] = "improved"
        else:
            new_tolerance = current_tolerance
            stats["reason"] = "unchanged"

    stats["new_tolerance"] = float(new_tolerance)
    stats["good_count"] = sum(loss <= new_tolerance for loss in valid_losses)
    stats["changed"] = (
        current_tolerance is None
        or float(new_tolerance) != float(current_tolerance)
    )
    return float(new_tolerance), stats


def _relabel_archive_trajectory(env, trajectory, tolerance):
    if not trajectory:
        return []

    chain_label = env.relabel_chain(
        losses=[
            float(transition["current_loss"])
            for transition in trajectory
        ],
        tolerance=tolerance,
    )
    selected = trajectory[: chain_label["end_index"] + 1]
    done = chain_label["done"]
    relabeled = []
    for index, (transition, reward) in enumerate(
        zip(selected, chain_label["rewards"])
    ):
        transition_done = done if index == len(selected) - 1 else 0
        reward = float(reward)
        relabeled.append((
            transition["state"],
            transition["next_state"],
            transition["action"],
            reward,
            transition_done,
            reward,
            transition["opt_model_i"],
        ))
    return relabeled


def _select_recent_complete_chains(relabeled_trajectories, capacity):
    selected_reversed = []
    used = 0
    for trajectory in reversed(relabeled_trajectories):
        if len(trajectory) > capacity:
            continue
        if used + len(trajectory) > capacity:
            break
        selected_reversed.append(trajectory)
        used += len(trajectory)
    return list(reversed(selected_reversed))


def _rebuild_adaptive_replay_buffer(
    agent,
    env,
    archive_transitions,
    tolerance,
    priority_batch_size,
):
    old_buffer = agent.replay_buffer
    archive_trajectories = []
    trajectory = []
    for transition in archive_transitions:
        trajectory.append(transition)
        if int(transition.get("done", 0)) != 0:
            archive_trajectories.append(trajectory)
            trajectory = []
    if trajectory:
        archive_trajectories.append(trajectory)

    relabeled = [
        _relabel_archive_trajectory(env, trajectory, tolerance)
        for trajectory in archive_trajectories
    ]
    selected = _select_recent_complete_chains(relabeled, old_buffer.capacity)

    new_buffer = PrioritizedReplayBuffer(
        capacity=old_buffer.capacity,
        alpha=old_buffer.alpha,
        eps=old_buffer.eps,
    )
    new_buffer.success_threshold = old_buffer.success_threshold
    for trajectory in selected:
        for transition in trajectory:
            new_buffer.push(*transition, priority=1.0)

    agent.replay_buffer = new_buffer
    if len(new_buffer) > 0:
        recalc_all_priorities_batched(agent, batch_size=priority_batch_size)
    return {
        "archive_chains": len(archive_trajectories),
        "selected_chains": len(selected),
        "buffer_transitions": len(new_buffer),
    }


def _build_torch_optimizer(opt_name: str, params, action: Dict[str, Any], model=None):

    name = (opt_name or "").lower()
    opt_params = action.get("params", {})
    if name == "adam":
        lr = float(opt_params.get("lr", 1e-3))
        return torch.optim.Adam(
            params, lr=lr,
        )

    if name == "soap":
        lr = float(opt_params.get("lr", 3e-4))
        return SOAP(
            params,
            lr=lr,
        )

    if name == "muon":
        lr = float(opt_params.get("lr", 2e-2))
        set_MUON_options(
            momentum=float(opt_params.get("momentum", 0.95)),
            ns_steps=int(opt_params.get("ns_steps", 5)),
            adam_lr=float(opt_params.get("adam_lr", 3e-4)),
        )
        opt, _ = get_pytorch_optimizer(
            params,
            "muon",
            learning_rate=lr,
            model=model,
        )
        return opt

    if name == "pcgrad":
        lr = float(opt_params.get("lr", 1e-3))
        return PCGrad(
            torch.optim.Adam(params, lr=lr),
        )

    if name in ["ssbroyden", "ss-broyden", "ss_broyden"]:
        lr = float(opt_params.get("lr", 1.0))
        tolerance_grad = float(opt_params.get("tolerance_grad", 1e-10))
        debug = bool(opt_params.get("debug", False))
        debug_every = int(opt_params.get("debug_every", 100))
        return SSBroyden(
            params,
            lr=lr,
            tolerance_grad=tolerance_grad,
            debug=debug,
            debug_every=debug_every,
        )

    if name in ["lbfgs", "l-bfgs", "l_bfgs", "LBFGS"]:
        # torch LBFGS (для pytorch backend DeepXDE норм)
        opt = torch.optim.LBFGS(
            params,
            lr=action["params"]["lr"],
            line_search_fn="strong_wolfe",
            max_iter = 10
        )

        return opt 

    if name in ["pso", "PSO", "Pso"]:
        # Передаём гиперпараметры через глобальные PSO_options
        set_PSO_options(
            lr=float(opt_params.get("lr", 1e-3)),
        )
        return "PSO"  # deepxde/optimizers/pytorch/pso.PSO

    raise ValueError(f"Unknown optimizer type: {opt_name}. Expected Adam / SOAP / Muon / PCGrad / SSBroyden / LBFGS / PSO.")


def _extract_weighted_train_loss(model) -> float:
    loss_train = getattr(getattr(model, "train_state", None), "loss_train", None)
    if loss_train is None:
        return float("nan")

    loss_train = np.asarray(loss_train, dtype=np.float64)
    loss_value = float(np.sum(loss_train))
    if not np.isfinite(loss_value) or loss_value < 0.0:
        return float("nan")
    return loss_value


def _evaluate_weighted_train_loss(model) -> float:
    """Evaluate the current network instead of reading stale train_state data."""
    batch_size = getattr(model, "batch_size", None)
    batch = model.data.train_next_batch(batch_size)
    x_train, y_train = batch[:2]
    aux_vars = batch[2] if len(batch) > 2 else None
    _, losses = model._outputs_losses(True, x_train, y_train, aux_vars)
    losses = np.asarray(losses, dtype=np.float64)
    value = float(np.sum(losses))
    if not np.isfinite(value) or value < 0.0:
        return float("nan")
    return value


def _compute_progress_stats(initial_loss, losses, eps=1e-12):
    initial_loss = float(initial_loss)
    losses = np.asarray(losses, dtype=np.float64)
    eps = float(eps)
    all_losses = np.concatenate(([initial_loss], losses))
    if eps <= 0.0:
        raise ValueError("progress replay eps must be greater than zero")
    if not np.all(np.isfinite(all_losses)) or np.any(all_losses < 0.0):
        raise ValueError("progress replay requires finite non-negative losses")

    best_step = int(np.argmin(all_losses))
    best_loss = float(all_losses[best_step])
    final_loss = float(losses[-1])
    best_score = float(np.log((initial_loss + eps) / (best_loss + eps)))
    final_score = float(np.log((initial_loss + eps) / (final_loss + eps)))
    stability_gap = float(np.log((final_loss + eps) / (best_loss + eps)))
    return {
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "best_score": best_score,
        "final_score": final_score,
        "stability_gap": stability_gap,
        "best_step": best_step,
        "best_index": None if best_step == 0 else best_step - 1,
    }


def _progress_transition_tuple(transition):
    reward = float(transition["reward_model"])
    return (
        transition["state"],
        transition["next_state"],
        transition["action"],
        reward,
        int(transition["done"]),
        reward,
        int(transition["opt_model_i"]),
    )


def _select_recent_progress_records(records, capacity):
    selected_reversed = []
    used = 0
    for record in reversed(records):
        size = len(record["transitions"])
        if size > capacity:
            continue
        if used + size > capacity:
            break
        selected_reversed.append(record)
        used += size
    return list(reversed(selected_reversed))


def _rebuild_progress_replay_buffer(agent, records, priority_batch_size):
    old_buffer = agent.replay_buffer
    selected = _select_recent_progress_records(records, old_buffer.capacity)
    new_buffer = PrioritizedReplayBuffer(
        capacity=old_buffer.capacity,
        alpha=old_buffer.alpha,
        eps=old_buffer.eps,
    )
    new_buffer.success_threshold = old_buffer.success_threshold

    for record in selected:
        indexes = [
            new_buffer.push(
                *_progress_transition_tuple(transition), priority=1.0
            )
            for transition in record["transitions"]
        ]
        best_index = record["best_index"]
        if best_index is not None and record["best_score"] > record["min_score"]:
            new_buffer.register_progress_endpoint(
                indexes[best_index], record["best_score"]
            )

    agent.replay_buffer = new_buffer
    if len(new_buffer) > 0:
        recalc_all_priorities_batched(agent, batch_size=priority_batch_size)
    return {
        "selected_chains": len(selected),
        "buffer_transitions": len(new_buffer),
        "progress_endpoints": len(new_buffer.progress_endpoints),
    }


def _log_replay_action_diagnostics(agent):
    action_stats = {
        action_idx: {
            "count": 0,
            "reward_sum": 0.0,
            "success_nonterminal_count": 0,
            "success_terminal_count": 0,
        }
        for action_idx in agent.i2opt
    }

    current_chain = []

    def action_idx_of(transition):
        action = transition.action
        return int(action[0])

    def record_success_chain(chain):
        if not chain or int(chain[-1].done) != 1:
            return
        for transition in chain[:-1]:
            action_stats[action_idx_of(transition)]["success_nonterminal_count"] += 1
        action_stats[action_idx_of(chain[-1])]["success_terminal_count"] += 1

    for transition in agent.replay_buffer.memory:
        action_idx = action_idx_of(transition)
        stats = action_stats[action_idx]
        stats["count"] += 1
        stats["reward_sum"] += float(transition.reward)

        current_chain.append(transition)
        if int(transition.done) != 0:
            record_success_chain(current_chain)
            current_chain = []

    print("\nPost-processed replay diagnostics:")
    for action_idx, optim_name in agent.i2opt.items():
        stats = action_stats[action_idx]
        count = stats["count"]
        reward_mean = stats["reward_sum"] / max(count, 1)
        print(
            f"  {action_idx} -> {optim_name}: count={count}, "
            f"reward_mean={reward_mean:.6f}, "
            f"success_nonterminal={stats['success_nonterminal_count']}, "
            f"success_terminal={stats['success_terminal_count']}"
        )


def _print_offline_greedy_chain_diagnostic(agent, max_states=16):
    success_indexes = set(agent.replay_buffer.success_indexes)
    successful_chains = []
    current_chain = []

    for index, transition in enumerate(agent.replay_buffer.memory):
        current_chain.append((index, transition))
        if int(transition.done) != 0:
            if index in success_indexes:
                successful_chains.append(current_chain)
            current_chain = []

    if not successful_chains:
        print("\nOffline greedy-chain diagnostic: no successful chain found.")
        return

    successful_chains.sort(key=lambda chain: (len(chain), chain[0][0]))
    chain = successful_chains[len(successful_chains) // 2]

    state_rows = [
        (position, transition.state, int(transition.action[0]))
        for position, (_, transition) in enumerate(chain)
    ]

    if len(state_rows) > max_states:
        selected = np.linspace(
            0, len(state_rows) - 1, num=max_states, dtype=int
        )
        state_rows = [state_rows[index] for index in np.unique(selected)]

    state_batch = torch.stack(
        [agent._stack_state(state) for _, state, _ in state_rows],
        dim=0,
    )

    was_training = agent.model_optim.training
    agent.model_optim.eval()
    with torch.no_grad():
        _, q_values = agent.model_optim(state_batch)
    if was_training:
        agent.model_optim.train()

    q_values = q_values.detach().cpu()
    greedy_actions = q_values.argmax(dim=1)
    top_values = torch.topk(q_values, k=min(2, q_values.shape[1]), dim=1).values

    action_counts = {action_idx: 0 for action_idx in agent.i2opt}
    print(
        "\nOffline greedy-chain diagnostic: "
        f"chain_len={len(chain)}, evaluated_states={len(state_rows)}"
    )
    print("  pos | replay_action | greedy_action | q_gap | q_values")
    for row_index, (position, _, replay_action) in enumerate(state_rows):
        greedy_action = int(greedy_actions[row_index])
        action_counts[greedy_action] += 1
        q_gap = (
            float(top_values[row_index, 0] - top_values[row_index, 1])
            if top_values.shape[1] > 1
            else float("nan")
        )
        replay_name = agent.i2opt[replay_action]
        q_text = ", ".join(
            f"{agent.i2opt[action_idx]}={float(q_values[row_index, action_idx]):.3f}"
            for action_idx in agent.i2opt
        )
        print(
            f"  {position:>3} | {replay_name:<15} | "
            f"{agent.i2opt[greedy_action]:<13} | {q_gap:>5.3f} | {q_text}"
        )

    selected_counts = {
        agent.i2opt[action_idx]: count
        for action_idx, count in action_counts.items()
        if count > 0
    }
    print(
        "  greedy_counts="
        f"{selected_counts}, unique_actions={len(selected_counts)}, "
        f"single_action_collapse={len(selected_counts) == 1}"
    )


def run_deepxde_rl_training(
    model,
    loss_weights,
    train_args: Dict[str, Any],
    rl_agent_params,
    optimizers_dict: Dict[str, Any],
    AE_model_params=None,
    AE_train_params=None,
    loss_surface_params=None,
    save_path: str = ".",

):
    """
    model: deepxde.Model (уже созданный get_model())
    train_args: то, что раньше шло в model.train(**train_args)
    env_ctor: класс/фабрика EnvRLOptimizer
    agent_ctor: класс/фабрика DQNAgent
    """

    # callbacks базовые (Tester/Loss/Plot и т.п.)
    base_callbacks = train_args.get("callbacks", [])
    equation_params = train_args.get("equation_params", [])
    display_every = int(train_args.get("display_every", 100))
    adaptive_tolerance_config = dict(
        rl_agent_params.get("adaptive_tolerance", {}) or {}
    )
    adaptive_tolerance_enabled = bool(
        adaptive_tolerance_config.get("enabled", False)
    )
    progress_replay_config = dict(
        rl_agent_params.get("progress_replay", {}) or {}
    )
    progress_replay_enabled = bool(
        progress_replay_config.get("enabled", False)
    )
    if adaptive_tolerance_enabled and progress_replay_enabled:
        raise ValueError(
            "adaptive_tolerance and progress_replay cannot be enabled together"
        )
    progress_fraction = float(progress_replay_config.get("fraction", 0.30))
    progress_rank_alpha = float(
        progress_replay_config.get("rank_alpha", 0.70)
    )
    progress_train_all_suffixes = bool(
        progress_replay_config.get("train_all_suffixes", False)
    )
    recursive_soft_watkins = bool(
        progress_replay_config.get("recursive_soft_watkins", False)
    )
    update_priorities_from_progress = bool(
        progress_replay_config.get("update_priorities_from_progress", True)
    )
    progress_min_score = float(progress_replay_config.get("min_score", 0.0))
    progress_eps = float(progress_replay_config.get("eps", 1e-12))
    if not np.isfinite(progress_fraction) or not 0.0 <= progress_fraction <= 1.0:
        raise ValueError("progress_replay fraction must be in [0, 1]")
    if not np.isfinite(progress_rank_alpha) or progress_rank_alpha < 0.0:
        raise ValueError("progress_replay rank_alpha cannot be negative")
    if not np.isfinite(progress_min_score) or progress_min_score < 0.0:
        raise ValueError("progress_replay min_score cannot be negative")
    if not np.isfinite(progress_eps) or progress_eps <= 0.0:
        raise ValueError("progress_replay eps must be greater than zero")
    max_actions_per_trajectory = int(
        rl_agent_params.get("max_actions_per_trajectory", 12)
    )
    if max_actions_per_trajectory < 2:
        raise ValueError("max_actions_per_trajectory must be at least 2.")
    priority_recalc_batch_size = int(
        adaptive_tolerance_config.get("priority_recalc_batch_size", 256)
        if adaptive_tolerance_enabled
        else rl_agent_params.get("recalc_batch_size", 32)
    )
    if adaptive_tolerance_enabled and priority_recalc_batch_size < 1:
        raise ValueError("priority_recalc_batch_size must be at least 1.")

    # создаём env/agent (как раньше внутри model.py, только теперь снаружи)
    env = EnvRLOptimizer(optimizers=optimizers_dict,
                         equation_params=equation_params,
                         callbacks=None,
                         AE_model_params=AE_model_params,
                         AE_train_params=AE_train_params,
                         loss_surface_params=loss_surface_params,
                         n_save_models=rl_agent_params['n_save_models'],
                         tolerance=rl_agent_params["tolerance"],
                         terminate_on_tolerance=not (
                             adaptive_tolerance_enabled
                             or progress_replay_enabled
                         ))
    env.configure_chain_reward(
        alpha=rl_agent_params.get("chain_reward_alpha", 0.2),
        dense_clip=rl_agent_params.get("chain_reward_dense_clip", 5.0),
        success_bonus=rl_agent_params.get("chain_success_bonus", 10.0),
        fail_penalty=rl_agent_params.get("chain_fail_penalty", -5.0),
    )

    # These objects must be created after the first optimizer is started
    n_observation = env.observation_space
    # state_dim = np.prod(env.observation_space.shape)
    n_action = env.action_space

    rl_agent = DQNAgent(n_observation,
                        n_action,
                        optimizer_dict=optimizers_dict,
                        memory_size=rl_agent_params["rl_buffer_size"],
                        gamma=rl_agent_params["gamma"],
                        lr=rl_agent_params["lr"],
                        device=device,
                        batch_size=rl_agent_params["rl_batch_size"],
                        n_transitions_reinit = rl_agent_params["n_transitions_reinit"],
                        include_terminal_starts=rl_agent_params.get(
                            "include_terminal_starts", False
                        ),
                        warmup_updates=(
                            0
                            if adaptive_tolerance_enabled
                            else int(rl_agent_params.get("warmup_updates", 30))
                        ),
                        recalc_batch_size=priority_recalc_batch_size,
                        epsilon_start=rl_agent_params.get("epsilon_start"),
                        epsilon_end=rl_agent_params.get("epsilon_end"),
                        epsilon_decay_steps=rl_agent_params.get("epsilon_decay"),
                        epsilon_random_steps=rl_agent_params.get(
                            "epsilon_random_steps"
                        ),
                        progress_replay_enabled=progress_replay_enabled,
                        progress_fraction=progress_fraction,
                        progress_rank_alpha=progress_rank_alpha,
                        progress_train_all_suffixes=progress_train_all_suffixes,
                        recursive_soft_watkins=recursive_soft_watkins,
                        update_priorities_from_progress=(
                            update_priorities_from_progress
                        ),
                        exp = rl_agent_params["exp"],
                        model_snapshot_dir=f"{save_path}/rl_model_snapshots")

    # init state (как у тебя в model.py: нулевые карты)
    state_shape = get_state_shape(loss_surface_params)
    def zero_state():
        z = torch.zeros(state_shape, device=device)
        return {"loss_total": z.clone(), "loss_oper": z.clone(), "loss_bnd": z.clone()}
    
    # rl_agent.replay_buffer = collect_all_comet_transitions(rl_agent.replay_buffer, max_exps_last=500, tolerance = rl_agent_params["tolerance"],
    #                                                        prev_tol= rl_agent_params["prev_tol"], use_tol = rl_agent_params["use_tol"], new_tol = rl_agent_params["new_tol"],
    #                                                        use_log_state=rl_agent_params["log_key"], 
    #                                                        proj_name=rl_agent_params["proj_name"],
    #                                                        reset_success_done_to_failure=rl_agent_params.get("reset_success_done_to_failure", False),
    #                                                        recompute_chain_rewards=rl_agent_params.get("recompute_chain_rewards", False),
    #                                                        set_reward_from_next_loss=rl_agent_params.get("set_reward_from_next_loss", False),
    #                                                        recover_current_loss_from_solver_models=rl_agent_params.get("recover_current_loss_from_solver_models", False),
    #                                                        dde_pde_model=rl_agent_params.get("dde_pde_model", loss_surface_params.get("dde_pde_model")))
    # _log_replay_action_diagnostics(rl_agent)

    # offline_pretrain_steps = int(rl_agent_params.get("offline_pretrain_steps", 0))
    # offline_pretrain_iters = int(rl_agent_params.get("offline_pretrain_iters", 5))
    # if offline_pretrain_steps > 0:
    #     if len(rl_agent.replay_buffer) < rl_agent.batch_size:
    #         raise RuntimeError(
    #             "Not enough replay transitions for offline pretraining: "
    #             f"{len(rl_agent.replay_buffer)} < batch_size({rl_agent.batch_size})"
    #         )

    #     print(
    #         "\nStarting offline RL pretraining: "
    #         f"steps={offline_pretrain_steps}, iters_per_step={offline_pretrain_iters}."
    #     )
    #     for step in range(1, offline_pretrain_steps + 1):
    #         loss_optim, loss_param = rl_agent.optim_(iters=offline_pretrain_iters)
    #         rl_agent.steps_done += 1
    #         if not loss_optim or not loss_param:
    #             raise RuntimeError(
    #                 f"Offline pretraining stopped at step {step}: no updates were made."
    #             )
    #         print(
    #             f"[offline {step}/{offline_pretrain_steps}] "
    #             f"optim_loss_mean={np.mean(loss_optim):.6f}, "
    #             f"param_loss_mean={np.mean(loss_param):.6f}"
    #         )

    #     _print_offline_greedy_chain_diagnostic(rl_agent)
    #     rl_agent.reinit_target()
    #     rl_agent.transition_counter = 0

    rl_agent.start_epsilon_schedule()
    print(
        "Epsilon schedule starts at global metric step "
        f"{rl_agent.epsilon_step_offset}; first online interaction uses step 0."
    )
    # if backup_params is not None:
    #     optim_state, params_state = load_rl_agent_from_comet(backup_params["experiment_key"], map_location=device_type())
    #     rl_agent.model_optim.load_state_dict(optim_state)
    #     rl_agent.model_params.load_state_dict(params_state)

    idx_traj = 0
    archive_transitions = []
    progress_archive_records = []
    current_adaptive_tolerance = None

    for traj in range(train_args["n_trajectories"]):
        # реинициализация сети на новую траекторию
        if hasattr(model.net, "apply"):
            model.net.apply(reinit_torch_weights)

        # сброс локальных переменных траектории
        state = zero_state()
        prev_reward = -1.0
        last_opt = None
        same_opt_streak = 0
        optimizers_history = []
        rl_penalty = 0
        total_reward = 0.0
        trajectory_transitions = []
        trajectory_losses = []
        trajectory_initial_loss = None
        progress_rollout_metrics = None
        final_done = 0

        print('\n############################################################################' +
        f'\nStarting trajectory {idx_traj + 1}/{rl_agent_params["n_trajectories"]} ' +
        'with a new initial point.')


        for t in itertools.count():

            # --- agent action ---
            action, action_raw, is_model = rl_agent.select_action(state)
            agent_step = rl_agent.steps_done
            action_raw[2]['epochs'] = action_raw[1]
            action_raw = (action_raw[0], action_raw[2])

            # штраф за повтор оптимизатора (как у тебя было)
            if last_opt == action["type"]:
                same_opt_streak += 1
            else:
                same_opt_streak = 0
            last_opt = action["type"]

            if is_model:
                print("Action by model")
            else:
                print("Action by epsilon-greedy")
            print(f"\naction = {action}")

            # --- compile optimizer for this chunk ---
            chunk_iters = int(action["epochs"])
            torch_opt = _build_torch_optimizer(action["type"], model.net.parameters(), action, model=model.net)


            model.compile(torch_opt, loss_weights=loss_weights)
            model.optimizer = torch_opt
            if progress_replay_enabled and trajectory_initial_loss is None:
                trajectory_initial_loss = _evaluate_weighted_train_loss(model)
                if not np.isfinite(trajectory_initial_loss):
                    raise RuntimeError(
                        "Could not evaluate a finite initial weighted train loss "
                        "for progress replay."
                    )
                print(
                    "Progress replay initial weighted loss: "
                    f"{trajectory_initial_loss}"
                )
            saver = ModelSaverCallback(total_iterations=chunk_iters, n_save_models=train_args['n_save_models'])
            callbacks = list(base_callbacks) + [saver]

            print('\n===========================================================================\n' +
                    f'\nRL agent training: step {t + 1}.'
                    f'\nTime: {datetime.datetime.now()}.'
                    f'\nUsing optimizer: {action["type"]} for {action["epochs"]} epochs.'
                    f'\nTotal Reward = {total_reward}.\n')

            model.train(
                iterations=chunk_iters,
                display_every=display_every,
                callbacks=callbacks,
                model_save_path=save_path,
                save_model=False,
            )

            solver_models = saver.saved_models
            tester_callback = callbacks[0]
            rmse = tester_callback.rmse
            b_rmse = tester_callback.brmse
            train_loss = _extract_weighted_train_loss(model)
            transition_ready = False

            if np.isfinite(train_loss):
                print(f"Operator RMSE: {rmse}, Boundary RMSE: {b_rmse}")
                print(f"Weighted train loss: {train_loss}")
                if (
                    adaptive_tolerance_enabled
                    and current_adaptive_tolerance is not None
                    and train_loss <= current_adaptive_tolerance
                ):
                    print(
                        "Success by current tol: "
                        f"loss={train_loss} <= "
                        f"tolerance={current_adaptive_tolerance}"
                    )
                old_raw_reward = float(
                    (rmse if np.isfinite(rmse) else 0.0)
                    + (b_rmse if np.isfinite(b_rmse) else 0.0)
                )

                env.solver_models = solver_models
                env.reward_params = {
                    "loss": train_loss,
                }
                env.rl_penalty = rl_penalty

                optimizers_history.append(action["type"])
                print(f'\nPassed optimizer {action["type"]}.')


                env.set_step_context(
                    prev_state=state,
                    step_i=t,
                    same_opt_streak=same_opt_streak,
                    is_model=is_model,
                    rl_opt_step=rl_agent.opt_step,
                    prev_reward_scalar=None if prev_reward == -1 else prev_reward,
                )

                next_state, reward_shaped, done, info = env.step()
                final_done = done
                transition_ready = True

                # prev_reward — теперь просто хранит reward_scalar из info
                prev_reward = info["reward_scalar"]

                trajectory_transitions.append({
                    "state": state,
                    "next_state": next_state,
                    "solver_models": _serialize_solver_models(solver_models),
                    "action_raw": action_raw,
                    "agent_step": agent_step,
                    "done": done,
                    "opt_model_i": info["opt_model_i"],
                    "reward_scalar": float(info["reward_scalar"]),
                    "old_reward_model": float(reward_shaped.item()),
                    "old_raw_reward": old_raw_reward,
                    "current_loss": float(train_loss),
                })
                trajectory_losses.append(float(train_loss))

                total_reward += float(reward_shaped.item())
            else:
                done = -1
                final_done = -1
                reward_shaped = torch.tensor(-10.0, device=device)
                info = {
                    "reward_scalar": 0.0,
                    "opt_model_i": -1,
                }
                print(f"Operator RMSE: {rmse}, Boundary RMSE: {b_rmse}. Stopping trajectory with done = -1.")
                print(f"Weighted train loss: {train_loss}. Stopping trajectory with done = -1.")

            print(f'\nCurrent reward after {action["type"]} optimizer: {info["reward_scalar"]}.\n'
                    f'Reward after taking prev reward and penalty: {reward_shaped}\n'
                    f'Total reward after using {", ".join(optimizers_history)} '
                    f'{"optimizers" if len(optimizers_history) > 1 else "optimizer"}: {total_reward}.\n'
                    f'\ndone = {done}')
            
            if len(rl_agent.replay_buffer) >= rl_agent_params["agent_min_buffer"]:
                rl_agent.optim_(iters=rl_agent_params["agent_update_iters"])

            # callbacks.callbacks[1].save_every = self.t
            # env.render()
            if transition_ready:
                state = next_state
            if done == 1:
                break
            elif done == 0:
                # Keep the legacy termination mechanism: set the penalty after
                # the penultimate action so the following action ends the chain.
                if t == max_actions_per_trajectory - 2:
                    rl_penalty = -1 
            elif done == -1:
                rl_penalty = 0
                break

        if len(trajectory_transitions) > 0:
            if final_done == -1:
                trajectory_transitions[-1]["done"] = -1

            if progress_replay_enabled:
                trajectory_rewards = env.compute_progress_rewards(
                    initial_loss=trajectory_initial_loss,
                    losses=trajectory_losses,
                    eps=progress_eps,
                )
            else:
                trajectory_rewards = env.compute_chain_rewards(
                    losses=trajectory_losses,
                    done=final_done,
                )

            assert len(trajectory_rewards) == len(trajectory_transitions), (
                f"len(trajectory_rewards)={len(trajectory_rewards)} != "
                f"len(trajectory_transitions)={len(trajectory_transitions)}"
            )

            chain_total_reward = 0.0

            for tr, chain_reward in zip(trajectory_transitions, trajectory_rewards):
                chain_reward = float(chain_reward)
                chain_total_reward += chain_reward

                if not adaptive_tolerance_enabled and not progress_replay_enabled:
                    rl_agent.push_memory((
                        tr["state"],
                        tr["next_state"],
                        tr["action_raw"],
                        chain_reward,
                        tr["done"],
                        chain_reward,
                        tr["opt_model_i"],
                    ))

                step_done = tr["agent_step"]

                entry = {
                    'state': tr["state"],
                    'next_state': tr["next_state"],
                    'solver_models': tr["solver_models"],
                    'action': tr["action_raw"],
                    'reward': tr["reward_scalar"],
                    'current_loss': tr["current_loss"],
                    'done': tr["done"],
                    'reward_model_raw': chain_reward,
                    'reward_model': chain_reward,
                    'reward_scheme': (
                        "progress_log_reward"
                        if progress_replay_enabled
                        else "env_chain_reward"
                    ),
                    'old_reward_model': tr["old_reward_model"],
                    'old_raw_reward': tr["old_raw_reward"],
                    'opt_model_i': tr["opt_model_i"],
                }

                try:
                    file_path = os.path.join(output_dir, f'transitions_{step_done}.pt')
                    torch.save(entry, file_path)

                    rl_agent_params['exp'].log_asset(
                        file_path,
                        file_name=f"entry_step_{step_done}.pt",
                        step=step_done,
                        overwrite=True
                    )

                except Exception as e:
                    print(e)

                if adaptive_tolerance_enabled or progress_replay_enabled:
                    archive_transitions.append(to_cpu({
                        key: value
                        for key, value in entry.items()
                        if key != "solver_models"
                    }))

            if adaptive_tolerance_enabled:
                archived_trajectory = archive_transitions[-len(trajectory_transitions):]
                previous_tolerance = current_adaptive_tolerance
                current_adaptive_tolerance, tolerance_stats = (
                    _update_adaptive_tolerance(
                        archive_transitions,
                        current_adaptive_tolerance,
                        adaptive_tolerance_config,
                    )
                )

                rebuild_stats = None
                rebuild_reason = None
                if current_adaptive_tolerance is not None:
                    env.tolerance = current_adaptive_tolerance
                    if tolerance_stats["changed"]:
                        rebuild_reason = tolerance_stats["reason"]
                    else:
                        relabeled_trajectory = _relabel_archive_trajectory(
                            env,
                            archived_trajectory,
                            current_adaptive_tolerance,
                        )
                        if (
                            len(rl_agent.replay_buffer)
                            + len(relabeled_trajectory)
                            > rl_agent.replay_buffer.capacity
                        ):
                            rebuild_reason = "capacity"
                        else:
                            for transition in relabeled_trajectory:
                                rl_agent.replay_buffer.push(*transition, coeff=1.5)

                    if rebuild_reason is not None:
                        rebuild_stats = _rebuild_adaptive_replay_buffer(
                            agent=rl_agent,
                            env=env,
                            archive_transitions=archive_transitions,
                            tolerance=current_adaptive_tolerance,
                            priority_batch_size=priority_recalc_batch_size,
                        )

                print(
                    "\nAdaptive tolerance: "
                    f"valid={tolerance_stats['valid_count']}, "
                    f"target_good={tolerance_stats['target_count']}, "
                    f"actual_good={tolerance_stats['good_count']}, "
                    f"target={tolerance_stats['target_tolerance']}, "
                    f"previous={previous_tolerance}, "
                    f"current={current_adaptive_tolerance}, "
                    f"reason={tolerance_stats['reason']}, "
                    f"rebuild={rebuild_reason}\n"
                )
                if rebuild_stats is not None:
                    print(f"Adaptive replay rebuild: {rebuild_stats}\n")

                experiment = rl_agent_params.get("exp")
                if experiment is not None:
                    adaptive_metrics = {
                        "adaptive_tolerance_valid_count": tolerance_stats["valid_count"],
                        "adaptive_tolerance_target_count": tolerance_stats["target_count"],
                        "adaptive_tolerance_good_count": tolerance_stats["good_count"],
                        "adaptive_archive_trajectories": sum(
                            int(transition.get("done", 0)) != 0
                            for transition in archive_transitions
                        ),
                        "adaptive_archive_transitions": len(archive_transitions),
                        "adaptive_replay_transitions": len(rl_agent.replay_buffer),
                    }
                    if current_adaptive_tolerance is not None:
                        adaptive_metrics["adaptive_tolerance"] = current_adaptive_tolerance
                    if tolerance_stats["target_tolerance"] is not None:
                        adaptive_metrics["adaptive_target_tolerance"] = tolerance_stats[
                            "target_tolerance"
                        ]
                    experiment.log_metrics(adaptive_metrics, step=rl_agent.steps_done)

            elif progress_replay_enabled:
                archived_trajectory = archive_transitions[-len(trajectory_transitions):]
                progress_stats = _compute_progress_stats(
                    trajectory_initial_loss,
                    trajectory_losses,
                    eps=progress_eps,
                )
                record = {
                    "transitions": archived_trajectory,
                    "best_index": progress_stats["best_index"],
                    "best_score": progress_stats["best_score"],
                    "min_score": progress_min_score,
                }
                progress_archive_records.append(record)

                if (
                    len(rl_agent.replay_buffer) + len(archived_trajectory)
                    > rl_agent.replay_buffer.capacity
                ):
                    rebuild_stats = _rebuild_progress_replay_buffer(
                        agent=rl_agent,
                        records=progress_archive_records,
                        priority_batch_size=priority_recalc_batch_size,
                    )
                    print(f"Progress replay rebuild: {rebuild_stats}\n")
                else:
                    indexes = [
                        rl_agent.push_memory(
                            _progress_transition_tuple(transition)
                        )
                        for transition in archived_trajectory
                    ]
                    best_index = progress_stats["best_index"]
                    if (
                        best_index is not None
                        and progress_stats["best_score"] > progress_min_score
                    ):
                        rl_agent.replay_buffer.register_progress_endpoint(
                            indexes[best_index], progress_stats["best_score"]
                        )

                progress_rollout_metrics = {
                    "progress_initial_loss": progress_stats["initial_loss"],
                    "progress_best_loss": progress_stats["best_loss"],
                    "progress_final_loss": progress_stats["final_loss"],
                    "progress_best_score": progress_stats["best_score"],
                    "progress_final_score": progress_stats["final_score"],
                    "progress_stability_gap": progress_stats["stability_gap"],
                    "progress_best_step": progress_stats["best_step"],
                    "progress_pool_size": len(
                        rl_agent.replay_buffer.progress_endpoints
                    ),
                }
                print(
                    "\nProgress replay: "
                    f"initial={progress_stats['initial_loss']}, "
                    f"best={progress_stats['best_loss']} at step "
                    f"{progress_stats['best_step']}, "
                    f"final={progress_stats['final_loss']}, "
                    f"score={progress_stats['best_score']}, "
                    f"pool={len(rl_agent.replay_buffer.progress_endpoints)}\n"
                )

            print(
                f"\nPushed trajectory with "
                f"{'progress log rewards' if progress_replay_enabled else 'env chain rewards'}. "
                f"steps={len(trajectory_transitions)}, "
                f"final_loss={trajectory_losses[-1]}, "
                f"final_done={final_done}, "
                f"chain_total_reward={chain_total_reward}\n"
            )

            if len(rl_agent.replay_buffer) >= rl_agent_params["agent_min_buffer"]:
                rl_agent.optim_(iters=rl_agent_params["agent_update_iters"])

            if progress_rollout_metrics is not None:
                progress_rollout_metrics.update(
                    rl_agent.consume_progress_sampling_metrics()
                )
                experiment = rl_agent_params.get("exp")
                if experiment is not None:
                    experiment.log_metrics(
                        progress_rollout_metrics,
                        step=rl_agent.steps_done,
                    )


        if adaptive_tolerance_enabled or progress_replay_enabled:
            idx_traj += 1
        elif done == 1:
            idx_traj += 1


def train_process_rl(data, save_path, device, seed, rl_agent_params):
    """
    drop-in replacement for train_process(...)
    """
    # hooked = HookedStdout(f"{save_path}/log.txt")
    # sys.stdout = hooked
    # sys.stderr = HookedStdout(f"{save_path}/logerr.txt", sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dde.config.set_default_float("float32")
    # dde.config.set_random_seed(seed)

    payload = dill.loads(data)

    # совместимость: если раньше data был (get_model, train_args)
    # теперь можно передать (get_model, train_args, rl_payload)
    if len(payload) == 2:
        get_model, train_args = payload
        model = get_model()
        model.train(**train_args, model_save_path=save_path)
        return

    get_model, train_args, optimizers, AE_model_params, AE_train_params,  loss_surface_params = payload
    model, loss_weights = get_model()
    # rl_payload структура:
    # {
    #   "train_args": {...},
    #   "optimizers_dict": {...},
    #   "equation_params": ...,
    #   "AE_model_params": ...,
    #   "AE_train_params": ...,
    #   "loss_surface_params": ...
    # }

    run_deepxde_rl_training(
        model=model,
        loss_weights=loss_weights,
        train_args=train_args,
        rl_agent_params=rl_agent_params,
        optimizers_dict=optimizers,
        AE_model_params=AE_model_params,
        AE_train_params=AE_train_params,
        loss_surface_params=loss_surface_params,
        save_path=save_path,
    )
