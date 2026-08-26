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
import copy

dill.settings["recurse"] = True
import torch
import deepxde as dde
from RL.rl_environment import EnvRLOptimizer
from RL.rl_algorithms import DQNAgent
from src.utils.callbacks import ModelSaverCallback  
from deepxde.optimizers.config import set_LBFGS_options, set_MUON_options, set_PSO_options, LBFGS_options, PSO_options
from deepxde.optimizers.pytorch.optimizers import get as get_pytorch_optimizer
from deepxde.optimizers.pytorch.pcgrad import PCGrad
from deepxde.optimizers.pytorch.soap import SOAP
from deepxde.optimizers.pytorch.ssbroyden import SSBroyden
from typing import Any, Dict
from RL.rl_utils.load_buffer.load_exps_from_comet import collect_all_comet_transitions

# Enforce single-precision defaults before any model/layer creation.
dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_dir = os.path.join('.', 'transitions')

os.makedirs(output_dir, exist_ok=True)


def _build_action_space(optimizers, physics_forms=None):
    """Build either the legacy optimizer space or a physics-form action space."""

    action_space = copy.deepcopy(optimizers)
    if physics_forms is None:
        return action_space
    if len(action_space) != 1:
        raise ValueError(
            "Physics-form action mode requires exactly one fixed optimizer."
        )
    if not isinstance(physics_forms, dict) or not physics_forms:
        raise ValueError("physics_forms must be a non-empty dictionary.")

    fixed_optimizer_params = next(iter(action_space.values()))
    fixed_optimizer_params.pop("epochs", None)
    action_space = {}
    for form_name, form_config in physics_forms.items():
        normalized_name = str(form_name).strip().lower()
        if normalized_name not in {"weak", "strong"}:
            raise ValueError(
                f"Unknown physics form {form_name!r}; expected 'weak' or 'strong'."
            )
        if not isinstance(form_config, dict):
            raise ValueError(f"physics_forms[{form_name!r}] must be a dictionary.")
        epochs = form_config.get("epochs")
        if not isinstance(epochs, (list, tuple)) or not epochs:
            raise ValueError(
                f"physics_forms[{form_name!r}]['epochs'] must be a non-empty sequence."
            )
        normalized_epochs = [int(epoch_count) for epoch_count in epochs]
        if any(epoch_count <= 0 for epoch_count in normalized_epochs):
            raise ValueError("Physics-form epoch counts must be positive.")
        params = copy.deepcopy(fixed_optimizer_params)
        params.update(copy.deepcopy(form_config))
        params["epochs"] = normalized_epochs
        action_space[normalized_name] = params
    return action_space


def _set_model_physics_form(model, physics_form):
    """Switch the DeepXDE data/loss provider before recompiling a chunk."""

    if physics_form is None:
        return getattr(model, "physics_loss_kind", "strong")
    normalized_name = str(physics_form).strip().lower()
    if normalized_name not in {"weak", "strong"}:
        raise ValueError(f"Unknown physics form: {physics_form!r}.")

    data_attr = f"{normalized_name}_data"
    data = getattr(model, data_attr, None)
    if data is None:
        raise RuntimeError(
            f"Model does not provide {data_attr}. "
            "Attach the weak-form loss before starting RL training."
        )
    model.data = data
    model.physics_loss_kind = normalized_name
    return normalized_name

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


def _evaluate_strong_train_loss(model, loss_weights) -> float:
    """Evaluate every action on one canonical strong-form objective."""

    strong_data = getattr(model, "strong_data", None)
    if strong_data is None:
        return _extract_weighted_train_loss(model)

    inputs_np, targets_np, auxiliary_vars = strong_data.train_next_batch(
        getattr(model, "batch_size", None)
    )
    parameter = next(model.net.parameters())
    tensor_kwargs = {"device": parameter.device, "dtype": parameter.dtype}
    if isinstance(inputs_np, tuple):
        inputs = tuple(
            torch.as_tensor(value, **tensor_kwargs).requires_grad_()
            for value in inputs_np
        )
    else:
        inputs = torch.as_tensor(inputs_np, **tensor_kwargs).requires_grad_()
    targets = (
        None
        if targets_np is None
        else torch.as_tensor(targets_np, **tensor_kwargs)
    )
    model.net.auxiliary_vars = (
        [] if auxiliary_vars is None else auxiliary_vars
    )
    model.net.train(mode=False)
    outputs = model.net(inputs)
    losses = strong_data.losses_train(
        targets,
        outputs,
        dde.losses.get("MSE"),
        inputs,
        model,
        model.net.auxiliary_vars,
    )
    if not isinstance(losses, (list, tuple)):
        losses = [losses]
    weighted_losses = torch.stack(losses)
    if loss_weights is not None:
        weighted_losses = weighted_losses * torch.as_tensor(
            loss_weights,
            device=weighted_losses.device,
            dtype=weighted_losses.dtype,
        )
    loss_value = float(weighted_losses.sum().detach().cpu())
    dde.grad.clear()
    if not np.isfinite(loss_value) or loss_value < 0.0:
        return float("nan")
    return loss_value


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
    physics_forms=None,
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

    action_space_dict = _build_action_space(optimizers_dict, physics_forms)
    fixed_optimizer_name = (
        next(iter(optimizers_dict)) if physics_forms is not None else None
    )

    # callbacks базовые (Tester/Loss/Plot и т.п.)
    base_callbacks = train_args.get("callbacks", [])
    equation_params = train_args.get("equation_params", [])
    display_every = int(train_args.get("display_every", 100))

    # создаём env/agent (как раньше внутри model.py, только теперь снаружи)
    env = EnvRLOptimizer(optimizers=action_space_dict,
                         equation_params=equation_params,
                         callbacks=None,
                         AE_model_params=AE_model_params,
                         AE_train_params=AE_train_params,
                         loss_surface_params=loss_surface_params,
                         n_save_models=rl_agent_params['n_save_models'],
                         tolerance=rl_agent_params["tolerance"])
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
                        optimizer_dict=action_space_dict,
                        memory_size=rl_agent_params["rl_buffer_size"],
                        gamma=rl_agent_params["gamma"],
                        lr=rl_agent_params["lr"],
                        device=device,
                        batch_size=rl_agent_params["rl_batch_size"],
                        n_transitions_reinit = rl_agent_params["n_transitions_reinit"],
                        include_terminal_starts=rl_agent_params.get(
                            "include_terminal_starts", False
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

    for traj in range(train_args["n_trajectories"]):
        # реинициализация сети на новую траекторию
        if hasattr(model.net, "apply"):
            model.net.apply(reinit_torch_weights)

        # сброс локальных переменных траектории
        state = zero_state()
        prev_reward = -1.0
        last_action_key = None
        same_opt_streak = 0
        optimizers_history = []
        physics_forms_history = []
        rl_penalty = 0
        total_reward = 0.0
        trajectory_transitions = []
        trajectory_losses = []
        final_done = 0

        print('\n############################################################################' +
        f'\nStarting trajectory {idx_traj + 1}/{rl_agent_params["n_trajectories"]} ' +
        'with a new initial point.')


        for t in itertools.count():

            # --- agent action ---
            action, action_raw, is_model = rl_agent.select_action(state)
            agent_step = rl_agent.steps_done
            raw_params = dict(action_raw[2])
            if action_raw[1] is not None:
                raw_params['epochs'] = action_raw[1]
            action_raw = (action_raw[0], raw_params)

            # In the ablation mode the main head selects weak/strong. The
            # optimizer is fixed, while the selected form's parameter head
            # supplies epochs, lr, and any other Adam parameters.
            if fixed_optimizer_name is not None:
                selected_form = action["type"]
                action = {
                    **action,
                    "type": fixed_optimizer_name,
                    "physics_form": selected_form,
                }

            # Repeating the same optimizer with another physics form is a
            # meaningful switch, so the repeat penalty uses the joint action.
            action_key = (action["type"], action.get("physics_form"))
            if last_action_key == action_key:
                same_opt_streak += 1
            else:
                same_opt_streak = 0
            last_action_key = action_key

            if is_model:
                print("Action by model")
            else:
                print("Action by epsilon-greedy")
            print(f"\naction = {action}")

            # --- compile optimizer for this chunk ---
            chunk_iters = int(action["epochs"])
            physics_form = _set_model_physics_form(
                model, action.get("physics_form")
            )
            torch_opt = _build_torch_optimizer(action["type"], model.net.parameters(), action, model=model.net)


            model.compile(torch_opt, loss_weights=loss_weights)
            model.optimizer = torch_opt
            saver = ModelSaverCallback(total_iterations=chunk_iters, n_save_models=train_args['n_save_models'])
            callbacks = list(base_callbacks) + [saver]

            print('\n===========================================================================\n' +
                    f'\nRL agent training: step {t + 1}.'
                    f'\nTime: {datetime.datetime.now()}.'
                    f'\nUsing optimizer: {action["type"]} for {action["epochs"]} epochs.'
                    f'\nPhysics form: {physics_form}.'
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
            objective_train_loss = _extract_weighted_train_loss(model)
            train_loss = (
                objective_train_loss
                if physics_form == "strong"
                else _evaluate_strong_train_loss(model, loss_weights)
            )
            transition_ready = False

            if np.isfinite(train_loss) and np.isfinite(objective_train_loss):
                print(f"Operator RMSE: {rmse}, Boundary RMSE: {b_rmse}")
                print(
                    f"Selected-form weighted train loss: {objective_train_loss}\n"
                    f"Canonical strong-form reward loss: {train_loss}"
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
                physics_forms_history.append(physics_form)
                print(
                    f'\nPassed optimizer {action["type"]} '
                    f'with {physics_form} physics form.'
                )


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
                    "physics_form": physics_form,
                    "agent_step": agent_step,
                    "done": done,
                    "opt_model_i": info["opt_model_i"],
                    "reward_scalar": float(info["reward_scalar"]),
                    "old_reward_model": float(reward_shaped.item()),
                    "old_raw_reward": old_raw_reward,
                    "current_loss": float(train_loss),
                    "objective_train_loss": float(objective_train_loss),
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
                print(
                    f"Selected-form weighted train loss: {objective_train_loss}. "
                    f"Canonical strong-form reward loss: {train_loss}. "
                    "Stopping trajectory with done = -1."
                )

            history = ", ".join(
                f"{optimizer}/{form}"
                for optimizer, form in zip(optimizers_history, physics_forms_history)
            )
            print(f'\nCurrent reward after {action["type"]}/{physics_form}: {info["reward_scalar"]}.\n'
                    f'Reward after taking prev reward and penalty: {reward_shaped}\n'
                    f'Total reward after using {history}: {total_reward}.\n'
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
                if t == 10:
                    rl_penalty = -1 
            elif done == -1:
                rl_penalty = 0
                break

        if len(trajectory_transitions) > 0:
            if final_done == -1:
                trajectory_transitions[-1]["done"] = -1

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

                try:
                    file_path = os.path.join(output_dir, f'transitions_{step_done}.pt')

                    entry = {
                        'state': tr["state"],
                        'next_state': tr["next_state"],
                        'solver_models': tr["solver_models"],
                        'action': tr["action_raw"],
                        'physics_form': tr["physics_form"],
                        'reward': tr["reward_scalar"],
                        'current_loss': tr["current_loss"],
                        'objective_train_loss': tr["objective_train_loss"],
                        'done': tr["done"],
                        'reward_model_raw': chain_reward,
                        'reward_model': chain_reward,
                        'reward_scheme': "env_chain_reward",
                        'old_reward_model': tr["old_reward_model"],
                        'old_raw_reward': tr["old_raw_reward"],
                        'opt_model_i': tr["opt_model_i"],
                    }
                    torch.save(entry, file_path)

                    rl_agent_params['exp'].log_asset(
                        file_path,
                        file_name=f"entry_step_{step_done}.pt",
                        step=step_done,
                        overwrite=True
                    )

                except Exception as e:
                    print(e)

            print(
                f"\nPushed trajectory with env chain rewards. "
                f"steps={len(trajectory_transitions)}, "
                f"final_loss={trajectory_losses[-1]}, "
                f"final_done={final_done}, "
                f"chain_total_reward={chain_total_reward}\n"
            )

            if len(rl_agent.replay_buffer) >= rl_agent_params["agent_min_buffer"]:
                rl_agent.optim_(iters=rl_agent_params["agent_update_iters"])


        if done == 1:
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

    if len(payload) == 6:
        get_model, train_args, optimizers, AE_model_params, AE_train_params, loss_surface_params = payload
        physics_forms = None
    elif len(payload) == 7:
        get_model, train_args, optimizers, physics_forms, AE_model_params, AE_train_params, loss_surface_params = payload
    else:
        raise ValueError(f"Unsupported RL training payload length: {len(payload)}")
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
        physics_forms=physics_forms,
        AE_model_params=AE_model_params,
        AE_train_params=AE_train_params,
        loss_surface_params=loss_surface_params,
        save_path=save_path,
    )
