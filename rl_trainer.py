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
from src.model import RWFMLP, materialize_effective_mlp
from src.utils.callbacks import (
    LongHorizonMetricsCallback,
    ModelSaverCallback,
    SolutionImageCallback,
)
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

# --- утилита: реинициализация torch модулей (для "новой траектории") ---
def reinit_torch_weights(module):
    import torch

    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def reinit_network(network):
    """Start a fresh trajectory while preserving structured RWF initialization."""
    if isinstance(network, RWFMLP):
        network.reset_parameters()
    elif hasattr(network, "apply"):
        network.apply(reinit_torch_weights)


def _landscape_snapshots(solver_models):
    if solver_models and isinstance(solver_models[0], RWFMLP):
        return [materialize_effective_mlp(snapshot) for snapshot in solver_models]
    return solver_models

def get_state_shape(loss_surface_params):
    min_x, max_x, xnum = loss_surface_params["x_range"]
    min_y, max_y = min_x, max_x
    step_size = (max_x - min_x) / xnum

    x_coords = torch.arange(min_x, max_x + step_size, step_size)
    y_coords = torch.arange(min_y, max_y + step_size, step_size)

    return tuple(torch.meshgrid(x_coords, y_coords)[0].shape)


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
    image_log_every = int(train_args.get("image_log_every", display_every))
    save_solution_images = bool(train_args.get("save_solution_images", False))
    log_long_horizon_metrics = bool(
        train_args.get("log_long_horizon_metrics", False)
    )
    transitions_dir = os.path.join(save_path, "transitions")
    os.makedirs(transitions_dir, exist_ok=True)

    # создаём env/agent (как раньше внутри model.py, только теперь снаружи)
    env = EnvRLOptimizer(optimizers=optimizers_dict,
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
                        optimizer_dict=optimizers_dict,
                        memory_size=rl_agent_params["rl_buffer_size"],
                        gamma=rl_agent_params["gamma"],
                        lr=rl_agent_params["lr"],
                        device=device,
                        batch_size=rl_agent_params["rl_batch_size"],
                        n_transitions_reinit = rl_agent_params["n_transitions_reinit"],
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
    #                                                        recompute_chain_rewards=rl_agent_params.get("recompute_chain_rewards", True),
    #                                                     set_reward_from_next_loss=rl_agent_params.get("set_reward_from_next_loss", True))
    # if backup_params is not None:
    #     optim_state, params_state = load_rl_agent_from_comet(backup_params["experiment_key"], map_location=device_type())
    #     rl_agent.model_optim.load_state_dict(optim_state)
    #     rl_agent.model_params.load_state_dict(params_state)

    idx_traj = 0
    validation_epoch = 0

    for traj in range(train_args["n_trajectories"]):
        # реинициализация сети на новую траекторию
        reinit_network(model.net)

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
            saver = ModelSaverCallback(total_iterations=chunk_iters, n_save_models=train_args['n_save_models'])
            optimizer_slug = str(action["type"]).strip().lower().replace(" ", "_")
            step_save_path = save_path
            callbacks = list(base_callbacks)
            diagnostics_dir = os.path.join(
                save_path,
                f"trajectory_{traj + 1:04d}",
                "diagnostics",
            )
            metrics_path = os.path.join(
                diagnostics_dir,
                f"step_{t + 1:02d}_{optimizer_slug}_metrics.csv",
            )
            metrics_callback = None
            if log_long_horizon_metrics:
                metrics_callback = LongHorizonMetricsCallback(
                    metrics_path,
                    log_every=image_log_every,
                    experiment=rl_agent_params.get("exp"),
                    metric_step=agent_step,
                    epoch_offset=validation_epoch,
                )
                callbacks.insert(
                    0,
                    metrics_callback,
                )
            if save_solution_images:
                step_save_path = os.path.join(
                    save_path,
                    f"trajectory_{traj + 1:04d}",
                    f"step_{t + 1:02d}_{optimizer_slug}",
                )
                solution_callback = SolutionImageCallback(
                    step_save_path,
                    log_every=image_log_every,
                    metrics_path=metrics_path,
                    experiment=rl_agent_params.get("exp"),
                    metric_step=agent_step,
                )
                callbacks.insert(0, solution_callback)
            callbacks.append(saver)

            print('\n===========================================================================\n' +
                    f'\nRL agent training: step {t + 1}.'
                    f'\nTime: {datetime.datetime.now()}.'
                    f'\nUsing optimizer: {action["type"]} for {action["epochs"]} epochs.'
                    f'\nTotal Reward = {total_reward}.\n')

            model.train(
                iterations=chunk_iters,
                display_every=display_every,
                callbacks=callbacks,
                model_save_path=step_save_path,
                save_model=False,
            )
            if metrics_callback is not None:
                validation_epoch += metrics_callback.local_epoch

            solver_models = _landscape_snapshots(saver.saved_models)
            tester_callback = next(
                callback for callback in callbacks if hasattr(callback, "rmse")
            )
            rmse = tester_callback.rmse
            b_rmse = tester_callback.brmse
            train_loss = _extract_weighted_train_loss(model)
            transition_ready = False

            if np.isfinite(train_loss):
                print(f"Operator RMSE: {rmse}, Boundary RMSE: {b_rmse}")
                print(f"Weighted train loss: {train_loss}")
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
                    file_path = os.path.join(transitions_dir, f'transitions_{step_done}.pt')

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
