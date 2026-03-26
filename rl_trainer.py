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
from src.utils.callbacks import ModelSaverCallback  
from deepxde.optimizers.config import set_LBFGS_options, set_PSO_options, LBFGS_options, PSO_options
from typing import Any, Dict
from RL.rl_utils.load_buffer.load_exps_from_comet import collect_all_comet_transitions

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
    min_x, max_x, xnum = loss_surface_params["x_range"]
    min_y, max_y = min_x, max_x
    step_size = (max_x - min_x) / xnum

    x_coords = torch.arange(min_x, max_x + step_size, step_size)
    y_coords = torch.arange(min_y, max_y + step_size, step_size)

    return tuple(torch.meshgrid(x_coords, y_coords)[0].shape)


def _build_torch_optimizer(opt_name: str, params, action: Dict[str, Any]):

    name = (opt_name or "").lower()
    opt_params = action.get("params", {})
    if name == "adam":
        lr = float(opt_params.get("lr", 1e-3))
        return torch.optim.Adam(
            params, lr=lr,
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

    raise ValueError(f"Unknown optimizer type: {opt_name}. Expected Adam / LBFGS / PSO.")


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

    # создаём env/agent (как раньше внутри model.py, только теперь снаружи)
    env = EnvRLOptimizer(optimizers=optimizers_dict,
                                 equation_params=equation_params,
                                 callbacks=None,
                                 AE_model_params=AE_model_params,
                                 AE_train_params=AE_train_params,
                                 loss_surface_params=loss_surface_params,
                                 n_save_models=rl_agent_params['n_save_models'],
                                 tolerance=rl_agent_params["tolerance"])

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
                        exp = rl_agent_params["exp"])

    # init state (как у тебя в model.py: нулевые карты)
    state_shape = get_state_shape(loss_surface_params)
    def zero_state():
        z = torch.zeros(state_shape, device=device)
        return {"loss_total": z.clone(), "loss_oper": z.clone(), "loss_bnd": z.clone()}
    
    rl_agent.replay_buffer = collect_all_comet_transitions(rl_agent.replay_buffer, max_exps_last=500, tolerance = rl_agent_params["tolerance"],
                                                           prev_tol= rl_agent_params["prev_tol"], use_tol = rl_agent_params["use_tol"], new_tol = rl_agent_params["new_tol"],
                                                           use_log_state=rl_agent_params["log_key"], 
                                                           proj_name=rl_agent_params["proj_name"])
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
        last_opt = None
        same_opt_streak = 0
        optimizers_history = []
        rl_penalty = 0
        total_reward = 0.0

        print('\n############################################################################' +
        f'\nStarting trajectory {idx_traj + 1}/{rl_agent_params["n_trajectories"]} ' +
        'with a new initial point.')


        for t in itertools.count():

            # --- agent action ---
            action, action_raw, is_model = rl_agent.select_action(state)
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
            torch_opt = _build_torch_optimizer(action["type"], model.net.parameters(), action)


            model.compile(torch_opt, loss_weights=loss_weights)
            model.optimizer = torch_opt
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

            if np.isfinite(rmse) or np.isfinite(b_rmse):

                print(f"Operator RMSE: {rmse}, Boundary RMSE: {b_rmse}")

                env.solver_models = solver_models
                env.reward_params = {
                    "operator": {
                        "coeff": train_args["operator_coeff"] if np.isfinite(rmse) else 0.0,
                        "error": rmse if np.isfinite(rmse) else 0.0,
                    },
                    "bconds": {
                        "coeff": train_args["bnd_coeff"] if np.isfinite(b_rmse) else 0.0,
                        "error": b_rmse if np.isfinite(b_rmse) else 0.0,
                    },
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

                # prev_reward — теперь просто хранит reward_scalar из info
                prev_reward = info["reward_scalar"]

                # reward уже финальный (reward_model_i)
                rl_agent.push_memory((state, next_state, action_raw, float(reward_shaped.item()),
                                    done, float(reward_shaped.item()), info["opt_model_i"]))

                # update agent
                if len(rl_agent.replay_buffer) >= rl_agent_params["agent_min_buffer"]:
                    rl_agent.optim_(iters=rl_agent_params["agent_update_iters"])

                state = next_state
                total_reward += float(reward_shaped.item())
            else:
                done = -1
                next_state = state
                reward_shaped = torch.tensor(-10.0, device=device)
                info = {
                    "reward_scalar": 0.0,
                    "opt_model_i": -1,
                }

                print(f"Operator RMSE: {rmse}, Boundary RMSE: {b_rmse}. Stopping trajectory with done = -1.")

            try:
                # Сохраняем entry локально
                file_path = os.path.join(output_dir, f'transitions_{rl_agent.steps_done}.pt')

                entry = {
                            'state': state,
                            'next_state': next_state,
                            'action': action_raw,
                            'reward': float(info["reward_scalar"]),
                            'done': done, 
                            'reward_model_raw': float(reward_shaped.item()),
                            'reward_model': float(reward_shaped.item()),
                            'opt_model_i': info["opt_model_i"]
                        }
                torch.save(entry, file_path)

                # Логируем тот же файл в comet
                rl_agent_params['exp'].log_asset(
                    file_path,
                    file_name=f"entry_step_{rl_agent.steps_done}.pt",
                    step=rl_agent.steps_done,
                    overwrite=True
                )

            except Exception as e:
                print(e)

            print(f'\nCurrent reward after {action["type"]} optimizer: {info["reward_scalar"]}.\n'
                    f'Reward after taking prev reward and penalty: {reward_shaped}\n'
                    f'Total reward after using {", ".join(optimizers_history)} '
                    f'{"optimizers" if len(optimizers_history) > 1 else "optimizer"}: {total_reward}.\n'
                    f'\ndone = {done}')

            # callbacks.callbacks[1].save_every = self.t
            # env.render()

            if done == 1:
                break
            elif done == 0:
                if t == 10:
                    rl_penalty = -1 
            elif done == -1:
                rl_penalty = 0
                break

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
