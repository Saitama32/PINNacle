import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from typing import List, Union

from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
from landscape_visualization._aux.visualization_model import VisualizationModel
from landscape_visualization._aux.early_stopping_plot import EarlyStopping

# from tedeous.optimizers.optimizer import Optimizer
# from tedeous.callbacks.callback_list import CallbackList
from deepxde.callbacks import CallbackList


def compute_reward(reward_params, prev_reward, method="diff"):
    """
    Calculates the reward for the agent.

    Args:
        reward_params (dict): dictionary with operator and boundary error value and coefficients.
        prev_reward (float): previous value of reward.
        method (str): The method for calculating the reward (“diff” or “absolute”).
    Returns:
        float: The value of the reward.
    """
    current_reward = reward_params["operator"]["coeff"] * reward_params["operator"]["error"] + \
        reward_params["bconds"]["coeff"] * reward_params["bconds"]["error"]

    if method == "diff":
        return prev_reward - current_reward
    elif method == "absolute":
        return -current_reward
    else:
        raise ValueError("Invalid reward method. Use 'diff' or 'absolute'.")


class EnvRLOptimizer(gym.Env):
    def __init__(self,
                 optimizers: dict,
                 equation_params: list = None,
                 loss_surface_params: dict = None,
                 AE_model_params: dict = None,
                 AE_train_params: dict = None,
                 reward_method: str = "absolute",
                 callbacks: Union[CallbackList, List, None] = None,
                 n_save_models: int = None,
                 tolerance: float = 1e-2):
        super(EnvRLOptimizer, self).__init__()

        self.optimizers = optimizers
        self.solver_models = None
        self.reward_params = None
        self.rl_penalty = 0
        self.raw_states_dict = {}

        self.AE_model_params = AE_model_params
        self.AE_train_params = AE_train_params
        self.loss_surface_params = loss_surface_params
        self.equation_params = equation_params
        self.reward_method = reward_method
        self.callbacks = callbacks

        self.visualization_model = VisualizationModel(**self.AE_model_params)
        self.plot_loss_surface = None

        # Размерность нужно вытягивать из кода loss landscape, она будет постоянной,
        # т.к. action_dim - список оптимизаторов, он не меняется
        # state_dim - размерность поверхности, мы используем латентное 2D пространство, для генерации поверхности

        # Action - selecting an optimizer with its parameters
        # self.action_space = spaces.Discrete(len(self.optimizer_configs))
        self.action_space = {key: len(value) for key, value in optimizers.items()}

        # # State - loss surface (can be an array)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.visualization_model.latent_dim,
        #                                     dtype=np.float32)
        # observation_space = 3
        self.observation_space = self.visualization_model.latent_dim + 1

        self.current_reward = None
        self.reward_history = []
        self.tolerance = tolerance
        self.counter = 1
        self.n_save_models = n_save_models

        # reward shaping config (можешь вынести наружу)
        self.repeat_k = 3
        self.repeat_penalty = 0.5
        self.time_penalty = 0.05
        self.done_bonus = 10.0
        self.fail_penalty = -5.0

        # step context (будем заполнять из training loop)
        self._ctx = {}
        self._prev_state = None

    
    def set_step_context(self, *, prev_state, step_i, same_opt_streak,
                         is_model, rl_opt_step=None, prev_reward_scalar=None):
            self._prev_state = prev_state
            self._ctx = dict(
                step_i=step_i,
                same_opt_streak=same_opt_streak,
                is_model=is_model,
                rl_opt_step=rl_opt_step,
                prev_reward_scalar=prev_reward_scalar,
            )

    def reset(self):
        """Reset environment - load error surface, reset history to zero, select starting point."""
        self.current_reward = self.reward_history[-1]
        self.counter += 1

    def step(self):
        """Applying an action (optimizer selection) and updating the state."""

        finetune_AE_model = self.AE_train_params['finetune_AE_model']
        batch_size = self.AE_train_params['batch_size']
        every_epoch = self.AE_train_params['every_epoch']
        learning_rate = self.AE_train_params['learning_rate']
        resume = self.AE_train_params['resume']
        AE_params = self.AE_train_params[
            'other_RL_epoch_AE_params' if finetune_AE_model else 'first_RL_epoch_AE_params'
        ]

        epochs = AE_params['epochs']
        patience_scheduler = AE_params['patience_scheduler']
        cosine_scheduler_patience = AE_params['cosine_scheduler_patience']

        cb_es = EarlyStopping(patience=patience_scheduler)

        AEmodel = self.visualization_model.train(
            learning_rate, cosine_scheduler_patience, epochs, every_epoch, batch_size, resume,
            callbacks=[cb_es], solver_models=self.solver_models, finetune_AE_model=finetune_AE_model
        )

        self.loss_surface_params['solver_models'] = self.solver_models
        self.loss_surface_params['AE_model'] = AEmodel

        self.plot_loss_surface = PlotLossSurface(**self.loss_surface_params)
        self.plot_loss_surface.counter = self.counter

        # 1) next_state + базовый reward (как было)
        self.raw_states_dict = self.plot_loss_surface.save_equation_loss_surface(log_key=self.AE_train_params['log_key'])

        prev_reward_env = 0 if len(self.reward_history) == 0 else self.reward_history[-1]
        base_reward = compute_reward(self.reward_params, prev_reward_env, method=self.reward_method) + self.rl_penalty

        self.current_reward = base_reward
        self.reward_history.append(self.current_reward)
        self.reward_history = self.reward_history[-5:]

        success = abs(self.current_reward.item()) < self.tolerance
        if self.rl_penalty == -1:
            done = -1
        elif success:
            done = 1
        else:
            done = 0

        # 2) delta (как у тебя снаружи)
        if self._prev_state is not None and "loss_total" in self.raw_states_dict and "loss_total" in self._prev_state:
            raw_delta = self.raw_states_dict["loss_total"] - self._prev_state["loss_total"]
            delta = torch.sign(raw_delta) * torch.log1p(torch.abs(raw_delta))
            delta = delta / (delta.abs().max() + 1e-6)
            delta = delta.clamp(-1, 1)
            self.raw_states_dict["delta"] = delta

        # 3) reward_model_i (полная твоя логика)
        ctx = self._ctx
        reward_scalar = float(base_reward.item()) if hasattr(base_reward, "item") else float(base_reward)

        prev_reward_scalar = ctx.get("prev_reward_scalar", None)
        is_model = bool(ctx.get("is_model", False))
        step_i = int(ctx.get("step_i", 0))
        same_opt_streak = int(ctx.get("same_opt_streak", 0))
        rl_opt_step = ctx.get("rl_opt_step", None)

        opt_model_i = -1
        if prev_reward_scalar is None:
            reward_model_i = reward_scalar
        else:
            if is_model:
                opt_model_i = int(rl_opt_step) if rl_opt_step is not None else -1
            reward_model_i = reward_scalar - float(prev_reward_scalar)

        # repeat penalty
        if same_opt_streak > self.repeat_k:
            over = same_opt_streak - self.repeat_k
            reward_model_i -= self.repeat_penalty * over

        reward_model_i_raw = reward_model_i

        # time penalty
        reward_model_i -= self.time_penalty * step_i

        # done shaping
        if done == 1:
            reward_model_i += self.done_bonus
        elif done == -1:
            reward_model_i = self.fail_penalty

        info = {
            "reward_scalar": reward_scalar,
            "reward_model_raw": float(reward_model_i_raw),
            "reward_model": float(reward_model_i),
            "opt_model_i": int(opt_model_i),
        }

        # Возвращаем сразу shaped reward (как тебе нужно для replay buffer)
        return self.raw_states_dict, torch.tensor(reward_model_i), done, info

    def render(self):
        """Display the current error and convergence history."""

        self.reset()

        # print(f"Optimizer: {self.current_optimizer['name']}, Loss: {self.current_loss}")

        # Plotting PDE solution
        self.callbacks.on_epoch_end()
        self.callbacks.callbacks[1].save_every = 0.1

        # # Plotting loss landscape
        # if self.rl_penalty != -1:
        #     self.plot_loss_surface.plotting_equation_loss_surface(*self.equation_params)

    def close(self):
        plt.close('all')
