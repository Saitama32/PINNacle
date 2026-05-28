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


def compute_loss_ratio_reward(prev_loss, curr_loss, eps=1e-12, clip=5.0):
    if prev_loss is None:
        return 0.0

    prev_loss = float(prev_loss)
    curr_loss = float(curr_loss)
    reward = np.log(prev_loss + eps) - np.log(curr_loss + eps)
    return float(np.clip(reward, -clip, clip))


class EnvRLOptimizer(gym.Env):
    def __init__(self,
                 optimizers: dict,
                 equation_params: list = None,
                 loss_surface_params: dict = None,
                 AE_model_params: dict = None,
                 AE_train_params: dict = None,
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
        self.callbacks = callbacks

        self.visualization_model = VisualizationModel(**self.AE_model_params)
        self.plot_loss_surface = None

        # Action dimension is fixed by the optimizer dictionary.
        # State dimension comes from the latent loss surface representation.

        # Action - selecting an optimizer with its parameters
        # self.action_space = spaces.Discrete(len(self.optimizer_configs))
        self.action_space = {key: len(value) for key, value in optimizers.items()}

        # # State - loss surface (can be an array)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.visualization_model.latent_dim,
        #                                     dtype=np.float32)
        # observation_space = 3
        self.observation_space = self.visualization_model.latent_dim + 1

        self.tolerance = tolerance
        self.counter = 1
        self.n_save_models = n_save_models

        # Reward shaping config.
        self.repeat_k = 3
        self.repeat_penalty = 0.5
        self.time_penalty = 0.05
        self.done_bonus = 10.0
        self.fail_penalty = -5.0

        # Chain reward config.
        self.chain_reward_alpha = 0.2
        self.chain_reward_dense_clip = 5.0
        self.chain_success_bonus = 5.0
        self.chain_fail_penalty = -5.0

        # Step context filled by the training loop.
        self._ctx = {}
        self._prev_state = None

    def configure_chain_reward(
        self,
        *,
        alpha=None,
        dense_clip=None,
        success_bonus=None,
        fail_penalty=None,
    ):
        if alpha is not None:
            self.chain_reward_alpha = float(alpha)
        if dense_clip is not None:
            self.chain_reward_dense_clip = float(dense_clip)
        if success_bonus is not None:
            self.chain_success_bonus = float(success_bonus)
        if fail_penalty is not None:
            self.chain_fail_penalty = float(fail_penalty)

    def compute_chain_rewards(
        self,
        losses,
        done=0,
        eps=1e-12,
        alpha=None,
        dense_clip=None,
        success_bonus=None,
        fail_penalty=None,
    ):
        """
        Computes trajectory-level rewards from the full loss chain.

        The final loss score is distributed uniformly over all transitions.
        A weak dense shaping term rewards local loss improvements.
        Terminal success/failure is applied to the trajectory-level final score,
        not only to the last transition.
        """
        losses = np.asarray(losses, dtype=np.float64)
        T = len(losses)

        if T == 0:
            return []

        alpha = self.chain_reward_alpha if alpha is None else float(alpha)
        dense_clip = self.chain_reward_dense_clip if dense_clip is None else float(dense_clip)
        success_bonus = self.chain_success_bonus if success_bonus is None else float(success_bonus)
        fail_penalty = self.chain_fail_penalty if fail_penalty is None else float(fail_penalty)

        final_score = -np.log(losses[-1] + eps)

        if done == 1:
            final_score += success_bonus
        elif done == -1:
            final_score += fail_penalty

        weights = np.ones(T, dtype=np.float64)
        weights = weights / weights.sum()
        terminal_rewards = final_score * weights

        dense_rewards = np.zeros(T, dtype=np.float64)
        for t in range(1, T):
            dense = np.log(losses[t - 1] + eps) - np.log(losses[t] + eps)
            dense_rewards[t] = np.clip(dense, -dense_clip, dense_clip)

        rewards = terminal_rewards + alpha * dense_rewards
        return rewards.tolist()

    def compute_trajectory_rewards(self, transitions, losses):
        """
        transitions: list of dicts collected by rl_trainer.
        losses: list of scalar train losses for the same transitions.

        Returns:
            list[float]: reward for each transition.
        """
        if len(transitions) == 0:
            return []

        final_done = int(transitions[-1].get("done", 0))

        return self.compute_chain_rewards(
            losses=losses,
            done=final_done,
        )

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

        # 1) next_state and current raw loss.
        self.raw_states_dict = self.plot_loss_surface.save_equation_loss_surface(log_key=self.AE_train_params['log_key'])

        if "loss" in self.reward_params:
            reward_scalar = float(self.reward_params["loss"])
        else:
            reward_scalar = float(
                self.reward_params["operator"]["coeff"] * self.reward_params["operator"]["error"] +
                self.reward_params["bconds"]["coeff"] * self.reward_params["bconds"]["error"]
            )

        success = abs(reward_scalar) < self.tolerance

        if success:
            done = 1
        elif self.rl_penalty == -1:
            done = -1
        else:
            done = 0

        # 2) delta.
        if self._prev_state is not None and "loss_total" in self.raw_states_dict and "loss_total" in self._prev_state:
            raw_delta = self.raw_states_dict["loss_total"] - self._prev_state["loss_total"]
            delta = torch.sign(raw_delta) * torch.log1p(torch.abs(raw_delta))
            delta = delta / (delta.abs().max() + 1e-6)
            delta = delta.clamp(-1, 1)
            self.raw_states_dict["delta"] = delta

        # 3) loss-ratio reward.
        ctx = self._ctx

        prev_reward_scalar = ctx.get("prev_reward_scalar", None)
        is_model = bool(ctx.get("is_model", False))
        step_i = int(ctx.get("step_i", 0))
        same_opt_streak = int(ctx.get("same_opt_streak", 0))
        rl_opt_step = ctx.get("rl_opt_step", None)

        opt_model_i = -1

        if prev_reward_scalar is not None and is_model:
            opt_model_i = int(rl_opt_step) if rl_opt_step is not None else -1

        reward_model_i = compute_loss_ratio_reward(
            prev_loss=prev_reward_scalar,
            curr_loss=reward_scalar,
        )

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

        # Return shaped reward for the replay buffer.
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
