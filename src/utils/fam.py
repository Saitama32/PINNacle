import json
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import deepxde as dde
from deepxde import display
from deepxde.callbacks import CallbackList


def split_fixed_movable_points(points, num_fixed, num_movable, rng=None, shuffle=True):
    points = np.asarray(points, dtype=np.float32)
    total = num_fixed + num_movable
    if total <= 0:
        raise ValueError("The total number of fixed and movable points must be positive.")
    if len(points) != total:
        raise ValueError(f"Expected {total} PDE points, got {len(points)}.")
    if shuffle:
        rng = np.random.default_rng() if rng is None else rng
        points = rng.permutation(points)
    return points[:num_fixed].copy(), points[num_fixed:].copy()


def build_refresh_schedule_every(iterations, refresh_every):
    if iterations <= 1:
        return []
    if refresh_every <= 0:
        raise ValueError("refresh_every must be positive.")
    if refresh_every >= iterations:
        raise ValueError("refresh_every must be smaller than iterations.")
    return list(range(refresh_every, iterations, refresh_every))


def normalize_brightness(values, power=1.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    clipped = np.clip(values, a_min=1e-12, a_max=None)
    weighted = np.power(clipped, power)
    total = float(np.sum(weighted))
    if not np.isfinite(total) or total <= 0:
        return np.full_like(weighted, 1.0 / max(len(weighted), 1))
    return weighted / total


def gaussian_time_window(times, mu, sigma, w0):
    times = np.asarray(times, dtype=np.float64)
    return float(w0) * np.exp(-((times - float(mu)) ** 2) / (2.0 * float(sigma) ** 2))


def famaw_weighted_loss(grouped_losses, c_params, group_names, base_weights=None):
    loss = torch.zeros((), dtype=c_params.dtype, device=c_params.device)
    for idx, name in enumerate(group_names):
        base_weight = 1.0 if base_weights is None else base_weights[name]
        loss = loss + base_weight * torch.exp(-c_params[idx]) * grouped_losses[name] + c_params[idx]
    return loss


def compute_gradient_brightness(model, points):
    first_param = next(model.net.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    inputs = torch.as_tensor(points, dtype=torch.float32, device=device)
    inputs.requires_grad_()
    outputs = model.net(inputs)
    grad_sq = torch.zeros(inputs.shape[0], dtype=outputs.dtype, device=outputs.device)
    for component in range(outputs.shape[1]):
        grad_component = torch.autograd.grad(
            outputs[:, component].sum(),
            inputs,
            retain_graph=component + 1 < outputs.shape[1],
            create_graph=False,
        )[0]
        grad_sq = grad_sq + torch.sum(grad_component * grad_component, dim=1)
    brightness = torch.sqrt(1.0 + grad_sq)
    return brightness.detach().cpu().numpy()


def _is_duplicate(candidate, points, index, atol=1e-7):
    if len(points) == 0:
        return False
    for point_index, point in enumerate(points):
        if point_index == index:
            continue
        if np.allclose(candidate, point, atol=atol, rtol=0):
            return True
    return False


def move_fireflies(points, brightness, lower, upper, alpha0, beta0, gamma, rng=None, max_attempts=128):
    points = np.asarray(points, dtype=np.float32).copy()
    brightness = np.asarray(brightness, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    search_range = upper - lower
    rng = np.random.default_rng() if rng is None else rng
    occupied = {tuple(np.round(point.astype(np.float64), 7)) for point in points}

    for i in range(len(points)):
        x_i = points[i].copy()
        x_i_key = tuple(np.round(x_i.astype(np.float64), 7))
        for j in range(len(points)):
            if brightness[i] >= brightness[j]:
                continue
            x_j = points[j]
            zeta = max(0.0, 1.0 - brightness[i] / max(brightness[j], 1e-12))
            if zeta <= 0:
                continue
            distance = np.linalg.norm(x_i - x_j)
            beta = zeta * beta0 * np.exp(-gamma * distance)
            alpha = alpha0 * min(zeta, 1.0)
            base_candidate = x_i + beta * (x_j - x_i)

            candidate = None
            for _ in range(max_attempts):
                random_step = alpha * (rng.random(size=x_i.shape) - 0.5) * search_range
                proposal = base_candidate + random_step
                if np.any(proposal < lower) or np.any(proposal > upper):
                    continue
                proposal_key = tuple(np.round(proposal.astype(np.float64), 7))
                if proposal_key in occupied and proposal_key != x_i_key:
                    continue
                candidate = proposal.astype(np.float32)
                break
            if candidate is not None:
                candidate_key = tuple(np.round(candidate.astype(np.float64), 7))
                occupied.discard(x_i_key)
                occupied.add(candidate_key)
                x_i = candidate
                x_i_key = candidate_key
        points[i] = x_i
    return points


def infer_loss_groups(loss_config):
    groups = {}
    for idx, config in enumerate(loss_config):
        loss_type = config.get("type", "")
        loss_name = str(config.get("name", "")).lower()
        if loss_type == "pde":
            group_name = "PDE"
        elif loss_type in {"ic", "initial"} or loss_name.startswith("ic"):
            group_name = "IC"
        elif loss_type == "boundary":
            group_name = "BC"
        else:
            group_name = loss_type.upper() if loss_type else f"LOSS_{idx}"
        groups.setdefault(group_name, []).append(idx)
    return groups


class LossWeightAdapter:
    def __init__(self, initial_weights):
        self._weights = np.asarray(initial_weights, dtype=np.float32).copy()

    def set(self, weights):
        self._weights = weights

    def get_numpy(self):
        if torch.is_tensor(self._weights):
            return self._weights.detach().cpu().numpy().astype(np.float32)
        return np.asarray(self._weights, dtype=np.float32)

    def __call__(self):
        return self._weights


@dataclass
class FAMTrainConfig:
    mode: str
    iterations: int
    refresh_every: int
    weight_lr: float
    alpha: float
    beta: float
    gamma: float
    num_fixed_points: int
    num_movable_points: int
    display_every: int
    save_model: bool
    save_diagnostics: bool
    save_point_plots: bool
    point_plot_every: int
    causal_window_enabled: bool = False
    causal_sigma: Optional[float] = None
    causal_w0: float = 1.0
    causal_threshold: float = 1.05
    causal_ema_beta: float = 0.9
    causal_required_success_checks: int = 3
    causal_log_brightness: bool = False
    pde_point_weighting_enabled: bool = False
    pde_point_weight_coeff: float = 1.0


class FAMTrainer:
    def __init__(
        self,
        model,
        config,
        loss_weight_adapter,
        callbacks=None,
        model_save_path=None,
        seed=None,
        static_loss_weights=None,
    ):
        if config.mode not in {"fam-w", "famaw-w"}:
            raise ValueError(f"Unsupported FAM mode: {config.mode}")
        self.model = model
        self.pde = model.pde
        self.data = model.data
        self.config = config
        self.loss_weight_adapter = loss_weight_adapter
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(model)
        self.model.model_save_path = model_save_path
        self.model.display_every = config.display_every
        self.model.batch_size = None
        self.model.stop_training = False
        self.model.callbacks = self.callbacks
        self.save_path = model_save_path
        self.rng = np.random.default_rng(seed)

        self.group_indices = infer_loss_groups(self.pde.loss_config)
        self.group_names = list(self.group_indices.keys())
        self.fixed_bc_points = None
        self.fixed_num_bcs = None
        self.fixed_points = None
        self.movable_points = None
        self.c_params = None
        self.weight_optimizer = None
        self.refresh_history = []
        self.current_loss_weights = np.ones(self.pde.num_loss, dtype=np.float32)
        self.static_loss_weights = (
            np.asarray(static_loss_weights, dtype=np.float32).copy()
            if static_loss_weights is not None
            else np.ones(self.pde.num_loss, dtype=np.float32)
        )
        self.refresh_steps = build_refresh_schedule_every(config.iterations, config.refresh_every)
        self.joint_weight_update_optimizers = {"adam", "sgd", "rmsprop", "adamw", "soap"}
        self.causal_window_enabled = bool(config.causal_window_enabled)
        self.causal_t_min, self.causal_t_max = self._time_bounds()
        time_span = self.causal_t_max - self.causal_t_min
        self.causal_sigma = (
            0.1 * time_span
            if config.causal_sigma is None
            else float(config.causal_sigma)
        )
        self.causal_w0 = float(config.causal_w0)
        self.causal_threshold = float(config.causal_threshold)
        self.causal_ema_beta = float(config.causal_ema_beta)
        self.causal_required_success_checks = int(config.causal_required_success_checks)
        self.causal_log_brightness = bool(config.causal_log_brightness)
        self.causal_mu = self.causal_t_min
        self.causal_brightness_ema = None
        self.causal_success_counter = 0
        self.causal_stage_index = 0
        self.causal_finished = False
        self.last_causal_state = None
        self.last_causal_move_state = None
        self.last_pde_point_brightness_norm = None
        self.last_pde_point_weight_state = None
        if self.causal_window_enabled:
            self._validate_causal_window_config()
        self._validate_pde_point_weighting_config()

    def _time_bounds(self):
        timedomain = getattr(self.data.geom, "timedomain", None)
        if timedomain is not None:
            return float(timedomain.t0), float(timedomain.t1)
        bbox = np.asarray(self.pde.bbox, dtype=np.float64)
        return float(bbox[-2]), float(bbox[-1])

    def _validate_causal_window_config(self):
        if self.config.mode not in {"fam-w", "famaw-w"}:
            raise ValueError("causal_window_enabled is only supported for fam-w or famaw-w mode.")
        if self.config.num_fixed_points <= 0:
            raise ValueError("causal_window_enabled requires at least one fixed point.")
        if self.causal_t_max <= self.causal_t_min:
            raise ValueError("Causal window requires a positive time span.")
        if self.causal_sigma <= 0:
            raise ValueError("causal_sigma must be positive.")
        if self.causal_w0 <= 0:
            raise ValueError("causal_w0 must be positive.")
        if not np.isfinite(self.causal_threshold):
            raise ValueError("causal_threshold must be finite.")
        if not 0 <= self.causal_ema_beta < 1:
            raise ValueError("causal_ema_beta must satisfy 0 <= beta < 1.")
        if self.causal_required_success_checks < 1:
            raise ValueError("causal_required_success_checks must be at least 1.")

    def _validate_pde_point_weighting_config(self):
        if not self.config.pde_point_weighting_enabled:
            return
        if self.config.mode not in {"fam-w", "famaw-w"}:
            raise ValueError("pde_point_weighting_enabled is only supported for fam-w or famaw-w mode.")
        if self.config.pde_point_weight_coeff < 0 or not np.isfinite(self.config.pde_point_weight_coeff):
            raise ValueError("pde_point_weight_coeff must be finite and non-negative.")

    def _causal_window_weights(self, points):
        times = np.asarray(points, dtype=np.float64)[:, -1]
        # Causal weighting only boosts the base brightness and never dims it.
        return np.maximum(
            gaussian_time_window(times, self.causal_mu, self.causal_sigma, self.causal_w0),
            0.1,
        )

    def _causal_check_interval(self):
        if self.causal_stage_index == 0:
            return self.causal_t_min, min(self.causal_t_min + self.causal_sigma, self.causal_t_max)
        return (
            max(self.causal_t_min, self.causal_mu - self.causal_sigma),
            min(self.causal_t_max, self.causal_mu + self.causal_sigma),
        )

    def _causal_check_points(self, left, right):
        times = self.fixed_points[:, -1]
        mask = (times >= left) & (times <= right)
        if np.any(mask):
            return self.fixed_points[mask], False
        return self.fixed_points, True

    def _causal_count_movable_points(self, left, right):
        times = self.movable_points[:, -1]
        return int(np.count_nonzero((times >= left) & (times <= right)))

    def _build_causal_move_state(self, brightness_raw, window_weights, brightness_for_move, brightness_norm):
        left, right = self._causal_check_interval()
        times = np.asarray(self.movable_points[:, -1], dtype=np.float64)
        interval_mask = (times >= left) & (times <= right)
        brightest_index = int(np.argmax(brightness_for_move))
        nearest_mu_index = int(np.argmin(np.abs(times - self.causal_mu)))
        return {
            "causal_window_weight_min": float(np.min(window_weights)),
            "causal_window_weight_mean": float(np.mean(window_weights)),
            "causal_window_weight_max": float(np.max(window_weights)),
            "causal_move_brightness_min": float(np.min(brightness_for_move)),
            "causal_move_brightness_mean": float(np.mean(brightness_for_move)),
            "causal_move_brightness_max": float(np.max(brightness_for_move)),
            "causal_move_norm_min": float(np.min(brightness_norm)),
            "causal_move_norm_mean": float(np.mean(brightness_norm)),
            "causal_move_norm_max": float(np.max(brightness_norm)),
            "causal_move_mass_in_interval": float(np.sum(brightness_norm[interval_mask])),
            "causal_move_points_in_interval": int(np.count_nonzero(interval_mask)),
            "causal_move_brightest_time": float(times[brightest_index]),
            "causal_move_brightest_weight": float(window_weights[brightest_index]),
            "causal_move_brightest_raw": float(brightness_raw[brightest_index]),
            "causal_move_brightest_total": float(brightness_for_move[brightest_index]),
            "causal_move_nearest_mu_time": float(times[nearest_mu_index]),
            "causal_move_nearest_mu_weight": float(window_weights[nearest_mu_index]),
            "causal_move_nearest_mu_raw": float(brightness_raw[nearest_mu_index]),
            "causal_move_nearest_mu_total": float(brightness_for_move[nearest_mu_index]),
            "causal_move_weighted_time_mean": float(np.sum(times * brightness_norm)),
        }

    def _log_causal_brightness_state(self, step, state):
        step_text = "?" if step is None else str(int(step))
        ema = state["causal_brightness_ema"]
        ema_text = "None" if ema is None else f"{ema:.6g}"
        print(
            "[FAMAW causal brightness] "
            f"step={step_text} "
            f"stage={state['causal_stage_index']} "
            f"mu={state['causal_mu']:.6g} "
            f"interval=[{state['causal_check_left']:.6g}, {state['causal_check_right']:.6g}] "
            f"points={state['causal_check_points']} "
            f"movable_points={state['causal_movable_points_in_interval']} "
            f"mean={state['causal_mean_brightness']:.6g} "
            f"ema={ema_text} "
            f"threshold={self.causal_threshold:.6g} "
            f"success={state['causal_success_counter']}/{self.causal_required_success_checks} "
            f"shifted={state['causal_shifted']} "
            f"finished={state['causal_finished']} "
            f"fallback={state['causal_used_fallback']}"
        )
        if "causal_move_brightness_max" in state:
            print(
                "[FAMAW causal move] "
                f"step={step_text} "
                f"mu={state['causal_mu']:.6g} "
                f"weight[min/mean/max]={state['causal_window_weight_min']:.6g}/{state['causal_window_weight_mean']:.6g}/{state['causal_window_weight_max']:.6g} "
                f"total_brightness[min/mean/max]={state['causal_move_brightness_min']:.6g}/{state['causal_move_brightness_mean']:.6g}/{state['causal_move_brightness_max']:.6g} "
                f"norm[min/max]={state['causal_move_norm_min']:.6g}/{state['causal_move_norm_max']:.6g} "
                f"mass_in_interval={state['causal_move_mass_in_interval']:.6g} "
                f"movable_in_interval={state['causal_move_points_in_interval']} "
                f"brightest_time={state['causal_move_brightest_time']:.6g} "
                f"brightest_weight={state['causal_move_brightest_weight']:.6g} "
                f"nearest_mu_time={state['causal_move_nearest_mu_time']:.6g} "
                f"nearest_mu_weight={state['causal_move_nearest_mu_weight']:.6g} "
                f"weighted_time_mean={state['causal_move_weighted_time_mean']:.6g}"
            )

    def _log_pde_point_weight_state(self, step, state):
        step_text = "?" if step is None else str(int(step))
        print(
            "[FAM PDE point weighting] "
            f"step={step_text} "
            f"coeff={state['pde_point_weight_coeff']:.6g} "
            f"weight[min/mean/max]={state['pde_point_weight_min']:.6g}/"
            f"{state['pde_point_weight_mean']:.6g}/{state['pde_point_weight_max']:.6g}"
        )

    def _update_causal_window(self, step=None):
        if not self.causal_window_enabled:
            self.last_causal_state = None
            return None

        left, right = self._causal_check_interval()
        check_points, used_fallback = self._causal_check_points(left, right)
        movable_points_in_interval = self._causal_count_movable_points(left, right)
        check_brightness = compute_gradient_brightness(self.model, check_points)
        mean_brightness = float(np.mean(check_brightness))
        if self.causal_brightness_ema is None:
            self.causal_brightness_ema = mean_brightness
        else:
            self.causal_brightness_ema = (
                self.causal_ema_beta * self.causal_brightness_ema
                + (1.0 - self.causal_ema_beta) * mean_brightness
            )
        ema_for_record = float(self.causal_brightness_ema)

        shifted = False
        if not self.causal_finished:
            if self.causal_brightness_ema < self.causal_threshold:
                self.causal_success_counter += 1
            else:
                self.causal_success_counter = 0

            if self.causal_success_counter >= self.causal_required_success_checks:
                self.causal_mu = min(self.causal_mu + self.causal_sigma, self.causal_t_max)
                self.causal_success_counter = 0
                self.causal_brightness_ema = None
                self.causal_stage_index += 1
                shifted = True
                if self.causal_mu >= self.causal_t_max or self.causal_mu + self.causal_sigma >= self.causal_t_max:
                    self.causal_mu = self.causal_t_max
                    self.causal_finished = True

        self.last_causal_state = {
            "causal_mu": float(self.causal_mu),
            "causal_sigma": float(self.causal_sigma),
            "causal_w0": float(self.causal_w0),
            "causal_check_left": float(left),
            "causal_check_right": float(right),
            "causal_check_points": int(len(check_points)),
            "causal_movable_points_in_interval": movable_points_in_interval,
            "causal_mean_brightness": mean_brightness,
            "causal_brightness_ema": ema_for_record,
            "causal_success_counter": int(self.causal_success_counter),
            "causal_stage_index": int(self.causal_stage_index),
            "causal_shifted": bool(shifted),
            "causal_finished": bool(self.causal_finished),
            "causal_used_fallback": bool(used_fallback),
        }
        if self.last_causal_move_state is not None:
            self.last_causal_state.update(self.last_causal_move_state)
        if self.causal_log_brightness:
            self._log_causal_brightness_state(step, self.last_causal_state)
        return self.last_causal_state

    def _sample_initial_pde_points(self):
        total = self.config.num_fixed_points + self.config.num_movable_points
        points = self.data.geom.random_points(total, random=self.data.train_distribution)
        self.fixed_points, self.movable_points = split_fixed_movable_points(
            points,
            self.config.num_fixed_points,
            self.config.num_movable_points,
            rng=self.rng,
            shuffle=True,
        )

    def _initialize_bc_points(self):
        self.fixed_bc_points = self.data.train_x_bc.copy()
        self.fixed_num_bcs = list(self.data.num_bcs)

    def _rebuild_train_data(self):
        pde_points = np.vstack((self.fixed_points, self.movable_points)).astype(np.float32)
        self.data.train_x_all = pde_points
        self.data.train_x_bc = self.fixed_bc_points.copy()
        self.data.num_bcs = list(self.fixed_num_bcs)
        self.data.train_x = np.vstack((self.data.train_x_bc, self.data.train_x_all))
        self.data.train_y = self.data.soln(self.data.train_x) if self.data.soln else None
        if self.data.auxiliary_var_fn is not None:
            self.data.train_aux_vars = self.data.auxiliary_var_fn(self.data.train_x).astype(np.float32)
        else:
            self.data.train_aux_vars = None
        self.data.test_x = None
        self.data.test_y = None
        self.data.test_aux_vars = None
        self.model.train_state.set_data_train(*self.data.train_next_batch(None))
        self.model.train_state.set_data_test(*self.data.test())

    def _get_group_losses(self, losses):
        group_losses = {}
        for label, indices in self.group_indices.items():
            group_losses[label] = torch.sum(losses[indices])
        return group_losses

    def _current_group_weight_values(self):
        base_group_weights = {
            name: float(self.static_loss_weights[indices[0]])
            for name, indices in self.group_indices.items()
        }
        if self.config.mode == "fam-w":
            return base_group_weights
        return {
            name: base_group_weights[name] * float(torch.exp(-self.c_params[idx]).detach().cpu().item())
            for idx, name in enumerate(self.group_names)
        }

    def _compute_current_loss_weights(self):
        weights = np.ones(self.pde.num_loss, dtype=np.float32)
        group_weights = self._current_group_weight_values()
        for label, indices in self.group_indices.items():
            for idx in indices:
                weights[idx] = group_weights[label]
        return weights

    def _sync_loss_weights(self):
        self.current_loss_weights = self._compute_current_loss_weights()
        self.loss_weight_adapter.set(self.current_loss_weights.copy())
        self.model.losshistory.set_loss_weights(self.current_loss_weights.copy())

    def _current_movable_pde_point_weights(self):
        if not self.config.pde_point_weighting_enabled:
            return None
        if self.last_pde_point_brightness_norm is None:
            return np.ones(self.config.num_movable_points, dtype=np.float32)
        brightness_norm = np.asarray(self.last_pde_point_brightness_norm, dtype=np.float32).reshape(-1)
        if len(brightness_norm) != len(self.movable_points):
            raise ValueError(
                f"Expected {len(self.movable_points)} movable PDE point weights, got {len(brightness_norm)}."
            )
        return 1.0 + float(self.config.pde_point_weight_coeff) * brightness_norm

    def _compute_pde_point_weight_state(self):
        point_weights = self._current_movable_pde_point_weights()
        if point_weights is None:
            self.last_pde_point_weight_state = None
            return None
        self.last_pde_point_weight_state = {
            "pde_point_weight_coeff": float(self.config.pde_point_weight_coeff),
            "pde_point_weight_min": float(np.min(point_weights)),
            "pde_point_weight_mean": float(np.mean(point_weights)),
            "pde_point_weight_max": float(np.max(point_weights)),
        }
        return self.last_pde_point_weight_state

    def _compute_pointwise_pde_losses(self):
        if self.pde.num_pde <= 0:
            return []
        first_param = next(self.model.net.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")

        def residual_squares(points):
            if len(points) == 0:
                return []
            inputs = torch.as_tensor(points, dtype=torch.float32, device=device)
            inputs.requires_grad_()
            outputs = self.model.net(inputs)
            residuals = self.pde.pde(inputs, outputs)
            if torch.is_tensor(residuals):
                residuals = [residuals]
            terms = []
            for residual in residuals:
                residual_flat = residual.reshape(residual.shape[0], -1)
                terms.append(torch.mean(torch.square(residual_flat), dim=1))
            return terms

        fixed_terms = residual_squares(self.fixed_points)
        movable_terms = residual_squares(self.movable_points)
        point_weights_np = self._current_movable_pde_point_weights()
        if point_weights_np is None:
            point_weights = None
        else:
            point_weights = torch.as_tensor(point_weights_np, dtype=torch.float32, device=device)
        losses = []
        num_fixed = len(self.fixed_points)
        num_movable = len(self.movable_points)
        total_points = num_fixed + num_movable
        if total_points <= 0:
            return [torch.zeros((), dtype=torch.float32, device=device) for _ in range(self.pde.num_pde)]
        for idx in range(self.pde.num_pde):
            fixed_term = fixed_terms[idx] if idx < len(fixed_terms) else None
            movable_term = movable_terms[idx] if idx < len(movable_terms) else None
            fixed_sum = torch.sum(fixed_term) if fixed_term is not None else torch.zeros((), dtype=torch.float32, device=device)
            if movable_term is None:
                movable_weighted_sum = torch.zeros((), dtype=torch.float32, device=device)
            elif point_weights is None:
                movable_weighted_sum = torch.sum(movable_term)
            else:
                movable_mean = torch.sum(point_weights * movable_term) / torch.clamp(torch.sum(point_weights), min=1e-12)
                movable_weighted_sum = movable_mean * float(num_movable)
            losses.append((fixed_sum + movable_weighted_sum) / float(total_points))
        return losses

    def _apply_pointwise_pde_weighting(self, weighted_losses):
        if not self.config.pde_point_weighting_enabled or self.pde.num_pde <= 0:
            return weighted_losses
        pde_losses = self._compute_pointwise_pde_losses()
        group_weights = torch.as_tensor(
            self.current_loss_weights[: self.pde.num_pde],
            dtype=weighted_losses.dtype,
            device=weighted_losses.device,
        )
        replaced = []
        for idx in range(len(weighted_losses)):
            if idx < self.pde.num_pde:
                replaced.append(group_weights[idx] * pde_losses[idx].to(weighted_losses.dtype))
            else:
                replaced.append(weighted_losses[idx])
        return torch.stack(replaced)

    def _compute_weighted_losses_tensor(self):
        self.model.net.auxiliary_vars = self.model.train_state.train_aux_vars
        try:
            _, weighted_losses = self.model.outputs_losses_train(
                self.model.train_state.X_train,
                self.model.train_state.y_train,
            )
        finally:
            self.model.net.auxiliary_vars = None
        return self._apply_pointwise_pde_weighting(weighted_losses)

    def _theta_closure(self, skip_backward=False):
        weighted_losses = self._compute_weighted_losses_tensor()
        theta_loss = torch.sum(weighted_losses)
        if not skip_backward:
            self.model.opt.zero_grad()
            theta_loss.backward()
        return theta_loss, weighted_losses

    def _step_theta_optimizer(self):
        self.model.net.auxiliary_vars = self.model.train_state.train_aux_vars
        try:
            def closure(*, skip_backward=False):
                theta_loss, weighted_losses = self._theta_closure(skip_backward=skip_backward)
                self.model.opt.losses = weighted_losses
                return theta_loss

            loss = self.model.opt.step(closure)
            if not torch.is_tensor(loss):
                loss, _ = self._theta_closure(skip_backward=True)
            if self.model.lr_scheduler is not None:
                if self.model.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.model.lr_scheduler.step(loss.detach())
                else:
                    self.model.lr_scheduler.step()
            return loss
        finally:
            self.model.net.auxiliary_vars = None

    def _compute_unweighted_losses_for_weights(self):
        weighted_losses = self._compute_weighted_losses_tensor()
        weights = torch.as_tensor(
            self.current_loss_weights,
            dtype=weighted_losses.dtype,
            device=weighted_losses.device,
        )
        return weighted_losses.detach() / torch.clamp(weights.detach(), min=1e-12)

    def _train_weight_step(self):
        self.weight_optimizer.zero_grad()
        unweighted_losses = self._compute_unweighted_losses_for_weights()
        grouped_losses = self._get_group_losses(unweighted_losses)
        base_weights = {
            name: float(self.static_loss_weights[indices[0]])
            for name, indices in self.group_indices.items()
        }
        weight_loss = famaw_weighted_loss(grouped_losses, self.c_params, self.group_names, base_weights=base_weights)
        weight_loss.backward()
        self.weight_optimizer.step()

    def _supports_joint_theta_weight_step(self):
        return (
            self.config.mode == "famaw-w"
            and self.weight_optimizer is not None
            and self.model.opt_name in self.joint_weight_update_optimizers
        )

    def _move_points(self):
        brightness_raw = compute_gradient_brightness(self.model, self.movable_points)
        brightness_for_move = brightness_raw
        window_weights = None
        self.last_causal_move_state = None
        if self.causal_window_enabled:
            window_weights = self._causal_window_weights(self.movable_points)
            brightness_for_move = brightness_raw * window_weights
        brightness_norm = normalize_brightness(brightness_for_move)
        if self.causal_window_enabled:
            self.last_causal_move_state = self._build_causal_move_state(
                brightness_raw,
                window_weights,
                brightness_for_move,
                brightness_norm,
            )
        if self.config.pde_point_weighting_enabled:
            self.last_pde_point_brightness_norm = brightness_norm.astype(np.float32).copy()
            state = self._compute_pde_point_weight_state()
            if state is not None:
                self._log_pde_point_weight_state(self.model.train_state.step, state)
        bbox = np.asarray(self.pde.bbox, dtype=np.float32)
        lower = bbox[::2]
        upper = bbox[1::2]
        before = self.movable_points.copy()
        self.movable_points = move_fireflies(
            self.movable_points,
            brightness_norm,
            lower,
            upper,
            alpha0=self.config.alpha,
            beta0=self.config.beta,
            gamma=self.config.gamma,
            rng=self.rng,
        )
        return before, self.movable_points.copy(), brightness_raw, brightness_norm

    def _plot_point_sets(self, path, title=None):
        if self.fixed_points is None or self.movable_points is None:
            return
        plt.figure(figsize=(7.2, 5.2))
        if isinstance(self.data.geom, dde.geometry.GeometryXTime) and self.fixed_points.shape[1] == 2:
            fixed_x = self.fixed_points[:, 1]
            fixed_y = self.fixed_points[:, 0]
            movable_x = self.movable_points[:, 1]
            movable_y = self.movable_points[:, 0]
            xlabel = "t"
            ylabel = "x"
        else:
            fixed_x = self.fixed_points[:, 0]
            fixed_y = self.fixed_points[:, 1]
            movable_x = self.movable_points[:, 0]
            movable_y = self.movable_points[:, 1]
            xlabel = "x0"
            ylabel = "x1"
        plt.scatter(fixed_x, fixed_y, c="royalblue", marker="x", s=16, linewidths=0.9, label=r"$D_{fn}$")
        plt.scatter(movable_x, movable_y, c="red", marker="x", s=20, linewidths=1.0, label=r"$D_{fm}$")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    def _resolve_point_plot_dir(self, step):
        if self.save_path is None:
            return None
        if self.config.point_plot_every is None or self.config.point_plot_every <= 0:
            return self.save_path
        bucket = int(np.ceil(step / self.config.point_plot_every) * self.config.point_plot_every)
        target_dir = os.path.join(self.save_path, str(bucket))
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def _record_refresh_state(self, step, brightness_raw, brightness_norm, moved_before, moved_after):
        record = {
            "step": int(step),
            "weights": self._current_group_weight_values(),
            "brightness_mean": float(np.mean(brightness_raw)),
            "brightness_max": float(np.max(brightness_raw)),
            "moved_distance_mean": float(np.mean(np.linalg.norm(moved_after - moved_before, axis=1))),
        }
        if self.last_pde_point_weight_state is not None:
            record.update(self.last_pde_point_weight_state)
        if self.last_causal_move_state is not None:
            record.update(self.last_causal_move_state)
        if self.last_causal_state is not None:
            record.update(self.last_causal_state)
        self.refresh_history.append(record)
        if self.save_path is None:
            return
        refresh_index = len(self.refresh_history) - 1
        if self.config.save_diagnostics:
            np.save(os.path.join(self.save_path, f"fam_points_before_{refresh_index:03d}.npy"), moved_before)
            np.save(os.path.join(self.save_path, f"fam_points_after_{refresh_index:03d}.npy"), moved_after)
            np.save(os.path.join(self.save_path, f"fam_brightness_raw_{refresh_index:03d}.npy"), brightness_raw)
            np.save(os.path.join(self.save_path, f"fam_brightness_norm_{refresh_index:03d}.npy"), brightness_norm)
        if self.config.save_point_plots:
            target_dir = self._resolve_point_plot_dir(int(step))
            self._plot_point_sets(
                os.path.join(target_dir, f"fam_points_step_{int(step):06d}.png"),
                title=f"Collocation points at step {int(step)}",
            )

    def _write_history(self):
        if self.save_path is None:
            return
        if self.config.save_diagnostics:
            with open(os.path.join(self.save_path, "fam_history.json"), "w", encoding="utf-8") as file_obj:
                json.dump(self.refresh_history, file_obj, indent=2)
        if self.config.save_point_plots:
            target_dir = self._resolve_point_plot_dir(int(self.model.train_state.step))
            self._plot_point_sets(
                os.path.join(target_dir, "fam_points_final.png"),
                title="Final collocation point distribution",
            )

    def _train_theta_step(self):
        if not self.config.pde_point_weighting_enabled:
            self.model._train_step(
                self.model.train_state.X_train,
                self.model.train_state.y_train,
                self.model.train_state.train_aux_vars,
            )
            return
        self._step_theta_optimizer()

    def _train_joint_theta_weight_step(self):
        self.weight_optimizer.zero_grad()
        self._step_theta_optimizer()
        weighted_losses = self._compute_weighted_losses_tensor()
        weights = torch.as_tensor(
            self.current_loss_weights,
            dtype=weighted_losses.dtype,
            device=weighted_losses.device,
        )
        unweighted_losses = weighted_losses.detach() / torch.clamp(weights, min=1e-12)
        grouped_losses = self._get_group_losses(unweighted_losses)
        base_weights = {
            name: float(self.static_loss_weights[indices[0]])
            for name, indices in self.group_indices.items()
        }
        weight_loss = famaw_weighted_loss(grouped_losses, self.c_params, self.group_names, base_weights=base_weights)
        weight_loss.backward()
        self.weight_optimizer.step()

    def train(self):
        self._sample_initial_pde_points()
        self._initialize_bc_points()
        self._rebuild_train_data()

        if self.config.mode == "famaw-w":
            param = next(self.model.net.parameters(), None)
            device = param.device if param is not None else torch.device("cpu")
            self.c_params = torch.nn.Parameter(torch.zeros(len(self.group_names), dtype=torch.float32, device=device))
            self.weight_optimizer = torch.optim.Adam([self.c_params], lr=self.config.weight_lr)

        self._sync_loss_weights()
        self.model.stop_training = False
        self.model._test()
        self.callbacks.on_train_begin()

        refresh_steps = set(self.refresh_steps)
        for iteration_index in range(self.config.iterations):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.model.train_state.set_data_train(*self.data.train_next_batch(self.model.batch_size))
            next_step = self.model.train_state.step + 1
            update_weights_now = self.config.mode == "famaw-w" and next_step in refresh_steps

            if self._supports_joint_theta_weight_step() and update_weights_now:
                self._train_joint_theta_weight_step()
                self._sync_loss_weights()
            else:
                if update_weights_now:
                    self._train_weight_step()
                    self._sync_loss_weights()
                self._train_theta_step()
            self.model.train_state.epoch += 1
            self.model.train_state.step += 1

            if (
                self.model.train_state.step % self.config.display_every == 0
                or iteration_index + 1 == self.config.iterations
            ):
                self.model._test()

            if self.model.train_state.step in refresh_steps:
                moved_before, moved_after, brightness_raw, brightness_norm = self._move_points()
                self._update_causal_window(self.model.train_state.step)
                self._record_refresh_state(
                    self.model.train_state.step,
                    brightness_raw,
                    brightness_norm,
                    moved_before,
                    moved_after,
                )
                self._rebuild_train_data()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.model.stop_training:
                break

        self.callbacks.on_train_end()
        self._write_history()
        print("")
        display.training_display.summary(self.model.train_state)
        if self.save_path is not None and self.config.save_model:
            self.model.save(self.save_path, verbose=1)
        return self.model.losshistory, self.model.train_state
