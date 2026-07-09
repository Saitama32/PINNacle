import json
import os
from dataclasses import dataclass

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


def build_refresh_schedule(iterations, refresh_count):
    if iterations <= 1:
        return []
    if refresh_count < 0:
        raise ValueError("refresh_count must be non-negative.")
    if refresh_count == 0:
        return []
    if refresh_count >= iterations:
        raise ValueError("refresh_count must be smaller than iterations.")
    steps = []
    for index in range(1, refresh_count + 1):
        step = int(round(index * iterations / (refresh_count + 1)))
        step = min(max(step, 1), iterations - 1)
        if not steps or steps[-1] != step:
            steps.append(step)
    return steps


def normalize_brightness(values, power=1.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    clipped = np.clip(values, a_min=1e-12, a_max=None)
    weighted = np.power(clipped, power)
    total = float(np.sum(weighted))
    if not np.isfinite(total) or total <= 0:
        return np.full_like(weighted, 1.0 / max(len(weighted), 1))
    return weighted / total


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
    refresh_count: int
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
        self.refresh_steps = build_refresh_schedule(config.iterations, config.refresh_count)
        self.joint_weight_update_optimizers = {"adam", "sgd", "rmsprop", "adamw", "soap"}

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

    def _compute_weighted_losses_tensor(self):
        self.model.net.auxiliary_vars = self.model.train_state.train_aux_vars
        try:
            _, weighted_losses = self.model.outputs_losses_train(
                self.model.train_state.X_train,
                self.model.train_state.y_train,
            )
        finally:
            self.model.net.auxiliary_vars = None
        return weighted_losses

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
        brightness_norm = normalize_brightness(brightness_raw)
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
        self.model._train_step(
            self.model.train_state.X_train,
            self.model.train_state.y_train,
            self.model.train_state.train_aux_vars,
        )

    def _train_joint_theta_weight_step(self):
        self.model.net.auxiliary_vars = self.model.train_state.train_aux_vars
        try:
            self.model.opt.zero_grad()
            self.weight_optimizer.zero_grad()
            _, weighted_losses = self.model.outputs_losses_train(
                self.model.train_state.X_train,
                self.model.train_state.y_train,
            )
            theta_loss = torch.sum(weighted_losses)
            theta_loss.backward()

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

            self.model.opt.step()
            if self.model.lr_scheduler is not None:
                if self.model.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.model.lr_scheduler.step(theta_loss.detach())
                else:
                    self.model.lr_scheduler.step()
            self.weight_optimizer.step()
        finally:
            self.model.net.auxiliary_vars = None

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
