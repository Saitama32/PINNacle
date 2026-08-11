import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import deepxde as dde
import numpy as np
import torch
from scipy import interpolate, stats

from .optimizer_adapter import MaskedOptimizerAdapter
from .weight_groups import WeightGroupCollection


@dataclass
class DynamicFreezingConfig:
    enabled: bool = False
    group_size: int = 256
    freeze_fraction: float = 0.25
    good_tolerance: float = 1e-3
    freeze_events: int = 3
    freeze_interval_steps: int = 2000
    diagnostic_nt: int = 16
    diagnostic_nx: int = 64
    relative_eps: float = 1e-12
    nullspace_enabled: bool = True
    nullspace_max_points: int = 256
    nullspace_damping: float = 1e-6
    responsibility_enabled: bool = True
    responsibility_nt: int = 16
    responsibility_nx: int = 64
    log_every: int = 100
    seed: int = 12345
    ic_weight: float = 100.0

    def validate(self):
        if self.group_size <= 0:
            raise ValueError("weight group size must be positive")
        if not 0 <= self.freeze_fraction < 1:
            raise ValueError("freeze fraction must satisfy 0 <= value < 1")
        if self.good_tolerance <= 0:
            raise ValueError("good tolerance must be positive")
        if self.freeze_events <= 0 or self.freeze_interval_steps <= 0:
            raise ValueError("freeze event count and interval must be positive")
        for value, name in (
            (self.diagnostic_nt, "diagnostic_nt"),
            (self.diagnostic_nx, "diagnostic_nx"),
            (self.responsibility_nt, "responsibility_nt"),
            (self.responsibility_nx, "responsibility_nx"),
            (self.nullspace_max_points, "nullspace_max_points"),
            (self.log_every, "log_every"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


class DynamicFreezingController(dde.callbacks.Callback):
    """KS-specific event controller; training remains in the existing trainer."""

    def __init__(self, model, config, log_dir):
        super().__init__()
        config.validate()
        self.model = model
        self.config = config
        self.log_dir = os.path.abspath(log_dir)
        self.groups = WeightGroupCollection(model.net, config.group_size)
        self.adapter = None
        self.event_count = 0
        self.first_event_step = None
        self.event_steps = []
        self.mask_history = {group.group_id: [] for group in self.groups.groups}
        self.training_rows = []
        self.temporal_rows = []
        self._last_logged_step = -1
        self._build_grids()

    @property
    def device(self):
        return next(self.model.net.parameters()).device

    @property
    def dtype(self):
        return next(self.model.net.parameters()).dtype

    def _build_grids(self):
        bbox = np.asarray(self.model.pde.bbox, dtype=np.float64)
        if bbox.shape != (4,):
            raise ValueError("Dynamic freezing currently expects KS bbox [x_min, x_max, t_min, t_max]")
        self.x_values = np.linspace(bbox[0], bbox[1], self.config.diagnostic_nx, endpoint=False)
        self.t_values = np.linspace(bbox[2], bbox[3], self.config.diagnostic_nt)
        xx, tt = np.meshgrid(self.x_values, self.t_values)
        self.grid_np = np.column_stack([xx.reshape(-1), tt.reshape(-1)]).astype(np.float32)
        self.interior_np = self.grid_np[self.grid_np[:, 1] > bbox[2] + 1e-12]
        self.ic_np = np.column_stack(
            [self.x_values, np.full_like(self.x_values, bbox[2])]
        ).astype(np.float32)
        self._reference_nearest = None
        reference = getattr(self.model.pde, "ref_data", None)
        if reference is not None:
            reference = np.asarray(reference)
            valid = reference.ndim == 2 and reference.shape[1] >= 3
            if valid:
                finite = np.isfinite(reference[:, :3]).all(axis=1)
                if np.any(finite):
                    self._reference_nearest = interpolate.NearestNDInterpolator(
                        reference[finite, :2], reference[finite, 2:]
                    )

    def _tensor(self, array, requires_grad=False):
        return torch.as_tensor(array, dtype=self.dtype, device=self.device).clone().detach().requires_grad_(requires_grad)

    @staticmethod
    def _ic_target(points):
        x = points[:, 0:1]
        return torch.cos(x) * (1 + torch.sin(x))

    def _ic_loss(self, keep_graph=False):
        points = self._tensor(self.ic_np, requires_grad=keep_graph)
        loss = torch.mean((self.model.net(points) - self._ic_target(points)) ** 2)
        return loss if keep_graph else loss.detach()

    def _residual(self, points_np, keep_graph=False):
        if len(points_np) == 0:
            return torch.empty(0, dtype=self.dtype, device=self.device)
        points = self._tensor(points_np, requires_grad=True)
        output = self.model.net(points)
        residual = self.model.pde.pde(points, output)
        if isinstance(residual, (list, tuple)):
            residual = residual[0]
        if residual.ndim == 1:
            residual = residual[:, None]
        if not keep_graph:
            residual = residual.detach()
        dde.grad.clear()
        return residual

    def _region(self, event_id):
        residual = self._residual(self.interior_np)
        squared = residual.square().reshape(-1)
        if event_id == 1:
            good_mask = torch.zeros_like(squared, dtype=torch.bool)
        else:
            good_mask = squared < self.config.good_tolerance
        bad_mask = ~good_mask
        return good_mask.cpu().numpy(), bad_mask.cpu().numpy(), squared.cpu().numpy()

    def _evaluate_losses(self, good_mask, bad_mask):
        ic_loss = self._ic_loss()
        residual_sq = self._residual(self.interior_np).square().reshape(-1)
        good_tensor = torch.as_tensor(good_mask, dtype=torch.bool, device=self.device)
        bad_tensor = torch.as_tensor(bad_mask, dtype=torch.bool, device=self.device)
        good_pde = residual_sq[good_tensor].mean() if good_tensor.any() else torch.zeros((), device=self.device, dtype=self.dtype)
        bad = residual_sq[bad_tensor].mean() if bad_tensor.any() else torch.zeros((), device=self.device, dtype=self.dtype)
        good = ic_loss + good_pde
        return {
            "ic": float(ic_loss.cpu()),
            "good_pde": float(good_pde.cpu()),
            "good": float(good.cpu()),
            "bad": float(bad.cpu()),
            "pde": float(residual_sq.mean().cpu()) if residual_sq.numel() else 0.0,
        }

    def should_trigger_before_step(self):
        if not self.config.enabled or self.event_count >= self.config.freeze_events:
            return False
        step = int(getattr(self.model.train_state, "step", 0)) + 1
        if self.event_count == 0:
            return float(self._ic_loss().cpu()) < self.config.good_tolerance
        return step >= self.first_event_step + self.event_count * self.config.freeze_interval_steps

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "dynamic_freezing_config.json"), "w", encoding="utf-8") as file_obj:
            json.dump(asdict(self.config), file_obj, indent=2, sort_keys=True)
        self._write_csv("weight_groups.csv", self.groups.metadata())
        if self.config.enabled:
            self.adapter = MaskedOptimizerAdapter(self.model.net, self.model.opt, self.groups, self).install()

    def on_epoch_end(self):
        step = int(self.model.train_state.step)
        if step == self._last_logged_step or step % self.config.log_every:
            return
        self._last_logged_step = step
        self._log_training(step)

    def on_train_end(self):
        final_step = int(getattr(self.model.train_state, "step", 0))
        if final_step != self._last_logged_step:
            self._last_logged_step = final_step
            self._log_training(final_step)
        if self.adapter is not None:
            self.adapter.uninstall()
        self._write_csv("training.csv", self.training_rows)
        self._write_csv("temporal_profiles.csv", self.temporal_rows)
        summary = {
            "enabled": self.config.enabled,
            "events_completed": self.event_count,
            "event_steps": self.event_steps,
            "first_event_reached": self.first_event_step is not None,
            "num_groups": len(self.groups),
            "num_frozen_groups": len(self.groups.frozen_groups),
        }
        if self.config.enabled and self.event_count == 0:
            summary["no_event_reason"] = "IC loss did not reach good_tolerance before training ended"
        with open(os.path.join(self.log_dir, "summary.json"), "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2)
        self._plot_training()

    def run_event(self, proposal):
        event_id = self.event_count + 1
        step = int(getattr(self.model.train_state, "step", 0)) + 1
        if event_id == 1:
            self.first_event_step = step
        good_mask, bad_mask, _ = self._region(event_id)
        base = self._evaluate_losses(good_mask, bad_mask)
        was_frozen = {group.group_id: group.is_frozen for group in self.groups.groups}
        rows = []
        eps = self.config.relative_eps
        for group in self.groups.groups:
            full_delta = proposal.get(group.parameter)
            if full_delta is None:
                delta = torch.zeros(group.num_weights, dtype=self.dtype, device=self.device)
            else:
                delta = full_delta.reshape(-1)[group.flat_start : group.flat_end].detach().clone()
            before = group.values().clone()
            self.groups.apply_group_delta(group, delta)
            changed = self._evaluate_losses(good_mask, bad_mask)
            with torch.no_grad():
                group.parameter.reshape(-1)[group.flat_start : group.flat_end].copy_(before)
            d_ic = (changed["ic"] - base["ic"]) / (base["ic"] + eps)
            d_good_pde = (changed["good_pde"] - base["good_pde"]) / (base["good_pde"] + eps)
            d_good = (changed["good"] - base["good"]) / (base["good"] + eps)
            d_bad = (changed["bad"] - base["bad"]) / (base["bad"] + eps)
            protected = abs(d_good)
            utility = max(0.0, -d_bad)
            score = utility / (protected + eps)
            rows.append({
                "event_id": event_id,
                "step": step,
                "group_id": group.group_id,
                "layer_name": group.layer_name,
                "parameter_name": group.parameter_name,
                "flat_start": group.flat_start,
                "flat_end": group.flat_end,
                "num_weights": group.num_weights,
                "is_frozen_before": was_frozen[group.group_id],
                "update_norm": float(torch.linalg.vector_norm(delta).cpu()),
                "parameter_norm": float(torch.linalg.vector_norm(before).cpu()),
                "loss_good_before": base["good"],
                "loss_bad_before": base["bad"],
                "delta_good_abs": changed["good"] - base["good"],
                "delta_bad_abs": changed["bad"] - base["bad"],
                "d_ic_rel": d_ic,
                "d_good_pde_rel": d_good_pde,
                "d_good_rel": d_good,
                "d_bad_rel": d_bad,
                "P": protected,
                "U": utility,
                "S": score,
                "force_train": d_good < 0 and d_bad < 0,
                "_delta": delta,
            })

        null_metrics = self._nullspace_metrics(good_mask, proposal)
        for row in rows:
            row.update(null_metrics.get(row["group_id"], {"jg_delta_norm": math.nan, "null_ratio": math.nan}))

        eligible = [row for row in rows if not row["force_train"]]
        eligible.sort(
            key=lambda row: (
                round(row["S"], 12) if np.isfinite(row["S"]) else math.inf,
                -row["P"],
                row["group_id"],
            )
        )
        freeze_count = min(int(round(self.config.freeze_fraction * len(rows))), len(eligible))
        selected = {row["group_id"] for row in eligible[:freeze_count]}
        self.groups.set_frozen(selected)
        for row in rows:
            row["selected_for_freeze"] = row["group_id"] in selected
            row["is_frozen_after"] = row["selected_for_freeze"]
            row.pop("_delta")
        self.event_count = event_id
        self.event_steps.append(step)
        for group in self.groups.groups:
            self.mask_history[group.group_id].append("freeze" if group.is_frozen else "train")
        self._write_csv(f"event_{event_id:02d}_groups.csv", rows)
        self._write_mask_history()
        self._write_correlations(event_id, rows)
        self._save_temporal_profile(step, event_id)
        if self.config.responsibility_enabled:
            self._save_responsibility(event_id)
        self._plot_event(event_id, rows)
        print(
            f"Dynamic freezing event {event_id}/{self.config.freeze_events} at step {step}: "
            f"frozen {len(selected)}/{len(rows)} groups."
        )

    def _selected_constraint_points(self, good_mask):
        max_points = self.config.nullspace_max_points
        ic_count = min(len(self.ic_np), max_points)
        ic_idx = np.linspace(0, len(self.ic_np) - 1, ic_count, dtype=int)
        remaining = max_points - ic_count
        good_idx = np.flatnonzero(good_mask)
        if remaining <= 0:
            good_idx = np.empty(0, dtype=int)
        elif len(good_idx) > remaining:
            good_idx = good_idx[np.linspace(0, len(good_idx) - 1, remaining, dtype=int)]
        return self.ic_np[ic_idx], self.interior_np[good_idx]

    def _nullspace_metrics(self, good_mask, proposal):
        if not self.config.nullspace_enabled:
            return {}
        try:
            ic_points_np, pde_points_np = self._selected_constraint_points(good_mask)
            ic_points = self._tensor(ic_points_np, requires_grad=True)
            constraints = [(self.model.net(ic_points) - self._ic_target(ic_points)).reshape(-1)]
            if len(pde_points_np):
                constraints.append(self._residual(pde_points_np, keep_graph=True).reshape(-1))
            constraint_vector = torch.cat(constraints)
            params = [parameter for _, parameter, _ in self.groups.weight_parameters]
            rows = []
            for index in range(constraint_vector.numel()):
                grads = torch.autograd.grad(
                    constraint_vector[index], params, retain_graph=True, allow_unused=True
                )
                rows.append(torch.cat([
                    torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1)
                    for parameter, grad in zip(params, grads)
                ]).detach())
            jacobian = torch.stack(rows).to(dtype=torch.float64)
            gram = jacobian @ jacobian.T
            gram.diagonal().add_(self.config.nullspace_damping)
            metrics = {}
            for group in self.groups.groups:
                full_delta = proposal.get(group.parameter)
                delta = (
                    torch.zeros(group.num_weights, device=self.device, dtype=self.dtype)
                    if full_delta is None
                    else full_delta.reshape(-1)[group.flat_start : group.flat_end]
                ).detach().to(torch.float64)
                norm = torch.linalg.vector_norm(delta)
                columns = slice(group.weight_offset, group.weight_offset + group.num_weights)
                j_group = jacobian[:, columns]
                j_delta = j_group @ delta
                j_norm = torch.linalg.vector_norm(j_delta)
                if norm <= self.config.relative_eps:
                    null_ratio = 0.0
                else:
                    solved = torch.linalg.solve(gram, j_delta)
                    full_delta_vector = torch.zeros(
                        self.groups.num_weights,
                        dtype=torch.float64,
                        device=jacobian.device,
                    )
                    full_delta_vector[columns] = delta
                    projection = full_delta_vector - jacobian.T @ solved
                    null_ratio = float(
                        (torch.linalg.vector_norm(projection) / (norm + self.config.relative_eps)).cpu()
                    )
                metrics[group.group_id] = {
                    "jg_delta_norm": float((j_norm / (norm + self.config.relative_eps)).cpu()),
                    "null_ratio": null_ratio,
                }
            return metrics
        except (RuntimeError, torch.linalg.LinAlgError) as error:
            print(f"Warning: null-space diagnostics failed: {error}")
            return {}
        finally:
            dde.grad.clear()

    def responsibility(self, points_np):
        params = [parameter for _, parameter, _ in self.groups.weight_parameters]
        result = np.zeros((len(points_np), len(self.groups)), dtype=np.float64)
        for point_index, point in enumerate(points_np):
            point_tensor = self._tensor(point[None, :], requires_grad=True)
            output = self.model.net(point_tensor).reshape(-1)[0]
            grads = torch.autograd.grad(output, params, allow_unused=True)
            for group in self.groups.groups:
                parameter_index = next(i for i, (_, parameter, _) in enumerate(self.groups.weight_parameters) if parameter is group.parameter)
                grad = grads[parameter_index]
                if grad is not None:
                    part = grad.reshape(-1)[group.flat_start : group.flat_end]
                    result[point_index, group.group_id] = float(torch.sum(part.square()).detach().cpu())
        denominator = result.sum(axis=1, keepdims=True)
        zero = denominator[:, 0] <= self.config.relative_eps
        result[~zero] /= denominator[~zero]
        if np.any(zero):
            result[zero] = 1.0 / len(self.groups)
        return result

    def _save_responsibility(self, event_id):
        bbox = np.asarray(self.model.pde.bbox, dtype=np.float64)
        x = np.linspace(bbox[0], bbox[1], self.config.responsibility_nx, endpoint=False)
        t = np.linspace(bbox[2], bbox[3], self.config.responsibility_nt)
        xx, tt = np.meshgrid(x, t)
        points = np.column_stack([xx.reshape(-1), tt.reshape(-1)]).astype(np.float32)
        responsibility = self.responsibility(points).reshape(len(t), len(x), len(self.groups))
        dominant = np.argmax(responsibility, axis=-1)
        if len(self.groups) == 1:
            entropy = np.zeros((len(t), len(x)))
        else:
            entropy = -np.sum(
                responsibility * np.log(responsibility + self.config.relative_eps), axis=-1
            ) / np.log(len(self.groups))
        np.savez_compressed(
            os.path.join(self.log_dir, f"event_{event_id:02d}_responsibility.npz"),
            responsibility=responsibility,
            dominant_group=dominant,
            entropy=entropy,
            entropy_by_time=entropy.mean(axis=1),
            mean_responsibility_by_time=responsibility.mean(axis=1),
            mean_responsibility_entropy=float(entropy.mean()),
            time=t,
            x=x,
        )

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            dominant_image = axes[0].imshow(
                dominant, origin="lower", aspect="auto", extent=[x[0], x[-1], t[0], t[-1]]
            )
            axes[0].set_title("Dominant weight group")
            fig.colorbar(dominant_image, ax=axes[0])
            entropy_image = axes[1].imshow(
                entropy, origin="lower", aspect="auto", extent=[x[0], x[-1], t[0], t[-1]], vmin=0, vmax=1
            )
            axes[1].set_title("Responsibility entropy")
            fig.colorbar(entropy_image, ax=axes[1])
            for axis in axes:
                axis.set_xlabel("x")
                axis.set_ylabel("t")
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, f"event_{event_id:02d}_responsibility.png"), dpi=150)
            plt.close(fig)
        except Exception as error:
            print(f"Warning: could not plot responsibility maps: {error}")

    def _temporal_profile(self):
        residual_sq = self._residual(self.grid_np).square().reshape(self.config.diagnostic_nt, self.config.diagnostic_nx).cpu().numpy()
        relative_l2 = np.full(self.config.diagnostic_nt, np.nan, dtype=float)
        if self._reference_nearest is not None:
            with torch.no_grad():
                prediction = self.model.net(self._tensor(self.grid_np)).detach().cpu().numpy()
            reference = np.asarray(self._reference_nearest(self.grid_np))
            prediction = prediction.reshape(self.config.diagnostic_nt, self.config.diagnostic_nx, -1)
            reference = reference.reshape(self.config.diagnostic_nt, self.config.diagnostic_nx, -1)
            numerator = np.linalg.norm(prediction - reference, axis=(1, 2))
            denominator = np.linalg.norm(reference, axis=(1, 2)) + self.config.relative_eps
            relative_l2 = numerator / denominator
        return (
            residual_sq.mean(axis=1),
            (residual_sq < self.config.good_tolerance).mean(axis=1),
            relative_l2,
        )

    def _save_temporal_profile(self, step, event_id=None):
        mean_residual, fraction_good, relative_l2 = self._temporal_profile()
        for time_value, residual_value, fraction, l2_value in zip(
            self.t_values, mean_residual, fraction_good, relative_l2
        ):
            self.temporal_rows.append({
                "step": step,
                "event_id": "" if event_id is None else event_id,
                "time": time_value,
                "mean_pde_residual_sq": residual_value,
                "fraction_good_points": fraction,
                "relative_l2": l2_value,
            })

    def _log_training(self, step):
        good_mask, bad_mask, squared = self._region(max(2, self.event_count + 1))
        losses = self._evaluate_losses(good_mask, bad_mask)
        self.training_rows.append({
            "step": step,
            "total_loss": self.config.ic_weight * losses["ic"] + losses["pde"],
            "ic_loss": losses["ic"],
            "pde_loss": losses["pde"],
            "current_num_good_points": int(np.sum(squared < self.config.good_tolerance)),
            "current_fraction_good": float(np.mean(squared < self.config.good_tolerance)),
            "num_frozen_groups": len(self.groups.frozen_groups),
            "num_trainable_groups": len(self.groups.trainable_groups),
            "global_relative_l2": self._global_relative_l2(),
        })
        self._save_temporal_profile(step)
        self._write_csv("training.csv", self.training_rows)
        self._write_csv("temporal_profiles.csv", self.temporal_rows)

    def _global_relative_l2(self):
        if self._reference_nearest is None:
            return math.nan
        with torch.no_grad():
            prediction = self.model.net(self._tensor(self.grid_np)).detach().cpu().numpy()
        reference = np.asarray(self._reference_nearest(self.grid_np))
        return float(
            np.linalg.norm(prediction - reference)
            / (np.linalg.norm(reference) + self.config.relative_eps)
        )

    def _write_mask_history(self):
        rows = []
        for group_id, values in self.mask_history.items():
            row = {"group_id": group_id}
            row.update({f"event{index + 1}": value for index, value in enumerate(values)})
            rows.append(row)
        self._write_csv("mask_history.csv", rows)

    def _write_correlations(self, event_id, rows):
        protected = np.asarray([row["P"] for row in rows], dtype=float)
        jg = np.asarray([row["jg_delta_norm"] for row in rows], dtype=float)
        null_complement = 1 - np.asarray([row["null_ratio"] for row in rows], dtype=float)
        payload = {"event_id": event_id}
        for name, values in (("jg_delta_norm", jg), ("one_minus_null_ratio", null_complement)):
            valid = np.isfinite(protected) & np.isfinite(values)
            if valid.sum() >= 2 and np.std(protected[valid]) > 0 and np.std(values[valid]) > 0:
                payload[f"pearson_P_vs_{name}"] = float(stats.pearsonr(protected[valid], values[valid]).statistic)
                payload[f"spearman_P_vs_{name}"] = float(stats.spearmanr(protected[valid], values[valid]).statistic)
            else:
                payload[f"pearson_P_vs_{name}"] = math.nan
                payload[f"spearman_P_vs_{name}"] = math.nan
        with open(os.path.join(self.log_dir, f"event_{event_id:02d}_correlations.json"), "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)

    def _write_csv(self, filename, rows):
        if not rows:
            return
        path = os.path.join(self.log_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _plot_event(self, event_id, rows):
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            ids = [row["group_id"] for row in rows]
            fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
            for axis, field in zip(axes, ("S", "d_good_rel", "d_bad_rel", "null_ratio")):
                colors = ["tab:red" if row["selected_for_freeze"] else "tab:blue" for row in rows]
                axis.scatter(ids, [row[field] for row in rows], c=colors, s=10)
                axis.set_ylabel(field)
                axis.grid(alpha=0.2)
            axes[-1].set_xlabel("group_id")
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, f"event_{event_id:02d}_group_scores.png"), dpi=150)
            plt.close(fig)
        except Exception as error:
            print(f"Warning: could not plot dynamic-freezing event: {error}")

    def _plot_training(self):
        if not self.training_rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            steps = [row["step"] for row in self.training_rows]
            fig, axis = plt.subplots(figsize=(9, 5))
            for field in ("total_loss", "ic_loss", "pde_loss"):
                axis.semilogy(steps, [max(row[field], self.config.relative_eps) for row in self.training_rows], label=field)
            for step in self.event_steps:
                axis.axvline(step, color="black", alpha=0.2)
            axis.legend()
            axis.set_xlabel("step")
            axis.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, "loss_curves.png"), dpi=150)
            plt.close(fig)

            if self.temporal_rows:
                snapshots = {}
                for row in self.temporal_rows:
                    key = (row["step"], row["event_id"])
                    snapshots.setdefault(key, {})[row["time"]] = row
                snapshot_keys = sorted(snapshots, key=lambda key: (key[0], str(key[1])))
                residual = np.asarray([
                    [snapshots[key][time]["mean_pde_residual_sq"] for time in self.t_values]
                    for key in snapshot_keys
                ])
                good = np.asarray([
                    [snapshots[key][time]["fraction_good_points"] for time in self.t_values]
                    for key in snapshot_keys
                ])
                relative_l2 = np.asarray([
                    [snapshots[key][time]["relative_l2"] for time in self.t_values]
                    for key in snapshot_keys
                ])
                x_positions = np.arange(len(snapshot_keys))
                fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
                axes[0].imshow(
                    np.log10(residual.T + self.config.relative_eps),
                    origin="lower",
                    aspect="auto",
                    extent=[-0.5, len(snapshot_keys) - 0.5, self.t_values[0], self.t_values[-1]],
                )
                axes[0].set_ylabel("t")
                axes[0].set_title("log10 temporal PDE residual")
                axes[1].imshow(
                    good.T,
                    origin="lower",
                    aspect="auto",
                    vmin=0,
                    vmax=1,
                    extent=[-0.5, len(snapshot_keys) - 0.5, self.t_values[0], self.t_values[-1]],
                )
                axes[1].set_ylabel("t")
                axes[1].set_title("Good-region fraction")
                axes[2].imshow(
                    relative_l2.T,
                    origin="lower",
                    aspect="auto",
                    extent=[-0.5, len(snapshot_keys) - 0.5, self.t_values[0], self.t_values[-1]],
                )
                axes[2].set_ylabel("t")
                axes[2].set_xlabel("step")
                axes[2].set_xticks(x_positions)
                axes[2].set_xticklabels(
                    [f"{step}" if event == "" else f"{step}/E{event}" for step, event in snapshot_keys],
                    rotation=45,
                    ha="right",
                )
                axes[2].set_title("Temporal relative L2 error")
                fig.tight_layout()
                fig.savefig(os.path.join(self.log_dir, "temporal_diagnostics.png"), dpi=150)
                plt.close(fig)
        except Exception as error:
            print(f"Warning: could not plot dynamic-freezing training logs: {error}")
