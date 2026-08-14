import csv
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass

import deepxde as dde
import numpy as np
import torch
from scipy import interpolate, stats

from src.losses.causal import (
    causal_loss_with_fixed_weights,
    causal_residual_loss,
    temporal_chunk_losses,
)

from .optimizer_adapter import MaskedOptimizerAdapter
from .weight_groups import WeightGroupCollection


@dataclass
class DynamicFreezingConfig:
    enabled: bool = False
    group_size: int = 256
    max_freeze_fraction: float = 0.25
    good_tolerance: float = 1e-3
    protected_pde_tolerance: float = 5e-2
    freeze_events: int = 3
    max_freeze_refresh_steps: int = 2000
    causal_protect_weight: float = 0.999
    causal_unprotect_weight: float = 0.995
    causal_front_patience: int = 100
    transfer_boundary_enabled: bool = True
    transfer_boundary_weight_threshold: float = 0.9
    transfer_boundary_drift_threshold: float = 0.01
    transfer_boundary_patience: int = 5
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
        if not 0 <= self.max_freeze_fraction <= 1:
            raise ValueError("max freeze fraction must satisfy 0 <= value <= 1")
        if self.good_tolerance <= 0:
            raise ValueError("good tolerance must be positive")
        if self.protected_pde_tolerance <= 0:
            raise ValueError("protected PDE tolerance must be positive")
        if self.freeze_events <= 0 or self.max_freeze_refresh_steps <= 0:
            raise ValueError("freeze event count and max refresh steps must be positive")
        if not 0 < self.causal_unprotect_weight <= self.causal_protect_weight <= 1:
            raise ValueError(
                "causal weights must satisfy 0 < unprotect weight <= protect weight <= 1"
            )
        if self.causal_front_patience <= 0:
            raise ValueError("causal front patience must be positive")
        if not 0 < self.transfer_boundary_weight_threshold <= 1:
            raise ValueError("transfer boundary weight threshold must satisfy 0 < value <= 1")
        if self.transfer_boundary_drift_threshold < 0:
            raise ValueError("transfer boundary drift threshold must be non-negative")
        if self.transfer_boundary_patience <= 0:
            raise ValueError("transfer boundary patience must be positive")
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
        self.model.dynamic_freezing_relative_eps = config.relative_eps
        self.log_dir = os.path.abspath(log_dir)
        self.groups = WeightGroupCollection(model.net, config.group_size)
        self.adapter = None
        self.event_count = 0
        self.first_event_step = None
        self.event_steps = []
        self.mask_history = {group.group_id: [] for group in self.groups.groups}
        self.training_rows = []
        self.temporal_rows = []
        self.previous_boundary_states = None
        self.previous_boundary_chunk_losses = None
        self.previous_boundary_step = None
        self.boundary_drift_history = {}
        self.transfer_ready_counts = {}
        self._warned_missing_ks_cache = False
        self._last_logged_step = -1
        self.last_event_step = None
        self.candidate_front = -1
        self.weight_enter_front = -1
        self.pde_gated_enter_front = -1
        self.pending_front = None
        self.pending_count = 0
        self.committed_front = -1
        self.last_event_front = None
        self._pending_event_reason = None
        self._build_grids()

    @property
    def device(self):
        return next(self.model.net.parameters()).device

    @property
    def dtype(self):
        return next(self.model.net.parameters()).dtype

    @property
    def causal_enabled(self):
        options = getattr(self.model, "causal_loss_options", None)
        return bool(options and options.get("enabled", False))

    @property
    def causal_options(self):
        return getattr(self.model, "causal_loss_options", {}) or {}

    @staticmethod
    def _front_at_threshold(details, threshold):
        if not details:
            return -1
        post_weights = details["post_chunk_weights"].detach().reshape(-1)
        front = -1
        for index, weight in enumerate(post_weights):
            if float(weight.cpu()) < threshold:
                break
            front = index
        return front

    def _cached_protected_front(self):
        details = getattr(self.model, "causal_loss_details", None)
        return self._front_at_threshold(details, self.config.causal_protect_weight)

    def _pde_gated_front(self, details, weight_front):
        if not details or weight_front < 0:
            return -1
        chunk_losses = details["chunk_losses"].detach().reshape(-1)
        prefix_means = torch.cumsum(chunk_losses, dim=0) / torch.arange(
            1,
            chunk_losses.numel() + 1,
            dtype=chunk_losses.dtype,
            device=chunk_losses.device,
        )
        tolerance = torch.as_tensor(
            self.config.protected_pde_tolerance,
            dtype=chunk_losses.dtype,
            device=chunk_losses.device,
        )
        front = -1
        for index in range(min(weight_front + 1, prefix_means.numel())):
            if bool(prefix_means[index] > tolerance) or bool(
                chunk_losses[index] > tolerance
            ):
                break
            front = index
        return front

    def _update_causal_front(self):
        details = getattr(self.model, "causal_loss_details", None)
        self.weight_enter_front = self._front_at_threshold(
            details, self.config.causal_protect_weight
        )
        self.pde_gated_enter_front = self._pde_gated_front(
            details, self.weight_enter_front
        )
        enter_front = self.pde_gated_enter_front
        exit_front = self._front_at_threshold(details, self.config.causal_unprotect_weight)
        if enter_front > self.committed_front:
            candidate = enter_front
        elif exit_front < self.committed_front:
            candidate = exit_front
        else:
            candidate = self.committed_front
        self.candidate_front = candidate
        if candidate == self.committed_front:
            self.pending_front = None
            self.pending_count = 0
            return False
        if candidate == self.pending_front:
            self.pending_count += 1
        else:
            self.pending_front = candidate
            self.pending_count = 1
        if self.pending_count < self.config.causal_front_patience:
            return False
        self.committed_front = candidate
        self.pending_front = None
        self.pending_count = 0
        return True

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

    def _causal_target(self, residual=None, fixed_weights=None, return_details=False):
        options = self.causal_options
        time_index = int(options.get("time_index", -1))
        try:
            time_values = self.interior_np[:, time_index]
        except IndexError as error:
            raise ValueError(
                f"causal time_index={time_index} is invalid for diagnostic points "
                f"with {self.interior_np.shape[1]} coordinates"
            ) from error
        if residual is None:
            residual = self._residual(self.interior_np)
        num_chunks = int(options.get("num_chunks", 16))
        if fixed_weights is not None:
            if return_details:
                raise ValueError("return_details is not supported with fixed causal weights")
            return causal_loss_with_fixed_weights(
                residual=residual,
                t=self._tensor(time_values),
                num_chunks=num_chunks,
                fixed_weights=fixed_weights,
            )
        return causal_residual_loss(
            residual=residual,
            t=self._tensor(time_values),
            num_chunks=num_chunks,
            tol=float(options.get("tol", 0.1)),
            include_ic_in_weights=bool(options.get("include_ic_in_weights", False)),
            ic_loss=self._ic_loss(),
            ic_weight_in_causal=float(options.get("ic_weight_in_causal", 0.0)),
            return_details=return_details,
        )

    def _causal_chunk_losses(self, residual):
        time_index = int(self.causal_options.get("time_index", -1))
        time_values = self._tensor(self.interior_np[:, time_index])
        return temporal_chunk_losses(
            residual,
            time_values,
            int(self.causal_options.get("num_chunks", 16)),
        )

    def _evaluate_losses(
        self,
        good_mask,
        bad_mask,
        causal_weights=None,
        return_causal_details=False,
        protected_front_chunk=-1,
    ):
        ic_loss = self._ic_loss()
        residual = self._residual(self.interior_np)
        residual_sq = residual.square().reshape(-1)
        if self.causal_enabled:
            if causal_weights is None:
                causal_result = self._causal_target(
                    residual=residual,
                    return_details=return_causal_details,
                )
                if return_causal_details:
                    causal_loss, _, causal_details = causal_result
                else:
                    causal_loss, _ = causal_result
                    causal_details = None
            else:
                causal_loss = self._causal_target(
                    residual=residual,
                    fixed_weights=causal_weights,
                )
                causal_details = None
            if protected_front_chunk >= 0:
                chunk_losses = (
                    causal_details["chunk_losses"]
                    if causal_details is not None
                    else self._causal_chunk_losses(residual)
                )
                protected_pde = chunk_losses[: protected_front_chunk + 1].mean()
            else:
                protected_pde = torch.zeros((), dtype=self.dtype, device=self.device)
            protected_loss = ic_loss + protected_pde
            result = {
                "ic": float(ic_loss.cpu()),
                "good_pde": float(protected_pde.detach().cpu()),
                "good": float(protected_loss.detach().cpu()),
                "bad": float(causal_loss.detach().cpu()),
                "pde": float(residual_sq.mean().cpu()) if residual_sq.numel() else 0.0,
            }
            if causal_details is not None:
                result["causal_details"] = causal_details
            return result

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
        if not self.config.enabled:
            return False
        step = int(getattr(self.model.train_state, "step", 0)) + 1
        if self.causal_enabled:
            if self.event_count == 0:
                if float(self._ic_loss().cpu()) >= self.config.good_tolerance:
                    return False
                details = getattr(self.model, "causal_loss_details", None)
                self.weight_enter_front = self._front_at_threshold(
                    details, self.config.causal_protect_weight
                )
                self.pde_gated_enter_front = self._pde_gated_front(
                    details, self.weight_enter_front
                )
                self.candidate_front = self.pde_gated_enter_front
                self.committed_front = -1
                self.pending_front = None
                self.pending_count = 0
                self._pending_event_reason = "initial_ic_ready"
                return True
            front_changed = self._update_causal_front()
            if front_changed and self.committed_front != self.last_event_front:
                self._pending_event_reason = "front_changed"
                return True
            if step - self.last_event_step >= self.config.max_freeze_refresh_steps:
                self._pending_event_reason = "max_refresh"
                return True
            return False
        if self.event_count >= self.config.freeze_events:
            return False
        if self.event_count == 0:
            return float(self._ic_loss().cpu()) < self.config.good_tolerance
        return step >= self.first_event_step + self.event_count * self.config.max_freeze_refresh_steps

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        for filename in (
            "dynamic_freezing_events.csv",
            "dynamic_freezing_groups.csv",
            "dynamic_freezing_chunks.csv",
            "boundary_diagnostics.csv",
            "transfer_boundary_history.csv",
        ):
            open(os.path.join(self.log_dir, filename), "w", encoding="utf-8").close()
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
        if self.causal_enabled:
            self._log_boundary_diagnostics(step)

    def on_train_end(self):
        final_step = int(getattr(self.model.train_state, "step", 0))
        if final_step != self._last_logged_step:
            self._last_logged_step = final_step
            self._log_training(final_step)
        if self.config.responsibility_enabled and self.causal_enabled:
            self._save_responsibility("final")
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
        event_reason = self._pending_event_reason or "legacy_schedule"
        protected_front = self.committed_front if self.causal_enabled else -1
        base = self._evaluate_losses(
            good_mask,
            bad_mask,
            return_causal_details=self.causal_enabled,
            protected_front_chunk=protected_front,
        )
        causal_weights = (
            base["causal_details"]["weights"] if self.causal_enabled else None
        )
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
            changed = self._evaluate_losses(
                good_mask,
                bad_mask,
                causal_weights=causal_weights,
                protected_front_chunk=protected_front,
            )
            with torch.no_grad():
                group.parameter.reshape(-1)[group.flat_start : group.flat_end].copy_(before)
            d_ic = (changed["ic"] - base["ic"]) / (base["ic"] + eps)
            d_good_pde = (
                math.nan
                if self.causal_enabled
                else (changed["good_pde"] - base["good_pde"])
                / (base["good_pde"] + eps)
            )
            d_good = (changed["good"] - base["good"]) / (base["good"] + eps)
            d_bad = (changed["bad"] - base["bad"]) / (base["bad"] + eps)
            protected = abs(d_good)
            utility = max(0.0, -d_bad)
            score = utility / (protected + eps)
            if self.causal_enabled:
                protection = self._causal_protection_metrics(
                    base, changed, protected_front
                )
                violates = protection["violates_ic_tolerance"] or protection[
                    "violates_protected_pde_tolerance"
                ]
                worsens = protection["worsens_ic"] or protection[
                    "worsens_protected_pde"
                ]
                protected_risk = protection["protected_risk"]
                incremental_worsening = protection["incremental_worsening"]
                risky_group = protection["risky"]
            else:
                violates = changed["good"] > self.config.good_tolerance
                worsens = changed["good"] > base["good"] + eps
                protected_risk = max(
                    0.0,
                    (changed["good"] - self.config.good_tolerance)
                    / self.config.good_tolerance,
                )
                incremental_worsening = max(
                    0.0,
                    (changed["good"] - base["good"])
                    / (max(base["good"], self.config.good_tolerance) + eps),
                )
                risky_group = violates and worsens
                protection = {}
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
                "loss_good_base": base["good"],
                "loss_good_after": changed["good"],
                "loss_bad_before": base["bad"],
                "protected_tolerance": (
                    math.nan if self.causal_enabled else self.config.good_tolerance
                ),
                "protected_risk": protected_risk,
                "incremental_worsening": incremental_worsening,
                "violates_protected_tolerance": violates,
                "worsens_protected": worsens,
                "risky": risky_group,
                "loss_ic_base": base["ic"],
                "loss_ic_after": changed["ic"],
                "ic_tolerance": self.config.good_tolerance,
                "loss_protected_pde_base": base["good_pde"],
                "loss_protected_pde_after": changed["good_pde"],
                "protected_pde_tolerance": self.config.protected_pde_tolerance,
                "delta_good_abs": changed["good"] - base["good"],
                "delta_bad_abs": changed["bad"] - base["bad"],
                "d_ic_rel": d_ic,
                "d_good_pde_rel": d_good_pde,
                "d_good_rel": d_good,
                "d_bad_rel": d_bad,
                "d_causal_fixed": d_bad if self.causal_enabled else math.nan,
                "P": protected,
                "U": utility,
                "S": score,
                "force_train": d_good < 0 and d_bad < 0,
                "_delta": delta,
                **protection,
            })

        protected_mask = (
            self._protected_point_mask(protected_front)
            if self.causal_enabled
            else good_mask
        )
        null_metrics = self._nullspace_metrics(protected_mask, proposal)
        for row in rows:
            row.update(null_metrics.get(row["group_id"], {"jg_delta_norm": math.nan, "null_ratio": math.nan}))

        if self.causal_enabled:
            selected, risky = self._select_causal_groups(rows)
        else:
            risky = []
            eligible = [row for row in rows if not row["force_train"]]
            eligible.sort(
                key=lambda row: (
                    round(row["S"], 12) if np.isfinite(row["S"]) else math.inf,
                    -row["P"],
                    row["group_id"],
                )
            )
            freeze_count = min(
                int(round(self.config.max_freeze_fraction * len(rows))),
                len(eligible),
            )
            selected = {row["group_id"] for row in eligible[:freeze_count]}
        self.groups.set_frozen(selected)
        for row in rows:
            row["selected_for_freeze"] = row["group_id"] in selected
            row["is_frozen_after"] = row["selected_for_freeze"]
            row["frozen_before"] = row["is_frozen_before"]
            row["frozen_after"] = row["is_frozen_after"]
            row["ic_loss_after"] = row["loss_ic_after"]
            row["protected_pde_loss_after"] = row["loss_protected_pde_after"]
            row["ic_risk"] = row.get("ic_protected_risk", math.nan)
            row["pde_risk"] = row.get("pde_protected_risk", math.nan)
            row.pop("_delta")
        self.event_count = event_id
        self.event_steps.append(step)
        self.last_event_step = step
        self.last_event_front = protected_front
        self._pending_event_reason = None
        for group in self.groups.groups:
            self.mask_history[group.group_id].append("freeze" if group.is_frozen else "train")
        self._append_csv("dynamic_freezing_groups.csv", rows)
        if self.causal_enabled:
            self._write_causal_chunks(event_id, step, base["causal_details"], protected_front)
        correlations = self._correlations(rows)
        previous_selected = {group_id for group_id, frozen in was_frozen.items() if frozen}
        union = previous_selected | selected
        mask_jaccard = len(previous_selected & selected) / len(union) if union else 1.0
        causal_details = base.get("causal_details") or {}
        causal_weights = causal_details.get("weights")
        causal_weight_values = (
            causal_weights.detach().cpu().numpy() if causal_weights is not None else np.asarray([])
        )
        max_frozen = int(math.floor(len(rows) * self.config.max_freeze_fraction))
        self._append_csv("dynamic_freezing_events.csv", [{
            "event_id": event_id,
            "step": step,
            "trigger": event_reason,
            "candidate_front": self.candidate_front if self.causal_enabled else -1,
            "weight_enter_front": self.weight_enter_front if self.causal_enabled else -1,
            "pde_gated_enter_front": self.pde_gated_enter_front if self.causal_enabled else -1,
            "committed_front": protected_front,
            "front_changed": event_reason == "front_changed",
            "num_risky_groups": len(risky),
            "num_frozen_groups": len(selected),
            "num_newly_frozen": len(selected - previous_selected),
            "num_unfrozen": len(previous_selected - selected),
            "ic_loss_base": base["ic"],
            "protected_pde_loss_base": base["good_pde"],
            "causal_loss_base": base["bad"] if self.causal_enabled else math.nan,
            "causal_weight_min": float(np.min(causal_weight_values)) if causal_weight_values.size else math.nan,
            "causal_weight_mean": float(np.mean(causal_weight_values)) if causal_weight_values.size else math.nan,
            "causal_weight_max": float(np.max(causal_weight_values)) if causal_weight_values.size else math.nan,
            "mask_jaccard": mask_jaccard,
            "cap_reached": self.causal_enabled and len(risky) > max_frozen,
            **correlations,
        }])
        self._save_temporal_profile(step, event_id)
        if not self.causal_enabled or event_reason == "max_refresh":
            if self.config.responsibility_enabled:
                self._save_responsibility(event_id)
            self._plot_event(event_id, rows)
        event_total = "unbounded" if self.causal_enabled else str(self.config.freeze_events)
        risky_text = f", risky={len(risky)}" if self.causal_enabled else ""
        cap_text = ""
        if self.causal_enabled:
            max_frozen = int(math.floor(len(rows) * self.config.max_freeze_fraction))
            if len(risky) > max_frozen:
                cap_text = f", cap_reached={max_frozen}"
        print(
            f"Dynamic freezing event {event_id}/{event_total} at step {step} "
            f"({event_reason}, protected_front={protected_front}): "
            f"frozen {len(selected)}/{len(rows)} groups{risky_text}{cap_text}."
        )

    def _protected_point_mask(self, front_chunk):
        mask = np.zeros(len(self.interior_np), dtype=bool)
        if front_chunk < 0:
            return mask
        time_index = int(self.causal_options.get("time_index", -1))
        order = np.argsort(self.interior_np[:, time_index], kind="stable")
        num_chunks = int(self.causal_options.get("num_chunks", 16))
        chunk_size = len(order) // num_chunks
        protected_count = min((front_chunk + 1) * chunk_size, len(order))
        mask[order[:protected_count]] = True
        return mask

    def _select_causal_groups(self, rows):
        risky = [row for row in rows if row["risky"]]
        risky.sort(
            key=lambda row: (
                -row["protected_risk"],
                -row["incremental_worsening"],
                row["group_id"],
            )
        )
        max_frozen = int(math.floor(len(rows) * self.config.max_freeze_fraction))
        selected = {row["group_id"] for row in risky[:max_frozen]}
        return selected, risky

    def _causal_protection_metrics(self, base, changed, protected_front_chunk):
        """Evaluate IC and causal-prefix protection as independent constraints."""
        eps = self.config.relative_eps
        ic_tolerance = self.config.good_tolerance
        pde_tolerance = self.config.protected_pde_tolerance

        ic_violates = changed["ic"] > ic_tolerance
        ic_worsens = changed["ic"] > base["ic"] + eps
        pde_is_protected = protected_front_chunk >= 0
        pde_violates = pde_is_protected and changed["good_pde"] > pde_tolerance
        pde_worsens = pde_is_protected and changed["good_pde"] > base["good_pde"] + eps

        ic_risk = max(0.0, (changed["ic"] - ic_tolerance) / ic_tolerance)
        pde_risk = (
            max(0.0, (changed["good_pde"] - pde_tolerance) / pde_tolerance)
            if pde_is_protected
            else 0.0
        )
        ic_increment = max(
            0.0,
            (changed["ic"] - base["ic"])
            / (max(base["ic"], ic_tolerance) + eps),
        )
        pde_increment = (
            max(
                0.0,
                (changed["good_pde"] - base["good_pde"])
                / (max(base["good_pde"], pde_tolerance) + eps),
            )
            if pde_is_protected
            else 0.0
        )
        ic_is_risky = ic_violates and ic_worsens
        pde_is_risky = pde_violates and pde_worsens
        risky = ic_is_risky or pde_is_risky
        active_risks = []
        active_increments = []
        if ic_is_risky:
            active_risks.append(ic_risk)
            active_increments.append(ic_increment)
        if pde_is_risky:
            active_risks.append(pde_risk)
            active_increments.append(pde_increment)
        return {
            "violates_ic_tolerance": ic_violates,
            "worsens_ic": ic_worsens,
            "ic_protected_risk": ic_risk,
            "ic_incremental_worsening": ic_increment,
            "violates_protected_pde_tolerance": pde_violates,
            "worsens_protected_pde": pde_worsens,
            "pde_protected_risk": pde_risk,
            "pde_incremental_worsening": pde_increment,
            "protected_risk": max(active_risks, default=0.0),
            "incremental_worsening": max(active_increments, default=0.0),
            "risky": risky,
        }

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
        layer_groups = {}
        for group in self.groups.groups:
            layer_groups.setdefault(group.parameter_name, []).append(group.group_id)
        for group_ids in layer_groups.values():
            denominator = result[:, group_ids].sum(axis=1, keepdims=True)
            nonzero = denominator[:, 0] > self.config.relative_eps
            result[np.ix_(np.flatnonzero(nonzero), group_ids)] /= denominator[nonzero]
            if np.any(~nonzero):
                result[np.ix_(np.flatnonzero(~nonzero), group_ids)] = 1.0 / len(group_ids)
        return result

    def _save_responsibility(self, event_id):
        bbox = np.asarray(self.model.pde.bbox, dtype=np.float64)
        x = np.linspace(bbox[0], bbox[1], self.config.responsibility_nx, endpoint=False)
        t = np.linspace(bbox[2], bbox[3], self.config.responsibility_nt)
        xx, tt = np.meshgrid(x, t)
        points = np.column_stack([xx.reshape(-1), tt.reshape(-1)]).astype(np.float32)
        responsibility = self.responsibility(points).reshape(len(t), len(x), len(self.groups))
        layer_names = []
        dominant_layers = []
        entropy_layers = []
        for parameter_name in dict.fromkeys(group.parameter_name for group in self.groups.groups):
            group_ids = [
                group.group_id for group in self.groups.groups
                if group.parameter_name == parameter_name
            ]
            layer_values = responsibility[:, :, group_ids]
            dominant_layers.append(np.take(np.asarray(group_ids), np.argmax(layer_values, axis=-1)))
            if len(group_ids) == 1:
                entropy_layers.append(np.zeros((len(t), len(x))))
            else:
                entropy_layers.append(
                    -np.sum(
                        layer_values * np.log(layer_values + self.config.relative_eps), axis=-1
                    ) / np.log(len(group_ids))
                )
            layer_names.append(parameter_name)
        dominant = np.stack(dominant_layers, axis=-1)
        entropy = np.stack(entropy_layers, axis=-1)
        label = f"event_{event_id:02d}" if isinstance(event_id, int) else str(event_id)
        np.savez_compressed(
            os.path.join(self.log_dir, f"{label}_responsibility.npz"),
            responsibility=responsibility,
            dominant_group=dominant,
            entropy=entropy,
            entropy_by_time=entropy.mean(axis=1),
            mean_responsibility_by_time=responsibility.mean(axis=1),
            mean_responsibility_entropy=float(entropy.mean()),
            layer_names=np.asarray(layer_names),
            time=t,
            x=x,
        )

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            dominant_image = axes[0].imshow(
                dominant[:, :, 0], origin="lower", aspect="auto", extent=[x[0], x[-1], t[0], t[-1]]
            )
            axes[0].set_title("Dominant weight group")
            fig.colorbar(dominant_image, ax=axes[0])
            entropy_image = axes[1].imshow(
                entropy.mean(axis=-1), origin="lower", aspect="auto", extent=[x[0], x[-1], t[0], t[-1]], vmin=0, vmax=1
            )
            axes[1].set_title("Responsibility entropy")
            fig.colorbar(entropy_image, ax=axes[1])
            for axis in axes:
                axis.set_xlabel("x")
                axis.set_ylabel("t")
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, f"{label}_responsibility.png"), dpi=150)
            plt.close(fig)
        except Exception as error:
            print(f"Warning: could not plot responsibility maps: {error}")

    def _predict_numpy(self, points_np):
        was_training = self.model.net.training
        self.model.net.train(False)
        try:
            with torch.no_grad():
                return self.model.net(self._tensor(points_np)).detach().cpu().numpy()
        finally:
            self.model.net.train(was_training)

    def _boundary_points(self, boundary_times):
        xx, tt = np.meshgrid(self.x_values, boundary_times)
        return np.column_stack([xx.reshape(-1), tt.reshape(-1)]).astype(np.float32)

    def _chunk_relative_l2(self, num_chunks):
        result = np.full(num_chunks, np.nan, dtype=float)
        if self._reference_nearest is None:
            return result
        prediction = self._predict_numpy(self.interior_np)
        reference = np.asarray(self._reference_nearest(self.interior_np)).reshape(prediction.shape)
        time_index = int(self.causal_options.get("time_index", -1))
        order = np.argsort(self.interior_np[:, time_index], kind="stable")
        chunk_size = len(order) // num_chunks
        if chunk_size <= 0:
            return result
        ordered = order[: chunk_size * num_chunks].reshape(num_chunks, chunk_size)
        for chunk_id, indices in enumerate(ordered):
            numerator = np.linalg.norm(prediction[indices] - reference[indices])
            denominator = np.linalg.norm(reference[indices]) + self.config.relative_eps
            result[chunk_id] = numerator / denominator
        return result

    @staticmethod
    def _empty_ks_term_metrics():
        return {
            "ut_rms": math.nan,
            "nonlinear_rms": math.nan,
            "uxx_rms": math.nan,
            "uxxxx_rms": math.nan,
            "ks_term_scale_rms": math.nan,
            "residual_to_term_scale_ratio": math.nan,
        }

    def _ks_term_metrics(self, step):
        cache = getattr(self.model, "ks_causal_chunk_diagnostics", None)
        if cache:
            metrics_step = int(cache.get("step", -1))
            return cache.get("chunks", {}), metrics_step, int(step) - metrics_step
        if not self._warned_missing_ks_cache:
            print(
                "Warning: KS chunk diagnostics are unavailable; "
                "boundary KS term metrics will be NaN."
            )
            self._warned_missing_ks_cache = True
        return {}, math.nan, math.nan

    def _log_boundary_diagnostics(self, step):
        details = getattr(self.model, "causal_loss_details", None)
        if not details:
            return
        chunk_losses = details["chunk_losses"].detach().cpu().numpy().reshape(-1)
        weights = details["weights"].detach().cpu().numpy().reshape(-1)
        post_weights = details["post_chunk_weights"].detach().cpu().numpy().reshape(-1)
        t_min = details["t_min"].detach().cpu().numpy().reshape(-1)
        t_max = details["t_max"].detach().cpu().numpy().reshape(-1)
        num_chunks = len(chunk_losses)
        boundary_points = self._boundary_points(t_max)
        boundary_states = self._predict_numpy(boundary_points).reshape(
            num_chunks, len(self.x_values), -1
        )

        drift_abs = np.full(num_chunks, np.nan, dtype=float)
        drift_rel = np.full(num_chunks, np.nan, dtype=float)
        loss_abs_change = np.full(num_chunks, np.nan, dtype=float)
        loss_rel_change = np.full(num_chunks, np.nan, dtype=float)
        compatible_previous = (
            self.previous_boundary_states is not None
            and self.previous_boundary_states.shape == boundary_states.shape
            and self.previous_boundary_chunk_losses is not None
            and self.previous_boundary_chunk_losses.shape == chunk_losses.shape
        )
        if compatible_previous:
            difference = boundary_states - self.previous_boundary_states
            drift_abs = np.sqrt(np.mean(difference ** 2, axis=(1, 2)))
            drift_rel = np.linalg.norm(difference, axis=(1, 2)) / (
                np.linalg.norm(boundary_states, axis=(1, 2)) + self.config.relative_eps
            )
            loss_abs_change = chunk_losses - self.previous_boundary_chunk_losses
            loss_rel_change = loss_abs_change / (
                self.previous_boundary_chunk_losses + self.config.relative_eps
            )

        chunk_l2re = self._chunk_relative_l2(num_chunks)
        boundary_l2re = np.full(num_chunks, np.nan, dtype=float)
        if self._reference_nearest is not None:
            boundary_reference = np.asarray(
                self._reference_nearest(boundary_points)
            ).reshape(boundary_states.shape)
            boundary_l2re = np.linalg.norm(
                boundary_states - boundary_reference, axis=(1, 2)
            ) / (
                np.linalg.norm(boundary_reference, axis=(1, 2))
                + self.config.relative_eps
            )

        ks_metrics, ks_metrics_step, ks_metrics_age_steps = self._ks_term_metrics(step)
        rolling_mean = np.full(num_chunks, np.nan, dtype=float)
        rolling_max = np.full(num_chunks, np.nan, dtype=float)
        for chunk_id in range(num_chunks):
            history = self.boundary_drift_history.setdefault(
                chunk_id, deque(maxlen=5)
            )
            if np.isfinite(drift_rel[chunk_id]):
                history.append(float(drift_rel[chunk_id]))
            if history:
                rolling_mean[chunk_id] = float(np.mean(history))
                rolling_max[chunk_id] = float(np.max(history))

        transfer_raw_ready = np.zeros(num_chunks, dtype=bool)
        transfer_ready_count = np.zeros(num_chunks, dtype=int)
        transfer_ready_stable = np.zeros(num_chunks, dtype=bool)
        if self.config.transfer_boundary_enabled:
            transfer_raw_ready = (
                (post_weights >= self.config.transfer_boundary_weight_threshold)
                & np.isfinite(drift_rel)
                & np.isfinite(rolling_mean)
                & (rolling_mean <= self.config.transfer_boundary_drift_threshold)
            )
            for chunk_id in range(num_chunks):
                previous_count = self.transfer_ready_counts.get(chunk_id, 0)
                count = previous_count + 1 if transfer_raw_ready[chunk_id] else 0
                self.transfer_ready_counts[chunk_id] = count
                transfer_ready_count[chunk_id] = count
            transfer_ready_stable = (
                transfer_ready_count >= self.config.transfer_boundary_patience
            )
        else:
            self.transfer_ready_counts.clear()

        domain_t_min = float(np.asarray(self.model.pde.bbox, dtype=float)[2])
        real_boundaries = t_max > domain_t_min + self.config.relative_eps
        transfer_candidate = -1
        for chunk_id in np.flatnonzero(real_boundaries):
            if not transfer_ready_stable[chunk_id]:
                break
            transfer_candidate = int(chunk_id)

        transfer_is_candidate = np.zeros(num_chunks, dtype=bool)
        if transfer_candidate >= 0:
            transfer_is_candidate[transfer_candidate] = True

        rows = []
        for chunk_id in range(num_chunks):
            terms = self._empty_ks_term_metrics()
            terms.update(ks_metrics.get(chunk_id, {}))
            rows.append({
                "step": step,
                "chunk_id": chunk_id,
                "t_min": float(t_min[chunk_id]),
                "t_max": float(t_max[chunk_id]),
                "boundary_t": float(t_max[chunk_id]),
                "chunk_loss": float(chunk_losses[chunk_id]),
                "causal_weight": float(weights[chunk_id]),
                "post_chunk_weight": float(post_weights[chunk_id]),
                "is_candidate_protected": chunk_id <= self.candidate_front,
                "is_committed_protected": chunk_id <= self.committed_front,
                "boundary_drift_abs": float(drift_abs[chunk_id]),
                "boundary_drift_rel": float(drift_rel[chunk_id]),
                "boundary_drift_rel_mean": float(rolling_mean[chunk_id]),
                "boundary_drift_rel_max": float(rolling_max[chunk_id]),
                "chunk_loss_delta": float(loss_abs_change[chunk_id]),
                "chunk_loss_rel_delta": float(loss_rel_change[chunk_id]),
                "transfer_raw_ready": bool(transfer_raw_ready[chunk_id]),
                "transfer_ready_count": int(transfer_ready_count[chunk_id]),
                "transfer_ready_stable": bool(transfer_ready_stable[chunk_id]),
                "transfer_is_candidate": bool(transfer_is_candidate[chunk_id]),
                "ks_metrics_step": ks_metrics_step,
                "ks_metrics_age_steps": ks_metrics_age_steps,
                **terms,
                "chunk_l2re": float(chunk_l2re[chunk_id]),
                "boundary_l2re": float(boundary_l2re[chunk_id]),
            })
        self._append_csv("boundary_diagnostics.csv", rows)

        oracle_fronts = {}
        for threshold in (0.1, 0.2):
            if not np.any(np.isfinite(chunk_l2re)):
                oracle_fronts[threshold] = math.nan
                continue
            oracle_front = -1
            for chunk_id in np.flatnonzero(real_boundaries):
                if not np.isfinite(chunk_l2re[chunk_id]) or chunk_l2re[chunk_id] >= threshold:
                    break
                oracle_front = int(chunk_id)
            oracle_fronts[threshold] = oracle_front

        candidate_values = (
            {
                "transfer_boundary_t": float(t_max[transfer_candidate]),
                "candidate_post_chunk_weight": float(post_weights[transfer_candidate]),
                "candidate_boundary_drift_rel": float(drift_rel[transfer_candidate]),
                "candidate_boundary_drift_rel_mean": float(rolling_mean[transfer_candidate]),
                "candidate_ready_count": int(transfer_ready_count[transfer_candidate]),
                "oracle_candidate_chunk_l2re": float(chunk_l2re[transfer_candidate]),
                "oracle_candidate_boundary_l2re": float(boundary_l2re[transfer_candidate]),
            }
            if transfer_candidate >= 0
            else {
                "transfer_boundary_t": math.nan,
                "candidate_post_chunk_weight": math.nan,
                "candidate_boundary_drift_rel": math.nan,
                "candidate_boundary_drift_rel_mean": math.nan,
                "candidate_ready_count": math.nan,
                "oracle_candidate_chunk_l2re": math.nan,
                "oracle_candidate_boundary_l2re": math.nan,
            }
        )
        eligible = np.flatnonzero(real_boundaries)
        self._append_csv("transfer_boundary_history.csv", [{
            "step": step,
            "transfer_boundary_candidate": transfer_candidate,
            **candidate_values,
            "num_raw_ready_boundaries": int(np.sum(transfer_raw_ready[eligible])),
            "num_stable_ready_boundaries": int(np.sum(transfer_ready_stable[eligible])),
            "oracle_farthest_chunk_l2re_lt_0_1": oracle_fronts[0.1],
            "oracle_farthest_chunk_l2re_lt_0_2": oracle_fronts[0.2],
        }])
        self.previous_boundary_states = boundary_states.copy()
        self.previous_boundary_chunk_losses = chunk_losses.copy()
        self.previous_boundary_step = int(step)

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
        causal_loss = losses["bad"] if self.causal_enabled else math.nan
        target_pde_loss = causal_loss if self.causal_enabled else losses["pde"]
        self.training_rows.append({
            "step": step,
            "total_loss": self.config.ic_weight * losses["ic"] + target_pde_loss,
            "ic_loss": losses["ic"],
            "causal_loss": causal_loss,
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

    def _write_causal_chunks(self, event_id, step, details, protected_front):
        rows = []
        for chunk_id, (chunk_loss, weight, post_weight, t_min, t_max) in enumerate(zip(
            details["chunk_losses"],
            details["weights"],
            details["post_chunk_weights"],
            details["t_min"],
            details["t_max"],
        )):
            loss_value = float(chunk_loss.cpu())
            weight_value = float(weight.cpu())
            rows.append({
                "event_id": event_id,
                "step": step,
                "chunk_id": chunk_id,
                "t_min": float(t_min.cpu()),
                "t_max": float(t_max.cpu()),
                "chunk_loss": loss_value,
                "causal_weight": weight_value,
                "post_chunk_weight": float(post_weight.cpu()),
                "weighted_chunk_loss": weight_value * loss_value,
                "is_protected": chunk_id <= protected_front,
                "is_candidate_protected": chunk_id <= self.candidate_front,
                "is_committed_protected": chunk_id <= protected_front,
                "is_front_chunk": chunk_id == protected_front,
            })
        self._append_csv("dynamic_freezing_chunks.csv", rows)

    def _correlations(self, rows):
        protected = np.asarray([row["P"] for row in rows], dtype=float)
        jg = np.asarray([row["jg_delta_norm"] for row in rows], dtype=float)
        null_complement = 1 - np.asarray([row["null_ratio"] for row in rows], dtype=float)
        payload = {}
        for name, values in (("jg_delta_norm", jg), ("one_minus_null_ratio", null_complement)):
            valid = np.isfinite(protected) & np.isfinite(values)
            if valid.sum() >= 2 and np.std(protected[valid]) > 0 and np.std(values[valid]) > 0:
                payload[f"pearson_P_vs_{name}"] = float(stats.pearsonr(protected[valid], values[valid]).statistic)
                payload[f"spearman_P_vs_{name}"] = float(stats.spearmanr(protected[valid], values[valid]).statistic)
            else:
                payload[f"pearson_P_vs_{name}"] = math.nan
                payload[f"spearman_P_vs_{name}"] = math.nan
        return payload

    def _append_csv(self, filename, rows):
        if not rows:
            return
        path = os.path.join(self.log_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

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
            fields = (
                ("protected_risk", "incremental_worsening", "d_causal_fixed", "jg_delta_norm")
                if self.causal_enabled
                else ("S", "d_good_rel", "d_bad_rel", "null_ratio")
            )
            for axis, field in zip(axes, fields):
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
            fields = ["total_loss", "ic_loss", "pde_loss"]
            if self.causal_enabled:
                fields.append("causal_loss")
            for field in fields:
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
