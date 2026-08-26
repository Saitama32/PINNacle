"""Hybrid VPINN weak loss with spatial integration by parts.

The weak residual is evaluated on local Legendre-bubble test functions.  Only
space is integrated by parts; time remains a sampled continuous coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Union

import numpy as np
import torch

from deepxde import backend as bkd
from vpinn.vpinn.weak_primitives import SpatialQuadrature, build_spatial_quadrature


class WeakFormAdapter(Protocol):
    spatial_bounds: Sequence[Sequence[float]]
    time_bounds: Sequence[float]

    def weak_residuals(
        self,
        net: torch.nn.Module,
        quadrature: SpatialQuadrature,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(time, cell, mode)`` or ``(equation, time, cell, mode)``."""


@dataclass(frozen=True)
class WeakFormConfig:
    spatial_cells: Union[int, tuple[int, ...]]
    quadrature_order: int
    test_function_count: int
    time_samples: int
    seed: int = 1234
    normalize_by_cell_volume: bool = True

    def __post_init__(self):
        if isinstance(self.spatial_cells, int):
            cells = (self.spatial_cells,)
        else:
            cells = tuple(self.spatial_cells)
        if not cells or any(int(value) <= 0 for value in cells):
            raise ValueError("spatial_cells must contain positive integers.")
        if int(self.quadrature_order) < 2:
            raise ValueError("quadrature_order must be at least 2.")
        if int(self.test_function_count) <= 0:
            raise ValueError("test_function_count must be positive.")
        if int(self.time_samples) <= 0:
            raise ValueError("time_samples must be positive.")


class WeakFormLoss:
    """Compute scalar or multi-equation weak losses with diagnostics."""

    def __init__(self, adapter: WeakFormAdapter, config: WeakFormConfig):
        self.adapter = adapter
        self.config = config
        self._quadrature_cache: dict[tuple[str, str], SpatialQuadrature] = {}

        generator = np.random.default_rng(int(config.seed))
        self._train_time_fractions = generator.random(
            int(config.time_samples), dtype=np.float64
        )
        # Midpoints avoid giving either endpoint a special weight in the fixed
        # deterministic evaluation pool.
        self._test_time_fractions = (
            np.arange(int(config.time_samples), dtype=np.float64) + 0.5
        ) / float(config.time_samples)
        self.last_diagnostics: dict[str, object] = {}
        self.last_train_diagnostics: dict[str, object] = {}
        self.last_test_diagnostics: dict[str, object] = {}

    def _device_dtype(self, net: torch.nn.Module):
        parameter = next(net.parameters(), None)
        if parameter is None:
            return torch.device("cpu"), torch.get_default_dtype()
        return parameter.device, parameter.dtype

    def _quadrature(self, device, dtype) -> SpatialQuadrature:
        key = (str(device), str(dtype))
        quadrature = self._quadrature_cache.get(key)
        if quadrature is None:
            quadrature = build_spatial_quadrature(
                bounds=self.adapter.spatial_bounds,
                spatial_cells=self.config.spatial_cells,
                quadrature_order=self.config.quadrature_order,
                test_function_count=self.config.test_function_count,
                device=device,
                dtype=dtype,
            )
            self._quadrature_cache[key] = quadrature
        return quadrature

    def _times(self, training: bool, device, dtype) -> torch.Tensor:
        lower, upper = (float(value) for value in self.adapter.time_bounds)
        fractions_array = (
            self._train_time_fractions if training else self._test_time_fractions
        )
        fractions = torch.as_tensor(fractions_array, device=device, dtype=dtype)
        return lower + (upper - lower) * fractions

    def __call__(self, net: torch.nn.Module, *, training: bool = True) -> torch.Tensor:
        device, dtype = self._device_dtype(net)
        quadrature = self._quadrature(device, dtype)
        times = self._times(training, device, dtype)
        raw_residuals = self.adapter.weak_residuals(net, quadrature, times)
        if raw_residuals.ndim == 3:
            raw_residuals = raw_residuals.unsqueeze(0)
        elif raw_residuals.ndim != 4:
            raise RuntimeError(
                "Weak-form adapter must return (time, cell, mode) or "
                "(equation, time, cell, mode) residuals."
            )
        normalized_residuals = (
            raw_residuals / quadrature.cell_volumes[None, None, :, None]
        )
        selected = (
            normalized_residuals
            if self.config.normalize_by_cell_volume
            else raw_residuals
        )
        component_losses = torch.mean(selected.square(), dim=(1, 2, 3))
        loss = component_losses[0] if component_losses.numel() == 1 else component_losses

        with torch.no_grad():
            diagnostics = {
                "training": bool(training),
                "raw_weak_loss": torch.mean(raw_residuals.detach().square()),
                "normalized_weak_loss": torch.mean(
                    normalized_residuals.detach().square()
                ),
                "weak_rms": torch.sqrt(torch.mean(selected.detach().square())),
                "equation_rms": torch.sqrt(
                    torch.mean(selected.detach().square(), dim=(1, 2, 3))
                ),
                "time_rms": torch.sqrt(
                    torch.mean(selected.detach().square(), dim=(0, 2, 3))
                ),
                "cell_rms": torch.sqrt(
                    torch.mean(selected.detach().square(), dim=(0, 1, 3))
                ),
                "mode_rms": torch.sqrt(
                    torch.mean(selected.detach().square(), dim=(0, 1, 2))
                ),
                "num_equations": int(raw_residuals.shape[0]),
                "num_times": int(raw_residuals.shape[1]),
                "num_cells": int(raw_residuals.shape[2]),
                "num_test_functions": int(raw_residuals.shape[3]),
                "points_per_cell": quadrature.points_per_cell,
            }
            self.last_diagnostics = diagnostics
            if training:
                self.last_train_diagnostics = diagnostics
            else:
                self.last_test_diagnostics = diagnostics
        return loss


class _WeakFormData:
    """DeepXDE data facade replacing only the PDE component of a dataset."""

    def __init__(self, base_data, weak_loss: WeakFormLoss):
        self.base_data = base_data
        self.weak_loss = weak_loss
        self.bcs = base_data.bcs
        self.num_bcs = list(base_data.num_bcs)

        # The weak PDE loss evaluates its own quadrature points.  Feeding only
        # the BC/IC prefix avoids an otherwise unused strong collocation pass.
        self.train_x = np.asarray(base_data.train_x_bc)
        # Keep DeepXDE's original test set (BC prefix + uniform interior
        # points). PlotCallback and validation need interior predictions even
        # though the strong residual is not evaluated there.
        self.test_x = np.asarray(base_data.test_x)
        self.train_y = base_data.soln(self.train_x) if base_data.soln else None
        self.test_y = base_data.soln(self.test_x) if base_data.soln else None
        self.train_aux_vars = None
        self.test_aux_vars = None

    def __getattr__(self, name):
        return getattr(self.base_data, name)

    def train_next_batch(self, batch_size=None):
        return self.train_x, self.train_y, self.train_aux_vars

    def test(self):
        return self.test_x, self.test_y, self.test_aux_vars

    def _losses(self, loss_fn, inputs, outputs, model, *, training: bool):
        weak_values = self.weak_loss(model.net, training=training)
        weak_losses = (
            [weak_values]
            if weak_values.ndim == 0
            else list(torch.unbind(weak_values))
        )
        weak_count = len(weak_losses)
        count = weak_count + len(self.bcs)
        if not isinstance(loss_fn, (list, tuple)):
            loss_functions = [loss_fn] * count
        elif len(loss_fn) == count:
            loss_functions = list(loss_fn)
        else:
            raise ValueError(
                f"Weak-form data has {count} errors, but received {len(loss_fn)} losses."
            )

        # Do not alias ``weak_losses`` here: appending BC terms would change
        # its length and shift the loss-function index on every iteration.
        losses = list(weak_losses)
        starts = list(map(int, np.cumsum([0] + self.num_bcs)))
        sample_points = self.train_x if training else self.test_x
        for index, bc in enumerate(self.bcs):
            begin, end = starts[index], starts[index + 1]
            error = bc.error(sample_points, inputs, outputs, begin, end)
            losses.append(
                loss_functions[weak_count + index](
                    bkd.zeros_like(error), error
                )
            )
        return losses

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return self._losses(loss_fn, inputs, outputs, model, training=True)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return self._losses(loss_fn, inputs, outputs, model, training=False)


def attach_weak_form_loss(model, weak_loss: WeakFormLoss):
    """Attach a weak PDE objective before compiling a PyTorch DeepXDE model."""

    if getattr(model, "opt", None) is not None:
        raise RuntimeError("Attach the weak-form loss before model.compile().")
    if hasattr(model.data, "base_data"):
        base_data = model.data.base_data
    else:
        base_data = model.data
    model.strong_data = base_data
    model.weak_data = _WeakFormData(base_data, weak_loss)
    model.weak_form_loss = weak_loss
    model.physics_loss_kind = "weak"
    model.data = model.weak_data
    # This PINNacle fork passes auxiliary variables through the PyTorch loss
    # closure unconditionally, while the stock FNN does not define the field.
    if not hasattr(model.net, "auxiliary_vars"):
        model.net.auxiliary_vars = []
    return model
