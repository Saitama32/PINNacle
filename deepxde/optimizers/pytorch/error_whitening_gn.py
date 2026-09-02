"""Explicit-Jacobian, low-rank Gauss--Newton error-whitening optimizer."""

from __future__ import annotations

import math

import torch


def _as_residual_vector(value):
    """Validate and flatten the residual closure result."""

    if isinstance(value, (tuple, list)):
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError("ErrorWhiteningGN closure must return a residual tensor")
    if value.ndim == 0:
        raise ValueError(
            "ErrorWhiteningGN closure must return residuals, not a scalar loss"
        )
    if value.numel() == 0:
        raise ValueError("ErrorWhiteningGN residual vector must not be empty")
    return value.reshape(-1)


class ErrorWhiteningGN(torch.optim.Optimizer):
    """Jacobian-only randomized low-rank Gauss--Newton.

    ``step`` requires a closure returning the weighted residual vector ``r``
    for a loss ``0.5 * ||r||^2``. The implementation explicitly forms the
    residual Jacobian in float64, but never forms the full ``J.T @ J`` matrix.
    """

    _DEFAULT_LINE_SEARCH_STEPS = (
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        *(2.0**-power for power in range(2, 31)),
    )

    def __init__(
        self,
        params,
        rank=100,
        oversketch=10,
        tol=1e-14,
        damping=1e-8,
        line_search=True,
        seed=0,
        line_search_steps=None,
    ):
        if rank <= 0:
            raise ValueError("ErrorWhiteningGN rank must be positive")
        if oversketch < 0:
            raise ValueError("ErrorWhiteningGN oversketch must be nonnegative")
        if tol < 0 or not math.isfinite(tol):
            raise ValueError("ErrorWhiteningGN tolerance must be finite and nonnegative")
        if damping < 0 or not math.isfinite(damping):
            raise ValueError("ErrorWhiteningGN damping must be finite and nonnegative")
        if line_search_steps is None:
            line_search_steps = self._DEFAULT_LINE_SEARCH_STEPS
        line_search_steps = tuple(float(step) for step in line_search_steps)
        if not line_search_steps or any(
            step <= 0 or not math.isfinite(step) for step in line_search_steps
        ):
            raise ValueError("ErrorWhiteningGN line-search steps must be positive and finite")

        # ``lr`` is retained in param_groups for scheduler/logger compatibility.
        # It is not an Adam-like learning rate: without line search the full
        # Gauss--Newton step is used.
        defaults = dict(lr=1.0)
        super().__init__(params, defaults)
        self.rank = int(rank)
        self.oversketch = int(oversketch)
        self.tol = float(tol)
        self.damping = float(damping)
        self.line_search = bool(line_search)
        self.seed = int(seed)
        self.line_search_steps = line_search_steps
        self._step_count = 0
        self.last_diagnostics = {}
        self.diagnostics_history = []

    def _parameters(self):
        parameters = []
        seen = set()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        if not parameters:
            raise ValueError("ErrorWhiteningGN has no trainable parameters")
        device = parameters[0].device
        if any(parameter.device != device for parameter in parameters):
            raise ValueError("ErrorWhiteningGN requires all parameters on one device")
        return parameters

    @staticmethod
    def _flatten_parameters(parameters):
        return torch.cat([parameter.detach().reshape(-1).double() for parameter in parameters])

    @staticmethod
    def _set_parameters(parameters, flat_values):
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                count = parameter.numel()
                parameter.copy_(
                    flat_values[offset : offset + count]
                    .reshape_as(parameter)
                    .to(dtype=parameter.dtype)
                )
                offset += count

    @staticmethod
    def _explicit_jacobian(residuals, parameters):
        parameter_count = sum(parameter.numel() for parameter in parameters)
        jacobian = torch.empty(
            residuals.numel(),
            parameter_count,
            dtype=torch.float64,
            device=residuals.device,
        )
        for row_index, residual in enumerate(residuals):
            if residual.requires_grad:
                gradients = torch.autograd.grad(
                    residual,
                    parameters,
                    retain_graph=row_index + 1 < residuals.numel(),
                    allow_unused=True,
                )
            else:
                gradients = (None,) * len(parameters)
            columns = []
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    columns.append(
                        torch.zeros(parameter.numel(), dtype=torch.float64, device=residuals.device)
                    )
                else:
                    columns.append(gradient.detach().reshape(-1).double())
            jacobian[row_index].copy_(torch.cat(columns))
        return jacobian

    @staticmethod
    def _loss_from_closure(closure):
        with torch.enable_grad():
            residuals = _as_residual_vector(closure())
        return 0.5 * torch.dot(residuals.detach().double(), residuals.detach().double())

    def _randomized_eigensystem(self, jacobian):
        parameter_count = jacobian.shape[1]
        sketch_size = min(parameter_count, self.rank + self.oversketch)
        generator = torch.Generator(device=jacobian.device)
        generator.manual_seed(self.seed + self._step_count)
        omega = torch.randn(
            parameter_count,
            sketch_size,
            dtype=torch.float64,
            device=jacobian.device,
            generator=generator,
        )
        # G @ Omega = J.T @ (J @ Omega), without materializing G.
        sketch = jacobian.mT @ (jacobian @ omega)
        basis = torch.linalg.qr(sketch, mode="reduced").Q
        projected_jacobian = jacobian @ basis
        small_matrix = projected_jacobian.mT @ projected_jacobian
        small_matrix = 0.5 * (small_matrix + small_matrix.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(small_matrix)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        keep = torch.isfinite(eigenvalues) & (eigenvalues > self.tol)
        retained_indices = torch.nonzero(keep, as_tuple=False).flatten()[: self.rank]
        if retained_indices.numel() == 0:
            return eigenvalues.new_empty(0), basis.new_empty(parameter_count, 0)
        retained_values = eigenvalues[retained_indices]
        retained_vectors = basis @ eigenvectors[:, retained_indices]
        return retained_values, retained_vectors

    @torch.no_grad()
    def _line_search(self, closure, parameters, base_parameters, direction, loss_before):
        best_loss = loss_before
        best_step = 0.0
        candidates = self.line_search_steps if self.line_search else (1.0,)
        for step_size in candidates:
            self._set_parameters(parameters, base_parameters + step_size * direction)
            candidate_loss = self._loss_from_closure(closure)
            if torch.isfinite(candidate_loss) and candidate_loss < best_loss:
                best_loss = candidate_loss
                best_step = step_size
        self._set_parameters(parameters, base_parameters + best_step * direction)
        return best_step, best_loss

    def step(self, closure=None):
        if closure is None:
            raise ValueError("ErrorWhiteningGN.step requires a residual closure")
        parameters = self._parameters()
        with torch.enable_grad():
            residuals = _as_residual_vector(closure())
        if residuals.device != parameters[0].device:
            raise ValueError("Residuals and parameters must be on the same device")

        detached_residuals = residuals.detach().double()
        loss_before = 0.5 * torch.dot(detached_residuals, detached_residuals)
        jacobian = self._explicit_jacobian(residuals, parameters)
        gradient = jacobian.mT @ detached_residuals
        eigenvalues, eigenvectors = self._randomized_eigensystem(jacobian)

        if eigenvalues.numel() == 0:
            direction = torch.zeros_like(gradient)
        else:
            projected_gradient = eigenvectors.mT @ gradient
            direction = -eigenvectors @ (
                projected_gradient / (eigenvalues + self.damping)
            )

        base_parameters = self._flatten_parameters(parameters)
        accepted_step, loss_after = self._line_search(
            closure, parameters, base_parameters, direction, loss_before
        )
        parameter_norm = torch.linalg.vector_norm(base_parameters)
        step_norm = torch.linalg.vector_norm(accepted_step * direction)
        gradient_norm = torch.linalg.vector_norm(gradient)
        explained_fraction = (
            torch.sum((eigenvectors.mT @ gradient).square()) / gradient_norm.square()
            if gradient_norm > 0 and eigenvalues.numel() > 0
            else gradient_norm.new_zeros(())
        )
        lambda_max = eigenvalues[0] if eigenvalues.numel() else gradient_norm.new_zeros(())
        lambda_min = eigenvalues[-1] if eigenvalues.numel() else gradient_norm.new_zeros(())
        condition = (
            lambda_max / lambda_min
            if eigenvalues.numel() and lambda_min > 0
            else gradient_norm.new_tensor(float("inf"))
        )
        diagnostics = {
            "loss_before": float(loss_before.cpu()),
            "loss_after": float(loss_after.cpu()),
            "accepted_step_size": float(accepted_step),
            "grad_norm": float(gradient_norm.cpu()),
            "gn_step_norm": float(step_norm.cpu()),
            "relative_step_norm": float((step_norm / parameter_norm.clamp_min(1e-30)).cpu()),
            "lambda_max": float(lambda_max.cpu()),
            "lambda_min_retained": float(lambda_min.cpu()),
            "condition_number_retained": float(condition.cpu()),
            "effective_rank": int(eigenvalues.numel()),
            "explained_gradient_fraction": float(explained_fraction.cpu()),
            "residual_count": int(residuals.numel()),
            "parameter_count": int(gradient.numel()),
        }
        self.last_diagnostics = diagnostics
        self.diagnostics_history.append(diagnostics.copy())
        self._step_count += 1
        return diagnostics


__all__ = ["ErrorWhiteningGN"]
