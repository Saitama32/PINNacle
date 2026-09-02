"""Matrix-free, low-rank Gauss--Newton error-whitening optimizer."""

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
    for a loss ``0.5 * ||r||^2``. Randomized sketching uses matrix-free
    ``J.T @ (J @ directions)`` products; neither ``J`` nor ``J.T @ J`` is
    materialized.
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
        operator_batch_size=8,
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
        if operator_batch_size <= 0:
            raise ValueError("ErrorWhiteningGN operator_batch_size must be positive")
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
        self.operator_batch_size = int(operator_batch_size)
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
    def _flatten_gradients(gradients, parameters, batch_size=None):
        columns = []
        for gradient, parameter in zip(gradients, parameters):
            if gradient is None:
                shape = (parameter.numel(),) if batch_size is None else (
                    batch_size,
                    parameter.numel(),
                )
                columns.append(
                    torch.zeros(shape, dtype=torch.float64, device=parameter.device)
                )
            elif batch_size is None:
                columns.append(gradient.reshape(-1).double())
            else:
                columns.append(gradient.reshape(batch_size, -1).double())
        return torch.cat(columns, dim=0 if batch_size is None else 1)

    def _gauss_newton_block(self, residuals, parameters, directions):
        """Return ``J.T @ (J @ directions)`` without constructing ``J``."""

        block_size = directions.shape[1]
        cotangent = torch.zeros_like(residuals, requires_grad=True)
        jt_cotangent = torch.autograd.grad(
            residuals,
            parameters,
            grad_outputs=cotangent,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        flat_jt_cotangent = self._flatten_gradients(jt_cotangent, parameters)
        directional_pairings = directions.mT @ flat_jt_cotangent
        identity = torch.eye(
            block_size, dtype=directions.dtype, device=directions.device
        )
        # Reverse-over-reverse computes all JVPs in this probe block. The
        # leading dimension is interpreted as the vmap batch by autograd.
        jvp_rows = torch.autograd.grad(
            directional_pairings,
            cotangent,
            grad_outputs=identity,
            is_grads_batched=True,
            retain_graph=True,
        )[0]
        vjp_gradients = torch.autograd.grad(
            residuals,
            parameters,
            grad_outputs=jvp_rows.detach(),
            is_grads_batched=True,
            retain_graph=True,
            allow_unused=True,
        )
        return self._flatten_gradients(
            vjp_gradients, parameters, batch_size=block_size
        ).mT

    def _gauss_newton_matmul(self, residuals, parameters, directions):
        blocks = []
        for start in range(0, directions.shape[1], self.operator_batch_size):
            stop = min(start + self.operator_batch_size, directions.shape[1])
            blocks.append(
                self._gauss_newton_block(
                    residuals, parameters, directions[:, start:stop]
                )
            )
        return torch.cat(blocks, dim=1)

    @staticmethod
    def _loss_from_closure(closure):
        closure = getattr(closure, "line_search_closure", closure)
        with torch.enable_grad():
            residuals = _as_residual_vector(closure())
        return 0.5 * torch.dot(residuals.detach().double(), residuals.detach().double())

    def _randomized_eigensystem(self, residuals, parameters, parameter_count):
        sketch_size = min(parameter_count, self.rank + self.oversketch)
        generator = torch.Generator(device=residuals.device)
        generator.manual_seed(self.seed + self._step_count)
        omega = torch.randn(
            parameter_count,
            sketch_size,
            dtype=torch.float64,
            device=residuals.device,
            generator=generator,
        )
        sketch = self._gauss_newton_matmul(
            residuals, parameters, omega
        )
        # One-pass generalized Nyström approximation for the PSD GN matrix:
        # G ~= (Y (Omega.T Y)^-1/2) (Y (Omega.T Y)^-1/2).T.
        small_matrix = omega.mT @ sketch
        small_matrix = 0.5 * (small_matrix + small_matrix.mT)
        gram_values, gram_vectors = torch.linalg.eigh(small_matrix)
        gram_scale = gram_values.abs().max().clamp_min(1.0)
        gram_tolerance = torch.finfo(gram_values.dtype).eps * sketch_size * gram_scale
        gram_keep = torch.isfinite(gram_values) & (gram_values > gram_tolerance)
        if not torch.any(gram_keep):
            return gram_values.new_empty(0), sketch.new_empty(parameter_count, 0)
        inverse_root = (
            gram_vectors[:, gram_keep]
            * gram_values[gram_keep].rsqrt().unsqueeze(0)
        ) @ gram_vectors[:, gram_keep].mT
        nystrom_factor = sketch @ inverse_root
        eigenvectors, singular_values, _ = torch.linalg.svd(
            nystrom_factor, full_matrices=False
        )
        eigenvalues = singular_values.square()
        keep = torch.isfinite(eigenvalues) & (eigenvalues > self.tol)
        retained_indices = torch.nonzero(keep, as_tuple=False).flatten()[: self.rank]
        if retained_indices.numel() == 0:
            return eigenvalues.new_empty(0), sketch.new_empty(parameter_count, 0)
        retained_values = eigenvalues[retained_indices]
        retained_vectors = eigenvectors[:, retained_indices]
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
        gradient_parts = torch.autograd.grad(
            residuals,
            parameters,
            grad_outputs=detached_residuals.to(residuals.dtype),
            retain_graph=True,
            allow_unused=True,
        )
        gradient = self._flatten_gradients(gradient_parts, parameters)
        eigenvalues, eigenvectors = self._randomized_eigensystem(
            residuals, parameters, gradient.numel()
        )

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
            "operator_batch_size": self.operator_batch_size,
            "gn_operator_passes": 1,
        }
        self.last_diagnostics = diagnostics
        self.diagnostics_history.append(diagnostics.copy())
        self._step_count += 1
        return diagnostics


__all__ = ["ErrorWhiteningGN"]
