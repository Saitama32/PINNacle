# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""REKLS V3 optimizer for the PyTorch backend.

The matrix update is ported from NVIDIA NeMo Emerging-Optimizers' ``ReklsV3``.
REKLS is restricted to 2D tensors upstream; this DeepXDE integration applies
the selected scalar optimizer to biases, scalars, and unsupported tensors too.
"""

from contextlib import contextmanager

import torch

from .madam import madam_update


@contextmanager
def _fp32_matmul_precision(precision):
    if not hasattr(torch, "get_float32_matmul_precision"):
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _eigh_with_fallback(matrix):
    """NVIDIA's descending-order eigh with a float64 fallback."""
    input_dtype = matrix.dtype
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    except (torch.linalg.LinAlgError, RuntimeError):
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix.double())
        eigenvalues = eigenvalues.to(input_dtype)
        eigenvectors = eigenvectors.to(input_dtype)
    return torch.flip(eigenvalues, [-1]), torch.flip(eigenvectors, [-1])


def _init_rekls_state(parameter, state):
    if parameter.ndim != 2:
        raise TypeError(
            f"ReklsV3 is only supported for 2D tensors, got {tuple(parameter.shape)}"
        )
    rows, columns = parameter.shape
    device = parameter.device
    dtype = torch.float32
    state.update(
        step=0,
        exp_avg=torch.zeros(rows, columns, device=device, dtype=dtype),
        exp_avg_sq=torch.zeros(rows, columns, device=device, dtype=dtype),
        L=torch.zeros(rows, rows, device=device, dtype=dtype),
        R=torch.zeros(columns, columns, device=device, dtype=dtype),
        Q_L=torch.eye(rows, device=device, dtype=dtype),
        Q_R=torch.eye(columns, device=device, dtype=dtype),
        eigvals_L=torch.zeros(rows, device=device, dtype=dtype),
        eigvals_R=torch.zeros(columns, device=device, dtype=dtype),
    )


def _project_in(tensor, state):
    return state["Q_L"].mT @ tensor @ state["Q_R"]


def _project_out(tensor, state):
    return state["Q_L"] @ tensor @ state["Q_R"].mT


def _update_kronecker_factors(gradient, state, shampoo_beta, epsilon):
    """Apply NVIDIA's KL-Shampoo correction to the L/R factors."""
    rows, columns = gradient.shape

    left_scale = state["eigvals_L"].clamp_min(epsilon).reciprocal() / rows
    left_inverse = (state["Q_L"] * left_scale[None, :]) @ state["Q_L"].mT
    right_update = gradient.mT @ left_inverse @ gradient

    right_scale = state["eigvals_R"].clamp_min(epsilon).reciprocal() / columns
    right_inverse = (
        state["Q_R"] * right_scale[None, :]
    ) @ state["Q_R"].mT
    left_update = gradient @ right_inverse @ gradient.mT

    state["L"].lerp_(left_update, 1 - shampoo_beta)
    state["R"].lerp_(right_update, 1 - shampoo_beta)


def _calculate_adam_update(gradient, exp_avg, exp_avg_sq, betas, epsilon, step):
    """NVIDIA KlSoapV3's bias-corrected inner Adam update."""
    beta1, beta2 = betas
    exp_avg.lerp_(gradient, 1 - beta1)
    exp_avg_sq.lerp_(gradient.square(), 1 - beta2)
    momentum = exp_avg / (1 - beta1**step)
    second_moment = exp_avg_sq / (1 - beta2**step)
    return momentum / (second_moment.sqrt() + epsilon)


class ReklsV3WithAuxAdam(torch.optim.Optimizer):
    """NVIDIA REKLS V3 with selectable Adam or MAdam scalar updates."""

    def __init__(
        self,
        params,
        lr,
        betas=(0.9, 0.95),
        shampoo_beta=0.95,
        eps=1e-8,
        weight_decay=0.01,
        base_optimizer="adam",
        scale_log2=16.0,
        auxiliary_betas=None,
        auxiliary_eps=1e-8,
        auxiliary_scale_log2=None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("REKLS V3 betas must be in [0, 1)")
        if not 0 <= shampoo_beta < 1:
            raise ValueError("REKLS V3 shampoo_beta must be in [0, 1)")
        if eps <= 0 or auxiliary_eps <= 0:
            raise ValueError("REKLS V3 epsilon values must be positive")
        if weight_decay < 0:
            raise ValueError("REKLS V3 weight decay must be nonnegative")
        if not isinstance(base_optimizer, str):
            raise ValueError("REKLS V3 base_optimizer must be 'adam' or 'madam'")
        base_optimizer = base_optimizer.lower()
        if base_optimizer not in {"adam", "madam"}:
            raise ValueError("REKLS V3 base_optimizer must be 'adam' or 'madam'")
        if scale_log2 // 2 != scale_log2 / 2:
            raise ValueError("REKLS V3 scale_log2 must be an even integer")
        if auxiliary_betas is None:
            auxiliary_betas = betas
        if auxiliary_scale_log2 is None:
            auxiliary_scale_log2 = scale_log2
        if auxiliary_scale_log2 // 2 != auxiliary_scale_log2 / 2:
            raise ValueError(
                "REKLS V3 auxiliary_scale_log2 must be an even integer"
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            base_optimizer=base_optimizer,
            scale_log2=scale_log2,
            use_rekls=True,
            auxiliary_betas=auxiliary_betas,
            auxiliary_eps=auxiliary_eps,
            auxiliary_scale_log2=auxiliary_scale_log2,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            if "use_rekls" not in group:
                raise ValueError("Each REKLS V3 group must define use_rekls")
            if group["use_rekls"]:
                bad_shapes = [
                    tuple(parameter.shape)
                    for parameter in group["params"]
                    if parameter.ndim != 2
                ]
                if bad_shapes:
                    raise ValueError(
                        f"REKLS V3 groups support only 2D tensors, got {bad_shapes}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if group["use_rekls"]:
                    self._rekls_step(parameter, group)
                else:
                    self._auxiliary_step(parameter, group)
        return loss

    def _rekls_step(self, parameter, group):
        if parameter.grad.is_sparse:
            raise RuntimeError("REKLS V3 does not support sparse gradients")
        gradient = parameter.grad.float()
        state = self.state[parameter]
        if len(state) == 0:
            _init_rekls_state(parameter, state)

        step = state["step"] + 1
        shampoo_beta = group["shampoo_beta"]
        shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**step)

        if state["step"] > 0:
            exp_avg = _project_out(state["exp_avg"], state)
        else:
            exp_avg = None

        with _fp32_matmul_precision("highest"):
            _update_kronecker_factors(
                gradient, state, shampoo_beta, group["eps"]
            )

        with _fp32_matmul_precision("high"):
            eigvals_left, basis_left = _eigh_with_fallback(state["L"])
            eigvals_right, basis_right = _eigh_with_fallback(state["R"])
            state["eigvals_L"], state["Q_L"] = eigvals_left, basis_left
            state["eigvals_R"], state["Q_R"] = eigvals_right, basis_right
            if exp_avg is not None:
                state["exp_avg"] = _project_in(exp_avg, state)

        if group["weight_decay"]:
            parameter.add_(
                parameter, alpha=-group["lr"] * group["weight_decay"]
            )

        with _fp32_matmul_precision("highest"):
            projected_gradient = _project_in(gradient, state)
            scalar_update = self._scalar_update(
                projected_gradient, state, group, step, auxiliary=False
            )
            update = _project_out(scalar_update, state)
        parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        state["step"] += 1

    @staticmethod
    def _scalar_update(gradient, state, group, step, auxiliary):
        betas = group["auxiliary_betas"] if auxiliary else group["betas"]
        if group["base_optimizer"] == "madam":
            scale_log2 = (
                group["auxiliary_scale_log2"] if auxiliary else group["scale_log2"]
            )
            return madam_update(
                gradient,
                state["exp_avg"],
                state["exp_avg_sq"],
                betas=betas,
                step=step,
                scale_log2=scale_log2,
                correct_bias=True,
            )
        epsilon = group["auxiliary_eps"] if auxiliary else group["eps"]
        return _calculate_adam_update(
            gradient,
            state["exp_avg"],
            state["exp_avg_sq"],
            betas,
            epsilon,
            step,
        )

    def _auxiliary_step(self, parameter, group):
        gradient = parameter.grad.float()
        if gradient.is_sparse:
            raise RuntimeError("REKLS V3 auxiliary optimizer does not support sparse gradients")
        state = self.state[parameter]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(gradient)
            state["exp_avg_sq"] = torch.zeros_like(gradient)
        state["step"] += 1
        if group["weight_decay"]:
            parameter.add_(
                parameter, alpha=-group["lr"] * group["weight_decay"]
            )
        update = self._scalar_update(
            gradient, state, group, state["step"], auxiliary=True
        )
        parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])


ReklsV3 = ReklsV3WithAuxAdam
ReklsV3WithAuxOptimizer = ReklsV3WithAuxAdam

__all__ = ["ReklsV3", "ReklsV3WithAuxAdam", "ReklsV3WithAuxOptimizer"]
