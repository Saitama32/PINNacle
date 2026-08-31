# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA KL-M-SOAP with an auxiliary MAdam path for non-matrix tensors."""

import torch

from .madam import madam_update
from .rekls_v3 import (
    _eigh_with_fallback,
    _fp32_matmul_precision,
    _init_rekls_state,
    _project_in,
    _project_out,
    _update_kronecker_factors,
)


def _orthogonal_iteration(matrix, basis):
    """One NVIDIA-style power iteration followed by QR and Rayleigh values."""
    refined_basis = torch.linalg.qr(matrix @ basis).Q
    eigenvalues = torch.diagonal(refined_basis.mT @ matrix @ refined_basis)
    return eigenvalues, refined_basis


class KlMSoapWithAuxMAdam(torch.optim.Optimizer):
    """KL-Shampoo eigenbasis with MAdam, plus MAdam for auxiliary tensors."""

    def __init__(
        self,
        params,
        lr,
        betas=(0.9, 0.95),
        shampoo_beta=0.95,
        eps=1e-8,
        weight_decay=0.01,
        scale_log2=16.0,
        auxiliary_betas=None,
        auxiliary_scale_log2=None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("KL-M-SOAP betas must be in [0, 1)")
        if not 0 <= shampoo_beta < 1:
            raise ValueError("KL-M-SOAP shampoo_beta must be in [0, 1)")
        if eps <= 0:
            raise ValueError("KL-M-SOAP epsilon must be positive")
        if weight_decay < 0:
            raise ValueError("KL-M-SOAP weight decay must be nonnegative")
        if scale_log2 // 2 != scale_log2 / 2:
            raise ValueError("KL-M-SOAP scale_log2 must be an even integer")
        if auxiliary_betas is None:
            auxiliary_betas = betas
        if auxiliary_scale_log2 is None:
            auxiliary_scale_log2 = scale_log2
        if len(auxiliary_betas) != 2 or any(
            not 0 <= beta < 1 for beta in auxiliary_betas
        ):
            raise ValueError("KL-M-SOAP auxiliary betas must be in [0, 1)")
        if auxiliary_scale_log2 // 2 != auxiliary_scale_log2 / 2:
            raise ValueError("auxiliary_scale_log2 must be an even integer")

        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            scale_log2=scale_log2,
            auxiliary_betas=auxiliary_betas,
            auxiliary_scale_log2=auxiliary_scale_log2,
            use_kl_m_soap=True,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            if "use_kl_m_soap" not in group:
                raise ValueError("Each KL-M-SOAP group must define use_kl_m_soap")
            if group["use_kl_m_soap"]:
                bad_shapes = [
                    tuple(parameter.shape)
                    for parameter in group["params"]
                    if parameter.ndim != 2
                ]
                if bad_shapes:
                    raise ValueError(
                        f"KL-M-SOAP groups support only 2D tensors, got {bad_shapes}"
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
                if group["use_kl_m_soap"]:
                    self._matrix_step(parameter, group)
                else:
                    self._auxiliary_step(parameter, group)
        return loss

    def _matrix_step(self, parameter, group):
        if parameter.grad.is_sparse:
            raise RuntimeError("KL-M-SOAP does not support sparse gradients")
        gradient = parameter.grad.float()
        state = self.state[parameter]
        if len(state) == 0:
            _init_rekls_state(parameter, state)

        step = state["step"] + 1
        shampoo_beta = group["shampoo_beta"]
        shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**step)

        if state["step"] == 0:
            with _fp32_matmul_precision("highest"):
                _update_kronecker_factors(
                    gradient, state, shampoo_beta, group["eps"]
                )
            with _fp32_matmul_precision("high"):
                state["eigvals_L"], state["Q_L"] = _eigh_with_fallback(state["L"])
                state["eigvals_R"], state["Q_R"] = _eigh_with_fallback(state["R"])
        else:
            exp_avg = _project_out(state["exp_avg"], state)
            with _fp32_matmul_precision("highest"):
                _update_kronecker_factors(
                    gradient, state, shampoo_beta, group["eps"]
                )
            with _fp32_matmul_precision("high"):
                state["eigvals_L"], state["Q_L"] = _orthogonal_iteration(
                    state["L"], state["Q_L"]
                )
                state["eigvals_R"], state["Q_R"] = _orthogonal_iteration(
                    state["R"], state["Q_R"]
                )
                state["exp_avg"] = _project_in(exp_avg, state)

        if group["weight_decay"]:
            parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])
        with _fp32_matmul_precision("highest"):
            projected_gradient = _project_in(gradient, state)
            scalar_update = madam_update(
                projected_gradient,
                state["exp_avg"],
                state["exp_avg_sq"],
                betas=group["betas"],
                step=step,
                scale_log2=group["scale_log2"],
                correct_bias=True,
            )
            update = _project_out(scalar_update, state)
        parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        state["step"] += 1

    def _auxiliary_step(self, parameter, group):
        gradient = parameter.grad.float()
        if gradient.is_sparse:
            raise RuntimeError("KL-M-SOAP auxiliary MAdam does not support sparse gradients")
        state = self.state[parameter]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(gradient)
            state["exp_avg_sq"] = torch.zeros_like(gradient)
        state["step"] += 1
        if group["weight_decay"]:
            parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])
        update = madam_update(
            gradient,
            state["exp_avg"],
            state["exp_avg_sq"],
            betas=group["auxiliary_betas"],
            step=state["step"],
            scale_log2=group["auxiliary_scale_log2"],
            correct_bias=True,
        )
        parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])


KlMSoap = KlMSoapWithAuxMAdam

__all__ = ["KlMSoap", "KlMSoapWithAuxMAdam"]
