# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""MOP optimizer for the PyTorch backend.

The matrix update follows NVIDIA NeMo Emerging-Optimizers' MOP implementation:
momentum is orthogonalized by an exact polar decomposition computed with SVD,
then scaled by the momentum's nuclear norm by default.  As with the local Muon
integration, non-matrix parameters are handled by auxiliary Adam or SOAP.
"""

import torch

from .muon import adam_update
from .soap import soap_step_parameter


_SCALE_MODES = {"nuclear_norm", "shape_scaling", "spectral", "unit_rms_norm"}


def polar_via_svd(A, return_p=False):
    """Compute the polar decomposition via SVD, matching NVIDIA's MOP."""
    U_svd, S, Vh = torch.linalg.svd(A, full_matrices=False)
    U_polar = U_svd @ Vh
    if not return_p:
        return U_polar, None, S
    p = Vh.mH @ torch.diag(S) @ Vh
    return U_polar, p, S


def _get_scale_factor(grad, singular_values, scale_mode):
    if scale_mode == "nuclear_norm":
        return singular_values.sum()
    size_out, size_in = grad.size(-2), grad.size(-1)
    if scale_mode == "shape_scaling":
        return max(1, size_out / size_in) ** 0.5
    if scale_mode == "spectral":
        return max(size_out, size_in) ** 0.5
    if scale_mode == "unit_rms_norm":
        return (size_out / size_in) ** 0.5
    raise ValueError(f"Invalid MOP scale mode: {scale_mode}")


def mop_update(
    grad,
    momentum,
    beta=0.95,
    nesterov=False,
    scale_mode="nuclear_norm",
    extra_scale_factor=1.0,
):
    """Return NVIDIA MOP's momentum/polar-decomposition matrix update."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    orth_grad, _, singular_values = polar_via_svd(update, False)
    scale_factor = _get_scale_factor(update, singular_values, scale_mode)
    return orth_grad * scale_factor * extra_scale_factor


class MOPWithAuxAdam(torch.optim.Optimizer):
    """MOP for matrix groups plus Adam or SOAP for auxiliary groups.

    Parameter groups must set ``use_mop``. Groups with ``use_mop=True`` must
    contain only 2D tensors; all other groups select the auxiliary algorithm
    with ``auxiliary_optimizer``.
    """

    def __init__(self, param_groups):
        if not isinstance(param_groups, (list, tuple)):
            raise TypeError("MOPWithAuxAdam expects a list of parameter groups.")
        if not param_groups:
            raise ValueError("MOPWithAuxAdam got an empty parameter group list.")

        prepared_groups = []
        for source_group in param_groups:
            if "use_mop" not in source_group:
                raise ValueError("Each MOPWithAuxAdam parameter group must define use_mop.")
            group = dict(source_group)
            params = list(group["params"])
            if group["use_mop"]:
                params = sorted(params, key=lambda x: x.size(), reverse=True)
                bad_shapes = [tuple(p.shape) for p in params if p.ndim != 2]
                if bad_shapes:
                    raise ValueError(f"MOP parameter groups only support 2D tensors, got {bad_shapes}.")
                group.setdefault("lr", 3e-4)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", False)
                group.setdefault("scale_mode", "nuclear_norm")
                group.setdefault("extra_scale_factor", 1.0)
                group.setdefault("weight_decay", 0.01)
                if group["scale_mode"] not in _SCALE_MODES:
                    raise ValueError(f"Invalid MOP scale mode: {group['scale_mode']}")
            else:
                group.setdefault("auxiliary_optimizer", "adam")
                if group["auxiliary_optimizer"] not in {"adam", "soap"}:
                    raise ValueError("MOP auxiliary optimizer must be 'adam' or 'soap'.")
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("shampoo_beta", 0.999)
                group.setdefault("precondition_frequency", 10)
                group.setdefault("max_precondition_dim", 4096)
                group.setdefault("bias_correction", True)
            group.setdefault("maximize", False)
            group["params"] = params
            if params:
                prepared_groups.append(group)

        if not prepared_groups:
            raise ValueError("MOPWithAuxAdam has no trainable parameters.")
        super().__init__(prepared_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_mop"]:
                self._mop_step(group)
            elif group["auxiliary_optimizer"] == "soap":
                self._soap_step(group)
            else:
                self._adam_step(group)
        return loss

    def _mop_step(self, group):
        lr = group["lr"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("MOP does not support sparse gradients.")
            if group["maximize"]:
                grad = -grad

            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
            update = mop_update(
                grad,
                state["momentum_buffer"],
                beta=group["momentum"],
                nesterov=group["nesterov"],
                scale_mode=group["scale_mode"],
                extra_scale_factor=group["extra_scale_factor"],
            )
            if group["weight_decay"]:
                p.mul_(1 - lr * group["weight_decay"])
            p.add_(update, alpha=-lr)

    def _soap_step(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = -p.grad if group["maximize"] else p.grad
            soap_step_parameter(p, grad, self.state[p], group)

    def _adam_step(self, group):
        lr = group["lr"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("MOP auxiliary Adam does not support sparse gradients.")
            if group["maximize"]:
                grad = -grad

            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            update = adam_update(
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                group["betas"],
                group["eps"],
            )
            if group["weight_decay"]:
                p.mul_(1 - lr * group["weight_decay"])
            p.add_(update, alpha=-lr)


__all__ = ["MOPWithAuxAdam", "mop_update", "polar_via_svd"]
