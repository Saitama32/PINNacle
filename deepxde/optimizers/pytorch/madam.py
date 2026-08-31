# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Magnitude-aware Adam for the PyTorch backend.

The scalar update follows NVIDIA NeMo Emerging-Optimizers.  MAdam removes
Adam's epsilon from the denominator and stores a power-of-two scaled second
moment so that small float32 gradients do not underflow when squared.
"""

import torch


def madam_update(
    gradient,
    exp_avg,
    exp_avg_sq_scaled,
    *,
    betas,
    step,
    scale_log2=16.0,
    correct_bias=True,
):
    """Apply NVIDIA's magnitude-aware Adam update in place."""
    if scale_log2 // 2 != scale_log2 / 2:
        raise ValueError("scale_log2 must be an even integer")

    beta1, beta2 = betas
    gradient_scale = 2.0 ** (scale_log2 // 2)
    exp_avg.lerp_(gradient, 1.0 - beta1)
    exp_avg_sq_scaled.lerp_((gradient * gradient_scale).square(), 1.0 - beta2)

    correction1 = 1.0 - beta1**step if correct_bias else 1.0
    correction2 = 1.0 - beta2**step if correct_bias else 1.0
    momentum = exp_avg / correction1
    second_moment_scaled = exp_avg_sq_scaled / correction2
    update = momentum / second_moment_scaled.sqrt() * gradient_scale
    update.masked_fill_(exp_avg == 0, 0.0)
    return update


class MAdam(torch.optim.Optimizer):
    """Standalone magnitude-aware Adam with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        scale_log2=16.0,
        correct_bias=True,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("MAdam betas must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("MAdam weight decay must be nonnegative")
        if scale_log2 // 2 != scale_log2 / 2:
            raise ValueError("scale_log2 must be an even integer")
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            scale_log2=scale_log2,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

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
                gradient = parameter.grad.float()
                if gradient.is_sparse:
                    raise RuntimeError("MAdam does not support sparse gradients")
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
                update = madam_update(
                    gradient,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    betas=group["betas"],
                    step=state["step"],
                    scale_log2=group["scale_log2"],
                    correct_bias=group["correct_bias"],
                )
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss


__all__ = ["MAdam", "madam_update"]
