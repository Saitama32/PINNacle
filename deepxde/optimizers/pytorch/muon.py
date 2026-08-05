"""Muon optimizer for the PyTorch backend.

Adapted from KellerJordan/Muon's public ``MuonWithAuxAdam`` implementation:
https://github.com/KellerJordan/Muon

Muon is intended for hidden weight matrices. Biases, external trainable
variables, and non-hidden parameters should be handled by the auxiliary Adam
branch.
"""

import math

import torch


def zeropower_via_newtonschulz5(g, steps=5, eps=1e-7):
    """Approximate the zeroth power of a matrix with Newton-Schulz iterations."""
    if g.ndim != 2:
        raise ValueError("Muon expects 2D matrix parameters.")

    dtype = g.dtype
    x = g.float()
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.T

    x = x / (x.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transpose:
        x = x.T
    return x.to(dtype=dtype)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for matrix groups plus AdamW-style updates for auxiliary groups.

    Parameter groups must set ``use_muon``. Groups with ``use_muon=True`` are
    updated by Muon and must contain only 2D tensors. Other groups are updated
    by the auxiliary Adam branch.
    """

    def __init__(self, param_groups):
        if not isinstance(param_groups, (list, tuple)):
            raise TypeError("MuonWithAuxAdam expects a list of parameter groups.")
        if not param_groups:
            raise ValueError("MuonWithAuxAdam got an empty parameter group list.")

        defaults = {}
        prepared_groups = []
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each MuonWithAuxAdam parameter group must define use_muon.")
            group = dict(group)
            params = list(group["params"])
            if group["use_muon"]:
                bad_shapes = [tuple(p.shape) for p in params if p.ndim != 2]
                if bad_shapes:
                    raise ValueError(f"Muon parameter groups only support 2D tensors, got {bad_shapes}.")
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 1e-3)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
            group.setdefault("maximize", False)
            group["params"] = params
            if params:
                prepared_groups.append(group)

        if not prepared_groups:
            raise ValueError("MuonWithAuxAdam has no trainable parameters.")
        super().__init__(prepared_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)
        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        weight_decay = group["weight_decay"]
        maximize = group["maximize"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients.")
            if maximize:
                grad = -grad

            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            update = grad.add(buf, alpha=momentum) if nesterov else buf
            update = zeropower_via_newtonschulz5(update, steps=ns_steps)
            update.mul_(max(1.0, p.size(0) / p.size(1)) ** 0.5)

            if weight_decay:
                p.mul_(1 - lr * weight_decay)
            p.add_(update, alpha=-lr)

    def _adam_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        maximize = group["maximize"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon auxiliary Adam does not support sparse gradients.")
            if maximize:
                grad = -grad

            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            if weight_decay:
                p.mul_(1 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)
