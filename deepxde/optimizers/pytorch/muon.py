"""Muon optimizer for the PyTorch backend.

Adapted from KellerJordan/Muon's public ``MuonWithAuxAdam`` implementation:
https://github.com/KellerJordan/Muon

Muon is intended for hidden weight matrices. Biases, external trainable
variables, and non-hidden parameters are handled by auxiliary Adam or SOAP.
"""

import torch

from .soap import soap_step_parameter


def zeropower_via_newtonschulz5(g, steps):
    """Approximate the zeroth power of a matrix with Newton-Schulz iterations."""
    if g.ndim < 2:
        raise ValueError("Muon expects matrix-like parameters.")

    x = g.bfloat16()
    transpose = g.size(-2) > g.size(-1)
    if transpose:
        x = x.mT

    a, b, c = 3.4445, -4.7750, 2.0315
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        xx_t = x @ x.mT
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transpose:
        x = x.mT
    return x


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for matrix groups plus Adam or SOAP for auxiliary groups.

    Parameter groups must set ``use_muon``. Groups with ``use_muon=True`` are
    updated by Muon and must contain only 2D tensors. Other groups select their
    update with ``auxiliary_optimizer``.
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
                params = sorted(params, key=lambda x: x.size(), reverse=True)
                bad_shapes = [tuple(p.shape) for p in params if p.ndim < 2]
                if bad_shapes:
                    raise ValueError(f"Muon parameter groups only support matrix-like tensors, got {bad_shapes}.")
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("auxiliary_optimizer", "adam")
                if group["auxiliary_optimizer"] not in {"adam", "soap"}:
                    raise ValueError("Muon auxiliary optimizer must be 'adam' or 'soap'.")
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
            elif group["auxiliary_optimizer"] == "soap":
                self._soap_step(group)
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

            update = muon_update(
                grad,
                state["momentum_buffer"],
                beta=momentum,
                ns_steps=ns_steps,
                nesterov=nesterov,
            )

            if weight_decay:
                p.mul_(1 - lr * weight_decay)
            p.add_(update.reshape(p.shape), alpha=-lr)

    def _soap_step(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = -p.grad if group["maximize"] else p.grad
            soap_step_parameter(p, grad, self.state[p], group)

    def _adam_step(self, group):
        lr = group["lr"]
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

            update = adam_update(grad, exp_avg, exp_avg_sq, state["step"], group["betas"], eps)
            if weight_decay:
                p.mul_(1 - lr * weight_decay)
            p.add_(update, alpha=-lr)
