"""PSGDPro optimizer for the PyTorch backend.

Adapted from NVIDIA NeMo Emerging-Optimizers (Apache-2.0):
https://github.com/NVIDIA-NeMo/Emerging-Optimizers/tree/main/emerging_optimizers/psgd
"""

import math

import torch


def _partial_contraction(first, second, axis):
    dimensions = [index for index in range(first.dim()) if index != axis]
    return torch.tensordot(first, second, dims=(dimensions, dimensions))


def _apply_single_factor(factors, tensor, axis):
    factor = factors[axis]
    if factor.dim() == 1:
        shape = [1] * tensor.dim()
        shape[axis] = factor.size(0)
        return tensor * factor.view(shape)
    result = torch.tensordot(factor, tensor, dims=([1], [axis]))
    order = list(range(1, axis + 1)) + [0] + list(
        range(axis + 1, tensor.dim())
    )
    return result.permute(order)


def apply_kronecker_factors(factors, tensor):
    if len(factors) != tensor.dim():
        raise ValueError("The number of PSGD factors must match the tensor rank")
    result = tensor
    for axis in range(len(factors)):
        result = _apply_single_factor(factors, result, axis)
    return result


def apply_preconditioner(factors, tensor):
    result = apply_kronecker_factors(factors, tensor)
    transposed = [factor if factor.dim() == 1 else factor.T for factor in factors]
    return apply_kronecker_factors(transposed, result)


def _subspace_iteration_bound(matrix, k=32, half_iters=2, eps=1e-8):
    vectors = torch.randn(
        k, matrix.shape[1], dtype=matrix.dtype, device=matrix.device
    )
    row_index = torch.argmax(torch.linalg.vector_norm(matrix, dim=1))
    dominant_row = matrix[row_index]
    alignment = torch.sign(torch.sum(dominant_row * vectors, dim=1, keepdim=True))
    vectors = dominant_row + alignment * vectors
    for _ in range(half_iters):
        vectors = vectors @ matrix
        vectors /= torch.linalg.vector_norm(
            vectors, dim=1, keepdim=True
        ) + eps
        vectors = vectors @ matrix
    return torch.amax(torch.linalg.vector_norm(vectors, dim=1))


def _norm_lower_bound_spd(matrix, eps=1e-8):
    scale = torch.clamp(matrix.diagonal().amax(), min=eps)
    return scale * _subspace_iteration_bound(matrix / scale, eps=eps)


def _norm_lower_bound_skew(matrix, eps=1e-8):
    scale = torch.clamp(matrix.abs().amax(), min=eps)
    return scale * _subspace_iteration_bound(matrix / scale, eps=eps)


def _procrustes_step(factor, max_step_size=0.125, eps=1e-8):
    generator = factor.T - factor
    generator /= torch.clamp(_norm_lower_bound_skew(generator), min=eps)
    first = generator @ factor
    trace_first = torch.trace(first)
    second = generator @ first
    trace_second = torch.trace(second)
    candidate = torch.clamp(
        -trace_first / trace_second, min=0.0, max=max_step_size
    )
    step_size = torch.where(
        trace_second < 0, candidate, factor.new_tensor(max_step_size)
    ).item()
    return torch.add(
        factor,
        torch.add(first, second, alpha=0.5 * step_size),
        alpha=step_size,
    )


def _uniformize_factors(factors):
    if len(factors) <= 1:
        return
    norms = [factor.abs().amax() for factor in factors]
    geometric_mean = torch.prod(torch.stack(norms)) ** (1.0 / len(factors))
    for factor, norm in zip(factors, norms):
        factor.mul_(geometric_mean / norm)


def _initialize_psgd_state(gradient, scale):
    factors = [
        torch.eye(size, device=gradient.device, dtype=torch.float32) * scale
        for size in gradient.shape
    ]
    lipschitz = [torch.ones((), device=gradient.device) for _ in gradient.shape]
    return factors, lipschitz


def _update_preconditioner(
    factors,
    lipschitz,
    momentum,
    damping_noise_scale,
    precond_lr,
    beta_lip,
):
    momentum = momentum.float()
    dampened = momentum + (
        damping_noise_scale + 1e-7 * momentum.abs()
    ) * torch.randn_like(momentum)
    preconditioned = apply_preconditioner(factors, dampened)
    total_numel = preconditioned.numel()
    new_factors = []
    new_lipschitz = []
    for axis, factor in enumerate(factors):
        covariance = _partial_contraction(
            preconditioned, preconditioned, axis
        )
        normalization = total_numel / factor.shape[0]
        bound = _norm_lower_bound_spd(covariance) + normalization
        lip = torch.maximum(
            beta_lip * lipschitz[axis] + (1.0 - beta_lip) * bound,
            bound,
        )
        factor = factor - precond_lr / lip * (
            covariance @ factor - normalization * factor
        )
        new_factors.append(_procrustes_step(factor))
        new_lipschitz.append(lip)
    return new_factors, new_lipschitz


def _scheduled_precond_lr(initial, step, minimum, warmup_steps):
    return max(initial / math.sqrt(1.0 + step / warmup_steps), minimum)


class PSGDPro(torch.optim.Optimizer):
    """PSGD-Kron-Whiten using NVIDIA's Procrustes preconditioner step."""

    def __init__(
        self,
        params,
        lr=3e-3,
        weight_decay=0.01,
        momentum=0.9,
        weight_decay_method="decoupled",
        beta_lip=0.9,
        precond_lr=0.1,
        precond_init_scale=1.0,
        damping_noise_scale=0.1,
        min_precond_lr=0.01,
        warmup_steps=10000,
        max_update_rms=0.0,
        auxiliary_betas=(0.9, 0.999),
        auxiliary_eps=1e-8,
    ):
        if lr < 0 or weight_decay < 0:
            raise ValueError("PSGDPro learning rate and weight decay must be non-negative")
        if not 0 <= momentum < 1 or not 0 <= beta_lip < 1:
            raise ValueError("PSGDPro momentum and beta_lip must be in [0, 1)")
        if weight_decay_method not in ("decoupled", "independent", "l2", "palm"):
            raise ValueError(f"Invalid weight decay method: {weight_decay_method}")
        self.weight_decay_method = weight_decay_method
        self.precond_init_scale = precond_init_scale
        self.damping_noise_scale = damping_noise_scale
        self.warmup_steps = warmup_steps
        self.max_update_rms = max_update_rms
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta_lip=beta_lip,
            precond_lr=precond_lr,
            min_precond_lr=min_precond_lr,
            use_psgd=True,
            auxiliary_betas=auxiliary_betas,
            auxiliary_eps=auxiliary_eps,
        )
        super().__init__(params, defaults)

    def _apply_weight_decay(self, parameter, gradient, group):
        decay = group["weight_decay"]
        if decay == 0:
            return
        lr = group["lr"]
        if self.weight_decay_method == "decoupled":
            parameter.add_(parameter, alpha=-decay * lr)
        elif self.weight_decay_method == "independent":
            parameter.add_(parameter, alpha=-decay)
        elif self.weight_decay_method == "l2":
            gradient.add_(parameter, alpha=decay)
        else:
            parameter.add_(parameter, alpha=-decay * lr * lr)

    def _psgd_step(self, parameter, group):
        gradient = parameter.grad
        if gradient.is_sparse:
            raise RuntimeError("PSGDPro does not support sparse gradients")
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(gradient)
            state["Q"], state["L"] = _initialize_psgd_state(
                gradient, self.precond_init_scale
            )
        self._apply_weight_decay(parameter, gradient, group)
        momentum = state["exp_avg"]
        momentum.lerp_(gradient, 1.0 - group["momentum"])
        precond_lr = _scheduled_precond_lr(
            group["precond_lr"],
            state["step"],
            group["min_precond_lr"],
            self.warmup_steps,
        )
        state["Q"], state["L"] = _update_preconditioner(
            state["Q"],
            state["L"],
            momentum,
            self.damping_noise_scale,
            precond_lr,
            group["beta_lip"],
        )
        _uniformize_factors(state["Q"])
        update = apply_preconditioner(state["Q"], momentum.float())
        if self.max_update_rms > 0:
            rms = update.square().mean().sqrt()
            update.div_(torch.clamp(rms / self.max_update_rms, min=1.0))
        parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        state["step"] += 1

    def _auxiliary_step(self, parameter, group):
        gradient = parameter.grad
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        state["step"] += 1
        beta1, beta2 = group["auxiliary_betas"]
        state["exp_avg"].lerp_(gradient, 1.0 - beta1)
        state["exp_avg_sq"].lerp_(gradient.square(), 1.0 - beta2)
        correction1 = 1.0 - beta1 ** state["step"]
        correction2 = 1.0 - beta2 ** state["step"]
        step_size = group["lr"] * correction2**0.5 / correction1
        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
        update = state["exp_avg"] / state["exp_avg_sq"].sqrt().add_(
            group["auxiliary_eps"]
        )
        parameter.add_(update, alpha=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("PSGDPro does not support closures")
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if group["use_psgd"]:
                    self._psgd_step(parameter, group)
                else:
                    self._auxiliary_step(parameter, group)
        return None


__all__ = ["PSGDPro", "apply_kronecker_factors", "apply_preconditioner"]
