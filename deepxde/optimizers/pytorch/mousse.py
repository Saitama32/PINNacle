"""Single-device Mousse optimizer for the PyTorch backend.

Adapted from the official MIT-licensed implementation:
https://github.com/Anti-Entrophic/Mousse/blob/main/dion/dion/mousse.py

The original optimizer includes DDP/FSDP2 communication and asynchronous task
machinery. This module retains its local matrix update exactly: Shampoo
curvature estimation, whitening, Newton--Schulz spectral projection,
unwhitening, norm grafting, and learning-rate scaling. Auxiliary parameters use
Lion, matching the official training configuration.
"""

import math

import torch


def zeropower_via_newton_schulz5(matrix, epsilon=1e-7):
    """Approximate the polar factor using Mousse's five NS5 iterations."""
    coefficients = (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    )
    work = matrix.to(torch.bfloat16)
    transposed = work.size(-2) > work.size(-1)
    if transposed:
        work = work.mT
    work = work / (work.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for a, b, c in coefficients:
        gram = work @ work.mT
        work = a * work + (b * gram + c * (gram @ gram)) @ work
    return work.mT if transposed else work


def _clean_eigenvalues(eigenvalues, epsilon):
    shift = torch.clamp(-eigenvalues.min(), min=0.0) + epsilon
    return eigenvalues + shift


def _adjusted_lr(lr, shape, mode):
    fan_out, fan_in = shape[-2:]
    if mode is None or mode == "None":
        return lr
    if mode == "spectral_norm":
        return lr * math.sqrt(fan_out / fan_in)
    if mode == "rms_norm":
        return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
    raise ValueError(f"Unknown Mousse adjust_lr mode: {mode}")


class MousseWithAuxLion(torch.optim.Optimizer):
    """Mousse for matrix groups and Lion for auxiliary groups.

    Parameter groups must provide ``algorithm='mousse'`` or
    ``algorithm='lion'``. The DeepXDE factory builds these groups
    automatically.
    """

    def __init__(
        self,
        params,
        lr=0.01,
        mu=0.95,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        epsilon=1e-8,
        nesterov=False,
        adjust_lr="spectral_norm",
        shampoo_epsilon=1e-10,
        shampoo_beta=0.95,
        shampoo_update_freq=10,
        shampoo_alpha=0.125,
        lr_correction=True,
        apply_norm=True,
        use_l_or_r=0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= mu < 1:
            raise ValueError(f"Invalid Mousse momentum: {mu}")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError(f"Invalid Lion betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", "None", None):
            raise ValueError(f"Invalid Mousse adjust_lr: {adjust_lr}")
        if shampoo_update_freq < 1:
            raise ValueError("shampoo_update_freq must be >= 1")
        if use_l_or_r not in (0, 1, 2):
            raise ValueError("use_l_or_r must be 0, 1, or 2")
        defaults = dict(
            lr=lr,
            mu=mu,
            betas=betas,
            weight_decay=weight_decay,
            epsilon=epsilon,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            shampoo_epsilon=shampoo_epsilon,
            shampoo_beta=shampoo_beta,
            shampoo_update_freq=shampoo_update_freq,
            shampoo_alpha=shampoo_alpha,
            lr_correction=lr_correction,
            apply_norm=apply_norm,
            use_l_or_r=use_l_or_r,
            algorithm="mousse",
        )
        super().__init__(params, defaults)

    @staticmethod
    def _initialize_mousse_state(parameter, state, use_l_or_r):
        rows, columns = parameter.shape
        state["momentum"] = torch.zeros_like(parameter)
        if use_l_or_r in (0, 1):
            state["L"] = torch.zeros(
                rows, rows, device=parameter.device, dtype=torch.float32
            )
            state["eig_L"] = (
                torch.zeros(rows, device=parameter.device),
                torch.eye(rows, device=parameter.device),
            )
        if use_l_or_r in (0, 2):
            state["R"] = torch.zeros(
                columns, columns, device=parameter.device, dtype=torch.float32
            )
            state["eig_R"] = (
                torch.zeros(columns, device=parameter.device),
                torch.eye(columns, device=parameter.device),
            )

    @staticmethod
    def _refresh_eigensystem(gradient, state, group, step):
        use_left = group["use_l_or_r"] in (0, 1)
        use_right = group["use_l_or_r"] in (0, 2)
        grad32 = gradient.float()
        beta = group["shampoo_beta"]
        if use_left:
            state["L"].mul_(beta).addmm_(
                grad32, grad32.T, beta=1.0, alpha=1.0 - beta
            )
        if use_right:
            state["R"].mul_(beta).addmm_(
                grad32.T, grad32, beta=1.0, alpha=1.0 - beta
            )
        frequency = group["shampoo_update_freq"]
        if step % frequency != 1 and frequency != 1:
            return

        correction = 1.0 - beta**step if group["lr_correction"] else 1.0
        epsilon = group["shampoo_epsilon"]
        if use_left:
            matrix = state["L"] / correction
            matrix = matrix * (matrix.shape[0] / matrix.trace())
            eigenvalues, eigenvectors = torch.linalg.eigh(
                matrix + epsilon * torch.eye(matrix.shape[0], device=matrix.device)
            )
            state["eig_L"] = (
                _clean_eigenvalues(eigenvalues, epsilon),
                eigenvectors,
            )
        if use_right:
            matrix = state["R"] / correction
            matrix = matrix * (matrix.shape[0] / matrix.trace())
            eigenvalues, eigenvectors = torch.linalg.eigh(
                matrix + epsilon * torch.eye(matrix.shape[0], device=matrix.device)
            )
            state["eig_R"] = (
                _clean_eigenvalues(eigenvalues, epsilon),
                eigenvectors,
            )

    @staticmethod
    def _mousse_direction(direction, state, group):
        original_dtype = direction.dtype
        direction = direction.float()
        use_left = group["use_l_or_r"] in (0, 1)
        use_right = group["use_l_or_r"] in (0, 2)
        alpha = group["shampoo_alpha"]
        if use_left:
            eigen_l, basis_l = state["eig_L"]
            scale_l = eigen_l.abs().pow(alpha)
            direction = basis_l.T @ direction
        if use_right:
            eigen_r, basis_r = state["eig_R"]
            scale_r = eigen_r.abs().pow(alpha)
            direction = direction @ basis_r
        if use_left:
            direction = direction / scale_l.unsqueeze(1)
        if use_right:
            direction = direction / scale_r.unsqueeze(0)

        direction = zeropower_via_newton_schulz5(
            direction, epsilon=group["epsilon"]
        )
        target_norm = direction.norm() if group["apply_norm"] else None
        if use_left:
            direction = direction / scale_l.unsqueeze(1)
        if use_right:
            direction = direction / scale_r.unsqueeze(0)
        if use_left:
            direction = basis_l @ direction
        if use_right:
            direction = direction @ basis_r.T
        if target_norm is not None:
            direction = direction * (target_norm / direction.norm().clamp_min(1e-30))
        return direction.to(original_dtype)

    def _mousse_step(self, parameter, group):
        gradient = parameter.grad
        if gradient.is_sparse:
            raise RuntimeError("Mousse does not support sparse gradients")
        state = self.state[parameter]
        if not state:
            self._initialize_mousse_state(parameter, state, group["use_l_or_r"])
            state["step"] = 0
        state["step"] += 1
        momentum = state["momentum"]
        momentum.mul_(group["mu"]).add_(gradient.to(momentum.dtype))
        direction = (
            momentum * group["mu"] + gradient.to(momentum.dtype)
            if group["nesterov"]
            else momentum
        )
        self._refresh_eigensystem(gradient, state, group, state["step"])
        direction = self._mousse_direction(direction.to(torch.bfloat16), state, group)
        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
        parameter.add_(
            direction.to(parameter.dtype),
            alpha=-_adjusted_lr(group["lr"], parameter.shape, group["adjust_lr"]),
        )

    def _lion_step(self, parameter, group):
        gradient = parameter.grad
        if gradient.is_sparse:
            raise RuntimeError("Mousse auxiliary Lion does not support sparse gradients")
        state = self.state[parameter]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(parameter)
        momentum = state["momentum"]
        beta1, beta2 = group["betas"]
        update = momentum.mul(beta1).add(gradient, alpha=1.0 - beta1).sign_()
        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
        parameter.add_(update, alpha=-group["lr"])
        momentum.mul_(beta2).add_(gradient, alpha=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            algorithm = group["algorithm"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if algorithm == "mousse":
                    self._mousse_step(parameter, group)
                elif algorithm == "lion":
                    self._lion_step(parameter, group)
                else:
                    raise ValueError(f"Unknown Mousse parameter algorithm: {algorithm}")
        return loss


__all__ = ["MousseWithAuxLion", "zeropower_via_newton_schulz5"]
