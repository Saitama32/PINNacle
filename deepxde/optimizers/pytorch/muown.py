# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA MuOwn with Adam/SOAP fallback for auxiliary parameters."""

import torch

from .muon import adam_update
from .rekls_v3 import _fp32_matmul_precision
from .soap import soap_step_parameter


_COEFFICIENT_SETS = {
    "simple": [(3.4445, -4.7750, 2.0315)],
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "cans": [
        (8.4703, -25.1081, 18.6293),
        (4.1828, -3.1087, 0.5806),
        (3.9619, -2.9541, 0.5630),
        (3.2866, -2.4647, 0.5074),
        (2.2737, -1.6447, 0.4162),
    ],
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
    "deepseekv4": [(3.4445, -4.7750, 2.0315)] * 8
    + [(2.0, -1.5, 0.5)] * 2,
    "cubic5": [
        (3.3656576, -3.3420992, 0.0),
        (2.5744352, -1.4957376, 0.0),
        (2.5368962, -1.4312570, 0.0),
        (2.4418906, -1.2764040, 0.0),
        (2.2230472, -0.9630650, 0.0),
    ],
}


def newton_schulz_muown(matrix, steps=5, coefficient_type="quintic"):
    """NVIDIA's float32 Newton--Schulz orthogonalization without Triton SYRK."""
    if matrix.ndim != 2:
        raise TypeError("MuOwn Newton-Schulz input must be 2D")
    if matrix.dtype != torch.float32:
        raise TypeError("MuOwn Newton-Schulz input must be float32")
    if coefficient_type not in _COEFFICIENT_SETS:
        raise ValueError(f"Unsupported coefficient type: {coefficient_type}")
    coefficients = _COEFFICIENT_SETS[coefficient_type]
    if coefficient_type == "cubic5" and steps != len(coefficients):
        raise ValueError("cubic5 requires exactly 5 Newton-Schulz steps")

    transpose = matrix.shape[0] > matrix.shape[1]
    x = matrix.mT if transpose else matrix
    x = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=1e-15)
    if torch.get_float32_matmul_precision() == "medium":
        x = x.bfloat16()

    repeat_last = coefficient_type in {"polar_express", "cans", "deepseekv4"}
    for index in range(steps):
        coefficient_index = min(index, len(coefficients) - 1) if repeat_last else index % len(coefficients)
        a, b, c = coefficients[coefficient_index]
        gram = x @ x.mT
        if c != 0.0:
            polynomial = torch.addmm(gram, gram, gram, alpha=c, beta=b)
            x = torch.addmm(x, polynomial, x, alpha=1.0, beta=a)
        else:
            x = torch.addmm(x, gram, x, alpha=b, beta=a)
    x = x.float()
    return x.mT if transpose else x


def _scale_factor(rows, columns, mode):
    if mode == "shape_scaling":
        return max(1, rows / columns) ** 0.5
    if mode == "spectral":
        return max(rows, columns) ** 0.5
    if mode == "unit_rms_norm":
        return (rows / columns) ** 0.5
    raise ValueError(f"Invalid MuOwn scale mode: {mode}")


def _weight_norm_decompose(weight, gradient, magnitude, direction_norm):
    unit_direction = weight / magnitude
    direction = unit_direction * direction_norm
    magnitude_gradient = (gradient * unit_direction).sum(dim=1, keepdim=True)
    direction_gradient = (magnitude / direction_norm) * (
        gradient - unit_direction * magnitude_gradient
    )
    return direction, magnitude_gradient, direction_gradient


class MuownWithAuxAdam(torch.optim.Optimizer):
    """MuOwn on selected matrices and Adam or SOAP on auxiliary parameters."""

    def __init__(self, param_groups):
        if not isinstance(param_groups, (list, tuple)) or not param_groups:
            raise ValueError("MuownWithAuxAdam expects non-empty parameter groups")
        prepared_groups = []
        for source_group in param_groups:
            if "use_muown" not in source_group:
                raise ValueError("Each MuOwn group must define use_muown")
            group = dict(source_group)
            params = list(group["params"])
            if group["use_muown"]:
                bad_shapes = [tuple(p.shape) for p in params if p.ndim != 2]
                if bad_shapes:
                    raise ValueError(f"MuOwn supports only 2D tensors, got {bad_shapes}")
                group.setdefault("lr", 3e-4)
                group.setdefault("momentum", 0.95)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("adam_eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("fp32_matmul_precision", "medium")
                group.setdefault("coefficient_type", "quintic")
                group.setdefault("ns_steps", 5)
                group.setdefault("scale_mode", "spectral")
                group.setdefault("extra_scale_factor", 1.0)
            else:
                group.setdefault("auxiliary_optimizer", "adam")
                if group["auxiliary_optimizer"] not in {"adam", "soap"}:
                    raise ValueError("MuOwn auxiliary optimizer must be 'adam' or 'soap'")
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
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
            raise ValueError("MuOwn has no trainable parameters")
        super().__init__(prepared_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MuOwn closure is not supported")
        for group in self.param_groups:
            if group["use_muown"]:
                self._muown_step(group)
            elif group["auxiliary_optimizer"] == "soap":
                self._soap_step(group)
            else:
                self._adam_step(group)
        return None

    def _muown_step(self, group):
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("MuOwn does not support sparse gradients")
            gradient = parameter.grad.float()
            if group["maximize"]:
                gradient = -gradient
            state = self.state[parameter]
            if len(state) == 0:
                row_norm = parameter.norm(dim=1, keepdim=True).float().clamp_min(1e-12)
                state.update(
                    step=0,
                    g=row_norm.clone(),
                    v_norm=row_norm.clone(),
                    momentum_buffer=torch.zeros_like(parameter, dtype=torch.float32),
                    m_g=torch.zeros_like(row_norm),
                    v_g=torch.zeros_like(row_norm),
                )

            step = state["step"] + 1
            direction, grad_g, grad_v = _weight_norm_decompose(
                parameter.float(), gradient, state["g"], state["v_norm"]
            )
            state["momentum_buffer"].lerp_(grad_v, 1 - group["momentum"])
            with _fp32_matmul_precision(group["fp32_matmul_precision"]):
                direction_update = newton_schulz_muown(
                    state["momentum_buffer"],
                    steps=group["ns_steps"],
                    coefficient_type=group["coefficient_type"],
                )
                direction_update.mul_(
                    _scale_factor(*parameter.shape, group["scale_mode"])
                    * group["extra_scale_factor"]
                )
            direction_new = direction.add(direction_update, alpha=-group["lr"])
            magnitude_update = adam_update(
                grad_g,
                state["m_g"],
                state["v_g"],
                step,
                group["betas"],
                group["adam_eps"],
            )
            state["g"].add_(magnitude_update, alpha=-group["lr"])
            if group["weight_decay"]:
                state["g"].mul_(1 - group["lr"] * group["weight_decay"])
            direction_norm_new = direction_new.norm(dim=1, keepdim=True)
            parameter.copy_(
                (state["g"] * direction_new / direction_norm_new).to(parameter.dtype)
            )
            state["v_norm"] = direction_norm_new
            state["step"] += 1

    def _soap_step(self, group):
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = -parameter.grad if group["maximize"] else parameter.grad
            soap_step_parameter(parameter, gradient, self.state[parameter], group)

    def _adam_step(self, group):
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if gradient.is_sparse:
                raise RuntimeError("MuOwn auxiliary Adam does not support sparse gradients")
            if group["maximize"]:
                gradient = -gradient
            state = self.state[parameter]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            update = adam_update(
                gradient,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                group["betas"],
                group["eps"],
            )
            if group["weight_decay"]:
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
            parameter.add_(update, alpha=-group["lr"])


Muown = MuownWithAuxAdam

__all__ = ["Muown", "MuownWithAuxAdam", "newton_schulz_muown"]
