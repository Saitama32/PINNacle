# Copyright 2025 Tim Tsz-Kit Lau.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# This module ports PolarGrad from https://github.com/timlautk/polargrad and
# adds only an auxiliary Adam branch for parameters that are not matrices.

import math
from itertools import repeat

import torch

from .soap import soap_step_parameter


_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
_POLAR_EXPRESS_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for a, b, c in _POLAR_EXPRESS_COEFFS[:-1]
] + [_POLAR_EXPRESS_COEFFS[-1]]


def _hermitian_factor(unitary, matrix):
    matrix = matrix.type_as(unitary)
    if matrix.size(-2) >= matrix.size(-1):
        factor = unitary.mH @ matrix
    else:
        factor = matrix @ unitary.mH
    return (factor + factor.mH) / 2


def _newton_schulz(
    matrix, max_iterations=5, a=3.4445, b=-4.7750, c=2.0315
):
    """Quintic Newton--Schulz routine used by upstream PolarGrad."""
    if matrix.is_complex():
        raise TypeError("Newton--Schulz currently supports real matrices only.")
    x = matrix.bfloat16()
    transposed = matrix.size(-2) > matrix.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(max_iterations):
        gram = x @ x.mT
        poly = b * gram + c * gram @ gram
        x = a * x + poly @ x
    return x.mT if transposed else x


def _preconditioned_newton_schulz(matrix):
    """Preconditioned Newton--Schulz routine from the PolarGrad repository."""
    if matrix.is_complex():
        raise TypeError("Preconditioned Newton--Schulz supports real matrices only.")
    x = matrix.bfloat16()
    transposed = matrix.size(-2) > matrix.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    sigma = torch.finfo(x.dtype).eps
    sigma_target = 0.1
    coefficient = 1.5 * math.sqrt(3) - sigma_target
    while sigma < sigma_target:
        sigma = coefficient * sigma * (
            1 - 4 / 27 * coefficient**2 * sigma**2
        )
        x = (
            coefficient * x
            - 4 / 27 * coefficient**3 * x @ x.mT @ x
        )

    delta = torch.tensor(float("inf"), device=x.device)
    tolerance = max(matrix.shape) * torch.finfo(x.dtype).eps
    while delta > tolerance:
        x_new = 1.5 * x - 0.5 * x @ x.mT @ x
        delta = torch.linalg.matrix_norm(x_new - x)
        x = x_new
    return x.mT if transposed else x


def _polar_express(matrix, max_iterations=5):
    """Polar Express polynomial sequence used by upstream PolarGrad."""
    if matrix.is_complex():
        raise TypeError("Polar Express currently supports real matrices only.")
    x = matrix.bfloat16()
    transposed = matrix.size(-2) > matrix.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    coefficients = _POLAR_EXPRESS_COEFFS[:max_iterations] + list(
        repeat(
            _POLAR_EXPRESS_COEFFS[-1],
            max(0, max_iterations - len(_POLAR_EXPRESS_COEFFS)),
        )
    )
    for a, b, c in coefficients:
        gram = x @ x.mT
        poly = b * gram + c * gram @ gram
        x = a * x + poly @ x
    return x.mT if transposed else x


def _qdwh_tall(matrix, max_iterations=5, eps=None):
    """QDWH for a tall matrix, matching PolarGrad's unpadded QDWH path."""
    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError("QDWH tall path requires rows >= columns.")
    if matrix.dtype in (torch.float16, torch.bfloat16):
        work = matrix.float()
    else:
        work = matrix
    if eps is None:
        eps = float(torch.finfo(work.dtype).eps)

    one_norm = torch.linalg.norm(work, ord=1)
    inf_norm = torch.linalg.norm(work, ord=float("inf"))
    alpha_inverse = 1 / torch.sqrt(one_norm * inf_norm)
    alpha_inverse = torch.where(one_norm == 0, 1.0, alpha_inverse)
    unitary = work * alpha_inverse

    lower_bound = eps
    tolerance_lower = 5.0 * eps
    tolerance_norm = tolerance_lower ** (1 / 3)
    qr_coefficients = []
    chol_coefficients = []
    iterations = 0
    while lower_bound + tolerance_lower < 1 and iterations < max_iterations:
        iterations += 1
        lower_sq = lower_bound * lower_bound
        delta = (4 * (1 / lower_sq - 1) / lower_sq) ** (1 / 3)
        sqrt_delta = math.sqrt(1 + delta)
        a = sqrt_delta + math.sqrt(
            2 - delta + 2 * (2 - lower_sq) / (lower_sq * sqrt_delta)
        )
        b = (a - 1) ** 2 / 4
        c = a + b - 1
        lower_bound = lower_bound * (a + b * lower_sq) / (1 + c * lower_sq)
        e = b / c
        if c > 100:
            qr_coefficients.append(((a - e) / math.sqrt(c), math.sqrt(c), e))
        else:
            chol_coefficients.append((a - e, c, e))

    rows, columns = unitary.shape
    identity = torch.eye(columns, dtype=unitary.dtype, device=unitary.device)

    for coefficient, sqrt_c, e in qr_coefficients:
        q, _ = torch.linalg.qr(
            torch.cat((sqrt_c * unitary, identity), dim=0), mode="reduced"
        )
        q_top = q[:rows, :]
        q_bottom = q[rows : rows + columns, :].mH
        unitary = e * unitary + coefficient * (q_top @ q_bottom)

    not_converged = True
    for coefficient, c, e in chol_coefficients:
        previous = unitary
        system = c * (unitary.mH @ unitary) + identity
        factor, info = torch.linalg.cholesky_ex(system, upper=False)
        if not bool(torch.all(info == 0)):
            raise RuntimeError("QDWH Cholesky factorization failed.")
        solved = torch.cholesky_solve(unitary.mH, factor, upper=False).mH
        unitary = e * unitary + coefficient * solved
        not_converged = bool(
            torch.linalg.matrix_norm(unitary - previous) > tolerance_norm
        )

    completed = len(qr_coefficients) + len(chol_coefficients)
    while not_converged and completed < max_iterations:
        previous = unitary
        system = 3 * (unitary.mH @ unitary) + identity
        factor, info = torch.linalg.cholesky_ex(system, upper=False)
        if not bool(torch.all(info == 0)):
            raise RuntimeError("QDWH Cholesky factorization failed.")
        solved = torch.cholesky_solve(unitary.mH, factor, upper=False).mH
        unitary = unitary / 3 + (8 / 3) * solved
        not_converged = bool(
            torch.linalg.matrix_norm(unitary - previous) > tolerance_norm
        )
        completed += 1

    unitary = 1.5 * unitary - 0.5 * unitary @ (unitary.mH @ unitary)
    return unitary.to(matrix.dtype)


def _qdwh(matrix, max_iterations=5, eps=None):
    if matrix.shape[0] >= matrix.shape[1]:
        return _qdwh_tall(matrix, max_iterations=max_iterations, eps=eps)
    return _qdwh_tall(
        matrix.mH, max_iterations=max_iterations, eps=eps
    ).mH


def _elliptic_k(angle, tolerance=None):
    if tolerance is None:
        tolerance = torch.finfo(torch.float64).eps
    a, b = 1.0, math.cos(angle)
    total = math.sin(angle) ** 2
    iteration = 0
    error = 1.0
    while error > tolerance:
        a_new = 0.5 * (a + b)
        b_new = math.sqrt(a * b)
        c_new = 0.5 * (a - b)
        iteration += 1
        error = 2**iteration * c_new**2
        total += error
        a, b = a_new, b_new
    return math.pi / (2 * a)


def _elliptic_jacobi(u, angle, tolerance=None):
    if tolerance is None:
        tolerance = torch.finfo(torch.float64).eps
    a_values = [1.0]
    b_values = [math.cos(angle)]
    c_values = [math.sin(angle)]
    while abs(c_values[-1]) > tolerance and len(c_values) < 1001:
        a_values.append(0.5 * (a_values[-1] + b_values[-1]))
        b_values.append(math.sqrt(a_values[-2] * b_values[-1]))
        c_values.append(0.5 * (a_values[-2] - b_values[-2]))
    phi = 2 ** (len(c_values) - 1) * a_values[-1] * u
    for index in range(len(c_values) - 2, -1, -1):
        value = c_values[index + 1] * math.sin(phi) / a_values[index + 1]
        phi = 0.5 * (math.asin(max(-1.0, min(1.0, value))) + phi)
    return math.sin(phi), math.cos(phi)


def _zolo_degree(condition):
    limits = ((1.001, 2), (1.01, 3), (1.1, 4), (1.2, 5), (1.5, 6), (2, 8),
              (6.5, 2), (180, 3), (1.5e4, 4), (2e6, 5), (1e9, 6),
              (3e12, 7))
    for limit, degree in limits:
        if condition < limit:
            return degree
    return 8


def _zolo_pd(matrix):
    """ZOLO-PD rational iteration, ported from upstream's default path."""
    if matrix.is_complex():
        raise TypeError("ZOLO-PD currently supports real matrices only.")
    original_dtype = matrix.dtype
    source = matrix.double()
    alpha = torch.linalg.matrix_norm(source, ord=2)
    if alpha == 0:
        return torch.zeros_like(matrix)
    unitary = source / alpha
    reduced = torch.linalg.qr(unitary, mode="reduced").R if source.shape[0] > source.shape[1] else unitary
    condition_one = torch.linalg.cond(reduced, p=1)
    lower_bound = torch.linalg.matrix_norm(reduced, ord=1) / condition_one
    lower_bound = float(lower_bound / math.sqrt(source.shape[1]))
    unitary = unitary / lower_bound
    condition = 1 / lower_bound

    for iteration in range(1 if condition < 2 else 2):
        angle = math.acos(1 / condition)
        elliptic_k = _elliptic_k(angle)
        degree = _zolo_degree(condition)
        coefficients = []
        for index in range(2 * degree):
            sn, cn = _elliptic_jacobi(
                (index + 1) * elliptic_k / (2 * degree + 1), angle
            )
            coefficients.append(sn**2 / cn**2)

        def rational(value):
            result = value
            for index in range(degree):
                result *= (value**2 + coefficients[2 * index + 1]) / (
                    value**2 + coefficients[2 * index]
                )
            return result

        correction = unitary.clone()
        identity = torch.eye(source.shape[1], dtype=source.dtype, device=source.device)
        for index in range(degree):
            numerator = math.prod(
                coefficients[2 * index] - coefficients[2 * other + 1]
                for other in range(degree)
            )
            denominator = math.prod(
                coefficients[2 * index] - coefficients[2 * other]
                for other in range(degree) if other != index
            )
            coefficient = coefficients[2 * index]
            if iteration == 0 and max(coefficients[:-1]) > 1e2:
                root = math.sqrt(coefficient)
                q, _ = torch.linalg.qr(
                    torch.cat((unitary, root * identity), dim=0), mode="reduced"
                )
                solved = (q[: source.shape[0]] @ q[source.shape[0] :].mH) / root
            else:
                system = unitary.mH @ unitary + coefficient * identity
                factor = torch.linalg.cholesky(system)
                solved = torch.cholesky_solve(unitary.mH, factor).mH
            correction -= (numerator / denominator) * solved
        unitary = correction / rational(1)
        condition = max(rational(condition) / rational(1), 1)
    unitary = 1.5 * unitary - 0.5 * unitary @ (unitary.mH @ unitary)
    return unitary.to(original_dtype)


def polar(matrix, method="qdwh", max_iterations=5, eps=None, ns_coeffs=(3.4445, -4.7750, 2.0315)):
    """Return the unitary polar factor using a PolarGrad repository method."""
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(f"Polar decomposition expects a nonempty matrix, got {tuple(matrix.shape)}.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative.")
    if method == "qdwh":
        return _qdwh(matrix, max_iterations=max_iterations, eps=eps)
    if method == "zolo-pd":
        transposed = matrix.shape[0] < matrix.shape[1]
        result = _zolo_pd(matrix.mH if transposed else matrix)
        return result.mH if transposed else result
    if method == "ns":
        return _newton_schulz(matrix, max_iterations, *ns_coeffs)
    if method == "precond_ns":
        return _preconditioned_newton_schulz(matrix)
    if method == "polar_express":
        return _polar_express(matrix, max_iterations)
    raise ValueError(f"Unknown polar decomposition method {method!r}.")


class PolarGradWithAuxAdam(torch.optim.Optimizer):
    """Upstream PolarGrad plus Adam or SOAP for auxiliary parameters."""

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        polar_first=False,
        method="qdwh",
        inner_steps=2,
        a=3.4445,
        b=-4.7750,
        c=2.031,
        betas=(0.9, 0.95),
        eps=1e-10,
    ):
        if lr < 0 or weight_decay < 0:
            raise ValueError("Learning rate and weight decay must be nonnegative.")
        if not 0 <= momentum < 1:
            raise ValueError("PolarGrad momentum must be in [0, 1).")
        if method not in {"qdwh", "zolo-pd", "ns", "precond_ns", "polar_express"}:
            raise ValueError(f"Unknown PolarGrad method {method!r}.")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            polar_first=polar_first,
            method=method,
            inner_steps=inner_steps,
            a=a,
            b=b,
            c=c,
            betas=betas,
            eps=eps,
            use_polargrad=True,
            auxiliary_optimizer="adam",
            shampoo_beta=0.999,
            precondition_frequency=10,
            max_precondition_dim=4096,
            bias_correction=True,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["auxiliary_optimizer"] not in {"adam", "soap"}:
                raise ValueError(
                    "PolarGrad auxiliary optimizer must be 'adam' or 'soap'."
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
                gradient = parameter.grad
                state = self.state[parameter]
                if group["use_polargrad"]:
                    if gradient.ndim != 2:
                        raise ValueError("PolarGrad parameters must be matrices.")
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(gradient)
                    momentum = state["momentum"]
                    coefficients = (group["a"], group["b"], group["c"])
                    if group["polar_first"]:
                        unitary = polar(
                            gradient,
                            method=group["method"],
                            max_iterations=group["inner_steps"],
                            ns_coeffs=coefficients,
                        )
                        nuclear_norm = torch.sum(gradient.type_as(unitary) * unitary)
                        momentum.lerp_(unitary.type_as(momentum), 1 - group["momentum"])
                        update = nuclear_norm * momentum.type_as(unitary)
                    else:
                        momentum.lerp_(gradient, 1 - group["momentum"])
                        unitary = polar(
                            momentum,
                            method=group["method"],
                            max_iterations=group["inner_steps"],
                            ns_coeffs=coefficients,
                        )
                        nuclear_norm = torch.sum(momentum.type_as(unitary) * unitary)
                        update = nuclear_norm * unitary
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
                else:
                    if group["auxiliary_optimizer"] == "soap":
                        soap_step_parameter(
                            parameter, gradient, state, group
                        )
                        continue
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(gradient, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                    bias1 = 1 - beta1 ** state["step"]
                    bias2 = 1 - beta2 ** state["step"]
                    denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(group["eps"])
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias1)
        return loss


# Preserve the upstream public class name for direct PyTorch use. Parameters
# passed without explicit groups use the PolarGrad branch by default.
PolarGrad = PolarGradWithAuxAdam
