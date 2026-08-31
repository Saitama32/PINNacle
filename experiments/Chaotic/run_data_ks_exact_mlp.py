"""Train a KS MLP on ``.dat`` values and derivatives from an exact teacher.

The teacher is the Fourier-in-space/cubic-spline-in-time representation of the
rectangular reference data.  A dense or RWF MLP (the student) is trained with
a normalized Sobolev loss on ``u``, ``u_t``, ``u_x``, ``u_xx`` and ``u_xxxx``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import deepxde as dde
import numpy as np
import torch
from deepxde.optimizers.pytorch.mousse import MousseWithAuxLion
from deepxde.optimizers.pytorch.muon import MuonWithAuxAdam
from deepxde.optimizers.pytorch.mop import MOPWithAuxAdam
from deepxde.optimizers.pytorch.polargrad import PolarGradWithAuxAdam
from scipy.interpolate import CubicSpline

from experiments.Chaotic.run_data_ks import (
    KS_ALPHA,
    KS_BETA,
    KS_GAMMA,
    build_network,
    build_optimizer as build_data_optimizer,
    evaluate_derivative_grid,
    evaluate_pinn_loss,
    ks_terms,
    load_data,
    prediction_metrics,
    save_checkpoint as save_student_checkpoint,
    save_solution_plot,
)
from src.utils.args import parse_hidden_layers


TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}
DERIVATIVE_KEYS = ("u_t", "u_x", "u_xx", "u_xxxx")


def parse_bool(value):
    """Parse an explicit true/false command-line value."""

    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value (true/false), got {value!r}"
    )


class CubicReLUCoefficientMLP(torch.nn.Module):
    """One-hidden-layer MLP representing a vector-valued cubic spline exactly."""

    def __init__(self, feature_knots, output_weights):
        super().__init__()
        feature_knots = torch.as_tensor(feature_knots, dtype=torch.float64)
        output_weights = torch.as_tensor(output_weights, dtype=torch.float64)
        self.hidden = torch.nn.Linear(1, len(feature_knots), bias=True, dtype=torch.float64)
        self.output = torch.nn.Linear(
            len(feature_knots), output_weights.shape[0], bias=False, dtype=torch.float64
        )
        with torch.no_grad():
            self.hidden.weight.fill_(1.0)
            self.hidden.bias.copy_(-feature_knots)
            self.output.weight.copy_(output_weights)

    def forward(self, t):
        return self.output(torch.relu(self.hidden(t)).pow(3))


class ExactFourierMLP(torch.nn.Module):
    """Cubic-ReLU MLP in time followed by exact Fourier synthesis in x."""

    def __init__(self, modes, feature_knots, output_weights):
        super().__init__()
        self.coefficient_mlp = CubicReLUCoefficientMLP(
            feature_knots, output_weights
        )
        self.register_buffer("modes", torch.as_tensor(modes, dtype=torch.float64))

    def forward(self, points):
        vector = self.coefficient_mlp(points[:, 1:2])
        count = self.modes.numel()
        real = vector[:, :count]
        imag = vector[:, count:]
        angle = points[:, 0:1] * self.modes.unsqueeze(0)
        multiplicity = torch.ones_like(self.modes)
        if multiplicity.numel() > 1:
            multiplicity[1:] = 2.0
        values = torch.sum(
            multiplicity.unsqueeze(0)
            * (real * torch.cos(angle) - imag * torch.sin(angle)),
            dim=1,
        )
        return values.unsqueeze(1)


def rectangular_field(points, values):
    x = np.unique(points[:, 0])
    t = np.unique(points[:, 1])
    field = values[:, 0].reshape(len(x), len(t)).T
    duplicate = np.allclose(field[:, 0], field[:, -1], rtol=0, atol=1e-12)
    if duplicate:
        x = x[:-1]
        field = field[:, :-1]
    return x, t, field, duplicate


def spline_mlp_weights(t, coefficient_values):
    real_spline = CubicSpline(t, coefficient_values.real, axis=0)
    imag_spline = CubicSpline(t, coefficient_values.imag, axis=0)

    outside_knots = np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float64)
    polynomial_matrix = np.stack(
        (
            -(outside_knots**3),
            3.0 * outside_knots**2,
            -3.0 * outside_knots,
            np.ones_like(outside_knots),
        ),
        axis=0,
    )

    def convert(spline):
        base_polynomial = np.stack(
            (spline.c[3, 0], spline.c[2, 0], spline.c[1, 0], spline.c[0, 0]),
            axis=0,
        )
        outside_weights = np.linalg.solve(polynomial_matrix, base_polynomial)
        hinge_weights = spline.c[0, 1:] - spline.c[0, :-1]
        return np.concatenate((outside_weights, hinge_weights), axis=0)

    feature_knots = np.concatenate((outside_knots, t[1:-1]))
    real_weights = convert(real_spline)
    imag_weights = convert(imag_spline)
    output_weights = np.concatenate((real_weights.T, imag_weights.T), axis=0)
    return feature_knots, output_weights


def save_teacher_checkpoint(path, model, configuration):
    torch.save(
        {
            "model": "ExactFourierMLP",
            "configuration": configuration,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        path,
    )


def reference_derivative_targets(points, values):
    """Return spline-time/spectral-space targets in original point order."""

    x = np.unique(points[:, 0])
    t = np.unique(points[:, 1])
    if len(points) != len(x) * len(t):
        raise ValueError("Derivative supervision requires a rectangular (x, t) grid")
    x_index = np.searchsorted(x, points[:, 0])
    t_index = np.searchsorted(t, points[:, 1])
    grid = np.empty((len(t), len(x)), dtype=np.float64)
    grid[t_index, x_index] = values[:, 0]

    duplicate = (
        np.isclose(x[-1] - x[0], 2.0 * np.pi)
        and np.allclose(grid[:, 0], grid[:, -1], rtol=1e-6, atol=1e-8)
    )
    spectral_x = x[:-1] if duplicate else x
    spectral_grid = grid[:, :-1] if duplicate else grid
    if len(spectral_x) < 2:
        raise ValueError("At least two distinct spatial points are required")
    spacing = np.diff(spectral_x)
    if not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
        raise ValueError("Spectral derivative targets require a uniform x grid")

    angular_modes = 2.0 * np.pi * np.fft.fftfreq(
        len(spectral_x), d=float(spacing[0])
    )
    coefficients = np.fft.fft(spectral_grid, axis=1)
    fields = {
        "u": spectral_grid,
        "u_t": CubicSpline(t, spectral_grid, axis=0)(t, 1),
        "u_x": np.fft.ifft((1j * angular_modes) * coefficients, axis=1).real,
        "u_xx": np.fft.ifft(-(angular_modes**2) * coefficients, axis=1).real,
        "u_xxxx": np.fft.ifft((angular_modes**4) * coefficients, axis=1).real,
    }
    if duplicate:
        fields = {key: np.c_[field, field[:, 0]] for key, field in fields.items()}
    targets = {
        key: np.asarray(field[t_index, x_index], dtype=np.float64).reshape(-1, 1)
        for key, field in fields.items()
    }
    scales = {
        key: max(float(np.sqrt(np.mean(target**2))), 1e-12)
        for key, target in targets.items()
    }
    return targets, scales


def build_student(args, points, values, device):
    hidden = parse_hidden_layers(args)
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden-layers must describe positive widths")
    input_min = np.min(points, axis=0)
    input_scale = np.max(points, axis=0) - input_min
    output_mean = float(np.mean(values, dtype=np.float64))
    output_std = float(np.std(values, dtype=np.float64))
    if not math.isfinite(output_std) or output_std <= 0.0:
        output_std = 1.0
    metadata = {
        "model": "RWFMLP" if args.network == "rwf" else "MLP",
        "precision": args.precision,
        "layer_sizes": [2, *hidden, 1],
        "rwf_mu": args.rwf_mu,
        "rwf_sigma": args.rwf_sigma,
        "input_min": input_min.tolist(),
        "input_scale": input_scale.tolist(),
        "output_mean": output_mean,
        "output_std": output_std,
        "alpha": KS_ALPHA,
        "beta": KS_BETA,
        "gamma": KS_GAMMA,
        "training_objective": "normalized_data_and_derivative_mse",
    }
    dde.config.set_default_float(args.precision)
    return build_network(metadata).to(device), metadata


def loss_weights(args):
    return {
        "u": args.data_weight,
        "u_t": args.ut_weight,
        "u_x": args.ux_weight,
        "u_xx": args.uxx_weight,
        "u_xxxx": args.uxxxx_weight,
    }


def matrix_auxiliary_group(
    args, params, use_flag, lr, adam_betas, adam_eps, weight_decay
):
    """Build the shared Adam/SOAP fallback group for matrix optimizers."""

    group = {
        "params": params,
        use_flag: False,
        "auxiliary_optimizer": args.matrix_fallback,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    if args.matrix_fallback == "soap":
        group.update(
            betas=(args.soap_beta1, args.soap_beta2),
            shampoo_beta=(
                args.soap_beta2
                if args.soap_shampoo_beta is None
                else args.soap_shampoo_beta
            ),
            eps=args.soap_epsilon,
            precondition_frequency=args.soap_precondition_frequency,
            max_precondition_dim=args.soap_max_precondition_dim,
            bias_correction=args.soap_bias_correction,
        )
    else:
        group.update(betas=adam_betas, eps=adam_eps)
    return group


def build_training_optimizer(student, args):
    """Build an optimizer with dense/RWF-aware routing for matrix optimizers."""

    hidden_matrices = [
        layer.V if hasattr(layer, "V") else layer.weight
        for layer in student.linears[:-1]
    ]

    if args.optimizer == "polargrad":
        hidden_ids = {id(parameter) for parameter in hidden_matrices}
        auxiliary = [
            parameter
            for parameter in student.parameters()
            if parameter.requires_grad and id(parameter) not in hidden_ids
        ]
        groups = []
        if hidden_matrices:
            groups.append(
                {
                    "params": hidden_matrices,
                    "use_polargrad": True,
                    "lr": args.lr,
                    "momentum": args.polargrad_momentum,
                    "polar_first": args.polargrad_polar_first,
                    "method": args.polargrad_method,
                    "inner_steps": args.polargrad_inner_steps,
                    "a": args.polargrad_a,
                    "b": args.polargrad_b,
                    "c": args.polargrad_c,
                    "weight_decay": args.polargrad_weight_decay,
                }
            )
        if auxiliary:
            groups.append(
                matrix_auxiliary_group(
                    args,
                    auxiliary,
                    "use_polargrad",
                    args.polargrad_adam_lr,
                    (
                        args.polargrad_adam_beta1,
                        args.polargrad_adam_beta2,
                    ),
                    args.polargrad_adam_epsilon,
                    args.polargrad_adam_weight_decay,
                )
            )
        return PolarGradWithAuxAdam(groups)

    if args.optimizer == "mousse":
        hidden_ids = {id(parameter) for parameter in hidden_matrices}
        auxiliary = [
            parameter
            for parameter in student.parameters()
            if parameter.requires_grad and id(parameter) not in hidden_ids
        ]
        groups = []
        if hidden_matrices:
            groups.append(
                {
                    "params": hidden_matrices,
                    "algorithm": "mousse",
                    "weight_decay": args.mousse_weight_decay,
                }
            )
        if auxiliary:
            groups.append(
                {
                    "params": auxiliary,
                    "algorithm": "lion",
                    "weight_decay": args.mousse_lion_weight_decay,
                }
            )
        adjust_lr = None if args.mousse_adjust_lr == "none" else args.mousse_adjust_lr
        return MousseWithAuxLion(
            groups,
            lr=args.lr,
            mu=args.mousse_momentum,
            betas=(args.mousse_lion_beta1, args.mousse_lion_beta2),
            epsilon=args.mousse_epsilon,
            nesterov=args.mousse_nesterov,
            adjust_lr=adjust_lr,
            shampoo_epsilon=args.mousse_shampoo_epsilon,
            shampoo_beta=args.mousse_shampoo_beta,
            shampoo_update_freq=args.mousse_shampoo_update_frequency,
            shampoo_alpha=args.mousse_shampoo_alpha,
            lr_correction=args.mousse_lr_correction,
            apply_norm=args.mousse_apply_norm,
            use_l_or_r=args.mousse_use_l_or_r,
        )

    if args.optimizer in {"psgdpro", "pcgpro"}:
        dde.optimizers.set_PSGDPRO_options(
            momentum=args.psgdpro_momentum,
            beta_lip=args.psgdpro_beta_lip,
            preconditioner_lr=args.psgdpro_preconditioner_lr,
            preconditioner_init_scale=args.psgdpro_preconditioner_init_scale,
            damping_noise_scale=args.psgdpro_damping_noise_scale,
            min_preconditioner_lr=args.psgdpro_min_preconditioner_lr,
            warmup_steps=args.psgdpro_warmup_steps,
            max_update_rms=args.psgdpro_max_update_rms,
            weight_decay_method=args.psgdpro_weight_decay_method,
            psgd_weight_decay=args.psgdpro_weight_decay,
            auxiliary_betas=(
                args.psgdpro_auxiliary_beta1,
                args.psgdpro_auxiliary_beta2,
            ),
            auxiliary_epsilon=args.psgdpro_auxiliary_epsilon,
            auxiliary_weight_decay=args.psgdpro_auxiliary_weight_decay,
        )
        optimizer_args = argparse.Namespace(**vars(args))
        optimizer_args.optimizer = "psgdpro"
        return build_data_optimizer(student, optimizer_args)

    if args.optimizer in {"kl-shampoo", "kl-soap"}:
        dde.optimizers.set_KLOPT_options(
            beta1=args.kl_beta1,
            beta2=args.kl_beta2,
            shampoo_beta=args.kl_shampoo_beta,
            epsilon=args.kl_epsilon,
            precondition_frequency=args.kl_precondition_frequency,
            using_klsoap=args.optimizer == "kl-soap",
            normalize_grads=args.kl_normalize_grads,
            init_factor=args.kl_init_factor,
            using_damping=args.kl_damping,
            using_clamping=args.kl_clamping,
            max_clamp_value=args.kl_max_clamp_value,
            cast_dtype=args.kl_cast_dtype,
        )
        return build_data_optimizer(student, args)

    if args.optimizer == "rekls-v3":
        dde.optimizers.set_REKLSV3_options(
            betas=(args.rekls_beta1, args.rekls_beta2),
            shampoo_beta=args.rekls_shampoo_beta,
            epsilon=args.rekls_epsilon,
            rekls_weight_decay=args.rekls_weight_decay,
            auxiliary_lr=args.rekls_auxiliary_lr,
            auxiliary_betas=(
                args.rekls_auxiliary_beta1,
                args.rekls_auxiliary_beta2,
            ),
            auxiliary_epsilon=args.rekls_auxiliary_epsilon,
            auxiliary_weight_decay=args.rekls_auxiliary_weight_decay,
        )
        return build_data_optimizer(student, args)

    if args.optimizer == "kl-m-soap":
        dde.optimizers.set_KLMSOAP_options(
            betas=(args.kl_m_soap_beta1, args.kl_m_soap_beta2),
            shampoo_beta=args.kl_m_soap_shampoo_beta,
            epsilon=args.kl_m_soap_epsilon,
            kl_m_soap_weight_decay=args.kl_m_soap_weight_decay,
            scale_log2=args.kl_m_soap_scale_log2,
            auxiliary_lr=args.kl_m_soap_auxiliary_lr,
            auxiliary_betas=(
                args.kl_m_soap_auxiliary_beta1,
                args.kl_m_soap_auxiliary_beta2,
            ),
            auxiliary_scale_log2=args.kl_m_soap_auxiliary_scale_log2,
            auxiliary_weight_decay=args.kl_m_soap_auxiliary_weight_decay,
        )
        return build_data_optimizer(student, args)

    if args.optimizer == "madam":
        dde.optimizers.set_MADAM_options(
            betas=(args.madam_beta1, args.madam_beta2),
            scale_log2=args.madam_scale_log2,
            correct_bias=args.madam_bias_correction,
        )
        return build_data_optimizer(student, args)

    if args.optimizer not in {"muon", "mop"}:
        return build_data_optimizer(student, args)

    hidden_ids = {id(parameter) for parameter in hidden_matrices}
    auxiliary = [
        parameter
        for parameter in student.parameters()
        if parameter.requires_grad and id(parameter) not in hidden_ids
    ]
    groups = []
    if hidden_matrices and args.optimizer == "muon":
        groups.append(
            {
                "params": hidden_matrices,
                "use_muon": True,
                "lr": args.lr,
                "momentum": args.muon_momentum,
                "nesterov": args.muon_nesterov,
                "ns_steps": args.muon_ns_steps,
                "weight_decay": args.muon_weight_decay,
            }
        )
    if hidden_matrices and args.optimizer == "mop":
        groups.append(
            {
                "params": hidden_matrices,
                "use_mop": True,
                "lr": args.lr,
                "momentum": args.mop_momentum,
                "nesterov": args.mop_nesterov,
                "scale_mode": args.mop_scale_mode,
                "extra_scale_factor": args.mop_extra_scale_factor,
                "weight_decay": args.mop_weight_decay,
            }
        )
    if auxiliary and args.optimizer == "muon":
        groups.append(
            matrix_auxiliary_group(
                args,
                auxiliary,
                "use_muon",
                args.muon_adam_lr,
                (args.muon_adam_beta1, args.muon_adam_beta2),
                args.muon_adam_epsilon,
                args.muon_adam_weight_decay,
            )
        )
    if auxiliary and args.optimizer == "mop":
        groups.append(
            matrix_auxiliary_group(
                args,
                auxiliary,
                "use_mop",
                args.mop_adam_lr,
                (args.mop_adam_beta1, args.mop_adam_beta2),
                args.mop_adam_epsilon,
                args.mop_adam_weight_decay,
            )
        )
    return MuonWithAuxAdam(groups) if args.optimizer == "muon" else MOPWithAuxAdam(groups)


def derivative_metrics(network, points, targets, scales, batch_size, device):
    """Evaluate normalized MSE and relative L2 for every supervised field."""

    dtype = next(network.parameters()).dtype
    square_error = {key: 0.0 for key in targets}
    square_target = {key: 0.0 for key in targets}
    count = 0
    network.eval()
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        batch = torch.as_tensor(points[start:stop], dtype=dtype, device=device)
        batch.requires_grad_(True)
        predicted = ks_terms(network, batch)
        for key, exact in targets.items():
            exact_tensor = torch.as_tensor(
                exact[start:stop], dtype=dtype, device=device
            )
            error = predicted[key].detach().double() - exact_tensor.double()
            square_error[key] += float(torch.sum(error.square()).cpu())
            square_target[key] += float(torch.sum(exact_tensor.double().square()).cpu())
        count += stop - start
    return {
        key: {
            "mse": square_error[key] / count,
            "normalized_mse": square_error[key] / count / (scales[key] ** 2),
            "relative_l2": math.sqrt(square_error[key] / square_target[key])
            if square_target[key] > 0.0 else None,
            "target_rms": scales[key],
        }
        for key in targets
    }


def train_student(args, student, points, targets, scales, device, run_dir, metadata):
    dtype = TORCH_DTYPES[args.precision]
    point_tensor = torch.as_tensor(points, dtype=dtype, device=device)
    target_tensors = {
        key: torch.as_tensor(value, dtype=dtype, device=device)
        for key, value in targets.items()
    }
    initial_mask = np.isclose(points[:, 1], np.min(points[:, 1]), rtol=0.0, atol=1e-12)
    if not np.any(initial_mask):
        raise ValueError("The reference data contain no initial-time points")
    initial_indices = torch.as_tensor(
        np.flatnonzero(initial_mask), dtype=torch.long, device=device
    )
    initial_points = point_tensor[initial_indices]
    initial_targets = target_tensors["u"][initial_indices]
    weights = loss_weights(args)
    active_derivatives = [key for key in DERIVATIVE_KEYS if weights[key] > 0.0]
    if weights["u"] <= 0.0 and not active_derivatives and args.pde_weight <= 0.0:
        raise ValueError("At least one data, derivative, or PDE weight must be positive")

    optimizer = build_training_optimizer(student, args)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.iterations, 1), eta_min=args.lr_min
        )
        if args.lr_min < args.lr else None
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    history = []
    for iteration in range(1, args.iterations + 1):
        student.train()
        if args.batch_size >= len(point_tensor):
            indices = torch.arange(len(point_tensor), device=device)
        else:
            indices = torch.randint(
                len(point_tensor), (args.batch_size,), generator=generator, device=device
            )
        batch_points = point_tensor[indices].detach().clone().requires_grad_(True)
        predicted = ks_terms(
            student, batch_points, create_graph_for_backward=True
        )
        data_target = target_tensors["u"][indices]
        data_error = predicted["u"] - data_target
        data_mse = torch.mean(data_error.square())
        data_l2re = torch.sqrt(
            torch.sum(data_error.square())
            / torch.clamp(torch.sum(data_target.square()), min=1e-30)
        )
        component_losses = {"u": data_mse / (scales["u"] ** 2)}
        for key in DERIVATIVE_KEYS:
            error = predicted[key] - target_tensors[key][indices]
            component_losses[key] = torch.mean(error.square()) / (scales[key] ** 2)

        if args.derivative_warmup <= 0:
            derivative_factor = 1.0
        else:
            derivative_factor = min(1.0, iteration / args.derivative_warmup)
        total = weights["u"] * component_losses["u"]
        for key in active_derivatives:
            total = total + derivative_factor * weights[key] * component_losses[key]
        # Keep this diagnostic/optional term in physical units, matching
        # evaluate_pinn_loss.  Its default weight is zero: this script's main
        # objective is supervised value-and-derivative matching.
        pde_loss = torch.mean(predicted["residual"].square())
        total = total + args.pde_weight * pde_loss

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            with torch.no_grad():
                ic_mse = torch.mean((student(initial_points) - initial_targets).square())
            row = {
                "iteration": iteration,
                "loss_total": float(total.detach().cpu()),
                "loss_data": float(data_mse.detach().cpu()),
                "l2re": float(data_l2re.detach().cpu()),
                "loss_data_normalized": float(component_losses["u"].detach().cpu()),
                "loss_ut_normalized": float(component_losses["u_t"].detach().cpu()),
                "loss_ux_normalized": float(component_losses["u_x"].detach().cpu()),
                "loss_uxx_normalized": float(component_losses["u_xx"].detach().cpu()),
                "loss_uxxxx_normalized": float(component_losses["u_xxxx"].detach().cpu()),
                "ic_loss": float(ic_mse.detach().cpu()),
                "pde_mse_diagnostic": float(pde_loss.detach().cpu()),
                "pde_weighted_contribution": float(
                    (args.pde_weight * pde_loss).detach().cpu()
                ),
                "derivative_factor": derivative_factor,
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            message = (
                f"step={iteration:7d} loss_total={row['loss_total']:.6e} "
                f"loss_data={row['loss_data']:.3e} l2re={row['l2re']:.3e} "
                f"loss_ut_norm={row['loss_ut_normalized']:.3e} "
                f"loss_ux_norm={row['loss_ux_normalized']:.3e} "
                f"loss_uxx_norm={row['loss_uxx_normalized']:.3e} "
                f"loss_uxxxx_norm={row['loss_uxxxx_normalized']:.3e} "
                f"ic_loss={row['ic_loss']:.3e} "
                f"derivative_factor={row['derivative_factor']:.3f}"
            )
            if args.pde_weight == 0.0:
                message += (
                    f" diagnostic_pinn_pde_loss={row['pde_mse_diagnostic']:.3e}"
                )
            else:
                message += (
                    f" pinn_pde_loss={row['pde_mse_diagnostic']:.3e} "
                    f"pde_contribution={row['pde_weighted_contribution']:.3e}"
                )
            print(message)

    save_student_checkpoint(run_dir / "weights_student.pt", student, metadata)
    if history:
        columns = list(history[0])
        np.savetxt(
            run_dir / "history.csv",
            np.asarray([[row[key] for key in columns] for row in history]),
            delimiter=",", header=",".join(columns), comments="",
        )
    return history


def run(args):
    args.adam_epsilon = 1e-8 if args.adam_epsilon is None else args.adam_epsilon
    args.soap_epsilon = 1e-8 if args.soap_epsilon is None else args.soap_epsilon
    args.muon_adam_epsilon = (
        1e-10 if args.muon_adam_epsilon is None else args.muon_adam_epsilon
    )
    args.mop_adam_epsilon = (
        1e-10 if args.mop_adam_epsilon is None else args.mop_adam_epsilon
    )
    args.kl_epsilon = 1e-8 if args.kl_epsilon is None else args.kl_epsilon
    if args.iterations <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("iterations, batch-size, and log-every must be positive")
    if args.lr <= 0.0 or args.lr_min < 0.0 or args.lr_min > args.lr:
        raise ValueError("Require 0 <= lr-min <= lr and lr > 0")
    if args.grad_clip <= 0.0:
        raise ValueError("grad-clip must be positive")
    if args.derivative_warmup < 0:
        raise ValueError("derivative-warmup must be non-negative")
    if args.derivative_eval_points <= 0 or args.derivative_eval_batch_size <= 0:
        raise ValueError("derivative evaluation sizes must be positive")
    if any(weight < 0.0 for weight in (*loss_weights(args).values(), args.pde_weight)):
        raise ValueError("Loss weights must be non-negative")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative")
    if args.adam_epsilon <= 0.0 or args.soap_epsilon <= 0.0:
        raise ValueError("Adam and SOAP epsilon values must be positive")
    if not 0.0 <= args.soap_beta1 < 1.0 or not 0.0 <= args.soap_beta2 < 1.0:
        raise ValueError("SOAP beta values must be in [0, 1)")
    if args.soap_shampoo_beta is not None and not 0.0 <= args.soap_shampoo_beta < 1.0:
        raise ValueError("soap-shampoo-beta must be in [0, 1)")
    if args.soap_precondition_frequency <= 0 or args.soap_max_precondition_dim <= 0:
        raise ValueError("SOAP precondition settings must be positive")
    if not 0.0 <= args.kl_beta1 < 1.0 or not 0.0 <= args.kl_beta2 < 1.0:
        raise ValueError("KL-Shampoo beta values must be in [0, 1)")
    if args.kl_shampoo_beta is not None and not 0.0 <= args.kl_shampoo_beta < 1.0:
        raise ValueError("kl-shampoo-beta must be in [0, 1)")
    if args.kl_epsilon <= 0.0 or not math.isfinite(args.kl_epsilon):
        raise ValueError("kl-epsilon must be positive and finite")
    if args.kl_precondition_frequency <= 0:
        raise ValueError("kl-precondition-frequency must be positive")
    if args.kl_init_factor <= 0.0 or not math.isfinite(args.kl_init_factor):
        raise ValueError("kl-init-factor must be positive and finite")
    if args.kl_max_clamp_value <= 0:
        raise ValueError("kl-max-clamp-value must be positive")
    if not 0.0 <= args.mousse_momentum < 1.0:
        raise ValueError("Mousse momentum must be in [0, 1)")
    if not 0.0 <= args.mousse_lion_beta1 < 1.0 or not 0.0 <= args.mousse_lion_beta2 < 1.0:
        raise ValueError("Mousse Lion beta values must be in [0, 1)")
    if args.mousse_epsilon <= 0.0 or args.mousse_shampoo_epsilon <= 0.0:
        raise ValueError("Mousse epsilon values must be positive")
    if not 0.0 <= args.mousse_shampoo_beta < 1.0:
        raise ValueError("mousse-shampoo-beta must be in [0, 1)")
    if args.mousse_shampoo_update_frequency <= 0 or args.mousse_shampoo_alpha < 0.0:
        raise ValueError("Mousse Shampoo frequency must be positive and alpha non-negative")
    if args.mousse_weight_decay < 0.0 or args.mousse_lion_weight_decay < 0.0:
        raise ValueError("Mousse weight decay values must be non-negative")
    if not 0.0 <= args.psgdpro_momentum < 1.0 or not 0.0 <= args.psgdpro_beta_lip < 1.0:
        raise ValueError("PSGDPro momentum and beta-lip must be in [0, 1)")
    if args.psgdpro_preconditioner_lr <= 0.0 or args.psgdpro_min_preconditioner_lr < 0.0:
        raise ValueError("PSGDPro preconditioner learning rates are invalid")
    if args.psgdpro_preconditioner_init_scale <= 0.0 or args.psgdpro_warmup_steps <= 0:
        raise ValueError("PSGDPro initialization scale and warmup must be positive")
    if args.psgdpro_damping_noise_scale < 0.0 or args.psgdpro_max_update_rms < 0.0:
        raise ValueError("PSGDPro damping and maximum update RMS must be non-negative")
    if not 0.0 <= args.psgdpro_auxiliary_beta1 < 1.0 or not 0.0 <= args.psgdpro_auxiliary_beta2 < 1.0:
        raise ValueError("PSGDPro auxiliary beta values must be in [0, 1)")
    if args.psgdpro_auxiliary_epsilon <= 0.0:
        raise ValueError("PSGDPro auxiliary epsilon must be positive")
    if args.psgdpro_weight_decay < 0.0 or args.psgdpro_auxiliary_weight_decay < 0.0:
        raise ValueError("PSGDPro weight decay values must be non-negative")
    if not 0.0 <= args.muon_momentum < 1.0 or args.muon_ns_steps <= 0:
        raise ValueError("Muon momentum must be in [0, 1) and ns-steps must be positive")
    if args.muon_adam_lr <= 0.0 or args.muon_adam_epsilon <= 0.0:
        raise ValueError("Muon auxiliary Adam lr and epsilon must be positive")
    if not 0.0 <= args.muon_adam_beta1 < 1.0 or not 0.0 <= args.muon_adam_beta2 < 1.0:
        raise ValueError("Muon auxiliary Adam beta values must be in [0, 1)")
    if args.muon_weight_decay < 0.0 or args.muon_adam_weight_decay < 0.0:
        raise ValueError("Muon weight decay values must be non-negative")
    if not 0.0 <= args.mop_momentum < 1.0:
        raise ValueError("MOP momentum must be in [0, 1)")
    if args.mop_adam_lr <= 0.0 or args.mop_adam_epsilon <= 0.0:
        raise ValueError("MOP auxiliary Adam lr and epsilon must be positive")
    if not 0.0 <= args.mop_adam_beta1 < 1.0 or not 0.0 <= args.mop_adam_beta2 < 1.0:
        raise ValueError("MOP auxiliary Adam beta values must be in [0, 1)")
    if args.mop_weight_decay < 0.0 or args.mop_adam_weight_decay < 0.0:
        raise ValueError("MOP weight decay values must be non-negative")
    if not math.isfinite(args.mop_extra_scale_factor):
        raise ValueError("MOP extra scale factor must be finite")
    if args.optimizer == "muon" and args.lr_min < args.lr and args.lr_min > args.muon_adam_lr:
        raise ValueError("lr-min cannot exceed muon-adam-lr when cosine decay is active")
    if args.optimizer == "mop" and args.lr_min < args.lr and args.lr_min > args.mop_adam_lr:
        raise ValueError("lr-min cannot exceed mop-adam-lr when cosine decay is active")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Always construct reference targets in float64.  Only the student follows
    # args.precision.
    dde.config.set_default_float("float64")
    points, values = load_data(args.data, precision="float64")
    x, t, field, duplicate = rectangular_field(points, values)
    modes = np.arange(len(x) // 2 + 1, dtype=np.float64)
    coefficients = np.fft.rfft(field, axis=1) / len(x)
    feature_knots, output_weights = spline_mlp_weights(t, coefficients)
    teacher = ExactFourierMLP(modes, feature_knots, output_weights).to(device)
    targets, target_scales = reference_derivative_targets(points, values)
    student, student_metadata = build_student(args, points, values, device)

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = (
        Path(args.out).expanduser().resolve()
        / f"{timestamp}-ks-data-derivative-{args.network}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        **vars(args),
        "data": str(Path(args.data).resolve()),
        "device": str(device),
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "teacher_parameters": sum(parameter.numel() for parameter in teacher.parameters()),
        "teacher_hidden_width": len(feature_knots),
        "teacher_output_width": output_weights.shape[0],
        "duplicate_periodic_endpoint": duplicate,
        "target_scales": target_scales,
        "loss_weights": {**loss_weights(args), "pde": args.pde_weight},
        "student_metadata": student_metadata,
    }
    student_metadata.update(
        derivative_target_method="cubic_spline_t_and_spectral_x",
        derivative_target_scales=target_scales,
        derivative_loss_weights=configuration["loss_weights"],
    )
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(configuration, file_obj, indent=2, sort_keys=True)
    np.savez_compressed(
        run_dir / "reference_targets.npz",
        x=points[:, 0], t=points[:, 1], **{key: value[:, 0] for key, value in targets.items()},
    )

    # Keep the exact representation as a reproducible record of how derivative
    # labels were produced.  It is frozen and is not the trained network.
    save_teacher_checkpoint(run_dir / "weights_reference_teacher.pt", teacher, configuration)
    scripted_buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(teacher), scripted_buffer)
    (run_dir / "reference_teacher_scripted.pt").write_bytes(scripted_buffer.getvalue())

    history = train_student(
        args, student, points, targets, target_scales, device, run_dir, student_metadata
    )

    numpy_dtype = np.float64 if args.precision == "float64" else np.float32
    student_points = points.astype(numpy_dtype)
    student_values = values.astype(numpy_dtype)
    data_metric = prediction_metrics(
        student, student_points, student_values, args.eval_batch_size, device
    )
    derivative_eval_count = min(args.derivative_eval_points, len(student_points))
    derivative_eval_indices = np.random.default_rng(args.seed + 31).choice(
        len(student_points), size=derivative_eval_count, replace=False
    )
    derivative_eval_targets = {
        key: value[derivative_eval_indices] for key, value in targets.items()
    }
    supervised_derivatives = derivative_metrics(
        student, student_points[derivative_eval_indices], derivative_eval_targets, target_scales,
        args.derivative_eval_batch_size, device,
    )

    class EvaluationArgs:
        precision = args.precision
        seed = args.seed
        pinn_points = args.pinn_points
        pinn_ic_points = args.pinn_ic_points
        pinn_batch_size = args.pinn_batch_size
        alpha = KS_ALPHA
        beta = KS_BETA
        gamma = KS_GAMMA
        ic_loss_weight = 100.0

    pinn_metric = evaluate_pinn_loss(
        student,
        (np.min(points, axis=0), np.max(points, axis=0)),
        EvaluationArgs,
        device,
    )
    diagnostic_dir = run_dir / "student_derivative_diagnostics"
    diagnostic_dir.mkdir()
    derivative_grid = evaluate_derivative_grid(
        student,
        (np.min(points, axis=0), np.max(points, axis=0)),
        KS_ALPHA, KS_BETA, KS_GAMMA,
        args.derivative_grid_nx, args.derivative_grid_nt,
        args.derivative_batch_size, diagnostic_dir, device,
    )
    predictions = []
    with torch.no_grad():
        for start in range(0, len(points), args.eval_batch_size):
            predictions.append(
                student(
                    torch.as_tensor(
                        student_points[start:start + args.eval_batch_size], device=device
                    )
                ).cpu().numpy()
            )
    prediction = np.vstack(predictions)[:, 0]
    np.savez_compressed(
        run_dir / "predictions.npz",
        x=points[:, 0], t=points[:, 1], exact=values[:, 0], prediction=prediction,
    )
    save_solution_plot(
        run_dir / "solution.png", points, values[:, 0], prediction,
        f"Data+derivative {'RWF MLP' if args.network == 'rwf' else 'MLP'}, "
        f"relative L2={data_metric['relative_l2']:.3e}",
    )
    metrics = {
        "student_data": data_metric,
        "student_pinn": pinn_metric,
        "student_supervised_derivatives": supervised_derivatives,
        "student_derivative_grid": derivative_grid,
        "last_training_row": history[-1] if history else None,
        "configuration": configuration,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)
    pde_role = "diagnostic_only" if args.pde_weight == 0.0 else "included_in_training"
    print(
        f"Data+derivative {'RWF MLP' if args.network == 'rwf' else 'MLP'}: "
        f"L2={data_metric['relative_l2']:.6e}; "
        f"PDE_MSE={pinn_metric['pde_mse']:.6e} "
        f"(pde_weight={args.pde_weight:.6g}, {pde_role}); "
        f"ut_rel={supervised_derivatives['u_t']['relative_l2']:.6e}; "
        f"parameters={configuration['student_parameters']}; "
        f"artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs_data_ks_exact_mlp"))
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument(
        "--network", "--network-type", choices=["mlp", "rwf"], default="rwf"
    )
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--optimizer",
        choices=[
            "adam", "rmsprop", "soap", "kl-shampoo", "kl-soap", "muon", "mop",
            "mousse", "psgdpro", "pcgpro", "polargrad", "rekls-v3",
            "kl-m-soap", "madam",
        ],
        default="kl-m-soap",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-min", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--matrix-fallback",
        choices=["adam", "soap"],
        default="soap",
        help="Fallback for non-matrix parameters of Muon, MOP, and PolarGrad.",
    )
    parser.add_argument("--adam-epsilon", type=float, default=None)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-epsilon", type=float, default=None)
    parser.add_argument("--soap-precondition-frequency", type=int, default=1)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)
    parser.add_argument("--soap-bias-correction", type=parse_bool, default=True)
    parser.add_argument("--kl-beta1", type=float, default=0.99)
    parser.add_argument("--kl-beta2", type=float, default=0.999)
    parser.add_argument("--kl-shampoo-beta", type=float, default=None)
    parser.add_argument("--kl-epsilon", type=float, default=None)
    parser.add_argument("--kl-precondition-frequency", type=int, default=1)
    parser.add_argument("--kl-normalize-grads", type=parse_bool, default=False)
    parser.add_argument("--kl-init-factor", type=float, default=0.1)
    parser.add_argument("--kl-damping", type=parse_bool, default=False)
    parser.add_argument("--kl-clamping", type=parse_bool, default=True)
    parser.add_argument("--kl-max-clamp-value", type=int, default=4000)
    parser.add_argument(
        "--kl-cast-dtype",
        choices=["float32", "float64", "float16", "bfloat16"],
        default="float64",
    )
    parser.add_argument("--rekls-beta1", type=float, default=0.99)
    parser.add_argument("--rekls-beta2", type=float, default=0.999)
    parser.add_argument("--rekls-shampoo-beta", type=float, default=0.999)
    parser.add_argument("--rekls-epsilon", type=float, default=1e-8)
    parser.add_argument("--rekls-weight-decay", type=float, default=0.01)
    parser.add_argument("--rekls-auxiliary-lr", type=float, default=None)
    parser.add_argument("--rekls-auxiliary-beta1", type=float, default=0.99)
    parser.add_argument("--rekls-auxiliary-beta2", type=float, default=0.999)
    parser.add_argument("--rekls-auxiliary-epsilon", type=float, default=1e-8)
    parser.add_argument("--rekls-auxiliary-weight-decay", type=float, default=0.0)
    parser.add_argument("--kl-m-soap-beta1", type=float, default=0.99)
    parser.add_argument("--kl-m-soap-beta2", type=float, default=0.999)
    parser.add_argument("--kl-m-soap-shampoo-beta", type=float, default=0.999)
    parser.add_argument("--kl-m-soap-epsilon", type=float, default=1e-8)
    parser.add_argument("--kl-m-soap-weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-m-soap-scale-log2", type=float, default=16.0)
    parser.add_argument("--kl-m-soap-auxiliary-lr", type=float, default=None)
    parser.add_argument("--kl-m-soap-auxiliary-beta1", type=float, default=0.99)
    parser.add_argument("--kl-m-soap-auxiliary-beta2", type=float, default=0.999)
    parser.add_argument("--kl-m-soap-auxiliary-scale-log2", type=float, default=16.0)
    parser.add_argument("--kl-m-soap-auxiliary-weight-decay", type=float, default=0.0)
    parser.add_argument("--madam-beta1", type=float, default=0.99)
    parser.add_argument("--madam-beta2", type=float, default=0.999)
    parser.add_argument("--madam-scale-log2", type=float, default=16.0)
    parser.add_argument("--madam-bias-correction", type=parse_bool, default=True)
    parser.add_argument("--mousse-momentum", type=float, default=0.95)
    parser.add_argument("--mousse-lion-beta1", type=float, default=0.9)
    parser.add_argument("--mousse-lion-beta2", type=float, default=0.95)
    parser.add_argument("--mousse-epsilon", type=float, default=1e-8)
    parser.add_argument("--mousse-nesterov", type=parse_bool, default=False)
    parser.add_argument(
        "--mousse-adjust-lr",
        choices=["spectral_norm", "rms_norm", "none"],
        default="spectral_norm",
    )
    parser.add_argument("--mousse-shampoo-epsilon", type=float, default=1e-10)
    parser.add_argument("--mousse-shampoo-beta", type=float, default=0.95)
    parser.add_argument("--mousse-shampoo-update-frequency", type=int, default=10)
    parser.add_argument("--mousse-shampoo-alpha", type=float, default=0.125)
    parser.add_argument("--mousse-lr-correction", type=parse_bool, default=True)
    parser.add_argument("--mousse-apply-norm", type=parse_bool, default=True)
    parser.add_argument("--mousse-use-l-or-r", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--mousse-weight-decay", type=float, default=0.01)
    parser.add_argument("--mousse-lion-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--psgdpro-momentum", "--pcgpro-momentum",
        dest="psgdpro_momentum", type=float, default=0.9,
    )
    parser.add_argument(
        "--psgdpro-beta-lip", "--pcgpro-beta-lip",
        dest="psgdpro_beta_lip", type=float, default=0.9,
    )
    parser.add_argument(
        "--psgdpro-preconditioner-lr", "--pcgpro-preconditioner-lr",
        dest="psgdpro_preconditioner_lr", type=float, default=0.03,
    )
    parser.add_argument(
        "--psgdpro-preconditioner-init-scale", "--pcgpro-preconditioner-init-scale",
        dest="psgdpro_preconditioner_init_scale", type=float, default=1.0,
    )
    parser.add_argument(
        "--psgdpro-damping-noise-scale", "--pcgpro-damping-noise-scale",
        dest="psgdpro_damping_noise_scale", type=float, default=0.1,
    )
    parser.add_argument(
        "--psgdpro-min-preconditioner-lr", "--pcgpro-min-preconditioner-lr",
        dest="psgdpro_min_preconditioner_lr", type=float, default=0.003,
    )
    parser.add_argument(
        "--psgdpro-warmup-steps", "--pcgpro-warmup-steps",
        dest="psgdpro_warmup_steps", type=int, default=2000,
    )
    parser.add_argument(
        "--psgdpro-max-update-rms", "--pcgpro-max-update-rms",
        dest="psgdpro_max_update_rms", type=float, default=0.0,
    )
    parser.add_argument(
        "--psgdpro-weight-decay-method", "--pcgpro-weight-decay-method",
        dest="psgdpro_weight_decay_method",
        choices=["decoupled", "independent", "l2", "palm"],
        default="decoupled",
    )
    parser.add_argument(
        "--psgdpro-weight-decay", "--pcgpro-weight-decay",
        dest="psgdpro_weight_decay", type=float, default=0.0,
    )
    parser.add_argument(
        "--psgdpro-auxiliary-beta1", "--pcgpro-auxiliary-beta1",
        dest="psgdpro_auxiliary_beta1", type=float, default=0.9,
    )
    parser.add_argument(
        "--psgdpro-auxiliary-beta2", "--pcgpro-auxiliary-beta2",
        dest="psgdpro_auxiliary_beta2", type=float, default=0.999,
    )
    parser.add_argument(
        "--psgdpro-auxiliary-epsilon", "--pcgpro-auxiliary-epsilon",
        dest="psgdpro_auxiliary_epsilon", type=float, default=1e-8,
    )
    parser.add_argument(
        "--psgdpro-auxiliary-weight-decay", "--pcgpro-auxiliary-weight-decay",
        dest="psgdpro_auxiliary_weight_decay", type=float, default=0.0,
    )
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", type=parse_bool, default=False)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-adam-epsilon", type=float, default=None)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--mop-momentum", type=float, default=0.95)
    parser.add_argument("--mop-nesterov", type=parse_bool, default=False)
    parser.add_argument(
        "--mop-scale-mode",
        choices=["nuclear_norm", "shape_scaling", "spectral", "unit_rms_norm"],
        default="nuclear_norm",
    )
    parser.add_argument("--mop-extra-scale-factor", type=float, default=1.0)
    parser.add_argument("--mop-adam-lr", type=float, default=3e-4)
    parser.add_argument("--mop-adam-beta1", type=float, default=0.9)
    parser.add_argument("--mop-adam-beta2", type=float, default=0.95)
    parser.add_argument("--mop-adam-epsilon", type=float, default=None)
    parser.add_argument("--mop-weight-decay", type=float, default=0.01)
    parser.add_argument("--mop-adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--polargrad-momentum", type=float, default=0.95)
    parser.add_argument("--polargrad-polar-first", type=parse_bool, default=False)
    parser.add_argument(
        "--polargrad-method",
        choices=["qdwh", "zolo-pd", "ns", "precond_ns", "polar_express"],
        default="zolo-pd",
    )
    parser.add_argument("--polargrad-inner-steps", type=int, default=2)
    parser.add_argument("--polargrad-a", type=float, default=3.4445)
    parser.add_argument("--polargrad-b", type=float, default=-4.7750)
    parser.add_argument("--polargrad-c", type=float, default=2.031)
    parser.add_argument("--polargrad-adam-lr", type=float, default=3e-4)
    parser.add_argument("--polargrad-adam-beta1", type=float, default=0.9)
    parser.add_argument("--polargrad-adam-beta2", type=float, default=0.95)
    parser.add_argument("--polargrad-adam-epsilon", type=float, default=1e-10)
    parser.add_argument("--polargrad-weight-decay", type=float, default=0.0)
    parser.add_argument("--polargrad-adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--derivative-warmup", type=int, default=3000)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--ut-weight", type=float, default=1.0)
    parser.add_argument("--ux-weight", type=float, default=1.0)
    parser.add_argument("--uxx-weight", type=float, default=1.0)
    parser.add_argument("--uxxxx-weight", type=float, default=1.0)
    parser.add_argument("--pde-weight", type=float, default=0.0)
    parser.add_argument("--pinn-points", type=int, default=20000)
    parser.add_argument("--pinn-ic-points", type=int, default=2048)
    parser.add_argument("--pinn-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--derivative-eval-points", type=int, default=8192)
    parser.add_argument("--derivative-eval-batch-size", type=int, default=128)
    parser.add_argument("--derivative-grid-nx", type=int, default=128)
    parser.add_argument("--derivative-grid-nt", type=int, default=64)
    parser.add_argument("--derivative-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
