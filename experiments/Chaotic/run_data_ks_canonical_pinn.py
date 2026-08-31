"""Train a KS MLP with the canonical strong-form residual and initial condition.

The reference ``x, t, u`` file defines the physical domain and is used for
diagnostics, plotting, and fixed input/output normalization. It does not
contribute a data or Sobolev loss. The optimized objective is

    MSE(v_tau + v * v_xi + v_xixi + v_xixixixi)
    + ic_weight * MSE(v(xi, tau_0) - v_0(xi)).
"""

from __future__ import annotations

import argparse
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

from src.model import RWFMLP
from src.pde.chaotic import CanonicalKuramotoSivashinskyEquation
from src.utils.args import parse_hidden_layers


TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def parse_bool(value):
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


def load_data(path, precision="float32"):
    """Load a finite three-column physical ``x, t, u`` data set."""

    raw = np.loadtxt(path, comments="%", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("KS data must have at least three columns: x, t, u")
    raw = raw[:, :3]
    if len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("KS data must contain at least two finite observations")
    dtype = np.float64 if precision == "float64" else np.float32
    points = raw[:, :2].astype(dtype)
    values = raw[:, 2:3].astype(dtype)
    if np.any(np.ptp(points, axis=0) <= 0):
        raise ValueError("Both x and t must vary in the KS data")
    return points, values


def _normalization_transform(lower, scale):
    lower_values = tuple(float(value) for value in lower)
    scale_values = tuple(float(value) for value in scale)

    def transform(inputs):
        lower_tensor = inputs.new_tensor(lower_values)
        scale_tensor = inputs.new_tensor(scale_values)
        return 2.0 * (inputs - lower_tensor) / scale_tensor - 1.0

    return transform


def _output_transform(mean, std):
    def transform(_, outputs):
        return outputs * std + mean

    return transform


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
    }
    dde.config.set_default_float(args.precision)
    if args.network == "rwf":
        network = RWFMLP(
            metadata["layer_sizes"], mu=args.rwf_mu, sigma=args.rwf_sigma
        )
    else:
        network = dde.nn.FNN(
            metadata["layer_sizes"], "tanh", "Glorot normal"
        )
    network = network.to(dtype=TORCH_DTYPES[args.precision], device=device)
    network.apply_feature_transform(
        _normalization_transform(input_min, input_scale)
    )
    network.apply_output_transform(_output_transform(output_mean, output_std))
    return network, metadata


def _cpu_state_dict(network):
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def save_checkpoint(path, network, metadata):
    torch.save(
        {"state_dict": _cpu_state_dict(network), "metadata": metadata}, path
    )


def save_solution_plot(path, points, exact, prediction, title):
    import matplotlib.pyplot as plt

    points = np.asarray(points)
    exact = np.asarray(exact).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if len(points) != len(exact) or len(exact) != len(prediction):
        raise ValueError("points, exact, and prediction must have the same length")
    x = np.unique(points[:, 0])
    t = np.unique(points[:, 1])
    error = np.abs(prediction - exact)
    solution_min = float(min(np.min(exact), np.min(prediction)))
    solution_max = float(max(np.max(exact), np.max(prediction)))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    if len(points) == len(x) * len(t):
        x_indices = np.searchsorted(x, points[:, 0])
        t_indices = np.searchsorted(t, points[:, 1])
        fields = []
        for values in (exact, prediction, error):
            field = np.empty((len(t), len(x)), dtype=values.dtype)
            field[t_indices, x_indices] = values
            fields.append(field)
        images = [
            axes[0].pcolormesh(
                x, t, fields[0], shading="auto", cmap="jet",
                vmin=solution_min, vmax=solution_max,
            ),
            axes[1].pcolormesh(
                x, t, fields[1], shading="auto", cmap="jet",
                vmin=solution_min, vmax=solution_max,
            ),
            axes[2].pcolormesh(x, t, fields[2], shading="auto", cmap="magma"),
        ]
    else:
        images = [
            axes[0].tricontourf(
                points[:, 0], points[:, 1], exact, levels=100, cmap="jet",
                vmin=solution_min, vmax=solution_max,
            ),
            axes[1].tricontourf(
                points[:, 0], points[:, 1], prediction, levels=100, cmap="jet",
                vmin=solution_min, vmax=solution_max,
            ),
            axes[2].tricontourf(
                points[:, 0], points[:, 1], error, levels=100, cmap="magma"
            ),
        ]
    for axis, image, label in zip(
        axes, images, ("Exact solution", "Canonical PINN prediction", "Absolute error")
    ):
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _base_optimizer(network, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(
            network.parameters(), lr=args.lr, eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    optimizer_name = args.optimizer
    if optimizer_name == "soap":
        dde.optimizers.set_SOAP_options(
            beta1=args.soap_beta1,
            beta2=args.soap_beta2,
            shampoo_beta=args.soap_shampoo_beta,
            epsilon=args.soap_epsilon,
            precondition_frequency=args.soap_precondition_frequency,
            max_precondition_dim=args.soap_max_precondition_dim,
            bias_correction=args.soap_bias_correction,
        )
    elif optimizer_name in {"kl-shampoo", "kl-soap"}:
        dde.optimizers.set_KLOPT_options(
            beta1=args.kl_beta1,
            beta2=args.kl_beta2,
            shampoo_beta=args.kl_shampoo_beta,
            epsilon=args.kl_epsilon,
            precondition_frequency=args.kl_precondition_frequency,
            using_klsoap=optimizer_name == "kl-soap",
            normalize_grads=args.kl_normalize_grads,
            init_factor=args.kl_init_factor,
            using_damping=args.kl_damping,
            using_clamping=args.kl_clamping,
            max_clamp_value=args.kl_max_clamp_value,
            cast_dtype=args.kl_cast_dtype,
        )
    elif optimizer_name == "rekls-v3":
        dde.optimizers.set_REKLSV3_options(
            betas=(args.rekls_beta1, args.rekls_beta2),
            shampoo_beta=args.rekls_shampoo_beta,
            epsilon=args.rekls_epsilon,
            rekls_weight_decay=args.rekls_weight_decay,
            auxiliary_lr=args.rekls_auxiliary_lr,
            auxiliary_betas=(args.rekls_auxiliary_beta1, args.rekls_auxiliary_beta2),
            auxiliary_epsilon=args.rekls_auxiliary_epsilon,
            auxiliary_weight_decay=args.rekls_auxiliary_weight_decay,
        )
    elif optimizer_name == "kl-m-soap":
        dde.optimizers.set_KLMSOAP_options(
            betas=(args.kl_m_soap_beta1, args.kl_m_soap_beta2),
            shampoo_beta=args.kl_m_soap_shampoo_beta,
            epsilon=args.kl_m_soap_epsilon,
            kl_m_soap_weight_decay=args.kl_m_soap_weight_decay,
            scale_log2=args.kl_m_soap_scale_log2,
            auxiliary_lr=args.kl_m_soap_auxiliary_lr,
            auxiliary_betas=(args.kl_m_soap_auxiliary_beta1, args.kl_m_soap_auxiliary_beta2),
            auxiliary_scale_log2=args.kl_m_soap_auxiliary_scale_log2,
            auxiliary_weight_decay=args.kl_m_soap_auxiliary_weight_decay,
        )
    elif optimizer_name == "madam":
        dde.optimizers.set_MADAM_options(
            betas=(args.madam_beta1, args.madam_beta2),
            scale_log2=args.madam_scale_log2,
            correct_bias=args.madam_bias_correction,
        )
    elif optimizer_name in {"psgdpro", "pcgpro"}:
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
                args.psgdpro_auxiliary_beta1, args.psgdpro_auxiliary_beta2
            ),
            auxiliary_epsilon=args.psgdpro_auxiliary_epsilon,
            auxiliary_weight_decay=args.psgdpro_auxiliary_weight_decay,
        )
        optimizer_name = "psgdpro"
    optimizer, _ = dde.optimizers.get(
        network.parameters(), optimizer_name, learning_rate=args.lr,
        weight_decay=args.weight_decay, model=network,
    )
    return optimizer


def matrix_auxiliary_group(
    args, params, use_flag, lr, adam_betas, adam_eps, weight_decay
):
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


def build_training_optimizer(network, args):
    hidden_matrices = [
        layer.V if hasattr(layer, "V") else layer.weight
        for layer in network.linears[:-1]
    ]
    hidden_ids = {id(parameter) for parameter in hidden_matrices}
    auxiliary = [
        parameter for parameter in network.parameters()
        if parameter.requires_grad and id(parameter) not in hidden_ids
    ]

    if args.optimizer == "polargrad":
        groups = [
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
        ]
        if auxiliary:
            groups.append(
                matrix_auxiliary_group(
                    args, auxiliary, "use_polargrad", args.polargrad_adam_lr,
                    (args.polargrad_adam_beta1, args.polargrad_adam_beta2),
                    args.polargrad_adam_epsilon,
                    args.polargrad_adam_weight_decay,
                )
            )
        return PolarGradWithAuxAdam(groups)

    if args.optimizer == "mousse":
        groups = []
        if hidden_matrices:
            groups.append(
                {"params": hidden_matrices, "algorithm": "mousse",
                 "weight_decay": args.mousse_weight_decay}
            )
        if auxiliary:
            groups.append(
                {"params": auxiliary, "algorithm": "lion",
                 "weight_decay": args.mousse_lion_weight_decay}
            )
        adjust_lr = None if args.mousse_adjust_lr == "none" else args.mousse_adjust_lr
        return MousseWithAuxLion(
            groups, lr=args.lr, mu=args.mousse_momentum,
            betas=(args.mousse_lion_beta1, args.mousse_lion_beta2),
            epsilon=args.mousse_epsilon, nesterov=args.mousse_nesterov,
            adjust_lr=adjust_lr,
            shampoo_epsilon=args.mousse_shampoo_epsilon,
            shampoo_beta=args.mousse_shampoo_beta,
            shampoo_update_freq=args.mousse_shampoo_update_frequency,
            shampoo_alpha=args.mousse_shampoo_alpha,
            lr_correction=args.mousse_lr_correction,
            apply_norm=args.mousse_apply_norm,
            use_l_or_r=args.mousse_use_l_or_r,
        )

    if args.optimizer not in {"muon", "mop"}:
        return _base_optimizer(network, args)

    groups = []
    if args.optimizer == "muon":
        groups.append(
            {"params": hidden_matrices, "use_muon": True, "lr": args.lr,
             "momentum": args.muon_momentum, "nesterov": args.muon_nesterov,
             "ns_steps": args.muon_ns_steps, "weight_decay": args.muon_weight_decay}
        )
        if auxiliary:
            groups.append(
                matrix_auxiliary_group(
                    args, auxiliary, "use_muon", args.muon_adam_lr,
                    (args.muon_adam_beta1, args.muon_adam_beta2),
                    args.muon_adam_epsilon, args.muon_adam_weight_decay,
                )
            )
        return MuonWithAuxAdam(groups)

    groups.append(
        {"params": hidden_matrices, "use_mop": True, "lr": args.lr,
         "momentum": args.mop_momentum, "nesterov": args.mop_nesterov,
         "scale_mode": args.mop_scale_mode,
         "extra_scale_factor": args.mop_extra_scale_factor,
         "weight_decay": args.mop_weight_decay}
    )
    if auxiliary:
        groups.append(
            matrix_auxiliary_group(
                args, auxiliary, "use_mop", args.mop_adam_lr,
                (args.mop_adam_beta1, args.mop_adam_beta2),
                args.mop_adam_epsilon, args.mop_adam_weight_decay,
            )
        )
    return MOPWithAuxAdam(groups)


def canonical_ks_terms(network, points, create_graph_for_backward=False):
    """Evaluate canonical KS derivatives and residual at ``(xi, tau)``."""

    values = network(points)
    first = torch.autograd.grad(
        values,
        points,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
    )[0]
    v_xi = first[:, 0:1]
    v_tau = first[:, 1:2]
    v_xixi = torch.autograd.grad(
        v_xi,
        points,
        grad_outputs=torch.ones_like(v_xi),
        create_graph=True,
    )[0][:, 0:1]
    v_xixixi = torch.autograd.grad(
        v_xixi,
        points,
        grad_outputs=torch.ones_like(v_xixi),
        create_graph=True,
    )[0][:, 0:1]
    v_xixixixi = torch.autograd.grad(
        v_xixixi,
        points,
        grad_outputs=torch.ones_like(v_xixixi),
        create_graph=create_graph_for_backward,
    )[0][:, 0:1]
    residual = v_tau + values * v_xi + v_xixi + v_xixixixi
    return {
        "v": values,
        "v_tau": v_tau,
        "v_xi": v_xi,
        "v_xixi": v_xixi,
        "v_xixixixi": v_xixixixi,
        "residual": residual,
    }


def _optimizer_defaults(args):
    args.adam_epsilon = 1e-8 if args.adam_epsilon is None else args.adam_epsilon
    args.soap_epsilon = 1e-8 if args.soap_epsilon is None else args.soap_epsilon
    args.muon_adam_epsilon = (
        1e-10 if args.muon_adam_epsilon is None else args.muon_adam_epsilon
    )
    args.mop_adam_epsilon = (
        1e-10 if args.mop_adam_epsilon is None else args.mop_adam_epsilon
    )
    args.kl_epsilon = 1e-8 if args.kl_epsilon is None else args.kl_epsilon


def _validate_args(args):
    if args.iterations <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("iterations, batch-size, and log-every must be positive")
    if args.pinn_points <= 0 or args.pinn_batch_size <= 0:
        raise ValueError("pinn-points and pinn-batch-size must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("eval-batch-size must be positive")
    if args.ic_batch_size <= 0 or args.ic_eval_points <= 0:
        raise ValueError("ic-batch-size and ic-eval-points must be positive")
    if not math.isfinite(args.ic_weight) or args.ic_weight < 0.0:
        raise ValueError("ic-weight must be finite and non-negative")
    if args.lr <= 0.0 or args.lr_min < 0.0 or args.lr_min > args.lr:
        raise ValueError("Require 0 <= lr-min <= lr and lr > 0")
    if args.grad_clip <= 0.0:
        raise ValueError("grad-clip must be positive")


def _predict_canonical(network, points, batch_size, device):
    dtype = next(network.parameters()).dtype
    predictions = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.as_tensor(
                points[start : start + batch_size], dtype=dtype, device=device
            )
            predictions.append(network(batch).detach().cpu().numpy())
    return np.vstack(predictions)


def physical_prediction_metrics(
    network, canonical_points, physical_values, pde, batch_size, device
):
    canonical_prediction = _predict_canonical(
        network, canonical_points, batch_size, device
    )
    physical_prediction = np.asarray(
        pde.to_physical_outputs(canonical_prediction), dtype=np.float64
    )
    physical_values = np.asarray(physical_values, dtype=np.float64)
    error = physical_prediction - physical_values
    squared_error = float(np.sum(error**2))
    squared_reference = float(np.sum(physical_values**2))
    mse = squared_error / error.size
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": (
            math.sqrt(squared_error / squared_reference)
            if squared_reference > 0.0
            else None
        ),
    }, physical_prediction


def evaluate_canonical_residual(
    network, bounds, count, batch_size, seed, device
):
    dtype = next(network.parameters()).dtype
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    lower = np.asarray(bounds[0], dtype=numpy_dtype)
    upper = np.asarray(bounds[1], dtype=numpy_dtype)
    rng = np.random.default_rng(seed)
    square_sum = 0.0
    residual_count = 0
    network.eval()
    for start in range(0, count, batch_size):
        current_count = min(batch_size, count - start)
        sample = rng.uniform(lower, upper, size=(current_count, 2)).astype(
            numpy_dtype
        )
        points = torch.as_tensor(sample, dtype=dtype, device=device).requires_grad_(
            True
        )
        residual = canonical_ks_terms(network, points)["residual"]
        square_sum += float(torch.sum(residual.detach().double().square()).cpu())
        residual_count += residual.numel()
    return square_sum / residual_count


def evaluate_canonical_ic(network, pde, bounds, count, batch_size, device):
    dtype = next(network.parameters()).dtype
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    xi = np.linspace(bounds[0][0], bounds[1][0], count, dtype=numpy_dtype)
    points_numpy = np.column_stack(
        (xi, np.full(count, bounds[0][1], dtype=numpy_dtype))
    )
    exact_numpy = pde.canonical_initial_condition(points_numpy)
    square_sum = 0.0
    network.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            points = torch.as_tensor(
                points_numpy[start:stop], dtype=dtype, device=device
            )
            exact = torch.as_tensor(
                exact_numpy[start:stop], dtype=dtype, device=device
            )
            error = network(points) - exact
            square_sum += float(torch.sum(error.detach().double().square()).cpu())
    return square_sum / count


def train_canonical_pinn(
    args,
    network,
    bounds,
    pde,
    canonical_reference_points,
    physical_reference_values,
    device,
    run_dir,
    metadata,
):
    dtype = TORCH_DTYPES[args.precision]
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    lower = np.asarray(bounds[0], dtype=numpy_dtype)
    upper = np.asarray(bounds[1], dtype=numpy_dtype)
    optimizer = build_training_optimizer(network, args)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.iterations, 1), eta_min=args.lr_min
        )
        if args.lr_min < args.lr
        else None
    )
    rng = np.random.default_rng(args.seed + 1)
    history = []

    for iteration in range(1, args.iterations + 1):
        network.train()
        sample = rng.uniform(lower, upper, size=(args.batch_size, 2)).astype(
            numpy_dtype
        )
        points = torch.as_tensor(sample, dtype=dtype, device=device).requires_grad_(
            True
        )
        residual = canonical_ks_terms(
            network, points, create_graph_for_backward=True
        )["residual"]
        residual_for_loss = residual.float() if residual.dtype == torch.float16 else residual
        canonical_residual_mse = torch.mean(residual_for_loss.square())

        xi = rng.uniform(
            lower[0], upper[0], size=(args.ic_batch_size, 1)
        ).astype(numpy_dtype)
        ic_points_numpy = np.hstack(
            (xi, np.full_like(xi, lower[1], dtype=numpy_dtype))
        )
        exact_ic_numpy = pde.canonical_initial_condition(ic_points_numpy)
        ic_points = torch.as_tensor(ic_points_numpy, dtype=dtype, device=device)
        exact_ic = torch.as_tensor(exact_ic_numpy, dtype=dtype, device=device)
        canonical_ic_mse = torch.mean((network(ic_points) - exact_ic).square())
        total_loss = canonical_residual_mse + args.ic_weight * canonical_ic_mse

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            network.parameters(), args.grad_clip
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            canonical_value = float(canonical_residual_mse.detach().cpu())
            physical_value = float(pde.to_physical_residual_mse(canonical_value))
            canonical_ic_value = float(canonical_ic_mse.detach().cpu())
            physical_ic_value = canonical_ic_value * pde.solution_scale**2
            total_value = float(total_loss.detach().cpu())
            reference_metric, _ = physical_prediction_metrics(
                network,
                canonical_reference_points,
                physical_reference_values,
                pde,
                args.eval_batch_size,
                device,
            )
            physical_l2re = reference_metric["relative_l2"]
            row = {
                "iteration": iteration,
                "loss_total": total_value,
                "canonical_residual_mse": canonical_value,
                "physical_residual_mse": physical_value,
                "canonical_ic_mse": canonical_ic_value,
                "physical_ic_mse": physical_ic_value,
                "ic_weight": args.ic_weight,
                "weighted_ic_contribution": args.ic_weight * canonical_ic_value,
                "physical_reference_l2re": physical_l2re,
                "data_weight": 0.0,
                "u_t_weight": 0.0,
                "u_x_weight": 0.0,
                "u_xx_weight": 0.0,
                "u_xxxx_weight": 0.0,
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"step={iteration:7d} "
                f"canonical_residual_mse={canonical_value:.6e} "
                f"physical_residual_mse={physical_value:.6e} "
                f"canonical_ic_mse={canonical_ic_value:.6e} "
                f"physical_ic_mse={physical_ic_value:.6e} "
                f"physical_reference_l2re={physical_l2re:.6e} "
                f"grad_norm={row['grad_norm']:.3e} lr={row['lr']:.3e}"
            )

    save_checkpoint(run_dir / "weights_student.pt", network, metadata)
    save_checkpoint(run_dir / "weights_last.pt", network, metadata)
    columns = list(history[0])
    np.savetxt(
        run_dir / "history.csv",
        np.asarray([[row[column] for column in columns] for row in history]),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    return history


def run(args):
    _optimizer_defaults(args)
    _validate_args(args)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    physical_points, physical_values = load_data(args.data, precision="float64")
    physical_bbox = [
        float(np.min(physical_points[:, 0])),
        float(np.max(physical_points[:, 0])),
        float(np.min(physical_points[:, 1])),
        float(np.max(physical_points[:, 1])),
    ]
    pde = CanonicalKuramotoSivashinskyEquation(
        datapath=args.data, bbox=physical_bbox
    )
    physical_data = np.column_stack((physical_points, physical_values))
    canonical_data = pde.to_canonical_data(physical_data)
    canonical_points = canonical_data[:, :2]
    canonical_values = canonical_data[:, 2:3]
    canonical_lower = np.asarray([pde.bbox[0], pde.bbox[2]], dtype=np.float64)
    canonical_upper = np.asarray([pde.bbox[1], pde.bbox[3]], dtype=np.float64)

    dde.config.set_default_float(args.precision)
    network, metadata = build_student(
        args, canonical_points, canonical_values, device
    )
    metadata.update(
        training_objective="canonical_strong_form_residual_and_initial_condition",
        coordinate_system="canonical_ks",
        canonical_coefficients={"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
        physical_coefficients={
            "alpha": pde.physical_alpha,
            "beta": pde.physical_beta,
            "gamma": pde.physical_gamma,
        },
        canonical_scales={
            "length": pde.length_scale,
            "time": pde.time_scale,
            "solution": pde.solution_scale,
            "residual": pde.residual_scale,
            "residual_mse": pde.residual_mse_scale,
        },
        loss_weights={
            "u": 0.0,
            "u_t": 0.0,
            "u_x": 0.0,
            "u_xx": 0.0,
            "u_xxxx": 0.0,
            "canonical_residual": 1.0,
            "canonical_ic": args.ic_weight,
        },
    )

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-ks-canonical-pinn-{args.network}-{args.optimizer}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        **vars(args),
        "data": str(Path(args.data).resolve()),
        "device": str(device),
        "physical_bbox": physical_bbox,
        "canonical_bbox": pde.bbox,
        "loss_weights": metadata["loss_weights"],
        "model_metadata": metadata,
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(configuration, file_obj, indent=2, sort_keys=True)

    history = train_canonical_pinn(
        args,
        network,
        (canonical_lower, canonical_upper),
        pde,
        canonical_points,
        physical_values,
        device,
        run_dir,
        metadata,
    )
    canonical_residual_mse = evaluate_canonical_residual(
        network,
        (canonical_lower, canonical_upper),
        args.pinn_points,
        args.pinn_batch_size,
        args.seed + 2,
        device,
    )
    physical_residual_mse = float(
        pde.to_physical_residual_mse(canonical_residual_mse)
    )
    canonical_ic_mse = evaluate_canonical_ic(
        network,
        pde,
        (canonical_lower, canonical_upper),
        args.ic_eval_points,
        args.eval_batch_size,
        device,
    )
    physical_ic_mse = canonical_ic_mse * pde.solution_scale**2
    data_metric, physical_prediction = physical_prediction_metrics(
        network,
        canonical_points,
        physical_values,
        pde,
        args.eval_batch_size,
        device,
    )

    np.savez_compressed(
        run_dir / "predictions.npz",
        x=physical_points[:, 0],
        t=physical_points[:, 1],
        exact=physical_values[:, 0],
        prediction=physical_prediction[:, 0],
    )
    save_solution_plot(
        run_dir / "solution.png",
        physical_points,
        physical_values[:, 0],
        physical_prediction[:, 0],
        f"Canonical KS PINN, physical L2RE={data_metric['relative_l2']:.3e}",
    )
    metrics = {
        "canonical_residual_mse": canonical_residual_mse,
        "physical_residual_mse": physical_residual_mse,
        "canonical_ic_mse": canonical_ic_mse,
        "physical_ic_mse": physical_ic_mse,
        "physical_data_diagnostic": data_metric,
        "last_training_row": history[-1],
        "configuration": configuration,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)

    print(
        "Canonical KS PINN: "
        f"canonical PDE MSE={canonical_residual_mse:.6e}; "
        f"physical PDE MSE={physical_residual_mse:.6e}; "
        f"canonical IC MSE={canonical_ic_mse:.6e}; "
        f"physical IC MSE={physical_ic_mse:.6e}; "
        f"physical L2RE={data_metric['relative_l2']:.6e}; "
        f"artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=str(PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat")
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "runs_data_ks_canonical_pinn")
    )
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument(
        "--network", "--network-type", choices=["mlp", "rwf"], default="rwf"
    )
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument(
        "--precision", choices=["float32", "float64"], default="float64"
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--optimizer",
        choices=[
            "adam", "rmsprop", "madam", "soap", "kl-shampoo", "kl-soap", "kl-m-soap", "muon",
            "mop", "mousse", "psgdpro", "pcgpro", "polargrad", "rekls-v3",
        ],
        default="rekls-v3",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-min", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--pinn-points", type=int, default=20000)
    parser.add_argument("--pinn-batch-size", type=int, default=32)
    parser.add_argument("--ic-weight", type=float, default=1.0)
    parser.add_argument("--ic-batch-size", type=int, default=512)
    parser.add_argument("--ic-eval-points", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument(
        "--matrix-fallback", choices=["adam", "soap"], default="soap"
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

    # Kept as hidden compatibility flags; this script always overrides them.
    for option in ("data", "ut", "ux", "uxx", "uxxxx"):
        parser.add_argument(
            f"--{option}-weight", type=float, default=0.0, help=argparse.SUPPRESS
        )
    parser.add_argument("--pde-weight", type=float, default=1.0, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    args.data_weight = 0.0
    args.ut_weight = 0.0
    args.ux_weight = 0.0
    args.uxx_weight = 0.0
    args.uxxxx_weight = 0.0
    args.pde_weight = 1.0
    return args


if __name__ == "__main__":
    run(parse_args())
