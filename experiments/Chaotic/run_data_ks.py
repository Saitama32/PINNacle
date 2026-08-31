"""Supervised (data-driven) training for the Kuramoto--Sivashinsky data set.

The network is trained only on ``(x, t) -> u`` observations.  After training,
the script evaluates the losses that the regular KS PINN would have seen:
the differential-equation residual and the initial-condition error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import deepxde as dde

from src.model import RWFMLP
from src.utils.args import parse_hidden_layers


KS_ALPHA = 100.0 / 16.0
KS_BETA = 100.0 / (16.0**2)
KS_GAMMA = 100.0 / (16.0**4)
TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
NUMPY_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


def load_data(path: os.PathLike, precision: str = "float32") -> tuple[np.ndarray, np.ndarray]:
    """Load a finite three-column ``x, t, u`` data set."""

    raw = np.loadtxt(path, comments="%", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("KS data must have at least three columns: x, t, u")
    raw = raw[:, :3]
    if len(raw) < 2:
        raise ValueError("KS data must contain at least two observations")
    if not np.isfinite(raw).all():
        raise ValueError("KS data contains NaN or infinite values")
    numpy_dtype = NUMPY_DTYPES[precision]
    points = raw[:, :2].astype(numpy_dtype)
    values = raw[:, 2:3].astype(numpy_dtype)
    if np.any(np.ptp(points, axis=0) <= 0):
        raise ValueError("Both x and t must vary in the KS data")
    return points, values


def split_indices(
    count: int,
    test_fraction: float,
    min_points_for_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a reproducible point-wise train/test split when data is sufficient."""

    if count < 2:
        raise ValueError("At least two points are required")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if min_points_for_test < 2:
        raise ValueError("min_points_for_test must be at least 2")
    permutation = np.random.default_rng(seed).permutation(count)
    if test_fraction == 0.0 or count < min_points_for_test:
        return permutation, np.empty(0, dtype=np.int64)
    test_count = min(max(1, int(round(count * test_fraction))), count - 1)
    return permutation[test_count:], permutation[:test_count]


def _normalization_transform(lower, scale):
    lower_values = tuple(float(value) for value in lower)
    scale_values = tuple(float(value) for value in scale)

    def transform(inputs):
        lower_tensor = inputs.new_tensor(lower_values)
        scale_tensor = inputs.new_tensor(scale_values)
        return 2.0 * (inputs - lower_tensor) / scale_tensor - 1.0

    return transform


def _output_transform(mean: float, std: float):
    def transform(_, outputs):
        return outputs * std + mean

    return transform


def build_network(metadata: dict) -> torch.nn.Module:
    """Build the dense or RWF MLP represented by checkpoint metadata."""

    precision = metadata.get("precision", "float32")
    dde.config.set_default_float(precision)
    model_type = metadata.get("model", "RWFMLP").lower()
    if model_type == "rwfmlp":
        network = RWFMLP(
            metadata["layer_sizes"],
            mu=metadata["rwf_mu"],
            sigma=metadata["rwf_sigma"],
        )
    elif model_type in {"mlp", "fnn"}:
        network = dde.nn.FNN(
            metadata["layer_sizes"], "tanh", "Glorot normal"
        )
    else:
        raise ValueError(f"Unsupported network model in metadata: {metadata.get('model')!r}")
    network = network.to(dtype=TORCH_DTYPES[precision])
    network.apply_feature_transform(
        _normalization_transform(metadata["input_min"], metadata["input_scale"])
    )
    network.apply_output_transform(
        _output_transform(metadata["output_mean"], metadata["output_std"])
    )
    return network


def build_optimizer(network: torch.nn.Module, args) -> torch.optim.Optimizer:
    """Build one of the repository's Adam, RMSprop, Muon, or SOAP optimizers."""

    if args.optimizer == "adam":
        return torch.optim.Adam(
            network.parameters(),
            lr=args.lr,
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "soap":
        dde.optimizers.set_SOAP_options(
            beta1=args.soap_beta1,
            beta2=args.soap_beta2,
            shampoo_beta=args.soap_shampoo_beta,
            epsilon=args.soap_epsilon,
            precondition_frequency=args.soap_precondition_frequency,
            max_precondition_dim=args.soap_max_precondition_dim,
            bias_correction=args.soap_bias_correction,
        )
    elif args.optimizer == "kl-m-soap":
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
    elif args.optimizer == "madam":
        dde.optimizers.set_MADAM_options(
            betas=(args.madam_beta1, args.madam_beta2),
            scale_log2=args.madam_scale_log2,
            correct_bias=args.madam_bias_correction,
        )
    elif args.optimizer == "muown":
        dde.optimizers.set_MUOWN_options(
            momentum=args.muown_momentum,
            betas=(args.muown_beta1, args.muown_beta2),
            adam_eps=args.muown_adam_epsilon,
            fp32_matmul_precision=args.muown_fp32_matmul_precision,
            coefficient_type=args.muown_coefficient_type,
            ns_steps=args.muown_ns_steps,
            scale_mode=args.muown_scale_mode,
            extra_scale_factor=args.muown_extra_scale_factor,
            muown_weight_decay=args.muown_weight_decay,
            auxiliary_optimizer=args.muown_auxiliary_optimizer,
            auxiliary_lr=args.muown_auxiliary_lr,
            auxiliary_betas=(args.muown_auxiliary_beta1, args.muown_auxiliary_beta2),
            auxiliary_eps=args.muown_auxiliary_epsilon,
            auxiliary_weight_decay=args.muown_auxiliary_weight_decay,
        )
    elif args.optimizer == "muon":
        dde.optimizers.set_MUON_options(
            momentum=args.muon_momentum,
            nesterov=args.muon_nesterov,
            ns_steps=args.muon_ns_steps,
            adam_lr=args.muon_adam_lr,
            adam_betas=(args.muon_adam_beta1, args.muon_adam_beta2),
            adam_eps=args.muon_adam_epsilon,
            muon_weight_decay=args.muon_weight_decay,
            adam_weight_decay=args.muon_adam_weight_decay,
        )
    optimizer, _ = dde.optimizers.get(
        network.parameters(),
        args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        model=network,
    )
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, args):
    """Build an iteration-based learning-rate scheduler."""

    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "exponential":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: args.lr_decay_rate
            ** (step / args.lr_decay_steps),
        )
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.iterations,
            eta_min=args.lr_min,
        )
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_steps,
            gamma=args.lr_decay_rate,
        )
    raise ValueError(f"Unsupported learning-rate scheduler: {args.lr_scheduler}")


def load_checkpoint(path: os.PathLike, device="cpu") -> tuple[torch.nn.Module, dict]:
    """Restore a model saved by this script."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint["metadata"]
    network = build_network(metadata).to(device)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.eval()
    return network, metadata


def _cpu_state_dict(network):
    return {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}


def save_checkpoint(path: Path, network: torch.nn.Module, metadata: dict) -> None:
    torch.save({"state_dict": _cpu_state_dict(network), "metadata": metadata}, path)


def save_solution_plot(path: os.PathLike, points, exact, prediction, title: str) -> None:
    """Save exact, predicted, and absolute-error KS fields as a PNG."""

    import matplotlib.pyplot as plt

    points = np.asarray(points)
    exact = np.asarray(exact).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if len(points) != len(exact) or len(exact) != len(prediction):
        raise ValueError("points, exact, and prediction must have the same length")

    x = np.unique(points[:, 0])
    t = np.unique(points[:, 1])
    is_rectangular_grid = len(points) == len(x) * len(t)
    error = np.abs(prediction - exact)
    solution_min = float(min(np.min(exact), np.min(prediction)))
    solution_max = float(max(np.max(exact), np.max(prediction)))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    if is_rectangular_grid:
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
            axes[0].tricontourf(points[:, 0], points[:, 1], exact, levels=100, cmap="jet",
                                vmin=solution_min, vmax=solution_max),
            axes[1].tricontourf(points[:, 0], points[:, 1], prediction, levels=100, cmap="jet",
                                vmin=solution_min, vmax=solution_max),
            axes[2].tricontourf(points[:, 0], points[:, 1], error, levels=100, cmap="magma"),
        ]

    for axis, image, label in zip(
        axes, images, ("Exact solution", "Data-driven RWF prediction", "Absolute error")
    ):
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_solution_plot_from_artifacts(run_dir: os.PathLike) -> Path:
    """Create ``solution.png`` for an already completed data-driven run."""

    run_dir = Path(run_dir)
    with np.load(run_dir / "predictions.npz") as data:
        points = np.column_stack((data["x"], data["t"]))
        exact = data["exact"]
        prediction = data["prediction"]
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as file_obj:
        relative_l2 = json.load(file_obj)["all_data"]["relative_l2"]
    output_path = run_dir / "solution.png"
    save_solution_plot(
        output_path,
        points,
        exact,
        prediction,
        f"Data-driven KS, relative L2={relative_l2:.3e}",
    )
    return output_path


def prediction_metrics(network, points, values, batch_size: int, device) -> dict:
    squared_error = 0.0
    squared_reference = 0.0
    absolute_error = 0.0
    count = 0
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            inputs = torch.as_tensor(points[start:stop], device=device)
            targets = torch.as_tensor(values[start:stop], device=device)
            error = network(inputs) - targets
            metric_error = error.double()
            metric_targets = targets.double()
            squared_error += float(torch.sum(metric_error.square()).cpu())
            squared_reference += float(torch.sum(metric_targets.square()).cpu())
            absolute_error += float(torch.sum(metric_error.abs()).cpu())
            count += error.numel()
    mse = squared_error / count
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": absolute_error / count,
        "relative_l2": math.sqrt(squared_error / squared_reference)
        if squared_reference > 0
        else None,
    }


def ks_terms(
    network,
    points,
    alpha=KS_ALPHA,
    beta=KS_BETA,
    gamma=KS_GAMMA,
    create_graph_for_backward: bool = False,
):
    """Evaluate the physical-coordinate KS derivatives and equation terms."""

    values = network(points)
    first = torch.autograd.grad(
        values, points, grad_outputs=torch.ones_like(values), create_graph=True
    )[0]
    u_x = first[:, 0:1]
    u_t = first[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x, points, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]
    u_xxx = torch.autograd.grad(
        u_xx, points, grad_outputs=torch.ones_like(u_xx), create_graph=True
    )[0][:, 0:1]
    u_xxxx = torch.autograd.grad(
        u_xxx,
        points,
        grad_outputs=torch.ones_like(u_xxx),
        create_graph=create_graph_for_backward,
    )[0][:, 0:1]
    term_adv = alpha * values * u_x
    term_diff = beta * u_xx
    term_hyper = gamma * u_xxxx
    residual = u_t + term_adv + term_diff + term_hyper
    return {
        "u": values,
        "u_t": u_t,
        "u_x": u_x,
        "u_xx": u_xx,
        "u_xxx": u_xxx,
        "u_xxxx": u_xxxx,
        "term_t": u_t,
        "term_adv": term_adv,
        "term_diff": term_diff,
        "term_hyper": term_hyper,
        "residual": residual,
    }


def ks_residual(network, points, alpha=KS_ALPHA, beta=KS_BETA, gamma=KS_GAMMA):
    """Evaluate the physical-coordinate KS residual with autograd."""

    return ks_terms(network, points, alpha=alpha, beta=beta, gamma=gamma)["residual"]


def _plot_grid_fields(path, x, t, fields, title: str, log_absolute: bool = False):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    for axis, (label, raw_values) in zip(axes.flat, fields):
        values = np.log10(1.0 + np.abs(raw_values)) if log_absolute else raw_values
        if log_absolute:
            image = axis.pcolormesh(x, t, values, shading="auto", cmap="magma")
        else:
            limit = float(np.quantile(np.abs(values), 0.99))
            if not math.isfinite(limit) or limit <= 0:
                limit = 1.0
            image = axis.pcolormesh(
                x, t, values, shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit
            )
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    for axis in axes.flat[len(fields) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_derivative_grid(
    network,
    bounds,
    alpha: float,
    beta: float,
    gamma: float,
    nx: int,
    nt: int,
    batch_size: int,
    output_dir: os.PathLike,
    device,
) -> dict:
    """Save post-training derivative, KS-term, and residual diagnostics."""

    network_dtype = next(network.parameters()).dtype
    numpy_dtype = NUMPY_DTYPES[str(network_dtype).split(".")[-1]]
    lower = np.asarray(bounds[0], dtype=numpy_dtype)
    upper = np.asarray(bounds[1], dtype=numpy_dtype)
    x = np.linspace(lower[0], upper[0], nx, dtype=numpy_dtype)
    t = np.linspace(lower[1], upper[1], nt, dtype=numpy_dtype)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    grid_points = np.column_stack((xx.reshape(-1), tt.reshape(-1))).astype(numpy_dtype)
    keys = (
        "u", "u_t", "u_x", "u_xx", "u_xxx", "u_xxxx",
        "term_t", "term_adv", "term_diff", "term_hyper", "residual",
    )
    chunks = {key: [] for key in keys}
    network.eval()
    for start in range(0, len(grid_points), batch_size):
        points = torch.as_tensor(
            grid_points[start : start + batch_size], device=device
        ).requires_grad_(True)
        terms = ks_terms(network, points, alpha=alpha, beta=beta, gamma=gamma)
        for key in keys:
            chunks[key].append(terms[key].detach().cpu().numpy().reshape(-1))
    fields = {
        key: np.concatenate(values).reshape(nt, nx) for key, values in chunks.items()
    }

    output_dir = Path(output_dir)
    np.savez_compressed(output_dir / "ks_derivative_grid.npz", x=x, t=t, **fields)
    derivative_fields = [
        ("u", fields["u"]),
        ("u_t", fields["u_t"]),
        ("u_x", fields["u_x"]),
        ("u_xx", fields["u_xx"]),
        ("u_xxx", fields["u_xxx"]),
        ("u_xxxx", fields["u_xxxx"]),
    ]
    _plot_grid_fields(
        output_dir / "ks_derivatives.png",
        x,
        t,
        derivative_fields,
        "KS derivatives from the trained data-driven network",
    )
    _plot_grid_fields(
        output_dir / "ks_log_abs_derivatives.png",
        x,
        t,
        derivative_fields,
        "KS log10(1 + absolute derivative)",
        log_absolute=True,
    )
    equation_fields = [
        ("u_t", fields["term_t"]),
        ("alpha * u * u_x", fields["term_adv"]),
        ("beta * u_xx", fields["term_diff"]),
        ("gamma * u_xxxx", fields["term_hyper"]),
        ("KS residual", fields["residual"]),
    ]
    _plot_grid_fields(
        output_dir / "ks_pde_terms.png",
        x,
        t,
        equation_fields,
        "KS equation terms and residual",
    )
    _plot_grid_fields(
        output_dir / "ks_log_abs_pde_terms.png",
        x,
        t,
        equation_fields,
        "KS log10(1 + absolute equation term)",
        log_absolute=True,
    )

    summary = {"nx": nx, "nt": nt, "num_points": nx * nt}
    for key in keys:
        values = fields[key].astype(np.float64)
        summary[f"rms_{key}"] = float(np.sqrt(np.mean(values**2)))
        summary[f"max_abs_{key}"] = float(np.max(np.abs(values)))
    return summary


def save_derivative_diagnostics_from_artifacts(
    run_dir: os.PathLike,
    device="auto",
    nx: int = 128,
    nt: int = 64,
    batch_size: int = 512,
) -> dict:
    """Generate derivative plots for an already completed run."""

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    network, metadata = load_checkpoint(Path(run_dir) / "weights_best.pt", device=device)
    numpy_dtype = NUMPY_DTYPES[metadata.get("precision", "float32")]
    lower = np.asarray(metadata["input_min"], dtype=numpy_dtype)
    upper = lower + np.asarray(metadata["input_scale"], dtype=numpy_dtype)
    summary = evaluate_derivative_grid(
        network,
        (lower, upper),
        alpha=float(metadata.get("alpha", KS_ALPHA)),
        beta=float(metadata.get("beta", KS_BETA)),
        gamma=float(metadata.get("gamma", KS_GAMMA)),
        nx=nx,
        nt=nt,
        batch_size=batch_size,
        output_dir=run_dir,
        device=torch.device(device),
    )
    metrics_path = Path(run_dir) / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file_obj:
            metrics = json.load(file_obj)
        metrics["derivative_grid"] = summary
        with metrics_path.open("w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, indent=2, sort_keys=True)
    return summary


def evaluate_pinn_loss(network, bounds, args, device) -> dict:
    """Estimate the same PDE and analytic IC MSEs used by the standard KS PINN."""

    rng = np.random.default_rng(args.seed + 1)
    numpy_dtype = NUMPY_DTYPES[args.precision]
    lower = np.asarray(bounds[0], dtype=numpy_dtype)
    upper = np.asarray(bounds[1], dtype=numpy_dtype)
    residual_square_sum = 0.0
    residual_count = 0
    network.eval()
    for start in range(0, args.pinn_points, args.pinn_batch_size):
        count = min(args.pinn_batch_size, args.pinn_points - start)
        sample = rng.uniform(lower, upper, size=(count, 2)).astype(numpy_dtype)
        points = torch.as_tensor(sample, device=device).requires_grad_(True)
        residual = ks_residual(
            network, points, alpha=args.alpha, beta=args.beta, gamma=args.gamma
        )
        residual_square_sum += float(torch.sum(residual.detach().double().square()).cpu())
        residual_count += residual.numel()

    x = rng.uniform(lower[0], upper[0], size=(args.pinn_ic_points, 1)).astype(numpy_dtype)
    t = np.full_like(x, lower[1])
    ic_points = torch.as_tensor(np.hstack((x, t)), device=device)
    exact_ic = torch.as_tensor(np.cos(x) * (1.0 + np.sin(x)), device=device)
    with torch.no_grad():
        ic_error = (network(ic_points) - exact_ic).double()
        ic_mse = float(torch.mean(ic_error.square()).cpu())
    pde_mse = residual_square_sum / residual_count
    return {
        "pde_mse": pde_mse,
        "ic_mse": ic_mse,
        "pinn_loss_unweighted": pde_mse + ic_mse,
        "pinn_loss_weighted": pde_mse + args.ic_loss_weight * ic_mse,
        "ic_loss_weight": args.ic_loss_weight,
        "num_domain_points": args.pinn_points,
        "num_initial_points": args.pinn_ic_points,
    }


def train_pinn_stage(
    network,
    bounds,
    validation_points,
    validation_values,
    args,
    device,
    metadata,
    run_dir: Path,
) -> dict:
    """Fine-tune supervised weights using only the KS PDE and IC losses."""

    stage_args = argparse.Namespace(**vars(args))
    stage_args.optimizer = args.pinn_optimizer
    stage_args.lr = args.pinn_lr
    stage_args.weight_decay = args.pinn_weight_decay
    stage_args.iterations = args.n_iter_pinn
    stage_args.lr_scheduler = args.pinn_lr_scheduler
    stage_args.lr_decay_steps = args.pinn_lr_decay_steps
    stage_args.lr_decay_rate = args.pinn_lr_decay_rate
    stage_args.lr_min = args.pinn_lr_min
    optimizer = build_optimizer(network, stage_args)
    scheduler = build_scheduler(optimizer, stage_args)

    numpy_dtype = NUMPY_DTYPES[args.precision]
    lower = np.asarray(bounds[0], dtype=numpy_dtype)
    upper = np.asarray(bounds[1], dtype=numpy_dtype)
    rng = np.random.default_rng(args.seed + 2)
    initial_metric = evaluate_pinn_loss(network, bounds, args, device)
    initial_data_metric = prediction_metrics(
        network, validation_points, validation_values, args.eval_batch_size, device
    )
    best_metric = dict(initial_metric)
    best_data_metric = dict(initial_data_metric)
    best_score = initial_metric["pinn_loss_weighted"]
    best_iteration = 0
    best_state = _cpu_state_dict(network)
    rows = []
    print(
        f"Starting PINN fine-tuning for {args.n_iter_pinn} iterations; "
        f"optimizer={args.pinn_optimizer}; lr={args.pinn_lr:.6e}; "
        f"initial weighted loss={best_score:.6e}"
    )

    for iteration in range(1, args.n_iter_pinn + 1):
        network.train()
        domain_numpy = rng.uniform(
            lower, upper, size=(args.pinn_train_domain_points, 2)
        ).astype(numpy_dtype)
        domain_points = torch.as_tensor(domain_numpy, device=device).requires_grad_(True)
        residual = ks_terms(
            network,
            domain_points,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            create_graph_for_backward=True,
        )["residual"]
        residual_for_loss = residual.float() if residual.dtype == torch.float16 else residual
        pde_loss = torch.mean(residual_for_loss.square())

        x_numpy = rng.uniform(
            lower[0], upper[0], size=(args.pinn_train_ic_points, 1)
        ).astype(numpy_dtype)
        t_numpy = np.full_like(x_numpy, lower[1])
        ic_points = torch.as_tensor(np.hstack((x_numpy, t_numpy)), device=device)
        exact_ic = torch.as_tensor(
            np.cos(x_numpy) * (1.0 + np.sin(x_numpy)), device=device
        )
        ic_error = network(ic_points) - exact_ic
        ic_error_for_loss = ic_error.float() if ic_error.dtype == torch.float16 else ic_error
        ic_loss = torch.mean(ic_error_for_loss.square())
        weighted_loss = pde_loss + args.ic_loss_weight * ic_loss

        optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        if args.pinn_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), args.pinn_grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if iteration % args.pinn_train_log_every == 0 or iteration == args.n_iter_pinn:
            fixed_metric = evaluate_pinn_loss(network, bounds, args, device)
            fixed_data_metric = prediction_metrics(
                network,
                validation_points,
                validation_values,
                args.eval_batch_size,
                device,
            )
            fixed_score = fixed_metric["pinn_loss_weighted"]
            if math.isfinite(fixed_score) and fixed_score < best_score:
                best_score = fixed_score
                best_iteration = iteration
                best_metric = dict(fixed_metric)
                best_data_metric = dict(fixed_data_metric)
                best_state = _cpu_state_dict(network)
            rows.append(
                [
                    iteration,
                    float(pde_loss.detach().cpu()),
                    float(ic_loss.detach().cpu()),
                    float(weighted_loss.detach().cpu()),
                    fixed_metric["pde_mse"],
                    fixed_metric["ic_mse"],
                    fixed_score,
                    fixed_data_metric["mse"],
                    fixed_data_metric["relative_l2"],
                    optimizer.param_groups[0]["lr"],
                ]
            )
            print(
                f"PINN fine-tune step={iteration:7d} "
                f"batch_weighted={rows[-1][3]:.6e} "
                f"fixed_pde={fixed_metric['pde_mse']:.6e} "
                f"fixed_ic={fixed_metric['ic_mse']:.6e} "
                f"fixed_weighted={fixed_score:.6e} "
                f"data_l2={fixed_data_metric['relative_l2']:.6e} "
                f"lr={rows[-1][9]:.6e}"
            )

    save_checkpoint(run_dir / "weights_pinn_last.pt", network, metadata)
    save_checkpoint(run_dir / "weights_last.pt", network, metadata)
    network.load_state_dict(best_state, strict=True)
    save_checkpoint(run_dir / "weights_pinn_best.pt", network, metadata)
    save_checkpoint(run_dir / "weights_best.pt", network, metadata)
    np.savetxt(
        run_dir / "pinn_finetune_history.csv",
        np.asarray(rows),
        delimiter=",",
        header=(
            "iteration,batch_pde_mse,batch_ic_mse,batch_weighted_loss,"
            "fixed_pde_mse,fixed_ic_mse,fixed_weighted_loss,"
            "data_mse,data_relative_l2,lr"
        ),
        comments="",
    )
    return {
        "enabled": True,
        "iterations": args.n_iter_pinn,
        "optimizer": args.pinn_optimizer,
        "best_iteration": best_iteration,
        "initial_pinn_loss": initial_metric,
        "initial_data_metric": initial_data_metric,
        "best_pinn_loss": best_metric,
        "best_data_metric": best_data_metric,
    }


def validate_args(args) -> None:
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.pinn_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.log_every <= 0:
        raise ValueError("log_every must be positive")
    if args.pinn_points <= 0 or args.pinn_ic_points <= 0:
        raise ValueError("PINN evaluation point counts must be positive")
    if args.pinn_log_every < 0:
        raise ValueError("pinn_log_every must be non-negative")
    if args.derivative_grid_nx < 2 or args.derivative_grid_nt < 2:
        raise ValueError("Derivative grid dimensions must be at least 2")
    if args.derivative_batch_size <= 0:
        raise ValueError("derivative_batch_size must be positive")
    if args.lr <= 0 or not math.isfinite(args.lr):
        raise ValueError("lr must be positive and finite")
    if args.lr_decay_steps <= 0:
        raise ValueError("lr_decay_steps must be positive")
    if args.lr_decay_rate <= 0 or not math.isfinite(args.lr_decay_rate):
        raise ValueError("lr_decay_rate must be positive and finite")
    if args.lr_min < 0 or not math.isfinite(args.lr_min):
        raise ValueError("lr_min must be finite and non-negative")
    if args.lr_scheduler == "cosine" and args.lr_min > args.lr:
        raise ValueError("lr_min cannot be greater than the initial lr")
    if (
        args.lr_scheduler == "cosine"
        and args.optimizer == "muon"
        and args.muon_adam_lr is not None
        and args.lr_min > args.muon_adam_lr
    ):
        raise ValueError("lr_min cannot be greater than the Muon auxiliary Adam lr")
    if args.adam_epsilon <= 0 or not math.isfinite(args.adam_epsilon):
        raise ValueError("adam_epsilon must be positive and finite")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError("weight_decay must be finite and non-negative")
    if args.ic_loss_weight < 0 or not math.isfinite(args.ic_loss_weight):
        raise ValueError("ic_loss_weight must be finite and non-negative")
    if not 0.0 <= args.soap_beta1 < 1.0 or not 0.0 <= args.soap_beta2 < 1.0:
        raise ValueError("SOAP beta values must be in [0, 1)")
    if args.soap_shampoo_beta is not None and not 0.0 <= args.soap_shampoo_beta < 1.0:
        raise ValueError("soap_shampoo_beta must be in [0, 1)")
    if args.soap_epsilon <= 0 or not math.isfinite(args.soap_epsilon):
        raise ValueError("soap_epsilon must be positive and finite")
    if args.soap_precondition_frequency <= 0 or args.soap_max_precondition_dim <= 0:
        raise ValueError("SOAP precondition settings must be positive")
    if not 0.0 <= args.muon_momentum < 1.0 or args.muon_ns_steps <= 0:
        raise ValueError("Muon momentum must be in [0, 1) and ns_steps must be positive")
    if args.muon_adam_lr is not None and args.muon_adam_lr <= 0:
        raise ValueError("muon_adam_lr must be positive")
    if not 0.0 <= args.muon_adam_beta1 < 1.0 or not 0.0 <= args.muon_adam_beta2 < 1.0:
        raise ValueError("Muon auxiliary Adam beta values must be in [0, 1)")
    if args.muon_adam_epsilon <= 0 or not math.isfinite(args.muon_adam_epsilon):
        raise ValueError("muon_adam_epsilon must be positive and finite")
    if args.muon_weight_decay < 0 or args.muon_adam_weight_decay < 0:
        raise ValueError("Muon weight decay values must be non-negative")


def run(args) -> Path:
    if args.precision == "float16":
        args.adam_epsilon = 1e-4 if args.adam_epsilon is None else args.adam_epsilon
        args.soap_epsilon = 1e-4 if args.soap_epsilon is None else args.soap_epsilon
        args.muon_adam_epsilon = (
            1e-4 if args.muon_adam_epsilon is None else args.muon_adam_epsilon
        )
        print(
            "Warning: float16 is experimental for fourth-order KS derivatives; "
            "overflow or loss of derivative accuracy is possible."
        )
    else:
        args.adam_epsilon = 1e-8 if args.adam_epsilon is None else args.adam_epsilon
        args.soap_epsilon = 1e-8 if args.soap_epsilon is None else args.soap_epsilon
        args.muon_adam_epsilon = (
            1e-10 if args.muon_adam_epsilon is None else args.muon_adam_epsilon
        )
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_path = Path(args.data).expanduser().resolve()
    points, values = load_data(data_path, precision=args.precision)
    train_indices, test_indices = split_indices(
        len(points), args.test_fraction, args.min_points_for_test, args.seed
    )
    train_points, train_values = points[train_indices], values[train_indices]
    test_points, test_values = points[test_indices], values[test_indices]

    hidden = parse_hidden_layers(args)
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden_layers must describe positive layer widths")
    input_min = np.min(points, axis=0)
    input_scale = np.max(points, axis=0) - input_min
    output_mean = float(np.mean(train_values, dtype=np.float64))
    output_std = float(np.std(train_values, dtype=np.float64))
    if not math.isfinite(output_std) or output_std <= 0:
        output_std = 1.0
    metadata = {
        "model": "RWFMLP",
        "precision": args.precision,
        "layer_sizes": [2, *hidden, 1],
        "rwf_mu": args.rwf_mu,
        "rwf_sigma": args.rwf_sigma,
        "input_min": input_min.tolist(),
        "input_scale": input_scale.tolist(),
        "output_mean": output_mean,
        "output_std": output_std,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
    }

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    network = build_network(metadata).to(device)
    optimizer = build_optimizer(network, args)
    scheduler = build_scheduler(optimizer, args)
    train_x = torch.as_tensor(train_points, device=device)
    train_y = torch.as_tensor(train_values, device=device)

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = (
        Path(args.out).expanduser().resolve()
        / f"{timestamp}-ks-data-rwf-{args.optimizer}-{args.precision}-lr-{args.lr_scheduler}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_dir / "split_indices.npz", train=train_indices, test=test_indices)
    resolved = vars(args).copy()
    resolved.update(
        data=str(data_path),
        device=str(device),
        num_points=len(points),
        num_train=len(train_indices),
        num_test=len(test_indices),
        model_metadata=metadata,
    )
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(resolved, file_obj, indent=2, sort_keys=True)

    history = []
    pinn_history = []
    best_score = math.inf
    best_iteration = 0
    best_state = _cpu_state_dict(network)
    print(
        f"Training RWF MLP on {len(train_indices)} points; test={len(test_indices)}; "
        f"optimizer={args.optimizer}; precision={args.precision}; "
        f"device={device}; artifacts={run_dir}"
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for iteration in range(1, args.iterations + 1):
        network.train()
        if args.batch_size >= len(train_x):
            batch_x, batch_y = train_x, train_y
        else:
            selection = torch.randint(
                len(train_x), (args.batch_size,), generator=generator, device=device
            )
            batch_x, batch_y = train_x[selection], train_y[selection]
        optimizer.zero_grad(set_to_none=True)
        batch_error = network(batch_x) - batch_y
        loss_error = batch_error.float() if batch_error.dtype == torch.float16 else batch_error
        loss = torch.mean(loss_error.square())
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if args.pinn_log_every and (
            iteration % args.pinn_log_every == 0 or iteration == args.iterations
        ):
            current_pinn = evaluate_pinn_loss(network, (input_min, input_min + input_scale), args, device)
            pinn_history.append(
                [
                    iteration,
                    current_pinn["pde_mse"],
                    current_pinn["ic_mse"],
                    current_pinn["pinn_loss_unweighted"],
                    current_pinn["pinn_loss_weighted"],
                ]
            )
            print(
                f"PINN step={iteration:7d} pde_mse={current_pinn['pde_mse']:.6e} "
                f"ic_mse={current_pinn['ic_mse']:.6e} "
                f"unweighted={current_pinn['pinn_loss_unweighted']:.6e} "
                f"weighted={current_pinn['pinn_loss_weighted']:.6e}"
            )

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            train_metric = prediction_metrics(
                network, train_points, train_values, args.eval_batch_size, device
            )
            test_metric = (
                prediction_metrics(network, test_points, test_values, args.eval_batch_size, device)
                if len(test_indices)
                else None
            )
            score = test_metric["mse"] if test_metric is not None else train_metric["mse"]
            if score < best_score:
                best_score = score
                best_iteration = iteration
                best_state = _cpu_state_dict(network)
            history.append(
                [iteration, float(loss.detach().cpu()), train_metric["mse"],
                 test_metric["mse"] if test_metric is not None else np.nan,
                 optimizer.param_groups[0]["lr"]]
            )
            test_text = f"{test_metric['mse']:.6e}" if test_metric else "disabled"
            print(
                f"step={iteration:7d} batch_mse={history[-1][1]:.6e} "
                f"train_mse={train_metric['mse']:.6e} test_mse={test_text} "
                f"lr={history[-1][4]:.6e}"
            )

    save_checkpoint(run_dir / "weights_supervised_last.pt", network, metadata)
    save_checkpoint(run_dir / "weights_last.pt", network, metadata)
    network.load_state_dict(best_state, strict=True)
    save_checkpoint(run_dir / "weights_supervised_best.pt", network, metadata)
    save_checkpoint(run_dir / "weights_best.pt", network, metadata)
    np.savetxt(
        run_dir / "history.csv",
        np.asarray(history),
        delimiter=",",
        header="iteration,batch_mse,train_mse,test_mse,lr",
        comments="",
    )
    if pinn_history:
        np.savetxt(
            run_dir / "pinn_history.csv",
            np.asarray(pinn_history),
            delimiter=",",
            header="iteration,pde_mse,ic_mse,pinn_loss_unweighted,pinn_loss_weighted",
            comments="",
        )

    train_metric = prediction_metrics(
        network, train_points, train_values, args.eval_batch_size, device
    )
    test_metric = (
        prediction_metrics(network, test_points, test_values, args.eval_batch_size, device)
        if len(test_indices)
        else None
    )
    all_metric = prediction_metrics(network, points, values, args.eval_batch_size, device)
    pinn_metric = evaluate_pinn_loss(
        network, (input_min, input_min + input_scale), args, device
    )
    derivative_metric = None
    if args.derivative_plots:
        derivative_metric = evaluate_derivative_grid(
            network,
            (input_min, input_min + input_scale),
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            nx=args.derivative_grid_nx,
            nt=args.derivative_grid_nt,
            batch_size=args.derivative_batch_size,
            output_dir=run_dir,
            device=device,
        )
    metrics = {
        "best_iteration": best_iteration,
        "supervised_best_iteration": best_iteration,
        "selection_metric": "test_mse" if test_metric is not None else "train_mse",
        "train": train_metric,
        "test": test_metric,
        "all_data": all_metric,
        "pinn_loss": pinn_metric,
        "derivative_grid": derivative_metric,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)

    predictions = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), args.eval_batch_size):
            inputs = torch.as_tensor(points[start : start + args.eval_batch_size], device=device)
            predictions.append(network(inputs).cpu().numpy())
    prediction = np.vstack(predictions)[:, 0]
    np.savez_compressed(
        run_dir / "predictions.npz",
        x=points[:, 0],
        t=points[:, 1],
        exact=values[:, 0],
        prediction=prediction,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    save_solution_plot(
        run_dir / "solution.png",
        points,
        values[:, 0],
        prediction,
        f"Data-driven KS, relative L2={all_metric['relative_l2']:.3e}",
    )
    print(
        f"Finished: supervised best step={best_iteration}, "
        f"all-data relative L2={all_metric['relative_l2']:.6e}, "
        f"weighted PINN loss={pinn_metric['pinn_loss_weighted']:.6e}; "
        f"artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Train a supervised RWF MLP on x,t,u KS data and evaluate its PINN loss."
    )
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"))
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "runs_data_ks"))
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "rmsprop", "madam", "muon", "muown", "soap", "kl-m-soap"],
        default="soap",
    )
    parser.add_argument(
        "--precision", choices=["float16", "float32", "float64"], default="float32"
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "exponential", "cosine", "step"],
        default="cosine",
    )
    parser.add_argument("--lr-decay-steps", type=int, default=1000)
    parser.add_argument("--lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=None)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-epsilon", type=float, default=None)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)
    soap_bias_group = parser.add_mutually_exclusive_group()
    soap_bias_group.add_argument(
        "--soap-bias-correction", dest="soap_bias_correction", action="store_true"
    )
    soap_bias_group.add_argument(
        "--no-soap-bias-correction", dest="soap_bias_correction", action="store_false"
    )
    parser.set_defaults(soap_bias_correction=True)
    parser.add_argument("--kl-m-soap-beta1", type=float, default=0.9)
    parser.add_argument("--kl-m-soap-beta2", type=float, default=0.95)
    parser.add_argument("--kl-m-soap-shampoo-beta", type=float, default=0.95)
    parser.add_argument("--kl-m-soap-epsilon", type=float, default=1e-8)
    parser.add_argument("--kl-m-soap-weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-m-soap-scale-log2", type=float, default=16.0)
    parser.add_argument("--kl-m-soap-auxiliary-lr", type=float, default=None)
    parser.add_argument("--kl-m-soap-auxiliary-beta1", type=float, default=0.9)
    parser.add_argument("--kl-m-soap-auxiliary-beta2", type=float, default=0.95)
    parser.add_argument("--kl-m-soap-auxiliary-scale-log2", type=float, default=16.0)
    parser.add_argument("--kl-m-soap-auxiliary-weight-decay", type=float, default=0.0)
    parser.add_argument("--madam-beta1", type=float, default=0.9)
    parser.add_argument("--madam-beta2", type=float, default=0.999)
    parser.add_argument("--madam-scale-log2", type=float, default=16.0)
    madam_bias_group = parser.add_mutually_exclusive_group()
    madam_bias_group.add_argument(
        "--madam-bias-correction", dest="madam_bias_correction", action="store_true"
    )
    madam_bias_group.add_argument(
        "--no-madam-bias-correction", dest="madam_bias_correction", action="store_false"
    )
    parser.set_defaults(madam_bias_correction=True)
    parser.add_argument("--muown-momentum", type=float, default=0.95)
    parser.add_argument("--muown-beta1", type=float, default=0.9)
    parser.add_argument("--muown-beta2", type=float, default=0.95)
    parser.add_argument("--muown-adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--muown-fp32-matmul-precision", choices=["medium", "high", "highest"], default="medium")
    parser.add_argument("--muown-coefficient-type", choices=["simple", "quintic", "polar_express", "cans", "aol", "deepseekv4", "cubic5"], default="quintic")
    parser.add_argument("--muown-ns-steps", type=int, default=5)
    parser.add_argument("--muown-scale-mode", choices=["shape_scaling", "spectral", "unit_rms_norm"], default="spectral")
    parser.add_argument("--muown-extra-scale-factor", type=float, default=1.0)
    parser.add_argument("--muown-weight-decay", type=float, default=0.0)
    parser.add_argument("--muown-auxiliary-optimizer", choices=["adam", "soap"], default="adam")
    parser.add_argument("--muown-auxiliary-lr", type=float, default=None)
    parser.add_argument("--muown-auxiliary-beta1", type=float, default=0.9)
    parser.add_argument("--muown-auxiliary-beta2", type=float, default=0.95)
    parser.add_argument("--muown-auxiliary-epsilon", type=float, default=1e-8)
    parser.add_argument("--muown-auxiliary-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    muon_nesterov_group = parser.add_mutually_exclusive_group()
    muon_nesterov_group.add_argument(
        "--muon-nesterov", dest="muon_nesterov", action="store_true"
    )
    muon_nesterov_group.add_argument(
        "--no-muon-nesterov", dest="muon_nesterov", action="store_false"
    )
    parser.set_defaults(muon_nesterov=True)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-adam-epsilon", type=float, default=None)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--min-points-for-test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--alpha", type=float, default=KS_ALPHA)
    parser.add_argument("--beta", type=float, default=KS_BETA)
    parser.add_argument("--gamma", type=float, default=KS_GAMMA)
    parser.add_argument("--ic-loss-weight", type=float, default=100.0)
    parser.add_argument("--pinn-points", type=int, default=8192)
    parser.add_argument("--pinn-ic-points", type=int, default=2048)
    parser.add_argument("--pinn-batch-size", type=int, default=512)
    parser.add_argument(
        "--pinn-log-every",
        type=int,
        default=1000,
        help="Evaluate and print fixed-point PINN losses every N iterations; 0 disables it.",
    )
    parser.add_argument(
        "--n-iter-pinn",
        type=int,
        default=1000,
        help="After supervised training, run this many KS PINN fine-tuning iterations.",
    )
    parser.add_argument(
        "--pinn-optimizer",
        choices=["adam", "rmsprop", "muon", "soap"],
        default="adam",
    )
    parser.add_argument("--pinn-lr", type=float, default=1e-6)
    parser.add_argument("--pinn-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--pinn-lr-scheduler",
        choices=["none", "exponential", "cosine", "step"],
        default="cosine",
    )
    parser.add_argument("--pinn-lr-decay-steps", type=int, default=1000)
    parser.add_argument("--pinn-lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--pinn-lr-min", type=float, default=1e-6)
    parser.add_argument("--pinn-train-domain-points", type=int, default=256)
    parser.add_argument("--pinn-train-ic-points", type=int, default=256)
    parser.add_argument("--pinn-train-log-every", type=int, default=100)
    parser.add_argument(
        "--pinn-grad-clip",
        type=float,
        default=1.0,
        help="Maximum PINN-stage gradient norm; 0 disables clipping.",
    )
    derivative_plot_group = parser.add_mutually_exclusive_group()
    derivative_plot_group.add_argument(
        "--derivative-plots", dest="derivative_plots", action="store_true"
    )
    derivative_plot_group.add_argument(
        "--no-derivative-plots", dest="derivative_plots", action="store_false"
    )
    parser.set_defaults(derivative_plots=True)
    parser.add_argument("--derivative-grid-nx", type=int, default=128)
    parser.add_argument("--derivative-grid-nt", type=int, default=64)
    parser.add_argument("--derivative-batch-size", type=int, default=512)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
