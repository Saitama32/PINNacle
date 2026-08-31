"""Train the two-field factorized KS system without data or Sobolev losses.

The network predicts ``(u, q)`` in physical coordinates and minimizes

    compatibility_weight * MSE(q - beta*u - gamma*u_xx)
    + dynamics_weight * MSE(u_t + alpha*u*u_x + q_xx)
    + ic_weight * MSE(u(x, t_0) - u_0(x)).

Reference values are used only for network normalization and diagnostics.
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

from src.model import RWFMLP
from src.pde.chaotic import (
    FactorizedKuramotoSivashinskyEquation,
    build_factorized_ks_reference,
)
from src.utils.args import parse_hidden_layers


TORCH_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def load_data(path):
    raw = np.loadtxt(path, comments="%", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("KS data must have at least three columns: x, t, u")
    raw = raw[:, :3]
    if len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("KS data must contain at least two finite observations")
    if np.any(np.ptp(raw[:, :2], axis=0) <= 0.0):
        raise ValueError("Both x and t must vary in the KS data")
    return raw


def _input_transform(lower, scale):
    lower = tuple(float(value) for value in lower)
    scale = tuple(float(value) for value in scale)

    def transform(inputs):
        return 2.0 * (inputs - inputs.new_tensor(lower)) / inputs.new_tensor(scale) - 1.0

    return transform


def _output_transform(mean, std):
    mean = tuple(float(value) for value in mean)
    std = tuple(float(value) for value in std)

    def transform(_, outputs):
        return outputs * outputs.new_tensor(std) + outputs.new_tensor(mean)

    return transform


def build_network(args, points, fields, device):
    hidden = parse_hidden_layers(args)
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden-layers must describe positive widths")
    input_min = np.min(points, axis=0)
    input_scale = np.ptp(points, axis=0)
    output_mean = np.mean(fields, axis=0, dtype=np.float64)
    output_std = np.std(fields, axis=0, dtype=np.float64)
    output_std = np.where(
        np.isfinite(output_std) & (output_std > 0.0), output_std, 1.0
    )
    layers = [2, *hidden, 2]
    dde.config.set_default_float(args.precision)
    if args.network == "rwf":
        network = RWFMLP(layers, mu=args.rwf_mu, sigma=args.rwf_sigma)
    else:
        network = dde.nn.FNN(layers, "tanh", "Glorot normal")
    network = network.to(dtype=TORCH_DTYPES[args.precision], device=device)
    network.apply_feature_transform(_input_transform(input_min, input_scale))
    network.apply_output_transform(_output_transform(output_mean, output_std))
    metadata = {
        "model": "RWFMLP" if args.network == "rwf" else "MLP",
        "precision": args.precision,
        "layer_sizes": layers,
        "rwf_mu": args.rwf_mu,
        "rwf_sigma": args.rwf_sigma,
        "input_min": input_min.tolist(),
        "input_scale": input_scale.tolist(),
        "output_mean": output_mean.tolist(),
        "output_std": output_std.tolist(),
        "outputs": ["u", "q"],
    }
    return network, metadata


def build_optimizer(network, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(
            network.parameters(), lr=args.lr, eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            network.parameters(), lr=args.lr, eps=args.adam_epsilon,
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
    elif args.optimizer == "rekls-v3":
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
    elif args.optimizer == "kl-m-soap":
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
    elif args.optimizer == "madam":
        dde.optimizers.set_MADAM_options(
            betas=(args.madam_beta1, args.madam_beta2),
            scale_log2=args.madam_scale_log2,
            correct_bias=args.madam_bias_correction,
        )
    optimizer, _ = dde.optimizers.get(
        network.parameters(), args.optimizer, learning_rate=args.lr,
        weight_decay=args.weight_decay, model=network,
    )
    return optimizer


def factorized_ks_terms(network, points, alpha, beta, gamma, backward=False):
    fields = network(points)
    u = fields[:, 0:1]
    q = fields[:, 1:2]
    first_u = torch.autograd.grad(
        u, points, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = first_u[:, 0:1]
    u_t = first_u[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x, points, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]
    q_x = torch.autograd.grad(
        q, points, grad_outputs=torch.ones_like(q), create_graph=True
    )[0][:, 0:1]
    q_xx = torch.autograd.grad(
        q_x,
        points,
        grad_outputs=torch.ones_like(q_x),
        create_graph=backward,
    )[0][:, 0:1]
    compatibility = q - beta * u - gamma * u_xx
    dynamics = u_t + alpha * u * u_x + q_xx
    return {
        "u": u,
        "q": q,
        "u_t": u_t,
        "u_x": u_x,
        "u_xx": u_xx,
        "q_xx": q_xx,
        "compatibility": compatibility,
        "dynamics": dynamics,
    }


def predict(network, points, batch_size, device):
    dtype = next(network.parameters()).dtype
    result = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.as_tensor(
                points[start : start + batch_size], dtype=dtype, device=device
            )
            result.append(network(batch).detach().cpu().numpy())
    return np.vstack(result)


def field_metrics(prediction, exact):
    prediction = np.asarray(prediction, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    error = prediction - exact
    squared_error = float(np.sum(error**2))
    squared_exact = float(np.sum(exact**2))
    mse = squared_error / error.size
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": (
            math.sqrt(squared_error / squared_exact) if squared_exact > 0.0 else None
        ),
    }


def evaluate_residuals(network, bounds, pde, args, device):
    dtype = next(network.parameters()).dtype
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    rng = np.random.default_rng(args.seed + 2)
    sums = {"compatibility": 0.0, "dynamics": 0.0}
    total_count = 0
    network.eval()
    for start in range(0, args.eval_points, args.eval_residual_batch_size):
        count = min(args.eval_residual_batch_size, args.eval_points - start)
        sample = rng.uniform(bounds[0], bounds[1], size=(count, 2)).astype(numpy_dtype)
        points = torch.as_tensor(sample, dtype=dtype, device=device).requires_grad_(True)
        terms = factorized_ks_terms(
            network, points, pde.alpha, pde.beta, pde.gamma
        )
        for name in sums:
            sums[name] += float(torch.sum(terms[name].detach().double().square()).cpu())
        total_count += count
    return {f"{name}_mse": value / total_count for name, value in sums.items()}


def evaluate_ic(network, bounds, pde, args, device):
    dtype = next(network.parameters()).dtype
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    x = np.linspace(bounds[0][0], bounds[1][0], args.ic_eval_points, dtype=numpy_dtype)
    points = np.column_stack((x, np.full_like(x, bounds[0][1])))
    exact = np.cos(x) * (1.0 + np.sin(x))
    prediction = predict(network, points, args.eval_batch_size, device)[:, 0]
    return field_metrics(prediction, exact)


def save_checkpoint(path, network, metadata):
    state = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }
    torch.save({"state_dict": state, "metadata": metadata}, path)


def save_solution_plot(path, points, exact, prediction, title):
    import matplotlib.pyplot as plt

    x = np.unique(points[:, 0])
    t = np.unique(points[:, 1])
    x_index = np.searchsorted(x, points[:, 0])
    t_index = np.searchsorted(t, points[:, 1])
    fields = []
    for values in (exact, prediction, np.abs(prediction - exact)):
        grid = np.empty((len(t), len(x)), dtype=np.float64)
        grid[t_index, x_index] = np.asarray(values).reshape(-1)
        fields.append(grid)
    value_min = float(min(np.min(exact), np.min(prediction)))
    value_max = float(max(np.max(exact), np.max(prediction)))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    images = [
        axes[0].pcolormesh(x, t, fields[0], shading="auto", cmap="jet",
                           vmin=value_min, vmax=value_max),
        axes[1].pcolormesh(x, t, fields[1], shading="auto", cmap="jet",
                           vmin=value_min, vmax=value_max),
        axes[2].pcolormesh(x, t, fields[2], shading="auto", cmap="magma"),
    ]
    for axis, image, label in zip(
        axes, images, ("Exact u", "Factorized PINN u", "Absolute error")
    ):
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_args(args):
    positive = {
        "iterations": args.iterations,
        "batch-size": args.batch_size,
        "ic-batch-size": args.ic_batch_size,
        "log-every": args.log_every,
        "eval-points": args.eval_points,
        "eval-residual-batch-size": args.eval_residual_batch_size,
        "eval-batch-size": args.eval_batch_size,
        "ic-eval-points": args.ic_eval_points,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("compatibility_weight", "dynamics_weight", "ic_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
    if args.compatibility_weight + args.dynamics_weight + args.ic_weight <= 0.0:
        raise ValueError("At least one training loss weight must be positive")
    if args.lr <= 0.0 or args.lr_min < 0.0 or args.lr_min > args.lr:
        raise ValueError("Require 0 <= lr-min <= lr and lr > 0")
    if args.grad_clip <= 0.0:
        raise ValueError("grad-clip must be positive")


def train(args, network, bounds, pde, reference, device, run_dir, metadata):
    dtype = TORCH_DTYPES[args.precision]
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    lower = np.asarray(bounds[0], dtype=numpy_dtype)
    upper = np.asarray(bounds[1], dtype=numpy_dtype)
    optimizer = build_optimizer(network, args)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.iterations, 1), eta_min=args.lr_min
        )
        if args.lr_min < args.lr else None
    )
    rng = np.random.default_rng(args.seed + 1)
    history = []

    for iteration in range(1, args.iterations + 1):
        network.train()
        domain_numpy = rng.uniform(lower, upper, size=(args.batch_size, 2)).astype(
            numpy_dtype
        )
        domain = torch.as_tensor(
            domain_numpy, dtype=dtype, device=device
        ).requires_grad_(True)
        terms = factorized_ks_terms(
            network, domain, pde.alpha, pde.beta, pde.gamma, backward=True
        )
        compatibility_mse = torch.mean(terms["compatibility"].square())
        dynamics_mse = torch.mean(terms["dynamics"].square())

        x = rng.uniform(lower[0], upper[0], size=(args.ic_batch_size, 1)).astype(
            numpy_dtype
        )
        ic_numpy = np.hstack((x, np.full_like(x, lower[1])))
        exact_ic = torch.as_tensor(
            np.cos(x) * (1.0 + np.sin(x)), dtype=dtype, device=device
        )
        ic_points = torch.as_tensor(ic_numpy, dtype=dtype, device=device)
        ic_mse = torch.mean((network(ic_points)[:, 0:1] - exact_ic).square())
        total = (
            args.compatibility_weight * compatibility_mse
            + args.dynamics_weight * dynamics_mse
            + args.ic_weight * ic_mse
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            prediction = predict(
                network, reference[:, :2], args.eval_batch_size, device
            )
            u_metric = field_metrics(prediction[:, 0], reference[:, 2])
            q_metric = field_metrics(prediction[:, 1], reference[:, 3])
            row = {
                "iteration": iteration,
                "loss_total": float(total.detach().cpu()),
                "compatibility_mse": float(compatibility_mse.detach().cpu()),
                "dynamics_mse": float(dynamics_mse.detach().cpu()),
                "ic_mse": float(ic_mse.detach().cpu()),
                "u_reference_l2re": u_metric["relative_l2"],
                "q_reference_l2re": q_metric["relative_l2"],
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"step={iteration:7d} total={row['loss_total']:.6e} "
                f"compatibility_mse={row['compatibility_mse']:.6e} "
                f"dynamics_mse={row['dynamics_mse']:.6e} "
                f"ic_mse={row['ic_mse']:.6e} "
                f"u_l2re={row['u_reference_l2re']:.6e} "
                f"q_l2re={row['q_reference_l2re']:.6e}"
            )

    save_checkpoint(run_dir / "weights_student.pt", network, metadata)
    save_checkpoint(run_dir / "weights_last.pt", network, metadata)
    columns = list(history[0])
    np.savetxt(
        run_dir / "history.csv",
        np.asarray([[row[column] for column in columns] for row in history]),
        delimiter=",", header=",".join(columns), comments="",
    )
    return history


def run(args):
    args.adam_epsilon = 1e-8 if args.adam_epsilon is None else args.adam_epsilon
    args.soap_epsilon = 1e-8 if args.soap_epsilon is None else args.soap_epsilon
    validate_args(args)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    physical = load_data(args.data)
    bbox = [
        float(np.min(physical[:, 0])), float(np.max(physical[:, 0])),
        float(np.min(physical[:, 1])), float(np.max(physical[:, 1])),
    ]
    pde = FactorizedKuramotoSivashinskyEquation(datapath=args.data, bbox=bbox)
    reference = build_factorized_ks_reference(
        physical, beta=pde.beta, gamma=pde.gamma
    )
    points = reference[:, :2]
    fields = reference[:, 2:4]
    lower = np.asarray([bbox[0], bbox[2]], dtype=np.float64)
    upper = np.asarray([bbox[1], bbox[3]], dtype=np.float64)
    network, metadata = build_network(args, points, fields, device)
    metadata.update(
        training_objective="factorized_ks_residuals_and_initial_condition",
        coefficients={"alpha": pde.alpha, "beta": pde.beta, "gamma": pde.gamma},
        loss_weights={
            "compatibility": args.compatibility_weight,
            "dynamics": args.dynamics_weight,
            "ic": args.ic_weight,
            "u_data": 0.0,
            "q_data": 0.0,
            "sobolev": 0.0,
        },
    )

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-ks-factorized-pinn-{args.network}-{args.optimizer}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        **vars(args),
        "data": str(Path(args.data).resolve()),
        "device": str(device),
        "bbox": bbox,
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "model_metadata": metadata,
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2, sort_keys=True)

    history = train(
        args, network, (lower, upper), pde, reference, device, run_dir, metadata
    )
    residual_metrics = evaluate_residuals(
        network, (lower, upper), pde, args, device
    )
    ic_metric = evaluate_ic(network, (lower, upper), pde, args, device)
    prediction = predict(network, points, args.eval_batch_size, device)
    u_metric = field_metrics(prediction[:, 0], fields[:, 0])
    q_metric = field_metrics(prediction[:, 1], fields[:, 1])

    np.savez_compressed(
        run_dir / "predictions.npz",
        x=points[:, 0], t=points[:, 1],
        exact_u=fields[:, 0], prediction_u=prediction[:, 0],
        exact_q=fields[:, 1], prediction_q=prediction[:, 1],
    )
    save_solution_plot(
        run_dir / "solution.png", points, fields[:, 0], prediction[:, 0],
        f"Factorized KS PINN, u L2RE={u_metric['relative_l2']:.3e}",
    )
    metrics = {
        **residual_metrics,
        "initial_condition": ic_metric,
        "u_reference": u_metric,
        "q_reference": q_metric,
        "last_training_row": history[-1],
        "configuration": configuration,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(
        "Factorized KS PINN: "
        f"compatibility MSE={residual_metrics['compatibility_mse']:.6e}; "
        f"dynamics MSE={residual_metrics['dynamics_mse']:.6e}; "
        f"IC MSE={ic_metric['mse']:.6e}; "
        f"u L2RE={u_metric['relative_l2']:.6e}; "
        f"q L2RE={q_metric['relative_l2']:.6e}; artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=str(PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat")
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "runs_data_ks_factorized_pinn")
    )
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument(
        "--network", "--network-type", choices=["mlp", "rwf"], default="rwf"
    )
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ic-batch-size", type=int, default=256)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "rmsprop", "madam", "soap", "kl-m-soap", "rekls-v3"],
        default="rekls-v3",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-min", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--compatibility-weight", type=float, default=1.0)
    parser.add_argument("--dynamics-weight", type=float, default=1.0)
    parser.add_argument("--ic-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-points", type=int, default=20000)
    parser.add_argument("--eval-residual-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--ic-eval-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--adam-epsilon", type=float, default=None)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-epsilon", type=float, default=None)
    parser.add_argument("--soap-precondition-frequency", type=int, default=1)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)
    parser.add_argument("--soap-bias-correction", type=parse_bool, default=True)
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
