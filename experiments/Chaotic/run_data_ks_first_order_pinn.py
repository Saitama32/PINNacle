"""Train the fully first-order four-field Kuramoto--Sivashinsky system.

The network predicts ``(u, p, q, r)`` and minimizes four equation residuals
plus the initial condition. Reference data do not contribute to the loss; they
are used only for output normalization, diagnostics, and plots.
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
    FirstOrderKuramotoSivashinskyEquation,
    build_first_order_ks_reference,
)
from src.utils.args import parse_hidden_layers


TORCH_DTYPES = {"float32": torch.float32, "float64": torch.float64}
RESIDUAL_NAMES = (
    "p_compatibility",
    "q_compatibility",
    "r_compatibility",
    "dynamics",
)


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
    data = np.loadtxt(path, comments="%", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("KS data must contain x, t, and u columns")
    data = data[:, :3]
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("KS data must contain at least two finite rows")
    if np.any(np.ptp(data[:, :2], axis=0) <= 0.0):
        raise ValueError("Both x and t must vary")
    return data


def build_network(args, points, fields, device):
    hidden = parse_hidden_layers(args)
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("hidden-layers must contain positive widths")
    lower = np.min(points, axis=0)
    scale = np.ptp(points, axis=0)
    mean = np.mean(fields, axis=0, dtype=np.float64)
    std = np.std(fields, axis=0, dtype=np.float64)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    layers = [2, *hidden, 4]
    dde.config.set_default_float(args.precision)
    network = (
        RWFMLP(layers, mu=args.rwf_mu, sigma=args.rwf_sigma)
        if args.network == "rwf"
        else dde.nn.FNN(layers, "tanh", "Glorot normal")
    )
    network = network.to(dtype=TORCH_DTYPES[args.precision], device=device)

    def input_transform(inputs):
        return 2.0 * (inputs - inputs.new_tensor(lower)) / inputs.new_tensor(scale) - 1.0

    def output_transform(_, outputs):
        return outputs * outputs.new_tensor(std) + outputs.new_tensor(mean)

    network.apply_feature_transform(input_transform)
    network.apply_output_transform(output_transform)
    metadata = {
        "model": "RWFMLP" if args.network == "rwf" else "MLP",
        "precision": args.precision,
        "layer_sizes": layers,
        "rwf_mu": args.rwf_mu,
        "rwf_sigma": args.rwf_sigma,
        "input_min": lower.tolist(),
        "input_scale": scale.tolist(),
        "output_mean": mean.tolist(),
        "output_std": std.tolist(),
        "outputs": ["u", "p", "q", "r"],
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
            beta1=args.soap_beta1, beta2=args.soap_beta2,
            shampoo_beta=args.soap_shampoo_beta, epsilon=args.soap_epsilon,
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
    optimizer, _ = dde.optimizers.get(
        network.parameters(), args.optimizer, learning_rate=args.lr,
        weight_decay=args.weight_decay, model=network,
    )
    return optimizer


def first_order_ks_terms(network, points, alpha, beta, gamma, backward=False):
    fields = network(points)
    u, p, q, r = (fields[:, index : index + 1] for index in range(4))
    u_gradient = torch.autograd.grad(
        u, points, grad_outputs=torch.ones_like(u), create_graph=backward,
        retain_graph=True,
    )[0]
    p_x = torch.autograd.grad(
        p, points, grad_outputs=torch.ones_like(p), create_graph=backward,
        retain_graph=True,
    )[0][:, 0:1]
    q_x = torch.autograd.grad(
        q, points, grad_outputs=torch.ones_like(q), create_graph=backward,
        retain_graph=True,
    )[0][:, 0:1]
    r_x = torch.autograd.grad(
        r, points, grad_outputs=torch.ones_like(r), create_graph=backward,
    )[0][:, 0:1]
    u_x, u_t = u_gradient[:, 0:1], u_gradient[:, 1:2]
    return {
        "u": u,
        "p": p,
        "q": q,
        "r": r,
        "p_compatibility": p - u_x,
        "q_compatibility": q - p_x,
        "r_compatibility": r - q_x,
        "dynamics": u_t + alpha * u * p + beta * q + gamma * r_x,
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


def metrics(prediction, exact):
    prediction = np.asarray(prediction, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    error = prediction - exact
    error_sq = float(np.sum(error**2))
    exact_sq = float(np.sum(exact**2))
    mse = error_sq / error.size
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": math.sqrt(error_sq / exact_sq) if exact_sq > 0.0 else None,
    }


def evaluate_residuals(network, bounds, pde, args, device):
    dtype = next(network.parameters()).dtype
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    rng = np.random.default_rng(args.seed + 2)
    sums = {name: 0.0 for name in RESIDUAL_NAMES}
    for start in range(0, args.eval_points, args.eval_residual_batch_size):
        count = min(args.eval_residual_batch_size, args.eval_points - start)
        sample = rng.uniform(bounds[0], bounds[1], size=(count, 2)).astype(numpy_dtype)
        points = torch.as_tensor(sample, dtype=dtype, device=device).requires_grad_(True)
        terms = first_order_ks_terms(
            network, points, pde.alpha, pde.beta, pde.gamma
        )
        for name in sums:
            sums[name] += float(torch.sum(terms[name].detach().double().square()).cpu())
    return {f"{name}_mse": value / args.eval_points for name, value in sums.items()}


def initial_condition_metrics(network, bounds, args, device):
    x = np.linspace(bounds[0][0], bounds[1][0], args.ic_eval_points)
    points = np.column_stack((x, np.full_like(x, bounds[0][1])))
    exact = np.cos(x) * (1.0 + np.sin(x))
    return metrics(predict(network, points, args.eval_batch_size, device)[:, 0], exact)


def periodic_boundary_metrics(network, bounds, args, device):
    t = np.linspace(bounds[0][1], bounds[1][1], args.bc_eval_points)
    left = np.column_stack((np.full_like(t, bounds[0][0]), t))
    right = np.column_stack((np.full_like(t, bounds[1][0]), t))
    jump = predict(network, left, args.eval_batch_size, device) - predict(
        network, right, args.eval_batch_size, device
    )
    component_mse = {
        f"periodic_{name}_mse": float(np.mean(jump[:, index] ** 2))
        for index, name in enumerate(("u", "p", "q", "r"))
    }
    return {**component_mse, "periodic_mse": sum(component_mse.values())}


def save_checkpoint(path, network, metadata):
    state = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }
    torch.save({"state_dict": state, "metadata": metadata}, path)


def save_plot(path, points, exact, prediction, title):
    import matplotlib.pyplot as plt

    x, t = np.unique(points[:, 0]), np.unique(points[:, 1])
    xi, ti = np.searchsorted(x, points[:, 0]), np.searchsorted(t, points[:, 1])
    arrays = []
    for values in (exact, prediction, np.abs(prediction - exact)):
        field = np.empty((len(t), len(x)))
        field[ti, xi] = values
        arrays.append(field)
    low = min(float(np.min(exact)), float(np.min(prediction)))
    high = max(float(np.max(exact)), float(np.max(prediction)))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    images = [
        axes[0].pcolormesh(x, t, arrays[0], shading="auto", cmap="jet", vmin=low, vmax=high),
        axes[1].pcolormesh(x, t, arrays[1], shading="auto", cmap="jet", vmin=low, vmax=high),
        axes[2].pcolormesh(x, t, arrays[2], shading="auto", cmap="magma"),
    ]
    for axis, image, label in zip(
        axes, images, ("Exact u", "First-order PINN u", "Absolute error")
    ):
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate(args):
    for name in (
        "iterations", "batch_size", "ic_batch_size", "log_every", "eval_points",
        "eval_residual_batch_size", "eval_batch_size", "ic_eval_points",
        "bc_batch_size", "bc_eval_points",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in (
        "p_weight", "q_weight", "r_weight", "dynamics_weight", "ic_weight",
        "bc_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
    if sum(
        getattr(args, name)
        for name in (
            "p_weight", "q_weight", "r_weight", "dynamics_weight", "ic_weight",
            "bc_weight",
        )
    ) <= 0.0:
        raise ValueError("At least one loss weight must be positive")
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
            optimizer, args.iterations, eta_min=args.lr_min
        )
        if args.lr_min < args.lr else None
    )
    rng, history = np.random.default_rng(args.seed + 1), []
    weights = {
        "p_compatibility": args.p_weight,
        "q_compatibility": args.q_weight,
        "r_compatibility": args.r_weight,
        "dynamics": args.dynamics_weight,
    }
    for iteration in range(1, args.iterations + 1):
        network.train()
        sample = rng.uniform(lower, upper, size=(args.batch_size, 2)).astype(numpy_dtype)
        points = torch.as_tensor(sample, dtype=dtype, device=device).requires_grad_(True)
        terms = first_order_ks_terms(
            network, points, pde.alpha, pde.beta, pde.gamma, backward=True
        )
        residual_mse = {
            name: torch.mean(terms[name].square()) for name in RESIDUAL_NAMES
        }
        x = rng.uniform(lower[0], upper[0], size=(args.ic_batch_size, 1)).astype(numpy_dtype)
        ic_numpy = np.hstack((x, np.full_like(x, lower[1])))
        ic_points = torch.as_tensor(ic_numpy, dtype=dtype, device=device)
        exact_ic = torch.as_tensor(
            np.cos(x) * (1.0 + np.sin(x)), dtype=dtype, device=device
        )
        ic_mse = torch.mean((network(ic_points)[:, 0:1] - exact_ic).square())
        boundary_t = rng.uniform(
            lower[1], upper[1], size=(args.bc_batch_size, 1)
        ).astype(numpy_dtype)
        left = np.hstack((np.full_like(boundary_t, lower[0]), boundary_t))
        right = np.hstack((np.full_like(boundary_t, upper[0]), boundary_t))
        left_fields = network(torch.as_tensor(left, dtype=dtype, device=device))
        right_fields = network(torch.as_tensor(right, dtype=dtype, device=device))
        periodic_component_mse = torch.mean(
            (left_fields - right_fields).square(), dim=0
        )
        periodic_mse = torch.sum(periodic_component_mse)
        total = sum(weights[name] * residual_mse[name] for name in RESIDUAL_NAMES)
        total = total + args.ic_weight * ic_mse + args.bc_weight * periodic_mse
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            prediction = predict(network, reference[:, :2], args.eval_batch_size, device)
            field_metrics = [
                metrics(prediction[:, index], reference[:, index + 2])
                for index in range(4)
            ]
            row = {
                "iteration": iteration,
                "loss_total": float(total.detach().cpu()),
                **{
                    f"{name}_mse": float(value.detach().cpu())
                    for name, value in residual_mse.items()
                },
                "ic_mse": float(ic_mse.detach().cpu()),
                "periodic_mse": float(periodic_mse.detach().cpu()),
                **{
                    f"periodic_{name}_mse": float(
                        periodic_component_mse[index].detach().cpu()
                    )
                    for index, name in enumerate(("u", "p", "q", "r"))
                },
                **{
                    f"{name}_reference_l2re": field_metrics[index]["relative_l2"]
                    for index, name in enumerate(("u", "p", "q", "r"))
                },
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"step={iteration:7d} total={row['loss_total']:.6e} "
                f"p_mse={row['p_compatibility_mse']:.6e} "
                f"q_mse={row['q_compatibility_mse']:.6e} "
                f"r_mse={row['r_compatibility_mse']:.6e} "
                f"dynamics_mse={row['dynamics_mse']:.6e} "
                f"ic_mse={row['ic_mse']:.6e} periodic_mse={row['periodic_mse']:.6e} "
                f"u_l2re={row['u_reference_l2re']:.6e}"
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
    validate(args)
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
    pde = FirstOrderKuramotoSivashinskyEquation(datapath=args.data, bbox=bbox)
    reference = build_first_order_ks_reference(physical)
    points, fields = reference[:, :2], reference[:, 2:6]
    lower = np.asarray([bbox[0], bbox[2]], dtype=np.float64)
    upper = np.asarray([bbox[1], bbox[3]], dtype=np.float64)
    network, metadata = build_network(args, points, fields, device)
    metadata.update(
        training_objective="first_order_ks_residuals_and_initial_condition",
        coefficients={"alpha": pde.alpha, "beta": pde.beta, "gamma": pde.gamma},
        loss_weights={
            "p_compatibility": args.p_weight,
            "q_compatibility": args.q_weight,
            "r_compatibility": args.r_weight,
            "dynamics": args.dynamics_weight,
            "ic": args.ic_weight,
            "periodic": args.bc_weight,
            "data": 0.0,
            "sobolev": 0.0,
        },
    )
    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-ks-first-order-pinn-{args.network}-{args.optimizer}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        **vars(args), "data": str(Path(args.data).resolve()), "device": str(device),
        "bbox": bbox, "parameters": sum(p.numel() for p in network.parameters()),
        "model_metadata": metadata,
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2, sort_keys=True)

    history = train(
        args, network, (lower, upper), pde, reference, device, run_dir, metadata
    )
    residuals = evaluate_residuals(network, (lower, upper), pde, args, device)
    ic_metric = initial_condition_metrics(network, (lower, upper), args, device)
    boundary_metrics = periodic_boundary_metrics(
        network, (lower, upper), args, device
    )
    prediction = predict(network, points, args.eval_batch_size, device)
    reference_metrics = {
        name: metrics(prediction[:, index], fields[:, index])
        for index, name in enumerate(("u", "p", "q", "r"))
    }
    np.savez_compressed(
        run_dir / "predictions.npz", x=points[:, 0], t=points[:, 1],
        exact_u=fields[:, 0], prediction_u=prediction[:, 0],
        exact_p=fields[:, 1], prediction_p=prediction[:, 1],
        exact_q=fields[:, 2], prediction_q=prediction[:, 2],
        exact_r=fields[:, 3], prediction_r=prediction[:, 3],
    )
    save_plot(
        run_dir / "solution.png", points, fields[:, 0], prediction[:, 0],
        f"First-order KS PINN, u L2RE={reference_metrics['u']['relative_l2']:.3e}",
    )
    result = {
        **residuals,
        "initial_condition": ic_metric,
        "periodic_boundary": boundary_metrics,
        "reference": reference_metrics,
        "last_training_row": history[-1],
        "configuration": configuration,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(
        "First-order KS PINN: "
        + "; ".join(f"{name}={value:.6e}" for name, value in residuals.items())
        + f"; IC MSE={ic_metric['mse']:.6e}"
        + f"; periodic MSE={boundary_metrics['periodic_mse']:.6e}"
        + f"; u L2RE={reference_metrics['u']['relative_l2']:.6e}"
        + f"; artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs_data_ks_first_order_pinn"))
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument("--network", "--network-type", choices=["mlp", "rwf"], default="rwf")
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ic-batch-size", type=int, default=256)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--optimizer", choices=["adam", "rmsprop", "madam", "soap", "kl-m-soap", "rekls-v3"], default="rekls-v3")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-min", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--p-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--r-weight", type=float, default=1.0)
    parser.add_argument("--dynamics-weight", type=float, default=1.0)
    parser.add_argument("--ic-weight", type=float, default=1.0)
    parser.add_argument("--bc-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-points", type=int, default=20000)
    parser.add_argument("--eval-residual-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--ic-eval-points", type=int, default=2048)
    parser.add_argument("--bc-eval-points", type=int, default=2048)
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
