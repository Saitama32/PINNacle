"""PINN fine-tuning of a data-pretrained Kuramoto--Sivashinsky RWF MLP."""

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

import numpy as np
import torch
import deepxde as dde

from experiments.Chaotic.run_data_ks import (
    KS_ALPHA,
    KS_BETA,
    KS_GAMMA,
    NUMPY_DTYPES,
    build_optimizer,
    build_scheduler,
    evaluate_derivative_grid,
    ks_terms,
    load_checkpoint,
    load_data,
    prediction_metrics,
    save_checkpoint,
    save_solution_plot,
)


def resolve_model_path(path: str) -> Path:
    model_path = Path(path).expanduser().resolve()
    if model_path.is_dir():
        model_path = model_path / "weights_best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return model_path


def resolve_data_path(explicit_path, source_dir: Path) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    config_path = source_dir / "run_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file_obj:
            configured = json.load(file_obj).get("data")
        if configured:
            return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"


def validation_subset(points, values, source_dir: Path):
    split_path = source_dir / "split_indices.npz"
    if split_path.exists():
        with np.load(split_path) as split:
            indices = np.asarray(split["test"], dtype=np.int64)
        if len(indices) and int(np.max(indices)) < len(points):
            return points[indices], values[indices]
    return points, values


def _cpu_state_dict(network) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def periodic_boundary_errors(
    network,
    times: torch.Tensor,
    x_lower: float,
    x_upper: float,
    create_graph_for_backward: bool = False,
) -> dict[str, torch.Tensor]:
    """Return periodic jumps of u and its first three spatial derivatives."""

    left_x = torch.full_like(times, x_lower)
    right_x = torch.full_like(times, x_upper)
    left = torch.cat((left_x, times), dim=1).requires_grad_(True)
    right = torch.cat((right_x, times), dim=1).requires_grad_(True)

    def values_and_derivatives(points):
        value = network(points)
        derivative_1 = torch.autograd.grad(
            value,
            points,
            grad_outputs=torch.ones_like(value),
            create_graph=True,
        )[0][:, 0:1]
        derivative_2 = torch.autograd.grad(
            derivative_1,
            points,
            grad_outputs=torch.ones_like(derivative_1),
            create_graph=True,
        )[0][:, 0:1]
        derivative_3 = torch.autograd.grad(
            derivative_2,
            points,
            grad_outputs=torch.ones_like(derivative_2),
            create_graph=create_graph_for_backward,
        )[0][:, 0:1]
        return value, derivative_1, derivative_2, derivative_3

    left_terms = values_and_derivatives(left)
    right_terms = values_and_derivatives(right)
    return {
        name: left_value - right_value
        for name, left_value, right_value in zip(
            ("u", "u_x", "u_xx", "u_xxx"), left_terms, right_terms
        )
    }


def periodic_boundary_losses(errors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    losses = {}
    for name, error in errors.items():
        error_for_loss = error.float() if error.dtype == torch.float16 else error
        losses[name] = torch.mean(error_for_loss.square())
    return losses


def normalized_periodic_losses(losses, normalization_scales):
    return {
        name: loss / float(normalization_scales[name])
        for name, loss in losses.items()
    }


def estimate_periodic_normalization_scales(network, bounds, args, device):
    """Use each initial boundary-jump MSE as its fixed normalization scale."""

    rng = np.random.default_rng(args.seed + 3)
    numpy_dtype = NUMPY_DTYPES[args.precision]
    lower = np.asarray(bounds[0], dtype=numpy_dtype)
    upper = np.asarray(bounds[1], dtype=numpy_dtype)
    square_sums = {name: 0.0 for name in ("u", "u_x", "u_xx", "u_xxx")}
    count_total = 0
    network.eval()
    for start in range(0, args.pinn_boundary_points, args.pinn_batch_size):
        count = min(args.pinn_batch_size, args.pinn_boundary_points - start)
        times_numpy = rng.uniform(lower[1], upper[1], size=(count, 1)).astype(
            numpy_dtype
        )
        errors = periodic_boundary_errors(
            network,
            torch.as_tensor(times_numpy, device=device),
            float(lower[0]),
            float(upper[0]),
        )
        for name, error in errors.items():
            square_sums[name] += float(
                torch.sum(error.detach().double().square()).cpu()
            )
        count_total += count
    return {
        name: max(square_sum / count_total, args.periodic_normalization_epsilon)
        for name, square_sum in square_sums.items()
    }


def add_data_anchor_metric(metric, data_metric, args):
    result = dict(metric)
    anchor_mse = float(data_metric["mse"])
    contribution = args.data_anchor_weight * anchor_mse if args.data_anchor else 0.0
    result.update(
        data_anchor_enabled=args.data_anchor,
        data_anchor_mse=anchor_mse,
        data_anchor_weight=args.data_anchor_weight,
        data_anchor_contribution=contribution,
        pinn_loss_weighted=result["pinn_loss_weighted"] + contribution,
        pinn_loss_unweighted=(
            result["pinn_loss_unweighted"] + anchor_mse
            if args.data_anchor
            else result["pinn_loss_unweighted"]
        ),
    )
    return result


def evaluate_pinn_loss_with_periodic(network, bounds, args, device) -> dict:
    """Evaluate fixed KS PDE, initial-condition, and periodic-boundary losses."""

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
        residual = ks_terms(
            network,
            points,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )["residual"]
        residual_square_sum += float(torch.sum(residual.detach().double().square()).cpu())
        residual_count += residual.numel()

    x = rng.uniform(lower[0], upper[0], size=(args.pinn_ic_points, 1)).astype(
        numpy_dtype
    )
    t_initial = np.full_like(x, lower[1])
    ic_points = torch.as_tensor(np.hstack((x, t_initial)), device=device)
    exact_ic = torch.as_tensor(np.cos(x) * (1.0 + np.sin(x)), device=device)
    with torch.no_grad():
        ic_error = (network(ic_points) - exact_ic).double()
        ic_mse = float(torch.mean(ic_error.square()).cpu())

    boundary_square_sums = {name: 0.0 for name in ("u", "u_x", "u_xx", "u_xxx")}
    boundary_count = 0
    for start in range(0, args.pinn_boundary_points, args.pinn_batch_size):
        count = min(args.pinn_batch_size, args.pinn_boundary_points - start)
        times_numpy = rng.uniform(
            lower[1], upper[1], size=(count, 1)
        ).astype(numpy_dtype)
        times = torch.as_tensor(times_numpy, device=device)
        errors = periodic_boundary_errors(
            network, times, float(lower[0]), float(upper[0])
        )
        for name, error in errors.items():
            boundary_square_sums[name] += float(
                torch.sum(error.detach().double().square()).cpu()
            )
        boundary_count += count

    boundary_mse = {
        name: square_sum / boundary_count
        for name, square_sum in boundary_square_sums.items()
    }
    boundary_normalized_mse = {
        name: value / float(args.periodic_normalization_scales[name])
        for name, value in boundary_mse.items()
    }
    periodic_mse = sum(boundary_mse.values())
    periodic_normalized_mse = sum(boundary_normalized_mse.values())
    periodic_objective = (
        periodic_normalized_mse if args.normalize_periodic_loss else periodic_mse
    )
    pde_mse = residual_square_sum / residual_count
    unweighted = pde_mse + ic_mse + periodic_objective
    weighted = (
        pde_mse
        + args.ic_loss_weight * ic_mse
        + args.periodic_loss_weight * periodic_objective
    )
    return {
        "pde_mse": pde_mse,
        "ic_mse": ic_mse,
        "periodic_mse": periodic_mse,
        "periodic_u_mse": boundary_mse["u"],
        "periodic_u_x_mse": boundary_mse["u_x"],
        "periodic_u_xx_mse": boundary_mse["u_xx"],
        "periodic_u_xxx_mse": boundary_mse["u_xxx"],
        "periodic_normalized_mse": periodic_normalized_mse,
        "periodic_u_normalized_mse": boundary_normalized_mse["u"],
        "periodic_u_x_normalized_mse": boundary_normalized_mse["u_x"],
        "periodic_u_xx_normalized_mse": boundary_normalized_mse["u_xx"],
        "periodic_u_xxx_normalized_mse": boundary_normalized_mse["u_xxx"],
        "periodic_objective": periodic_objective,
        "normalize_periodic_loss": args.normalize_periodic_loss,
        "periodic_normalization_scales": args.periodic_normalization_scales,
        "pinn_loss_unweighted": unweighted,
        "pinn_loss_weighted": weighted,
        "ic_loss_weight": args.ic_loss_weight,
        "periodic_loss_weight": args.periodic_loss_weight,
        "num_domain_points": args.pinn_points,
        "num_initial_points": args.pinn_ic_points,
        "num_boundary_points": args.pinn_boundary_points,
    }


def train_pinn_stage_with_periodic(
    network,
    bounds,
    anchor_points,
    anchor_values,
    validation_points,
    validation_values,
    args,
    device,
    metadata,
    run_dir: Path,
) -> dict:
    """Fine-tune using the KS PDE, IC, and periodic conditions through u_xxx."""

    stage_args = argparse.Namespace(**vars(args))
    stage_args.optimizer = args.pinn_optimizer
    stage_args.lr = args.pinn_lr
    stage_args.weight_decay = args.pinn_weight_decay
    stage_args.iterations = args.n_iter_pinn
    stage_args.lr_scheduler = args.pinn_lr_scheduler
    stage_args.lr_decay_steps = args.pinn_lr_decay_steps
    stage_args.lr_decay_rate = args.pinn_lr_decay_rate
    stage_args.lr_min = args.pinn_lr_min
    use_lbfgs = args.pinn_optimizer == "lbfgs"
    if use_lbfgs:
        line_search_fn = (
            None
            if args.pinn_lbfgs_line_search == "none"
            else args.pinn_lbfgs_line_search.replace("-", "_")
        )
        optimizer = torch.optim.LBFGS(
            network.parameters(),
            lr=args.pinn_lr,
            max_iter=args.pinn_lbfgs_max_iter,
            max_eval=args.pinn_lbfgs_max_eval,
            tolerance_grad=args.pinn_lbfgs_tolerance_grad,
            tolerance_change=args.pinn_lbfgs_tolerance_change,
            history_size=args.pinn_lbfgs_history_size,
            line_search_fn=line_search_fn,
        )
        scheduler = None
    else:
        optimizer = build_optimizer(network, stage_args)
        scheduler = build_scheduler(optimizer, stage_args)

    numpy_dtype = NUMPY_DTYPES[args.precision]
    lower = np.asarray(bounds[0], dtype=numpy_dtype)
    upper = np.asarray(bounds[1], dtype=numpy_dtype)
    rng = np.random.default_rng(args.seed + 2)
    initial_data_metric = prediction_metrics(
        network, validation_points, validation_values, args.eval_batch_size, device
    )
    initial_metric = add_data_anchor_metric(
        evaluate_pinn_loss_with_periodic(network, bounds, args, device),
        initial_data_metric,
        args,
    )
    best_metric = dict(initial_metric)
    best_data_metric = dict(initial_data_metric)
    best_score = initial_metric["pinn_loss_weighted"]
    best_iteration = 0
    best_state = _cpu_state_dict(network)
    rows = []
    print(
        f"Starting periodic PINN fine-tuning for {args.n_iter_pinn} iterations; "
        f"optimizer={args.pinn_optimizer}; lr={args.pinn_lr:.6e}; "
        f"initial weighted loss={best_score:.6e}"
    )
    if use_lbfgs:
        print(
            "LBFGS configuration: "
            f"max_iter={args.pinn_lbfgs_max_iter}; "
            f"max_eval={args.pinn_lbfgs_max_eval}; "
            f"history_size={args.pinn_lbfgs_history_size}; "
            f"line_search={args.pinn_lbfgs_line_search}. "
            "A fixed collocation batch is used for the complete LBFGS run; "
            "the LR scheduler and gradient clipping are disabled."
        )

    lbfgs_domain_numpy = None
    lbfgs_x_numpy = None
    lbfgs_anchor_indices = None
    lbfgs_times_numpy = None
    if use_lbfgs:
        lbfgs_domain_numpy = rng.uniform(
            lower, upper, size=(args.pinn_train_domain_points, 2)
        ).astype(numpy_dtype)
        lbfgs_x_numpy = rng.uniform(
            lower[0], upper[0], size=(args.pinn_train_ic_points, 1)
        ).astype(numpy_dtype)
        if args.data_anchor:
            lbfgs_anchor_indices = rng.integers(
                0, len(anchor_points), size=args.pinn_train_data_points
            )
        lbfgs_times_numpy = rng.uniform(
            lower[1], upper[1], size=(args.pinn_train_boundary_points, 1)
        ).astype(numpy_dtype)

    for iteration in range(1, args.n_iter_pinn + 1):
        network.train()
        domain_numpy = (
            lbfgs_domain_numpy
            if use_lbfgs
            else rng.uniform(
                lower, upper, size=(args.pinn_train_domain_points, 2)
            ).astype(numpy_dtype)
        )
        domain_points = torch.as_tensor(domain_numpy, device=device).requires_grad_(True)
        x_numpy = (
            lbfgs_x_numpy
            if use_lbfgs
            else rng.uniform(
                lower[0], upper[0], size=(args.pinn_train_ic_points, 1)
            ).astype(numpy_dtype)
        )
        t_initial = np.full_like(x_numpy, lower[1])
        ic_points = torch.as_tensor(np.hstack((x_numpy, t_initial)), device=device)
        exact_ic = torch.as_tensor(
            np.cos(x_numpy) * (1.0 + np.sin(x_numpy)), device=device
        )
        if args.data_anchor:
            anchor_indices = (
                lbfgs_anchor_indices
                if use_lbfgs
                else rng.integers(
                    0, len(anchor_points), size=args.pinn_train_data_points
                )
            )
            anchor_batch_points = torch.as_tensor(
                anchor_points[anchor_indices], device=device
            )
            anchor_batch_values = torch.as_tensor(
                anchor_values[anchor_indices], device=device
            )
        else:
            anchor_batch_points = None
            anchor_batch_values = None

        times_numpy = (
            lbfgs_times_numpy
            if use_lbfgs
            else rng.uniform(
                lower[1], upper[1], size=(args.pinn_train_boundary_points, 1)
            ).astype(numpy_dtype)
        )
        times = torch.as_tensor(times_numpy, device=device)
        def compute_batch_losses():
            residual = ks_terms(
                network,
                domain_points,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                create_graph_for_backward=True,
            )["residual"]
            residual_for_loss = (
                residual.float() if residual.dtype == torch.float16 else residual
            )
            current_pde_loss = torch.mean(residual_for_loss.square())

            ic_error = network(ic_points) - exact_ic
            ic_error_for_loss = (
                ic_error.float() if ic_error.dtype == torch.float16 else ic_error
            )
            current_ic_loss = torch.mean(ic_error_for_loss.square())

            if args.data_anchor:
                anchor_error = network(anchor_batch_points) - anchor_batch_values
                anchor_error_for_loss = (
                    anchor_error.float()
                    if anchor_error.dtype == torch.float16
                    else anchor_error
                )
                current_data_anchor_loss = torch.mean(anchor_error_for_loss.square())
            else:
                current_data_anchor_loss = torch.zeros(
                    (), dtype=current_pde_loss.dtype, device=device
                )

            current_periodic_raw_losses = periodic_boundary_losses(
                periodic_boundary_errors(
                    network,
                    times,
                    float(lower[0]),
                    float(upper[0]),
                    create_graph_for_backward=True,
                )
            )
            current_periodic_losses = (
                normalized_periodic_losses(
                    current_periodic_raw_losses, args.periodic_normalization_scales
                )
                if args.normalize_periodic_loss
                else current_periodic_raw_losses
            )
            current_periodic_loss = sum(current_periodic_losses.values())
            current_weighted_loss = (
                current_pde_loss
                + args.ic_loss_weight * current_ic_loss
                + args.periodic_loss_weight * current_periodic_loss
                + args.data_anchor_weight * current_data_anchor_loss
            )
            return {
                "pde": current_pde_loss,
                "ic": current_ic_loss,
                "data_anchor": current_data_anchor_loss,
                "periodic_raw": current_periodic_raw_losses,
                "periodic": current_periodic_loss,
                "weighted": current_weighted_loss,
            }

        if use_lbfgs:
            latest_losses = {}

            def closure():
                optimizer.zero_grad(set_to_none=True)
                closure_losses = compute_batch_losses()
                closure_losses["weighted"].backward()
                latest_losses.clear()
                latest_losses.update(closure_losses)
                return closure_losses["weighted"]

            optimizer.step(closure)
            batch_losses = latest_losses
        else:
            optimizer.zero_grad(set_to_none=True)
            batch_losses = compute_batch_losses()
            batch_losses["weighted"].backward()
            if args.pinn_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), args.pinn_grad_clip)
            optimizer.step()

        pde_loss = batch_losses["pde"]
        ic_loss = batch_losses["ic"]
        data_anchor_loss = batch_losses["data_anchor"]
        periodic_raw_losses = batch_losses["periodic_raw"]
        periodic_loss = batch_losses["periodic"]
        weighted_loss = batch_losses["weighted"]
        if scheduler is not None:
            scheduler.step()

        if iteration % args.pinn_train_log_every == 0 or iteration == args.n_iter_pinn:
            fixed_data_metric = prediction_metrics(
                network,
                validation_points,
                validation_values,
                args.eval_batch_size,
                device,
            )
            fixed_metric = add_data_anchor_metric(
                evaluate_pinn_loss_with_periodic(network, bounds, args, device),
                fixed_data_metric,
                args,
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
                    float(data_anchor_loss.detach().cpu()),
                    float(periodic_raw_losses["u"].detach().cpu()),
                    float(periodic_raw_losses["u_x"].detach().cpu()),
                    float(periodic_raw_losses["u_xx"].detach().cpu()),
                    float(periodic_raw_losses["u_xxx"].detach().cpu()),
                    float(periodic_loss.detach().cpu()),
                    float(weighted_loss.detach().cpu()),
                    fixed_metric["pde_mse"],
                    fixed_metric["ic_mse"],
                    fixed_metric["periodic_mse"],
                    fixed_metric["periodic_objective"],
                    fixed_score,
                    fixed_data_metric["mse"],
                    fixed_data_metric["relative_l2"],
                    optimizer.param_groups[0]["lr"],
                ]
            )
            print(
                f"PINN fine-tune step={iteration:7d} "
                f"batch_weighted={float(weighted_loss.detach().cpu()):.6e} "
                f"fixed_pde={fixed_metric['pde_mse']:.6e} "
                f"fixed_ic={fixed_metric['ic_mse']:.6e} "
                f"fixed_periodic_raw={fixed_metric['periodic_mse']:.6e} "
                f"fixed_periodic_objective={fixed_metric['periodic_objective']:.6e} "
                f"fixed_anchor={fixed_metric['data_anchor_mse']:.6e} "
                f"fixed_weighted={fixed_score:.6e} "
                f"data_l2={fixed_data_metric['relative_l2']:.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.6e}"
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
            "iteration,batch_pde_mse,batch_ic_mse,batch_data_anchor_mse,"
            "batch_periodic_u_raw_mse,batch_periodic_u_x_raw_mse,"
            "batch_periodic_u_xx_raw_mse,batch_periodic_u_xxx_raw_mse,"
            "batch_periodic_objective,batch_weighted_loss,fixed_pde_mse,"
            "fixed_ic_mse,fixed_periodic_raw_mse,fixed_periodic_objective,"
            "fixed_weighted_loss,"
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


def run(args) -> Path:
    if args.n_iter_pinn <= 0:
        raise ValueError("n_iter_pinn must be positive")
    if args.pinn_train_boundary_points <= 0 or args.pinn_boundary_points <= 0:
        raise ValueError("PINN boundary point counts must be positive")
    if args.periodic_loss_weight < 0 or not math.isfinite(args.periodic_loss_weight):
        raise ValueError("periodic_loss_weight must be finite and non-negative")
    if args.periodic_normalization_epsilon <= 0:
        raise ValueError("periodic_normalization_epsilon must be positive")
    if args.data_anchor_weight < 0 or not math.isfinite(args.data_anchor_weight):
        raise ValueError("data_anchor_weight must be finite and non-negative")
    if args.pinn_train_data_points <= 0:
        raise ValueError("pinn_train_data_points must be positive")
    if args.pinn_optimizer == "lbfgs":
        if args.pinn_lbfgs_max_iter <= 0:
            raise ValueError("pinn_lbfgs_max_iter must be positive")
        if args.pinn_lbfgs_max_eval is not None and args.pinn_lbfgs_max_eval <= 0:
            raise ValueError("pinn_lbfgs_max_eval must be positive when provided")
        if args.pinn_lbfgs_history_size <= 0:
            raise ValueError("pinn_lbfgs_history_size must be positive")
        if args.pinn_lbfgs_tolerance_grad < 0:
            raise ValueError("pinn_lbfgs_tolerance_grad must be non-negative")
        if args.pinn_lbfgs_tolerance_change < 0:
            raise ValueError("pinn_lbfgs_tolerance_change must be non-negative")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path = resolve_model_path(args.model)
    source_dir = model_path.parent
    network, metadata = load_checkpoint(model_path, device=device)
    source_precision = metadata.get("precision", "float32")
    args.precision = args.pinn_precision
    torch_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.precision]
    network = network.to(device=device, dtype=torch_dtype)
    dde.config.set_default_float(args.precision)
    metadata = dict(metadata)
    metadata["precision"] = args.precision
    args.alpha = float(metadata.get("alpha", KS_ALPHA))
    args.beta = float(metadata.get("beta", KS_BETA))
    args.gamma = float(metadata.get("gamma", KS_GAMMA))
    args.adam_epsilon = 1e-4 if args.precision == "float16" else 1e-8
    args.soap_epsilon = 1e-4 if args.precision == "float16" else 1e-8
    args.muon_adam_epsilon = 1e-4 if args.precision == "float16" else 1e-10

    data_path = resolve_data_path(args.data, source_dir)
    points, values = load_data(data_path, precision=args.precision)
    validation_points, validation_values = validation_subset(points, values, source_dir)
    numpy_dtype = np.float64 if args.precision == "float64" else np.float32
    lower = np.asarray(metadata["input_min"], dtype=numpy_dtype)
    upper = lower + np.asarray(metadata["input_scale"], dtype=numpy_dtype)
    args.periodic_normalization_scales = estimate_periodic_normalization_scales(
        network, (lower, upper), args, device
    )

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-{source_dir.name}-pinn-{args.pinn_optimizer}-{args.precision}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    save_checkpoint(run_dir / "weights_initial.pt", network, metadata)
    resolved = vars(args).copy()
    resolved.update(
        model=str(model_path),
        source_run=str(source_dir),
        data=str(data_path),
        device=str(device),
        source_model_precision=source_precision,
        model_metadata=metadata,
    )
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(resolved, file_obj, indent=2, sort_keys=True)

    initial_data = prediction_metrics(
        network, points, values, args.eval_batch_size, device
    )
    initial_pinn = add_data_anchor_metric(
        evaluate_pinn_loss_with_periodic(network, (lower, upper), args, device),
        initial_data,
        args,
    )
    fine_tune = train_pinn_stage_with_periodic(
        network,
        (lower, upper),
        points,
        values,
        validation_points,
        validation_values,
        args,
        device,
        metadata,
        run_dir,
    )
    final_data = prediction_metrics(network, points, values, args.eval_batch_size, device)
    final_pinn = add_data_anchor_metric(
        evaluate_pinn_loss_with_periodic(network, (lower, upper), args, device),
        final_data,
        args,
    )
    derivative_metric = None
    if args.derivative_plots:
        derivative_metric = evaluate_derivative_grid(
            network,
            (lower, upper),
            args.alpha,
            args.beta,
            args.gamma,
            args.derivative_grid_nx,
            args.derivative_grid_nt,
            args.derivative_batch_size,
            run_dir,
            device,
        )

    predictions = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), args.eval_batch_size):
            batch = torch.as_tensor(points[start : start + args.eval_batch_size], device=device)
            predictions.append(network(batch).cpu().numpy())
    prediction = np.vstack(predictions)[:, 0]
    np.savez_compressed(
        run_dir / "predictions.npz",
        x=points[:, 0],
        t=points[:, 1],
        exact=values[:, 0],
        prediction=prediction,
    )
    save_solution_plot(
        run_dir / "solution.png",
        points,
        values[:, 0],
        prediction,
        f"Data-pretrained + PINN KS, relative L2={final_data['relative_l2']:.3e}",
    )
    metrics = {
        "source_model": str(model_path),
        "initial_data": initial_data,
        "initial_pinn_loss": initial_pinn,
        "final_data": final_data,
        "final_pinn_loss": final_pinn,
        "pinn_finetune": fine_tune,
        "derivative_grid": derivative_metric,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)
    print(
        f"Finished PINN fine-tuning: best step={fine_tune['best_iteration']}, "
        f"relative L2={final_data['relative_l2']:.6e}, "
        f"weighted PINN loss={final_pinn['pinn_loss_weighted']:.6e}; "
        f"artifacts={run_dir}"
    )
    return run_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a data-pretrained KS RWF model using PDE, IC, and "
            "periodic boundary losses through the third spatial derivative."
        )
    )
    parser.add_argument("--model", default=r"C:\Users\Рустам\Documents\GitHub\PINNacle\runs_data_ks_local_basin_chain\08.25-12.50.50-ks-local-basin-chain\steps\08.25-13.44.48-ks-local-basin-float64\weights_local_best.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs_data_ks_pinn"))
    parser.add_argument(
        "--pinn-precision",
        choices=["float32", "float64"],
        default="float64",
        help="Precision used after loading the pretrained model and during PINN fine-tuning.",
    )
    parser.add_argument("--n-iter-pinn", type=int, default=1000)
    parser.add_argument(
        "--pinn-optimizer",
        choices=["adam", "rmsprop", "madam", "muon", "soap", "kl-m-soap", "lbfgs"],
        default="soap",
    )
    parser.add_argument("--pinn-lr", type=float, default=1e-5)
    parser.add_argument("--pinn-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--pinn-lr-scheduler",
        choices=["none", "exponential", "cosine", "step"],
        default="none",
    )
    parser.add_argument("--pinn-lr-decay-steps", type=int, default=1000)
    parser.add_argument("--pinn-lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--pinn-lr-min", type=float, default=1e-6)
    parser.add_argument("--pinn-train-domain-points", type=int, default=256)
    parser.add_argument("--pinn-train-ic-points", type=int, default=256)
    parser.add_argument("--pinn-train-boundary-points", type=int, default=256)
    parser.add_argument("--pinn-train-data-points", type=int, default=256)
    parser.add_argument("--pinn-train-log-every", type=int, default=100)
    parser.add_argument("--pinn-grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--pinn-lbfgs-max-iter",
        type=int,
        default=5,
        help="Maximum internal LBFGS iterations per outer PINN iteration.",
    )
    parser.add_argument(
        "--pinn-lbfgs-max-eval",
        type=int,
        default=None,
        help="Maximum closure evaluations per outer LBFGS step (PyTorch default if omitted).",
    )
    parser.add_argument("--pinn-lbfgs-history-size", type=int, default=100)
    parser.add_argument("--pinn-lbfgs-tolerance-grad", type=float, default=1e-7)
    parser.add_argument("--pinn-lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument(
        "--pinn-lbfgs-line-search",
        choices=["none", "strong-wolfe"],
        default="strong-wolfe",
    )
    parser.add_argument("--ic-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--periodic-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the normalized periodic objective. When normalization is "
            "disabled, this weights the raw sum of boundary MSEs."
        ),
    )
    parser.add_argument(
        "--normalize-periodic-loss",
        type=parse_bool,
        default=True,
        metavar="true|false",
        help="Normalize every periodic component by its initial boundary-jump MSE.",
    )
    parser.add_argument("--periodic-normalization-epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--data-anchor",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help="Add supervised MSE on random points sampled from the source .dat file.",
    )
    parser.add_argument("--data-anchor-weight", type=float, default=1e4)
    parser.add_argument("--pinn-points", type=int, default=20000)
    parser.add_argument("--pinn-ic-points", type=int, default=2048)
    parser.add_argument("--pinn-boundary-points", type=int, default=2048)
    parser.add_argument("--pinn-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--derivative-grid-nx", type=int, default=128)
    parser.add_argument("--derivative-grid-nt", type=int, default=64)
    parser.add_argument("--derivative-batch-size", type=int, default=512)
    parser.add_argument("--no-derivative-plots", dest="derivative_plots", action="store_false")
    parser.set_defaults(derivative_plots=True)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)
    parser.add_argument("--soap-bias-correction", action="store_true", default=True)
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
    parser.add_argument("--madam-bias-correction", action="store_true", default=True)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", action="store_true", default=True)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-adam-weight-decay", type=float, default=0.0)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
