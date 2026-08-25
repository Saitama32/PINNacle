"""Global-network Duhamel KS training with an optional strong PINN stage.

The mandatory first stage trains one global RWF MLP by local mild/Duhamel
links.  Every endpoint is evaluated directly by that same network; predictions
are never rolled out from a previous link.  The optional second stage delegates
unchanged strong-form optimization to ``run_data_ks_pinn.py``.
"""

from __future__ import annotations

import copy
import csv
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

from experiments.Chaotic import run_data_ks_pinn as strong_ks
from experiments.Chaotic.run_data_ks import (
    KS_ALPHA,
    KS_BETA,
    KS_GAMMA,
    NUMPY_DTYPES,
    build_network,
    build_optimizer,
    build_scheduler,
    evaluate_derivative_grid,
    evaluate_pinn_loss,
    load_checkpoint,
    load_data,
    prediction_metrics,
    save_checkpoint,
    save_solution_plot,
)
from src.utils.args import parse_hidden_layers


def cpu_state_dict(network) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def mean_square(value: torch.Tensor) -> torch.Tensor:
    value_for_loss = value.float() if value.dtype == torch.float16 else value
    return torch.mean(value_for_loss.square())


def exact_initial_condition_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x) * (1.0 + torch.sin(x))


def resolve_time_chain(t_lower: float, t_upper: float, requested_delta_t: float):
    duration = float(t_upper - t_lower)
    if duration <= 0:
        raise ValueError("The time domain must have positive length")
    if requested_delta_t <= 0 or not math.isfinite(requested_delta_t):
        raise ValueError("mild-delta-t must be positive and finite")
    intervals = max(1, int(math.ceil(duration / requested_delta_t - 1e-12)))
    resolved_delta_t = duration / intervals
    return intervals, resolved_delta_t


def build_mild_grid(lower, upper, args, device):
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.precision]
    length = float(upper[0] - lower[0])
    nx = args.mild_num_x_fft
    x = float(lower[0]) + length * torch.arange(nx, dtype=dtype, device=device) / nx
    # torch.fft.fftfreq returns cycles/unit; multiplying by 2*pi gives the
    # physical angular wavenumbers used by d_xx and d_xxxx.
    wave_numbers = 2.0 * math.pi * torch.fft.fftfreq(
        nx, d=length / nx, device=device, dtype=dtype
    )
    eigenvalues = args.beta * wave_numbers.square() - args.gamma * wave_numbers.pow(4)
    propagator = torch.exp(eigenvalues * args.resolved_mild_delta_t)

    nodes, weights = np.polynomial.legendre.leggauss(args.mild_quadrature_points)
    nodes = torch.as_tensor(nodes, dtype=dtype, device=device)
    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    relative_times = 0.5 * args.resolved_mild_delta_t * (nodes + 1.0)
    physical_weights = 0.5 * args.resolved_mild_delta_t * weights
    remaining_times = args.resolved_mild_delta_t - relative_times
    quadrature_kernel = torch.exp(remaining_times[:, None] * eigenvalues[None, :])
    return {
        "x": x,
        "wave_numbers": wave_numbers,
        "eigenvalues": eigenvalues,
        "propagator": propagator,
        "relative_times": relative_times,
        "quadrature_weights": physical_weights,
        "quadrature_kernel": quadrature_kernel,
        "max_linear_growth_rate": float(torch.max(eigenvalues).detach().cpu()),
        "max_linear_growth_factor": float(torch.max(propagator).detach().cpu()),
    }


def evaluate_space_time(network, x, times, need_x_derivative, backward):
    count_t, count_x = len(times), len(x)
    points = torch.stack(
        (
            x.reshape(1, -1).expand(count_t, count_x),
            times.reshape(-1, 1).expand(count_t, count_x),
        ),
        dim=-1,
    ).reshape(-1, 2)
    if need_x_derivative:
        points.requires_grad_(True)
    values = network(points).reshape(count_t, count_x)
    if not need_x_derivative:
        return values, None
    derivative_x = torch.autograd.grad(
        values,
        points,
        grad_outputs=torch.ones_like(values),
        create_graph=backward,
    )[0][:, 0].reshape(count_t, count_x)
    return values, derivative_x


def mild_duhamel_defects(network, left_times, grid, args, backward=True):
    """Compute local mild defects without u_t, u_xx, u_xxx, or u_xxxx."""

    left_times = left_times.to(device=grid["x"].device, dtype=grid["x"].dtype)
    dt = args.resolved_mild_delta_t
    right_times = left_times + dt
    endpoint_times = torch.cat((left_times, right_times), dim=0)
    endpoint_values, _ = evaluate_space_time(
        network, grid["x"], endpoint_times, False, backward
    )
    count = len(left_times)
    value_left = endpoint_values[:count]
    value_right = endpoint_values[count:]
    # norm="forward" makes these discrete approximations of Fourier-series
    # coefficients, so the loss does not change merely because Nx changes.
    hat_left = torch.fft.fft(value_left, dim=1, norm="forward")
    hat_right = torch.fft.fft(value_right, dim=1, norm="forward")

    quadrature_times = (
        left_times[:, None] + grid["relative_times"][None, :]
    ).reshape(-1)
    values_q, derivative_x_q = evaluate_space_time(
        network, grid["x"], quadrature_times, True, backward
    )
    nonlinear = -args.alpha * values_q * derivative_x_q
    nonlinear_hat = torch.fft.fft(nonlinear, dim=1, norm="forward").reshape(
        count, args.mild_quadrature_points, -1
    )
    integral = torch.sum(
        grid["quadrature_weights"][None, :, None]
        * grid["quadrature_kernel"][None, :, :]
        * nonlinear_hat,
        dim=1,
    )
    defect = hat_right - grid["propagator"][None, :] * hat_left - integral
    squared_magnitude = defect.real.square() + defect.imag.square()
    return {
        "defect": defect,
        "loss": torch.mean(squared_magnitude),
        "per_interval_rms": torch.sqrt(torch.mean(squared_magnitude, dim=1)),
        "per_interval_mean_abs": torch.mean(torch.abs(defect), dim=1),
        "per_interval_max_abs": torch.max(torch.abs(defect), dim=1).values,
    }


def iter_interval_chunks(args, device, dtype):
    count = args.resolved_mild_intervals
    chunk = min(args.mild_interval_batch_size, count)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        indices = torch.arange(start, stop, dtype=dtype, device=device)
        yield start, stop, args.time_lower + indices * args.resolved_mild_delta_t


def initial_condition_loss(network, x, time_lower):
    times = torch.full_like(x, time_lower)
    points = torch.cat((x, times), dim=1)
    return mean_square(network(points) - exact_initial_condition_torch(x))


def fixed_mild_diagnostics(network, fixed_ic_x, grid, args, device):
    dtype = grid["x"].dtype
    losses = []
    rms = []
    means = []
    maxima = []
    for _, _, left_times in iter_interval_chunks(args, device, dtype):
        result = mild_duhamel_defects(network, left_times, grid, args, backward=False)
        losses.append(result["loss"].detach() * len(left_times))
        rms.append(result["per_interval_rms"].detach())
        means.append(result["per_interval_mean_abs"].detach())
        maxima.append(result["per_interval_max_abs"].detach())
    mild_loss = torch.sum(torch.stack(losses)) / args.resolved_mild_intervals
    ic_loss = initial_condition_loss(network, fixed_ic_x, args.time_lower).detach()
    interval_rms = torch.cat(rms)
    return {
        "mild_loss": float(mild_loss.cpu()),
        "ic_loss": float(ic_loss.cpu()),
        "total_loss": float(
            (args.mild_loss_weight * mild_loss + args.ic_loss_weight * ic_loss).cpu()
        ),
        "mean_defect_per_interval": float(torch.mean(interval_rms).cpu()),
        "max_defect_per_interval": float(torch.max(interval_rms).cpu()),
        "interval_rms": interval_rms.cpu().numpy(),
        "interval_mean_abs": torch.cat(means).cpu().numpy(),
        "interval_max_abs": torch.cat(maxima).cpu().numpy(),
    }


def build_mild_optimizer(network, args):
    if args.pinn_optimizer == "lbfgs":
        return torch.optim.LBFGS(
            network.parameters(),
            lr=args.pinn_lr,
            max_iter=args.pinn_lbfgs_max_iter,
            max_eval=args.pinn_lbfgs_max_eval,
            tolerance_grad=args.pinn_lbfgs_tolerance_grad,
            tolerance_change=args.pinn_lbfgs_tolerance_change,
            history_size=args.pinn_lbfgs_history_size,
            line_search_fn=(
                None
                if args.pinn_lbfgs_line_search == "none"
                else args.pinn_lbfgs_line_search
            ),
        ), None
    stage_args = copy.copy(args)
    stage_args.optimizer = args.pinn_optimizer
    stage_args.lr = args.pinn_lr
    stage_args.weight_decay = args.pinn_weight_decay
    stage_args.lr_scheduler = args.pinn_lr_scheduler
    stage_args.lr_decay_steps = args.pinn_lr_decay_steps
    stage_args.lr_decay_rate = args.pinn_lr_decay_rate
    stage_args.lr_min = args.pinn_lr_min
    stage_args.iterations = args.mild_epochs
    optimizer = build_optimizer(network, stage_args)
    return optimizer, build_scheduler(optimizer, stage_args)


def save_history(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_mild(
    network,
    points,
    values,
    validation_points,
    validation_values,
    bounds,
    args,
    device,
    metadata,
    run_dir,
):
    grid = build_mild_grid(bounds[0], bounds[1], args, device)
    optimizer, scheduler = build_mild_optimizer(network, args)
    rng = np.random.default_rng(args.seed + 211)
    dtype_np = NUMPY_DTYPES[args.precision]
    dtype_torch = grid["x"].dtype
    fixed_ic_x = torch.linspace(
        float(bounds[0][0]),
        float(bounds[1][0]),
        args.pinn_train_ic_points,
        dtype=dtype_torch,
        device=device,
    ).reshape(-1, 1)
    if args.pinn_optimizer == "lbfgs":
        sampled = rng.uniform(
            bounds[0][0], bounds[1][0], size=(args.pinn_train_ic_points, 1)
        ).astype(dtype_np)
        lbfgs_ic_x = torch.as_tensor(sampled, device=device)
    else:
        lbfgs_ic_x = None
    rows = []
    best_score = math.inf
    best_iteration = 0
    best_state = cpu_state_dict(network)
    gradients_verified = False

    print(
        f"Starting global Duhamel KS for {args.mild_epochs} iterations; "
        f"optimizer={args.pinn_optimizer}; intervals={args.resolved_mild_intervals}; "
        f"dt={args.resolved_mild_delta_t:.8g}; Q={args.mild_quadrature_points}; "
        f"Nx={args.mild_num_x_fft}; max_growth="
        f"{grid['max_linear_growth_factor']:.6e}; derivatives=u_x only."
    )

    def backward_objective(ic_x):
        optimizer.zero_grad(set_to_none=True)
        ic_loss = initial_condition_loss(network, ic_x, args.time_lower)
        (args.ic_loss_weight * ic_loss).backward()
        mild_value = 0.0
        for _, _, left_times in iter_interval_chunks(args, device, dtype_torch):
            result = mild_duhamel_defects(network, left_times, grid, args, backward=True)
            fraction = len(left_times) / args.resolved_mild_intervals
            (args.mild_loss_weight * fraction * result["loss"]).backward()
            mild_value += fraction * float(result["loss"].detach().cpu())
        total_value = args.ic_loss_weight * float(ic_loss.detach().cpu())
        total_value += args.mild_loss_weight * mild_value
        return torch.tensor(total_value, dtype=dtype_torch, device=device)

    for iteration in range(1, args.mild_epochs + 1):
        network.train()
        if args.pinn_optimizer == "lbfgs":
            optimizer.step(lambda: backward_objective(lbfgs_ic_x))
        else:
            sampled = rng.uniform(
                bounds[0][0], bounds[1][0], size=(args.pinn_train_ic_points, 1)
            ).astype(dtype_np)
            ic_x = torch.as_tensor(sampled, device=device)
            backward_objective(ic_x)
            if args.pinn_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), args.pinn_grad_clip)
            optimizer.step()
        if not gradients_verified:
            gradients = [
                parameter.grad
                for parameter in network.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(item).all() for item in gradients):
                raise RuntimeError("Duhamel loss produced missing/non-finite gradients")
            if not any(bool(torch.any(item != 0)) for item in gradients):
                raise RuntimeError("Duhamel loss produced only zero gradients")
            gradients_verified = True
        if scheduler is not None:
            scheduler.step()

        if iteration % args.mild_log_every == 0 or iteration == args.mild_epochs:
            network.eval()
            diagnostics = fixed_mild_diagnostics(
                network, fixed_ic_x, grid, args, device
            )
            data_metric = prediction_metrics(
                network,
                validation_points,
                validation_values,
                args.eval_batch_size,
                device,
            )
            if math.isfinite(diagnostics["total_loss"]) and diagnostics["total_loss"] < best_score:
                best_score = diagnostics["total_loss"]
                best_iteration = iteration
                best_state = cpu_state_dict(network)
            row = {
                "iteration": iteration,
                "mild_loss": diagnostics["mild_loss"],
                "ic_loss": diagnostics["ic_loss"],
                "total_loss": diagnostics["total_loss"],
                "mean_defect_per_interval": diagnostics["mean_defect_per_interval"],
                "max_defect_per_interval": diagnostics["max_defect_per_interval"],
                "delta_t": args.resolved_mild_delta_t,
                "intervals": args.resolved_mild_intervals,
                "quadrature_points": args.mild_quadrature_points,
                "max_linear_growth": grid["max_linear_growth_factor"],
                "data_relative_l2": data_metric["relative_l2"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            rows.append(row)
            print(
                f"Duhamel step={iteration:7d} mild={diagnostics['mild_loss']:.6e} "
                f"ic={diagnostics['ic_loss']:.6e} total={diagnostics['total_loss']:.6e} "
                f"mean_interval={diagnostics['mean_defect_per_interval']:.6e} "
                f"max_interval={diagnostics['max_defect_per_interval']:.6e} "
                f"L2={data_metric['relative_l2']:.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.6e}"
            )

    last_state = cpu_state_dict(network)
    save_checkpoint(run_dir / "weights_mild_last.pt", network, metadata)
    network.load_state_dict(best_state, strict=True)
    save_checkpoint(run_dir / "weights_mild_best.pt", network, metadata)
    network.load_state_dict(last_state, strict=True)
    save_history(run_dir / "mild_history.csv", rows)
    final_diagnostics = fixed_mild_diagnostics(network, fixed_ic_x, grid, args, device)
    np.savez_compressed(
        run_dir / "mild_defects_final.npz",
        t_left=args.time_lower
        + np.arange(args.resolved_mild_intervals) * args.resolved_mild_delta_t,
        t_right=args.time_lower
        + (np.arange(args.resolved_mild_intervals) + 1) * args.resolved_mild_delta_t,
        rms=final_diagnostics["interval_rms"],
        mean_abs=final_diagnostics["interval_mean_abs"],
        max_abs=final_diagnostics["interval_max_abs"],
    )
    return {
        "iterations": args.mild_epochs,
        "best_iteration": best_iteration,
        "best_total_loss": best_score,
        "last_mild_loss": final_diagnostics["mild_loss"],
        "last_ic_loss": final_diagnostics["ic_loss"],
        "gradients_verified": gradients_verified,
        "resolved_delta_t": args.resolved_mild_delta_t,
        "resolved_intervals": args.resolved_mild_intervals,
        "quadrature_points": args.mild_quadrature_points,
        "max_linear_growth_rate": grid["max_linear_growth_rate"],
        "max_linear_growth_factor": grid["max_linear_growth_factor"],
        "training_max_spatial_derivative": "u_x",
        "uses_u_t": False,
        "uses_u_xx": False,
        "uses_u_xxx": False,
        "uses_u_xxxx": False,
        "uses_rollout": False,
        "history_rows": len(rows),
    }


def save_mild_predictions(network, points, values, args, device, run_dir):
    predictions = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), args.eval_batch_size):
            batch = torch.as_tensor(
                points[start : start + args.eval_batch_size], device=device
            )
            predictions.append(network(batch).cpu().numpy())
    prediction = np.vstack(predictions)[:, 0]
    np.savez_compressed(
        run_dir / "predictions_mild.npz",
        x=points[:, 0],
        t=points[:, 1],
        exact=values[:, 0],
        prediction=prediction,
    )
    metric = prediction_metrics(network, points, values, args.eval_batch_size, device)
    save_solution_plot(
        run_dir / "solution_mild.png",
        points,
        values[:, 0],
        prediction,
        f"Duhamel KS RWF, relative L2={metric['relative_l2']:.3e}",
    )
    return metric


def checkpoints_equal(first: Path, second: Path) -> bool:
    def read(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        except TypeError:
            return torch.load(path, map_location="cpu")["state_dict"]

    left, right = read(first), read(second)
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def run(args) -> Path:
    if args.mild_epochs <= 0 or args.mild_log_every <= 0:
        raise ValueError("mild-epochs and mild-log-every must be positive")
    if args.mild_num_x_fft < 3 or args.mild_quadrature_points <= 0:
        raise ValueError("mild Nx must be >=3 and quadrature points must be positive")
    if args.mild_interval_batch_size <= 0:
        raise ValueError("mild-interval-batch-size must be positive")
    if args.mild_loss_weight < 0 or not math.isfinite(args.mild_loss_weight):
        raise ValueError("mild-loss-weight must be finite and non-negative")
    if args.pinn_precision not in {"float32", "float64"}:
        raise ValueError("Duhamel KS supports float32 or float64")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    args.precision = args.pinn_precision
    dde.config.set_default_float(args.precision)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = None
    if args.model:
        model_path = strong_ks.resolve_model_path(args.model)
        source_dir = model_path.parent
        data_path = strong_ks.resolve_data_path(args.data, source_dir)
    else:
        source_dir = None
        data_path = Path(
            args.data or (PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat")
        ).resolve()
    points, values = load_data(data_path, precision=args.precision)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)

    if model_path is not None:
        network, metadata = load_checkpoint(model_path, device=device)
        source_precision = metadata.get("precision", "float32")
        network = network.to(
            device=device,
            dtype={"float32": torch.float32, "float64": torch.float64}[args.precision],
        )
        metadata = dict(metadata)
        metadata["precision"] = args.precision
        args.alpha = float(metadata.get("alpha", KS_ALPHA))
        args.beta = float(metadata.get("beta", KS_BETA))
        args.gamma = float(metadata.get("gamma", KS_GAMMA))
        validation_points, validation_values = strong_ks.validation_subset(
            points, values, source_dir
        )
    else:
        hidden = parse_hidden_layers(args)
        if not hidden or any(width <= 0 for width in hidden):
            raise ValueError("hidden-layers must contain positive widths")
        source_precision = None
        args.alpha, args.beta, args.gamma = KS_ALPHA, KS_BETA, KS_GAMMA
        metadata = {
            "model": "RWFMLP",
            "precision": args.precision,
            "layer_sizes": [2, *hidden, 1],
            "rwf_mu": args.rwf_mu,
            "rwf_sigma": args.rwf_sigma,
            "input_min": lower.tolist(),
            "input_scale": (upper - lower).tolist(),
            "output_mean": 0.0,
            "output_std": 1.0,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
        }
        network = build_network(metadata).to(device)
        validation_points, validation_values = points, values

    args.adam_epsilon = 1e-8
    args.soap_epsilon = 1e-8
    args.muon_adam_epsilon = 1e-10
    args.time_lower = float(lower[1])
    args.time_upper = float(upper[1])
    (
        args.resolved_mild_intervals,
        args.resolved_mild_delta_t,
    ) = resolve_time_chain(args.time_lower, args.time_upper, args.mild_delta_t)

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    origin = "pretrained" if source_dir is not None else "scratch"
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-{origin}-ks-duhamel-to-strong-"
        f"{args.pinn_optimizer}-{args.precision}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    save_checkpoint(run_dir / "weights_initial.pt", network, metadata)
    resolved = vars(args).copy()
    resolved.update(
        model=str(model_path) if model_path is not None else None,
        source_run=str(source_dir) if source_dir is not None else None,
        source_model_precision=source_precision,
        data=str(data_path),
        device=str(device),
        model_metadata=metadata,
        max_spatial_derivative_in_mild_training="u_x",
        mild_endpoint_evaluation="direct_global_network",
    )
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(resolved, file_obj, indent=2, sort_keys=True)

    initial_data = prediction_metrics(network, points, values, args.eval_batch_size, device)
    initial_strong = evaluate_pinn_loss(network, (lower, upper), args, device)
    mild_info = train_mild(
        network,
        points,
        values,
        validation_points,
        validation_values,
        (lower, upper),
        args,
        device,
        metadata,
        run_dir,
    )
    after_mild_data = save_mild_predictions(network, points, values, args, device, run_dir)
    after_mild_strong = evaluate_pinn_loss(network, (lower, upper), args, device)
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

    strong_dir = None
    strong_metrics = None
    transfer_equal = None
    if args.strong_enabled:
        print(
            "Duhamel stage finished. Starting unchanged strong KS from "
            f"{run_dir / 'weights_mild_last.pt'}"
        )
        strong_args = copy.deepcopy(args)
        strong_args.model = str(run_dir / "weights_mild_last.pt")
        strong_args.data = str(data_path)
        strong_args.out = str(run_dir / "strong")
        strong_dir = strong_ks.run(strong_args)
        transfer_equal = checkpoints_equal(
            run_dir / "weights_mild_last.pt", strong_dir / "weights_initial.pt"
        )
        if not transfer_equal:
            raise RuntimeError("Strong stage did not start from exact mild-last weights")
        with (strong_dir / "metrics.json").open("r", encoding="utf-8") as file_obj:
            strong_metrics = json.load(file_obj)

    metrics = {
        "configuration": resolved,
        "initial": {"data": initial_data, "strong_pinn": initial_strong},
        "mild_stage": mild_info,
        "after_mild": {
            "data": after_mild_data,
            "strong_pinn": after_mild_strong,
            "derivative_grid": derivative_metric,
        },
        "strong_enabled": args.strong_enabled,
        "strong_stage": {
            "run_dir": str(strong_dir) if strong_dir is not None else None,
            "weights_transfer_exact": transfer_equal,
            "metrics": strong_metrics,
        },
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)
    print(
        f"Finished Duhamel KS: L2 {initial_data['relative_l2']:.6e} -> "
        f"{after_mild_data['relative_l2']:.6e}; raw PDE "
        f"{initial_strong['pde_mse']:.6e} -> {after_mild_strong['pde_mse']:.6e}; "
        f"strong_enabled={args.strong_enabled}; artifacts={run_dir}"
    )
    return run_dir


def build_parser():
    parser = strong_ks.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        model=None,
        out=str(PROJECT_ROOT / "runs_data_ks_duhamel_to_strong"),
    )
    parser.add_argument(
        "--strong-enabled",
        type=strong_ks.parse_bool,
        default=True,
        metavar="true|false",
        help="After mandatory Duhamel training, run the unchanged strong PINN.",
    )
    parser.add_argument("--mild-epochs", type=int, default=1000)
    parser.add_argument("--mild-delta-t", type=float, default=0.01)
    parser.add_argument("--mild-quadrature-points", type=int, default=3)
    parser.add_argument("--mild-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--mild-num-x-fft",
        type=int,
        default=128,
        help="Periodic x grid size; the right endpoint is not duplicated.",
    )
    parser.add_argument(
        "--mild-interval-batch-size",
        type=int,
        default=8,
        help="Memory chunk only; gradients still average over every time interval.",
    )
    parser.add_argument("--mild-log-every", type=int, default=100)
    parser.add_argument(
        "--hidden-layers",
        default="100*5",
        help="RWF hidden layers used only when --model is omitted.",
    )
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
