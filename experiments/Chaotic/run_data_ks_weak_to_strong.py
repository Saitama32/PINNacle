"""Minimal weak-form KS training with an optional unchanged strong PINN stage.

The weak training graph integrates by parts only the fourth-order spatial term,
so the highest spatial derivative of the RWF MLP used during weak optimization
is ``u_xx``.  The optional strong stage is delegated verbatim to
``run_data_ks_pinn.py`` through the weak-stage checkpoint.
"""

from __future__ import annotations

import argparse
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
    ks_residual,
    load_checkpoint,
    load_data,
    prediction_metrics,
    save_checkpoint,
    save_solution_plot,
)
from src.utils.args import parse_hidden_layers


def parse_modes(value: str) -> int | None:
    if str(value).strip().lower() == "auto":
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected 'auto' or a non-negative integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("weak-num-modes must be non-negative")
    return result


def cpu_state_dict(network) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def exact_initial_condition_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x) * (1.0 + torch.sin(x))


def weak_spatial_terms(
    network,
    points: torch.Tensor,
    create_graph_for_backward: bool,
) -> dict[str, torch.Tensor]:
    """Evaluate only u, u_t, u_x and u_xx; never construct u_xxx/u_xxxx."""

    points = points.requires_grad_(True)
    value = network(points)
    gradient = torch.autograd.grad(
        value,
        points,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
    )[0]
    derivative_x = gradient[:, 0:1]
    derivative_t = gradient[:, 1:2]
    derivative_xx = torch.autograd.grad(
        derivative_x,
        points,
        grad_outputs=torch.ones_like(derivative_x),
        create_graph=create_graph_for_backward,
    )[0][:, 0:1]
    return {
        "u": value,
        "u_t": derivative_t,
        "u_x": derivative_x,
        "u_xx": derivative_xx,
    }


def weak_periodic_errors(
    network,
    times: torch.Tensor,
    x_lower: float,
    x_upper: float,
    create_graph_for_backward: bool,
) -> dict[str, torch.Tensor]:
    """Periodic jumps through u_xx, with no third or fourth derivative."""

    left = torch.cat((torch.full_like(times, x_lower), times), dim=1)
    right = torch.cat((torch.full_like(times, x_upper), times), dim=1)
    left_terms = weak_spatial_terms(network, left, create_graph_for_backward)
    right_terms = weak_spatial_terms(network, right, create_graph_for_backward)
    return {
        name: left_terms[name] - right_terms[name]
        for name in ("u", "u_x", "u_xx")
    }


def mean_square(value: torch.Tensor) -> torch.Tensor:
    value_for_loss = value.float() if value.dtype == torch.float16 else value
    return torch.mean(value_for_loss.square())


def resolve_modes(args, domain_length: float) -> tuple[int, float, float, int]:
    if args.beta <= 0 or args.gamma <= 0:
        raise ValueError("Automatic weak modes require positive beta and gamma")
    q_cutoff = math.sqrt(args.beta / args.gamma)
    q_max = args.weak_qmax_factor * q_cutoff
    automatic = int(math.ceil(q_max * domain_length / (2.0 * math.pi)))
    requested = automatic if args.weak_num_modes is None else args.weak_num_modes
    nyquist_limit = (args.weak_num_x_quad - 1) // 2
    resolved = min(requested, nyquist_limit)
    if resolved < requested:
        print(
            f"Clamped weak Fourier modes from K={requested} to K={resolved} "
            f"for Nx={args.weak_num_x_quad}."
        )
    return resolved, q_cutoff, q_max, requested


def build_weak_grid(lower, upper, args, device):
    length = float(upper[0] - lower[0])
    x = torch.arange(
        args.weak_num_x_quad,
        dtype={"float32": torch.float32, "float64": torch.float64}[args.precision],
        device=device,
    ).reshape(1, -1)
    x = float(lower[0]) + length * x / args.weak_num_x_quad
    modes = torch.arange(
        1,
        args.resolved_weak_modes + 1,
        dtype=x.dtype,
        device=device,
    ).reshape(-1, 1)
    q = 2.0 * math.pi * modes / length
    angle = q * (x - float(lower[0]))
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    q_square = q.square()
    return {
        "x": x,
        "dx": length / args.weak_num_x_quad,
        "cos": cosine,
        "sin": sine,
        "cos_xx": -q_square * cosine,
        "sin_xx": -q_square * sine,
    }


def weak_fourier_residuals(network, times, grid, args, backward):
    """Return constant/cosine/sine weak residuals at every supplied time."""

    count_t = times.shape[0]
    count_x = grid["x"].shape[1]
    x = grid["x"].expand(count_t, count_x)
    t = times.reshape(-1, 1).expand(count_t, count_x)
    points = torch.stack((x, t), dim=-1).reshape(-1, 2)
    terms = weak_spatial_terms(network, points, backward)
    shaped = {
        name: value.reshape(count_t, count_x)
        for name, value in terms.items()
    }
    base = (
        shaped["u_t"]
        + args.alpha * shaped["u"] * shaped["u_x"]
        + args.beta * shaped["u_xx"]
    )
    residual_constant = grid["dx"] * torch.sum(base, dim=1)
    if args.resolved_weak_modes:
        residual_cosine = grid["dx"] * (
            base @ grid["cos"].T
            + args.gamma * shaped["u_xx"] @ grid["cos_xx"].T
        )
        residual_sine = grid["dx"] * (
            base @ grid["sin"].T
            + args.gamma * shaped["u_xx"] @ grid["sin_xx"].T
        )
    else:
        residual_cosine = base.new_empty((count_t, 0))
        residual_sine = base.new_empty((count_t, 0))
    all_residuals = torch.cat(
        (residual_constant[:, None], residual_cosine, residual_sine), dim=1
    )
    return {
        "constant": residual_constant,
        "cosine": residual_cosine,
        "sine": residual_sine,
        "all": all_residuals,
        "loss": mean_square(all_residuals),
    }


def estimate_weak_periodic_scales(network, bounds, args, device):
    if not args.normalize_periodic_loss:
        return {name: 1.0 for name in ("u", "u_x", "u_xx")}
    rng = np.random.default_rng(args.seed + 31)
    lower, upper = bounds
    numpy_dtype = NUMPY_DTYPES[args.precision]
    sums = {name: 0.0 for name in ("u", "u_x", "u_xx")}
    total = 0
    for start in range(0, args.pinn_boundary_points, args.pinn_batch_size):
        count = min(args.pinn_batch_size, args.pinn_boundary_points - start)
        times = rng.uniform(lower[1], upper[1], size=(count, 1)).astype(numpy_dtype)
        errors = weak_periodic_errors(
            network,
            torch.as_tensor(times, device=device),
            float(lower[0]),
            float(upper[0]),
            False,
        )
        for name, error in errors.items():
            sums[name] += float(torch.sum(error.detach().double().square()).cpu())
        total += count
    return {
        name: max(value / total, args.periodic_normalization_epsilon)
        for name, value in sums.items()
    }


def weak_batch_losses(
    network,
    times,
    ic_x,
    boundary_times,
    anchor_points,
    anchor_values,
    grid,
    bounds,
    args,
    backward,
):
    residuals = weak_fourier_residuals(network, times, grid, args, backward)
    initial_time = torch.full_like(ic_x, float(bounds[0][1]))
    ic_points = torch.cat((ic_x, initial_time), dim=1)
    ic_loss = mean_square(network(ic_points) - exact_initial_condition_torch(ic_x))

    periodic_errors = weak_periodic_errors(
        network,
        boundary_times,
        float(bounds[0][0]),
        float(bounds[1][0]),
        backward,
    )
    periodic_raw = {name: mean_square(error) for name, error in periodic_errors.items()}
    periodic_components = {
        name: loss / float(args.weak_periodic_scales[name])
        if args.normalize_periodic_loss
        else loss
        for name, loss in periodic_raw.items()
    }
    periodic_loss = sum(periodic_components.values())

    if args.data_anchor:
        anchor_loss = mean_square(network(anchor_points) - anchor_values)
    else:
        anchor_loss = residuals["loss"].new_zeros(())
    total = (
        residuals["loss"]
        + args.ic_loss_weight * ic_loss
        + args.periodic_loss_weight * periodic_loss
        + args.data_anchor_weight * anchor_loss
    )
    return {
        "weak": residuals["loss"],
        "constant": residuals["constant"],
        "cosine": residuals["cosine"],
        "sine": residuals["sine"],
        "ic": ic_loss,
        "periodic_raw": periodic_raw,
        "periodic": periodic_loss,
        "anchor": anchor_loss,
        "total": total,
    }


def tensor_diagnostics(losses) -> dict[str, float]:
    def mean_abs(value):
        return float(torch.mean(torch.abs(value.detach())).cpu()) if value.numel() else 0.0

    return {
        "weak_loss": float(losses["weak"].detach().cpu()),
        "R_constant": mean_abs(losses["constant"]),
        "mean_abs_R_cos": mean_abs(losses["cosine"]),
        "mean_abs_R_sin": mean_abs(losses["sine"]),
        "max_abs_R": float(torch.max(torch.abs(torch.cat((
            losses["constant"].detach().reshape(-1),
            losses["cosine"].detach().reshape(-1),
            losses["sine"].detach().reshape(-1),
        )))).cpu()),
        "ic_loss": float(losses["ic"].detach().cpu()),
        "periodic_u": float(losses["periodic_raw"]["u"].detach().cpu()),
        "periodic_u_x": float(losses["periodic_raw"]["u_x"].detach().cpu()),
        "periodic_u_xx": float(losses["periodic_raw"]["u_xx"].detach().cpu()),
        "periodic_objective": float(losses["periodic"].detach().cpu()),
        "data_anchor": float(losses["anchor"].detach().cpu()),
        "total_loss": float(losses["total"].detach().cpu()),
    }


def weak_modal_rms(residuals) -> dict[str, torch.Tensor]:
    """RMS over time for every real Fourier test direction."""

    constant = torch.sqrt(torch.mean(residuals["constant"].detach().double().square()))
    cosine = torch.sqrt(torch.mean(residuals["cosine"].detach().double().square(), dim=0))
    sine = torch.sqrt(torch.mean(residuals["sine"].detach().double().square(), dim=0))
    combined = torch.sqrt(cosine.square() + sine.square())
    return {
        "constant": constant,
        "cosine": cosine,
        "sine": sine,
        "combined": combined,
    }


def _positive(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, np.finfo(np.float64).tiny)


def save_weak_modal_diagnostic(network, bounds, args, device, run_dir, stage):
    """Save E_k=sqrt(mean_t R_k(t)^2) without changing the training loss."""

    import matplotlib.pyplot as plt

    lower, upper = bounds
    grid = build_weak_grid(lower, upper, args, device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.precision]
    times = torch.linspace(
        float(lower[1]),
        float(upper[1]),
        args.weak_diagnostic_num_time_samples,
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)
    network.eval()
    residuals = weak_fourier_residuals(network, times, grid, args, backward=False)
    modal = weak_modal_rms(residuals)
    modes = np.arange(1, args.resolved_weak_modes + 1)
    constant = float(modal["constant"].cpu())
    cosine = modal["cosine"].cpu().numpy()
    sine = modal["sine"].cpu().numpy()
    combined = modal["combined"].cpu().numpy()
    np.savez_compressed(
        run_dir / f"weak_modal_residuals_{stage}.npz",
        times=times.detach().cpu().numpy()[:, 0],
        modes=modes,
        constant_rms=constant,
        cosine_rms=cosine,
        sine_rms=sine,
        combined_rms=combined,
        constant_residual=residuals["constant"].detach().cpu().numpy(),
        cosine_residual=residuals["cosine"].detach().cpu().numpy(),
        sine_residual=residuals["sine"].detach().cpu().numpy(),
    )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.scatter([0], [_positive(np.asarray([constant]))[0]], label="constant k=0", zorder=4)
    if modes.size:
        axis.semilogy(modes, _positive(cosine), label="cos RMS")
        axis.semilogy(modes, _positive(sine), label="sin RMS")
        axis.semilogy(modes, _positive(combined), label="pair magnitude", linewidth=2)
    axis.set(xlabel="Fourier mode k", ylabel=r"$E_k$", title=f"Weak residual modes ({stage})")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(run_dir / f"weak_modal_residuals_{stage}.png", dpi=180)
    plt.close(figure)
    return {
        "constant_rms": constant,
        "max_pair_rms": float(np.max(combined)) if combined.size else 0.0,
        "edge_pair_rms": float(combined[-1]) if combined.size else 0.0,
        "num_time_samples": args.weak_diagnostic_num_time_samples,
        "artifact": str(run_dir / f"weak_modal_residuals_{stage}.npz"),
        "plot": str(run_dir / f"weak_modal_residuals_{stage}.png"),
    }


def save_strong_residual_spectrum(network, bounds, args, device, run_dir, stage):
    """FFT-in-x diagnostic of the pointwise strong KS residual."""

    import matplotlib.pyplot as plt

    lower, upper = bounds
    nx = args.weak_diagnostic_num_x
    nt = args.weak_diagnostic_num_time_samples
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.precision]
    x = float(lower[0]) + (float(upper[0] - lower[0]) / nx) * torch.arange(
        nx, dtype=dtype, device=device
    )
    times = torch.linspace(float(lower[1]), float(upper[1]), nt, dtype=dtype, device=device)
    rows = []
    network.eval()
    for time_value in times:
        points = torch.stack((x, torch.full_like(x, time_value)), dim=1).requires_grad_(True)
        value = ks_residual(
            network, points, alpha=args.alpha, beta=args.beta, gamma=args.gamma
        )
        rows.append(value[:, 0].detach())
    residual = torch.stack(rows).double()
    coefficients = torch.fft.fft(residual, dim=1, norm="forward")
    mean_power_full = torch.mean(torch.abs(coefficients).square(), dim=0)
    signed_modes = torch.fft.fftfreq(nx, d=1.0 / nx, device=device).round().to(torch.int64)
    controlled = torch.abs(signed_modes) <= args.resolved_weak_modes
    inside_rms = torch.sqrt(torch.sum(mean_power_full[controlled]))
    outside_rms = torch.sqrt(torch.sum(mean_power_full[~controlled]))
    physical_rms = torch.sqrt(torch.mean(residual.square()))
    positive_count = nx // 2 + 1
    plotted_power = mean_power_full[:positive_count]
    plotted_rms = torch.sqrt(plotted_power)
    plotted_modes = np.arange(positive_count)
    rms_numpy = plotted_rms.cpu().numpy()
    np.savez_compressed(
        run_dir / f"strong_residual_spectrum_{stage}.npz",
        x=x.detach().cpu().numpy(),
        t=times.detach().cpu().numpy(),
        residual=residual.cpu().numpy(),
        modes=plotted_modes,
        modal_rms=rms_numpy,
        modal_mean_power=plotted_power.cpu().numpy(),
        controlled_K=args.resolved_weak_modes,
    )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.semilogy(plotted_modes, _positive(rms_numpy), linewidth=1.6)
    axis.axvline(
        args.resolved_weak_modes,
        color="tab:red",
        linestyle="--",
        label=f"weak K={args.resolved_weak_modes}",
    )
    axis.set(
        xlabel="Spatial Fourier mode |k|",
        ylabel=r"$\sqrt{\mathrm{mean}_t|\hat r_k|^2}$",
        title=f"Strong KS residual spectrum ({stage})",
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(run_dir / f"strong_residual_spectrum_{stage}.png", dpi=180)
    plt.close(figure)
    outside_value = float(outside_rms.cpu())
    inside_value = float(inside_rms.cpu())
    return {
        "physical_rms": float(physical_rms.cpu()),
        "inside_K_rms": inside_value,
        "outside_K_rms": outside_value,
        "outside_to_inside_ratio": outside_value / inside_value if inside_value > 0 else None,
        "peak_nonzero_mode": int(np.argmax(rms_numpy[1:]) + 1) if len(rms_numpy) > 1 else 0,
        "num_x": nx,
        "num_t": nt,
        "artifact": str(run_dir / f"strong_residual_spectrum_{stage}.npz"),
        "plot": str(run_dir / f"strong_residual_spectrum_{stage}.png"),
    }


def optimizer_for_weak(network, args):
    optimizer = torch.optim.LBFGS(
        network.parameters(),
        lr=args.pinn_lr,
        max_iter=args.pinn_lbfgs_max_iter,
        max_eval=args.pinn_lbfgs_max_eval,
        tolerance_grad=args.pinn_lbfgs_tolerance_grad,
        tolerance_change=args.pinn_lbfgs_tolerance_change,
        history_size=args.pinn_lbfgs_history_size,
        line_search_fn=(
            None if args.pinn_lbfgs_line_search == "none" else args.pinn_lbfgs_line_search
        ),
    )
    return optimizer, None


def build_non_lbfgs_optimizer(network, args):
    """Build optimizer and scheduler without accidentally constructing two optimizers."""

    stage_args = copy.copy(args)
    stage_args.optimizer = args.pinn_optimizer
    stage_args.lr = args.pinn_lr
    stage_args.weight_decay = args.pinn_weight_decay
    stage_args.lr_scheduler = args.pinn_lr_scheduler
    stage_args.lr_decay_steps = args.pinn_lr_decay_steps
    stage_args.lr_decay_rate = args.pinn_lr_decay_rate
    stage_args.lr_min = args.pinn_lr_min
    stage_args.iterations = args.weak_epochs
    optimizer = build_optimizer(network, stage_args)
    return optimizer, build_scheduler(optimizer, stage_args)


def sample_batches(rng, points, values, bounds, args, device):
    dtype = NUMPY_DTYPES[args.precision]
    lower, upper = bounds
    times = rng.uniform(
        lower[1], upper[1], size=(args.weak_num_time_samples, 1)
    ).astype(dtype)
    ic_x = rng.uniform(
        lower[0], upper[0], size=(args.pinn_train_ic_points, 1)
    ).astype(dtype)
    boundary_times = rng.uniform(
        lower[1], upper[1], size=(args.pinn_train_boundary_points, 1)
    ).astype(dtype)
    if args.data_anchor:
        indices = rng.integers(0, len(points), size=args.pinn_train_data_points)
        anchor_points = torch.as_tensor(points[indices], device=device)
        anchor_values = torch.as_tensor(values[indices], device=device)
    else:
        anchor_points = anchor_values = None
    return (
        torch.as_tensor(times, device=device),
        torch.as_tensor(ic_x, device=device),
        torch.as_tensor(boundary_times, device=device),
        anchor_points,
        anchor_values,
    )


def fixed_batches(points, values, bounds, args, device):
    dtype = NUMPY_DTYPES[args.precision]
    lower, upper = bounds
    times = np.linspace(
        lower[1], upper[1], args.weak_num_time_samples, dtype=dtype
    ).reshape(-1, 1)
    ic_x = np.linspace(
        lower[0], upper[0], args.pinn_train_ic_points, dtype=dtype
    ).reshape(-1, 1)
    boundary_times = np.linspace(
        lower[1], upper[1], args.pinn_train_boundary_points, dtype=dtype
    ).reshape(-1, 1)
    if args.data_anchor:
        count = min(args.pinn_train_data_points, len(points))
        indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
        anchor_points = torch.as_tensor(points[indices], device=device)
        anchor_values = torch.as_tensor(values[indices], device=device)
    else:
        anchor_points = anchor_values = None
    return (
        torch.as_tensor(times, device=device),
        torch.as_tensor(ic_x, device=device),
        torch.as_tensor(boundary_times, device=device),
        anchor_points,
        anchor_values,
    )


def save_history(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_weak(
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
    grid = build_weak_grid(bounds[0], bounds[1], args, device)
    if args.pinn_optimizer == "lbfgs":
        optimizer, scheduler = optimizer_for_weak(network, args)
    else:
        optimizer, scheduler = build_non_lbfgs_optimizer(network, args)
    rng = np.random.default_rng(args.seed + 101)
    fixed = fixed_batches(points, values, bounds, args, device)
    lbfgs_batch = sample_batches(rng, points, values, bounds, args, device)
    rows = []
    best_score = math.inf
    best_iteration = 0
    best_state = cpu_state_dict(network)
    gradients_verified = False

    print(
        f"Starting minimal weak KS for {args.weak_epochs} iterations; "
        f"optimizer={args.pinn_optimizer}; K={args.resolved_weak_modes}; "
        f"q_max={args.resolved_q_max:.6g}; Nx={args.weak_num_x_quad}; "
        "max_spatial_derivative=u_xx; u_xxx/u_xxxx disabled in weak training."
    )
    for iteration in range(1, args.weak_epochs + 1):
        network.train()
        batch = lbfgs_batch if args.pinn_optimizer == "lbfgs" else sample_batches(
            rng, points, values, bounds, args, device
        )

        def compute_losses():
            return weak_batch_losses(
                network, *batch, grid, bounds, args, backward=True
            )

        if args.pinn_optimizer == "lbfgs":
            latest = {}

            def closure():
                optimizer.zero_grad(set_to_none=True)
                losses = compute_losses()
                losses["total"].backward()
                latest.clear()
                latest.update(losses)
                return losses["total"]

            optimizer.step(closure)
            batch_losses = latest
        else:
            optimizer.zero_grad(set_to_none=True)
            batch_losses = compute_losses()
            batch_losses["total"].backward()
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
                raise RuntimeError("Weak loss produced missing or non-finite parameter gradients")
            if not any(bool(torch.any(item != 0)) for item in gradients):
                raise RuntimeError("Weak loss produced only zero parameter gradients")
            gradients_verified = True
        if scheduler is not None:
            scheduler.step()

        if iteration % args.weak_log_every == 0 or iteration == args.weak_epochs:
            network.eval()
            fixed_losses = weak_batch_losses(
                network, *fixed, grid, bounds, args, backward=False
            )
            diagnostics = tensor_diagnostics(fixed_losses)
            data_metric = prediction_metrics(
                network,
                validation_points,
                validation_values,
                args.eval_batch_size,
                device,
            )
            score = diagnostics["total_loss"]
            if math.isfinite(score) and score < best_score:
                best_score = score
                best_iteration = iteration
                best_state = cpu_state_dict(network)
            row = {
                "iteration": iteration,
                **diagnostics,
                "data_relative_l2": data_metric["relative_l2"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            rows.append(row)
            print(
                f"Weak step={iteration:7d} weak={diagnostics['weak_loss']:.6e} "
                f"total={diagnostics['total_loss']:.6e} "
                f"R0={diagnostics['R_constant']:.6e} "
                f"Rcos={diagnostics['mean_abs_R_cos']:.6e} "
                f"Rsin={diagnostics['mean_abs_R_sin']:.6e} "
                f"maxR={diagnostics['max_abs_R']:.6e} "
                f"L2={data_metric['relative_l2']:.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.6e}"
            )

    last_state = cpu_state_dict(network)
    save_checkpoint(run_dir / "weights_weak_last.pt", network, metadata)
    network.load_state_dict(best_state, strict=True)
    save_checkpoint(run_dir / "weights_weak_best.pt", network, metadata)
    network.load_state_dict(last_state, strict=True)
    save_history(run_dir / "weak_history.csv", rows)
    return {
        "iterations": args.weak_epochs,
        "best_iteration": best_iteration,
        "best_total_loss": best_score,
        "gradients_verified": gradients_verified,
        "training_max_spatial_derivative": "u_xx",
        "uses_u_xxx": False,
        "uses_u_xxxx": False,
        "history_rows": len(rows),
    }


def save_weak_predictions(network, points, values, args, device, run_dir):
    predictions = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), args.eval_batch_size):
            batch = torch.as_tensor(points[start : start + args.eval_batch_size], device=device)
            predictions.append(network(batch).cpu().numpy())
    prediction = np.vstack(predictions)[:, 0]
    np.savez_compressed(
        run_dir / "predictions_weak.npz",
        x=points[:, 0],
        t=points[:, 1],
        exact=values[:, 0],
        prediction=prediction,
    )
    metric = prediction_metrics(network, points, values, args.eval_batch_size, device)
    save_solution_plot(
        run_dir / "solution_weak.png",
        points,
        values[:, 0],
        prediction,
        f"Weak KS RWF, relative L2={metric['relative_l2']:.3e}",
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
    if args.weak_epochs <= 0 or args.weak_log_every <= 0:
        raise ValueError("weak-epochs and weak-log-every must be positive")
    if args.weak_num_time_samples <= 0 or args.weak_num_x_quad < 3:
        raise ValueError("weak time samples must be positive and Nx must be at least 3")
    if args.weak_diagnostic_num_time_samples <= 0:
        raise ValueError("weak-diagnostic-num-time-samples must be positive")
    if args.weak_diagnostic_num_x < 3:
        raise ValueError("weak-diagnostic-num-x must be at least 3")
    if args.weak_qmax_factor <= 0 or not math.isfinite(args.weak_qmax_factor):
        raise ValueError("weak-qmax-factor must be positive and finite")
    if args.pinn_precision not in {"float32", "float64"}:
        raise ValueError("Weak KS supports float32 or float64")
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
        data_path = Path(args.data or (PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat")).resolve()
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
    domain_length = float(upper[0] - lower[0])
    (
        args.resolved_weak_modes,
        args.resolved_q_cutoff,
        args.resolved_q_max,
        args.requested_weak_modes,
    ) = resolve_modes(args, domain_length)
    args.weak_periodic_scales = estimate_weak_periodic_scales(
        network, (lower, upper), args, device
    )

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    # Keep the wrapper name short: the unchanged strong runner embeds this name
    # once more in its own artifact directory, and Windows still commonly has a
    # short MAX_PATH limit.  The complete source path is retained in the config.
    origin = "pretrained" if source_dir is not None else "scratch"
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-{origin}-ks-weak-to-strong-{args.pinn_optimizer}-{args.precision}"
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
        max_spatial_derivative_in_weak_training="u_xx",
    )
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(resolved, file_obj, indent=2, sort_keys=True)

    initial_data = prediction_metrics(
        network, points, values, args.eval_batch_size, device
    )
    initial_strong = evaluate_pinn_loss(network, (lower, upper), args, device)
    initial_weak_modal = None
    initial_strong_spectrum = None
    if args.weak_spectral_diagnostics:
        initial_weak_modal = save_weak_modal_diagnostic(
            network, (lower, upper), args, device, run_dir, "initial"
        )
        initial_strong_spectrum = save_strong_residual_spectrum(
            network, (lower, upper), args, device, run_dir, "initial"
        )
    weak_info = train_weak(
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
    after_weak_data = save_weak_predictions(
        network, points, values, args, device, run_dir
    )
    after_weak_strong = evaluate_pinn_loss(network, (lower, upper), args, device)
    after_weak_modal = None
    after_weak_strong_spectrum = None
    if args.weak_spectral_diagnostics:
        after_weak_modal = save_weak_modal_diagnostic(
            network, (lower, upper), args, device, run_dir, "after_weak"
        )
        after_weak_strong_spectrum = save_strong_residual_spectrum(
            network, (lower, upper), args, device, run_dir, "after_weak"
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

    strong_dir = None
    strong_metrics = None
    transfer_equal = None
    if args.strong_enabled:
        print(
            "Weak stage finished. Starting the unchanged strong KS pipeline from "
            f"{run_dir / 'weights_weak_last.pt'}"
        )
        strong_args = copy.deepcopy(args)
        strong_args.model = str(run_dir / "weights_weak_last.pt")
        strong_args.data = str(data_path)
        strong_args.out = str(run_dir / "strong")
        strong_dir = strong_ks.run(strong_args)
        transfer_equal = checkpoints_equal(
            run_dir / "weights_weak_last.pt", strong_dir / "weights_initial.pt"
        )
        if not transfer_equal:
            raise RuntimeError("Strong stage did not start from the exact weak-last weights")
        with (strong_dir / "metrics.json").open("r", encoding="utf-8") as file_obj:
            strong_metrics = json.load(file_obj)

    metrics = {
        "configuration": resolved,
        "initial": {
            "data": initial_data,
            "strong_pinn": initial_strong,
            "weak_modal_residual": initial_weak_modal,
            "strong_residual_spectrum": initial_strong_spectrum,
        },
        "weak_stage": weak_info,
        "after_weak": {
            "data": after_weak_data,
            "strong_pinn": after_weak_strong,
            "weak_modal_residual": after_weak_modal,
            "strong_residual_spectrum": after_weak_strong_spectrum,
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
        f"Finished weak KS: L2 {initial_data['relative_l2']:.6e} -> "
        f"{after_weak_data['relative_l2']:.6e}; raw PDE "
        f"{initial_strong['pde_mse']:.6e} -> {after_weak_strong['pde_mse']:.6e}; "
        f"strong_enabled={args.strong_enabled}; artifacts={run_dir}"
    )
    return run_dir


def build_parser():
    parser = strong_ks.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        model=None,
        out=str(PROJECT_ROOT / "runs_data_ks_weak_to_strong"),
    )
    parser.add_argument(
        "--strong-enabled",
        type=strong_ks.parse_bool,
        default=True,
        metavar="true|false",
        help="After the mandatory weak stage, continue with the unchanged strong PINN.",
    )
    parser.add_argument("--weak-epochs", type=int, default=1000)
    parser.add_argument("--weak-num-time-samples", type=int, default=16)
    parser.add_argument(
        "--weak-num-x-quad",
        type=int,
        default=128,
        help="Number of non-duplicated points in the uniform periodic x quadrature.",
    )
    parser.add_argument(
        "--K",
        "--weak-num-modes",
        dest="weak_num_modes",
        type=parse_modes,
        default=None,
        metavar="auto|N",
        help="Fourier test modes; auto uses qmax_factor*sqrt(beta/gamma).",
    )
    parser.add_argument("--weak-qmax-factor", type=float, default=1.5)
    parser.add_argument("--weak-log-every", type=int, default=100)
    parser.add_argument(
        "--weak-spectral-diagnostics",
        type=strong_ks.parse_bool,
        default=True,
        metavar="true|false",
        help="Save per-mode weak RMS and FFT diagnostics of the strong residual.",
    )
    parser.add_argument(
        "--weak-diagnostic-num-time-samples",
        type=int,
        default=64,
        help="Uniform time samples used only by spectral diagnostics.",
    )
    parser.add_argument(
        "--weak-diagnostic-num-x",
        type=int,
        default=128,
        help="Periodic x-grid size used only by the strong-residual FFT diagnostic.",
    )
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
