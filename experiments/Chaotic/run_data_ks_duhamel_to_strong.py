"""Global-network Duhamel KS training with an optional strong PINN stage.

The mandatory first stage trains one global RWF MLP by local mild/Duhamel
links. The one-step term evaluates both endpoints directly. The optional
two-step term propagates exactly one intermediate endpoint, while both
nonlinear forcing integrals remain anchored to that same global network. The
optional strong stage delegates unchanged optimization to
``run_data_ks_pinn.py``.
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


class ReferenceFourierInterpolator(torch.nn.Module):
    """Differentiable periodic-x, cubic-time reference interpolant."""

    def __init__(self, times, coefficients, wave_numbers, x_lower):
        super().__init__()
        self.register_buffer("times", times)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("wave_numbers", wave_numbers)
        self.x_lower = float(x_lower)

    def forward(self, points):
        x = points[:, 0]
        t = points[:, 1].contiguous()
        insertion = torch.searchsorted(self.times, t, right=False)
        start = torch.clamp(insertion - 2, 0, len(self.times) - 4)
        indices = start[:, None] + torch.arange(4, device=t.device)[None, :]
        local_times = self.times[indices]
        coefficients = torch.zeros(
            (len(points), self.coefficients.shape[1]),
            dtype=self.coefficients.dtype,
            device=points.device,
        )
        # Four-point local Lagrange interpolation is diagnostic-only. It avoids
        # making the reference floor depend on piecewise-linear interpolation
        # of a data set whose stored temporal spacing is 0.004.
        for basis_index in range(4):
            weight = torch.ones_like(t)
            for other_index in range(4):
                if other_index != basis_index:
                    weight = weight * (
                        (t - local_times[:, other_index])
                        / (
                            local_times[:, basis_index]
                            - local_times[:, other_index]
                        )
                    )
            coefficients = coefficients + weight[:, None] * self.coefficients[
                indices[:, basis_index]
            ]
        phase = torch.exp(
            1j * (x[:, None] - self.x_lower) * self.wave_numbers[None, :]
        )
        return torch.sum(coefficients * phase, dim=1).real[:, None]


def build_reference_interpolator(points, values, bounds, device, dtype):
    """Build a diagnostic-only interpolant from a rectangular reference grid."""

    x_values = np.unique(points[:, 0].astype(np.float64))
    t_values = np.unique(points[:, 1].astype(np.float64))
    if len(points) != len(x_values) * len(t_values):
        raise ValueError("Reference Duhamel diagnostic requires a rectangular data grid")
    field = np.empty((len(t_values), len(x_values)), dtype=np.float64)
    x_indices = np.searchsorted(x_values, points[:, 0])
    t_indices = np.searchsorted(t_values, points[:, 1])
    field[t_indices, x_indices] = values[:, 0]

    duplicated_endpoint = bool(
        len(x_values) > 2
        and np.allclose(field[:, -1], field[:, 0], rtol=1e-8, atol=1e-10)
    )
    periodic_field = field[:, :-1] if duplicated_endpoint else field
    source_nx = periodic_field.shape[1]
    length = float(bounds[1][0] - bounds[0][0])
    field_tensor = torch.as_tensor(periodic_field, dtype=dtype, device=device)
    coefficients = torch.fft.fft(field_tensor, dim=1, norm="forward")
    wave_numbers = 2.0 * math.pi * torch.fft.fftfreq(
        source_nx, d=length / source_nx, dtype=dtype, device=device
    )
    network = ReferenceFourierInterpolator(
        torch.as_tensor(t_values, dtype=dtype, device=device),
        coefficients,
        wave_numbers,
        bounds[0][0],
    ).to(device)
    return network, {
        "source_num_x": int(source_nx),
        "source_num_t": int(len(t_values)),
        "duplicated_periodic_endpoint_removed": duplicated_endpoint,
        "space_interpolation": "periodic Fourier",
        "time_interpolation": "local four-point cubic Lagrange",
    }


def mild_step_from_state(network, state_hat, left_times, grid, args, backward=True):
    """Apply one unchanged Duhamel step to supplied Fourier endpoint states."""

    left_times = left_times.to(device=grid["x"].device, dtype=grid["x"].dtype)
    count = len(left_times)
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
    return grid["propagator"][None, :] * state_hat + integral


def mild_duhamel_defects(network, left_times, grid, args, backward=True):
    """Compute local mild defects without u_t, u_xx, u_xxx, or u_xxxx."""

    left_times = left_times.to(device=grid["x"].device, dtype=grid["x"].dtype)
    right_times = left_times + args.resolved_mild_delta_t
    endpoint_times = torch.cat((left_times, right_times), dim=0)
    endpoint_values, _ = evaluate_space_time(
        network, grid["x"], endpoint_times, False, backward
    )
    count = len(left_times)
    # norm="forward" makes these discrete approximations of Fourier-series
    # coefficients. Parseval then requires an additional Nx when a mean over
    # Fourier modes is used to recover the physical-space mean-square defect.
    hat_left = torch.fft.fft(endpoint_values[:count], dim=1, norm="forward")
    hat_right = torch.fft.fft(endpoint_values[count:], dim=1, norm="forward")
    propagated = mild_step_from_state(
        network, hat_left, left_times, grid, args, backward
    )
    defect = hat_right - propagated
    squared_magnitude = defect.real.square() + defect.imag.square()
    nx = defect.shape[1]
    fourier_mse = torch.mean(squared_magnitude)
    physical_mse = nx * fourier_mse
    per_interval_fourier_mse = torch.mean(squared_magnitude, dim=1)
    return {
        "defect": defect,
        "loss": physical_mse,
        "fourier_mse": fourier_mse,
        "physical_rms": torch.sqrt(physical_mse),
        "per_interval_fourier_mse": per_interval_fourier_mse,
        "per_interval_rms": torch.sqrt(nx * per_interval_fourier_mse),
        "per_interval_mean_abs": torch.mean(torch.abs(defect), dim=1),
        "per_interval_max_abs": torch.max(torch.abs(defect), dim=1).values,
    }


def two_step_duhamel_defects(network, left_times, grid, args, backward=True):
    """Propagate only one intermediate endpoint while anchoring both forcings."""

    left_times = left_times.to(device=grid["x"].device, dtype=grid["x"].dtype)
    dt = args.resolved_mild_delta_t
    end_times = left_times + 2.0 * dt
    endpoint_times = torch.cat((left_times, end_times), dim=0)
    endpoint_values, _ = evaluate_space_time(
        network, grid["x"], endpoint_times, False, backward
    )
    count = len(left_times)
    hat_start = torch.fft.fft(
        endpoint_values[:count], dim=1, norm="forward"
    )
    hat_end = torch.fft.fft(
        endpoint_values[count:], dim=1, norm="forward"
    )

    propagated_one = mild_step_from_state(
        network, hat_start, left_times, grid, args, backward
    )
    propagated_two = mild_step_from_state(
        network, propagated_one, left_times + dt, grid, args, backward
    )
    defect = hat_end - propagated_two
    squared_magnitude = defect.real.square() + defect.imag.square()
    nx = defect.shape[1]
    fourier_mse = torch.mean(squared_magnitude)
    physical_mse = nx * fourier_mse
    per_interval_fourier_mse = torch.mean(squared_magnitude, dim=1)
    return {
        "defect": defect,
        "loss": physical_mse,
        "fourier_mse": fourier_mse,
        "physical_rms": torch.sqrt(physical_mse),
        "per_interval_fourier_mse": per_interval_fourier_mse,
        "per_interval_rms": torch.sqrt(nx * per_interval_fourier_mse),
        "per_interval_mean_abs": torch.mean(torch.abs(defect), dim=1),
        "per_interval_max_abs": torch.max(torch.abs(defect), dim=1).values,
    }


def causal_chain_rollout(network, grid, args, backward=True):
    """Roll out H steps from the exact IC with an unbroken autograd graph."""

    steps = args.mild_chain_steps
    dtype = grid["x"].dtype
    device = grid["x"].device
    exact_ic = exact_initial_condition_torch(grid["x"][:, None])[:, 0]
    roll_hat = torch.fft.fft(exact_ic[None, :], dim=1, norm="forward")
    rollout_states = [exact_ic]
    all_times = args.time_lower + torch.arange(
        0, steps + 1, dtype=dtype, device=device
    ) * args.resolved_mild_delta_t
    network_states, _ = evaluate_space_time(
        network, grid["x"], all_times, False, backward
    )
    for step in range(steps):
        left_time = torch.as_tensor(
            [args.time_lower + step * args.resolved_mild_delta_t],
            dtype=dtype,
            device=device,
        )
        # Core invariant: this propagated state, not u_theta(t_j), starts the
        # next step. No detach is permitted anywhere in this loop.
        roll_hat = mild_step_from_state(
            network, roll_hat, left_time, grid, args, backward
        )
        rollout_states.append(
            torch.fft.ifft(roll_hat, dim=1, norm="forward").real[0]
        )
    rollout_endpoints = torch.stack(rollout_states[1:])
    difference = network_states[1:] - rollout_endpoints
    per_step_mse = torch.mean(difference.square(), dim=1)
    return {
        "loss": torch.mean(per_step_mse),
        "per_step_mse": per_step_mse,
        "per_step_rms": torch.sqrt(per_step_mse),
        "rollout_states": torch.stack(rollout_states),
        "network_states": network_states,
        "difference": difference,
        "times": all_times,
    }


def iter_interval_chunks(args, device, dtype, count=None, chunk_size=None):
    count = args.resolved_mild_intervals if count is None else count
    chunk = min(args.mild_interval_batch_size if chunk_size is None else chunk_size, count)
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
    one_losses = []
    one_fourier_losses = []
    one_rms = []
    one_means = []
    one_maxima = []
    for _, _, left_times in iter_interval_chunks(args, device, dtype):
        result = mild_duhamel_defects(network, left_times, grid, args, backward=False)
        one_losses.append(result["loss"].detach() * len(left_times))
        one_fourier_losses.append(result["fourier_mse"].detach() * len(left_times))
        one_rms.append(result["per_interval_rms"].detach())
        one_means.append(result["per_interval_mean_abs"].detach())
        one_maxima.append(result["per_interval_max_abs"].detach())
    one_step_loss = torch.sum(torch.stack(one_losses)) / args.resolved_mild_intervals
    one_step_fourier_mse = (
        torch.sum(torch.stack(one_fourier_losses)) / args.resolved_mild_intervals
    )

    two_count = args.resolved_mild_intervals - 1
    two_chunk_size = max(1, args.mild_interval_batch_size // 2)
    two_losses = []
    two_fourier_losses = []
    two_rms = []
    two_means = []
    two_maxima = []
    for _, _, left_times in iter_interval_chunks(
        args, device, dtype, count=two_count, chunk_size=two_chunk_size
    ):
        result = two_step_duhamel_defects(
            network, left_times, grid, args, backward=False
        )
        two_losses.append(result["loss"].detach() * len(left_times))
        two_fourier_losses.append(
            result["fourier_mse"].detach() * len(left_times)
        )
        two_rms.append(result["per_interval_rms"].detach())
        two_means.append(result["per_interval_mean_abs"].detach())
        two_maxima.append(result["per_interval_max_abs"].detach())
    two_step_loss = torch.sum(torch.stack(two_losses)) / two_count
    two_step_fourier_mse = torch.sum(torch.stack(two_fourier_losses)) / two_count
    two_step_weight = args.mild_two_step_weight
    mild_loss = one_step_loss + two_step_weight * two_step_loss
    fourier_mse = (
        one_step_fourier_mse + two_step_weight * two_step_fourier_mse
    )
    ic_loss = initial_condition_loss(network, fixed_ic_x, args.time_lower).detach()
    one_interval_rms = torch.cat(one_rms)
    two_interval_rms = torch.cat(two_rms)
    return {
        "mild_loss": float(mild_loss.cpu()),
        "fourier_mse": float(fourier_mse.cpu()),
        "physical_rms": float(torch.sqrt(mild_loss).cpu()),
        "one_step_mild_loss": float(one_step_loss.cpu()),
        "two_step_mild_loss": float(two_step_loss.cpu()),
        "total_mild_loss": float(mild_loss.cpu()),
        "one_step_fourier_mse": float(one_step_fourier_mse.cpu()),
        "two_step_fourier_mse": float(two_step_fourier_mse.cpu()),
        "one_step_physical_rms": float(torch.sqrt(one_step_loss).cpu()),
        "two_step_physical_rms": float(torch.sqrt(two_step_loss).cpu()),
        "ic_loss": float(ic_loss.cpu()),
        "total_loss": float(
            (args.mild_loss_weight * mild_loss + args.ic_loss_weight * ic_loss).cpu()
        ),
        "mean_defect_per_interval": float(torch.mean(one_interval_rms).cpu()),
        "max_defect_per_interval": float(torch.max(one_interval_rms).cpu()),
        "two_step_mean_defect_per_interval": float(
            torch.mean(two_interval_rms).cpu()
        ),
        "two_step_max_defect_per_interval": float(
            torch.max(two_interval_rms).cpu()
        ),
        "interval_rms": one_interval_rms.cpu().numpy(),
        "interval_mean_abs": torch.cat(one_means).cpu().numpy(),
        "interval_max_abs": torch.cat(one_maxima).cpu().numpy(),
        "two_step_interval_rms": two_interval_rms.cpu().numpy(),
        "two_step_interval_mean_abs": torch.cat(two_means).cpu().numpy(),
        "two_step_interval_max_abs": torch.cat(two_maxima).cpu().numpy(),
    }


def fixed_chain_diagnostics(network, fixed_ic_x, grid, args):
    result = causal_chain_rollout(network, grid, args, backward=False)
    chain_loss = result["loss"].detach()
    step_rms = result["per_step_rms"].detach()
    ic_loss = initial_condition_loss(network, fixed_ic_x, args.time_lower).detach()
    total = args.mild_loss_weight * chain_loss + args.ic_loss_weight * ic_loss
    return {
        "chain_loss": float(chain_loss.cpu()),
        "max_chain_error": float(torch.max(step_rms).cpu()),
        "chain_step_rms": step_rms.cpu().numpy(),
        "ic_loss": float(ic_loss.cpu()),
        "total_loss": float(total.cpu()),
    }


def save_chain_rollout_artifact(network, grid, args, run_dir):
    result = causal_chain_rollout(network, grid, args, backward=False)
    rollout = result["rollout_states"].detach().cpu().numpy()
    network_states = result["network_states"].detach().cpu().numpy()
    difference = network_states - rollout
    np.savez_compressed(
        run_dir / "mild_chain_rollout_final.npz",
        x=grid["x"].detach().cpu().numpy(),
        t=result["times"].detach().cpu().numpy(),
        rollout_states=rollout,
        network_states=network_states,
        difference=difference,
        endpoint_rms=result["per_step_rms"].detach().cpu().numpy(),
    )


def evaluate_reference_duhamel(points, values, bounds, args, device, run_dir):
    """Evaluate the exact-data trajectory with the same Duhamel implementation."""

    run_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.precision]
    reference, interpolation = build_reference_interpolator(
        points, values, bounds, device, dtype
    )
    fixed_ic_x = torch.linspace(
        float(bounds[0][0]),
        float(bounds[1][0]),
        args.pinn_train_ic_points,
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)
    results = {}
    for quadrature_points in sorted({3, 5, args.mild_quadrature_points}):
        diagnostic_args = copy.copy(args)
        diagnostic_args.mild_quadrature_points = quadrature_points
        grid = build_mild_grid(bounds[0], bounds[1], diagnostic_args, device)
        diagnostics = fixed_mild_diagnostics(
            reference, fixed_ic_x, grid, diagnostic_args, device
        )
        scalar_diagnostics = {
            name: value
            for name, value in diagnostics.items()
            if not isinstance(value, np.ndarray)
        }
        results[f"q{quadrature_points}"] = scalar_diagnostics
        np.savez_compressed(
            run_dir / f"reference_duhamel_defects_q{quadrature_points}.npz",
            t_left=args.time_lower
            + np.arange(args.resolved_mild_intervals) * args.resolved_mild_delta_t,
            t_right=args.time_lower
            + (np.arange(args.resolved_mild_intervals) + 1)
            * args.resolved_mild_delta_t,
            physical_rms=diagnostics["interval_rms"],
            two_step_t_left=args.time_lower
            + np.arange(args.resolved_mild_intervals - 1)
            * args.resolved_mild_delta_t,
            two_step_t_right=args.time_lower
            + (np.arange(args.resolved_mild_intervals - 1) + 2)
            * args.resolved_mild_delta_t,
            two_step_physical_rms=diagnostics["two_step_interval_rms"],
            fourier_mean_abs=diagnostics["interval_mean_abs"],
            fourier_max_abs=diagnostics["interval_max_abs"],
        )
        print(
            f"Reference Duhamel Q={quadrature_points}: "
            f"one_mse={diagnostics['one_step_mild_loss']:.6e} "
            f"one_rms={diagnostics['one_step_physical_rms']:.6e} "
            f"two_mse={diagnostics['two_step_mild_loss']:.6e} "
            f"two_rms={diagnostics['two_step_physical_rms']:.6e} "
            f"total_mild={diagnostics['total_mild_loss']:.6e}"
        )
    payload = {
        "interpolation": interpolation,
        "num_x_fft": args.mild_num_x_fft,
        "delta_t": args.resolved_mild_delta_t,
        "num_intervals": args.resolved_mild_intervals,
        "results": results,
    }
    with (run_dir / "reference_duhamel_metrics.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
    return payload


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
        f"{grid['max_linear_growth_factor']:.6e}; "
        f"two_step_weight={args.mild_two_step_weight:.6g}; derivatives=u_x only."
    )

    def backward_objective(ic_x):
        optimizer.zero_grad(set_to_none=True)
        ic_loss = initial_condition_loss(network, ic_x, args.time_lower)
        (args.ic_loss_weight * ic_loss).backward()
        one_step_value = 0.0
        for _, _, left_times in iter_interval_chunks(args, device, dtype_torch):
            result = mild_duhamel_defects(network, left_times, grid, args, backward=True)
            fraction = len(left_times) / args.resolved_mild_intervals
            (args.mild_loss_weight * fraction * result["loss"]).backward()
            one_step_value += fraction * float(result["loss"].detach().cpu())
        two_step_value = 0.0
        if args.mild_two_step_weight != 0.0:
            two_count = args.resolved_mild_intervals - 1
            two_chunk_size = max(1, args.mild_interval_batch_size // 2)
            for _, _, left_times in iter_interval_chunks(
                args,
                device,
                dtype_torch,
                count=two_count,
                chunk_size=two_chunk_size,
            ):
                result = two_step_duhamel_defects(
                    network, left_times, grid, args, backward=True
                )
                fraction = len(left_times) / two_count
                weighted = (
                    args.mild_loss_weight
                    * args.mild_two_step_weight
                    * fraction
                    * result["loss"]
                )
                weighted.backward()
                two_step_value += fraction * float(result["loss"].detach().cpu())
        mild_value = one_step_value + args.mild_two_step_weight * two_step_value
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
                "one_step_mild_loss": diagnostics["one_step_mild_loss"],
                "two_step_mild_loss": diagnostics["two_step_mild_loss"],
                "total_mild_loss": diagnostics["total_mild_loss"],
                "one_step_physical_rms": diagnostics["one_step_physical_rms"],
                "two_step_physical_rms": diagnostics["two_step_physical_rms"],
                "fourier_mse": diagnostics["fourier_mse"],
                "physical_rms": diagnostics["physical_rms"],
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
                f"Duhamel step={iteration:7d} "
                f"one_mse={diagnostics['one_step_mild_loss']:.6e} "
                f"two_mse={diagnostics['two_step_mild_loss']:.6e} "
                f"one_rms={diagnostics['one_step_physical_rms']:.6e} "
                f"two_rms={diagnostics['two_step_physical_rms']:.6e} "
                f"total_mild={diagnostics['total_mild_loss']:.6e} "
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
        two_step_t_left=args.time_lower
        + np.arange(args.resolved_mild_intervals - 1)
        * args.resolved_mild_delta_t,
        two_step_t_right=args.time_lower
        + (np.arange(args.resolved_mild_intervals - 1) + 2)
        * args.resolved_mild_delta_t,
        two_step_rms=final_diagnostics["two_step_interval_rms"],
        two_step_mean_abs=final_diagnostics["two_step_interval_mean_abs"],
        two_step_max_abs=final_diagnostics["two_step_interval_max_abs"],
    )
    return {
        "mode": "mild_local",
        "iterations": args.mild_epochs,
        "best_iteration": best_iteration,
        "best_total_loss": best_score,
        "last_mild_loss": final_diagnostics["mild_loss"],
        "last_one_step_mild_loss": final_diagnostics["one_step_mild_loss"],
        "last_two_step_mild_loss": final_diagnostics["two_step_mild_loss"],
        "last_total_mild_loss": final_diagnostics["total_mild_loss"],
        "last_one_step_physical_rms": final_diagnostics[
            "one_step_physical_rms"
        ],
        "last_two_step_physical_rms": final_diagnostics[
            "two_step_physical_rms"
        ],
        "last_fourier_mse": final_diagnostics["fourier_mse"],
        "last_physical_rms": final_diagnostics["physical_rms"],
        "last_ic_loss": final_diagnostics["ic_loss"],
        "loss_definition": "L1 + two_step_weight * L2; each Lj = Nx * mean |D_hat_j|^2",
        "fft_normalization": "forward",
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
        "two_step_weight": args.mild_two_step_weight,
        "uses_full_chain_rollout": False,
        "uses_two_step_endpoint_rollout": args.mild_two_step_weight != 0.0,
        "nonlinear_forcing_source": "single global u_theta at quadrature nodes",
        "history_rows": len(rows),
    }


def train_mild_chain(
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
    """Train one global u_theta against a causal rollout starting at exact IC."""

    grid = build_mild_grid(bounds[0], bounds[1], args, device)
    optimizer, scheduler = build_mild_optimizer(network, args)
    rng = np.random.default_rng(args.seed + 313)
    dtype_np = NUMPY_DTYPES[args.precision]
    dtype_torch = grid["x"].dtype
    fixed_ic_x = torch.linspace(
        float(bounds[0][0]),
        float(bounds[1][0]),
        args.pinn_train_ic_points,
        dtype=dtype_torch,
        device=device,
    ).reshape(-1, 1)
    rows = []
    best_score = math.inf
    best_iteration = 0
    best_state = cpu_state_dict(network)
    gradients_verified = False

    print(
        f"Starting causal-chain Duhamel KS for {args.mild_epochs} iterations; "
        f"optimizer={args.pinn_optimizer}; H={args.mild_chain_steps}; "
        f"dt={args.resolved_mild_delta_t:.8g}; Q={args.mild_quadrature_points}; "
        f"Nx={args.mild_num_x_fft}; hard_rollout_start=exact_IC; detach=false."
    )

    def closure_for(ic_x):
        optimizer.zero_grad(set_to_none=True)
        ic_loss = initial_condition_loss(network, ic_x, args.time_lower)
        chain = causal_chain_rollout(network, grid, args, backward=True)
        total = (
            args.ic_loss_weight * ic_loss
            + args.mild_loss_weight * chain["loss"]
        )
        total.backward()
        return total

    for iteration in range(1, args.mild_epochs + 1):
        network.train()
        sampled = rng.uniform(
            bounds[0][0], bounds[1][0], size=(args.pinn_train_ic_points, 1)
        ).astype(dtype_np)
        ic_x = torch.as_tensor(sampled, device=device)
        if args.pinn_optimizer == "lbfgs":
            optimizer.step(lambda: closure_for(ic_x))
        else:
            closure_for(ic_x)
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
                raise RuntimeError("Causal chain loss produced missing/non-finite gradients")
            if not any(bool(torch.any(item != 0)) for item in gradients):
                raise RuntimeError("Causal chain loss produced only zero gradients")
            gradients_verified = True
        if scheduler is not None:
            scheduler.step()

        if iteration % args.mild_log_every == 0 or iteration == args.mild_epochs:
            network.eval()
            diagnostics = fixed_chain_diagnostics(network, fixed_ic_x, grid, args)
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
            step_fields = {
                f"chain_error_step_{index}": float(value)
                for index, value in enumerate(diagnostics["chain_step_rms"], start=1)
            }
            row = {
                "iteration": iteration,
                "chain_loss": diagnostics["chain_loss"],
                "max_chain_error": diagnostics["max_chain_error"],
                "ic_loss": diagnostics["ic_loss"],
                "total_loss": diagnostics["total_loss"],
                **step_fields,
                "delta_t": args.resolved_mild_delta_t,
                "chain_steps": args.mild_chain_steps,
                "quadrature_points": args.mild_quadrature_points,
                "data_relative_l2": data_metric["relative_l2"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            rows.append(row)
            errors_text = " ".join(
                f"E{index}={value:.3e}"
                for index, value in enumerate(diagnostics["chain_step_rms"], start=1)
            )
            print(
                f"Mild chain step={iteration:7d} "
                f"chain={diagnostics['chain_loss']:.6e} "
                f"max={diagnostics['max_chain_error']:.6e} "
                f"ic={diagnostics['ic_loss']:.6e} "
                f"total={diagnostics['total_loss']:.6e} {errors_text} "
                f"L2={data_metric['relative_l2']:.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.6e}"
            )

    last_state = cpu_state_dict(network)
    save_checkpoint(run_dir / "weights_mild_last.pt", network, metadata)
    network.load_state_dict(best_state, strict=True)
    save_checkpoint(run_dir / "weights_mild_best.pt", network, metadata)
    network.load_state_dict(last_state, strict=True)
    save_history(run_dir / "mild_history.csv", rows)
    final_diagnostics = fixed_chain_diagnostics(network, fixed_ic_x, grid, args)
    save_chain_rollout_artifact(network, grid, args, run_dir)
    return {
        "mode": "mild_chain",
        "iterations": args.mild_epochs,
        "best_iteration": best_iteration,
        "best_total_loss": best_score,
        "last_chain_loss": final_diagnostics["chain_loss"],
        "last_max_chain_error": final_diagnostics["max_chain_error"],
        "last_chain_step_rms": final_diagnostics["chain_step_rms"].tolist(),
        "last_ic_loss": final_diagnostics["ic_loss"],
        "loss_definition": "mean_j mean_x |u_theta(x,t_j)-u_roll[j](x)|^2",
        "chain_steps": args.mild_chain_steps,
        "hard_rollout_start": "exact_initial_condition",
        "next_step_initial_state": "previous propagated endpoint",
        "detach_between_steps": False,
        "gradients_verified": gradients_verified,
        "resolved_delta_t": args.resolved_mild_delta_t,
        "quadrature_points": args.mild_quadrature_points,
        "training_max_spatial_derivative": "u_x",
        "nonlinear_forcing_source": "single global u_theta at quadrature nodes",
        "rollout_artifact": str(run_dir / "mild_chain_rollout_final.npz"),
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
    if args.mild_two_step_weight < 0 or not math.isfinite(args.mild_two_step_weight):
        raise ValueError("two-step-weight must be finite and non-negative")
    if args.mild_chain_steps <= 0:
        raise ValueError("mild-chain-steps must be positive")
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
    if args.resolved_mild_intervals < 2:
        raise ValueError("Two-step diagnostics require at least two time intervals")
    if args.mild_mode == "mild_chain" and args.mild_chain_steps > args.resolved_mild_intervals:
        raise ValueError(
            "mild-chain-steps exceeds the number of resolved intervals before time_upper"
        )

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    origin = "pretrained" if source_dir is not None else "scratch"
    mode_suffix = "-mild-chain" if args.mild_mode == "mild_chain" else ""
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-{origin}-ks-duhamel-to-strong{mode_suffix}-"
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
        mild_endpoint_evaluation=(
            "causal rollout from exact IC; every next step starts from the previous "
            "propagated endpoint"
            if args.mild_mode == "mild_chain"
            else "global u_theta at t_i and t_{i+2}; only the intermediate "
            "two-step endpoint is propagated"
        ),
        mild_nonlinear_forcing="global u_theta at every quadrature node",
        mild_loss_definition=(
            "mean_j mean_x |u_theta(x,t_j)-u_roll[j](x)|^2"
            if args.mild_mode == "mild_chain"
            else "L1 + two_step_weight * L2; each Lj = Nx * mean |D_hat_j|^2"
        ),
        mild_fft_normalization="forward",
    )
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(resolved, file_obj, indent=2, sort_keys=True)

    initial_data = prediction_metrics(network, points, values, args.eval_batch_size, device)
    initial_strong = evaluate_pinn_loss(network, (lower, upper), args, device)
    reference_duhamel = None
    if args.reference_duhamel_check:
        reference_duhamel = evaluate_reference_duhamel(
            points, values, (lower, upper), args, device, run_dir
        )
    mild_trainer = train_mild_chain if args.mild_mode == "mild_chain" else train_mild
    mild_info = mild_trainer(
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
        "reference_duhamel": reference_duhamel,
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
    parser.add_argument("--mild-epochs", type=int, default=5000)
    parser.add_argument("--mild-delta-t", type=float, default=0.01)
    parser.add_argument(
        "--mild-mode",
        choices=["mild_local", "mild_chain"],
        default="mild_local",
        help="Use existing independent local defects or a causal rollout from exact IC.",
    )
    parser.add_argument(
        "--mild-chain-steps",
        type=int,
        default=5,
        help="Causal rollout horizon H; used only when --mild-mode mild_chain.",
    )
    parser.add_argument("--mild-quadrature-points", type=int, default=5)
    parser.add_argument("--mild-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--two-step-weight",
        "--mild-two-step-weight",
        dest="mild_two_step_weight",
        type=float,
        default=1.0,
        help="Weight lambda_2 of the two-step endpoint-consistency loss.",
    )
    parser.add_argument(
        "--mild-num-x-fft",
        type=int,
        default=128,
        help="Periodic x grid size; the right endpoint is not duplicated.",
    )
    parser.add_argument(
        "--mild-interval-batch-size",
        type=int,
        default=100,
        help=(
            "One-step interval memory budget. Two-step chunks use half this "
            "many starts because every defect evaluates two forcing intervals."
        ),
    )
    parser.add_argument("--mild-log-every", type=int, default=100)
    parser.add_argument(
        "--reference-duhamel-check",
        type=strong_ks.parse_bool,
        default=True,
        metavar="true|false",
        help="Before training, measure the same Duhamel defect on the reference data for Q=3 and Q=5.",
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
