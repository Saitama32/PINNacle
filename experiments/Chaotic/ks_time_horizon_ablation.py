"""Diagnostic time-horizon ablation for the global KS PINN.

Every (seed, horizon) pair starts from a fresh network.  Diagnostics are
strictly observational: their point clouds and autograd calls never enter the
training objective or its random-number stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline

from experiments.Chaotic.run_data_ks import (
    KS_ALPHA,
    KS_BETA,
    KS_GAMMA,
    NUMPY_DTYPES,
    TORCH_DTYPES,
    _normalization_transform,
    build_optimizer,
    ks_terms,
    load_data,
)
from experiments.Chaotic.run_data_ks_pinn import periodic_boundary_errors
from src.model import RWFMLP
from src.utils.args import parse_hidden_layers


TERM_KEYS = ("term_t", "term_adv", "term_diff", "term_hyper", "residual")
SUMMARY_FIELDS = (
    "T", "seed", "final_relative_l2", "final_mse", "final_mae",
    "final_pde_mse", "final_ic_mse", "final_periodic_mse",
    "jacobian_condition_initial", "jacobian_condition_mid",
    "jacobian_condition_final", "jacobian_rank_initial",
    "jacobian_rank_final", "gradient_conflict_fraction_initial",
    "gradient_conflict_fraction_final", "gradient_cosine_mean_final",
    "first_layer_e_rank_initial", "first_layer_e_rank_final",
    "last_layer_e_rank_initial", "last_layer_e_rank_final",
    "term_error_u_t", "term_error_adv", "term_error_diff",
    "term_error_hyper",
)


def parse_number_list(value: str, cast):
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    if not rows:
        return
    fieldnames = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=json_value)


class ReferenceSolution:
    """Rectangular KS reference with spline time and spectral space derivatives."""

    def __init__(self, path: Path):
        points, values = load_data(path, precision="float64")
        self.x = np.unique(points[:, 0])
        self.t = np.unique(points[:, 1])
        if len(points) != len(self.x) * len(self.t):
            raise ValueError("KS reference must be a rectangular (x, t) grid")
        x_index = np.searchsorted(self.x, points[:, 0])
        t_index = np.searchsorted(self.t, points[:, 1])
        grid = np.empty((len(self.t), len(self.x)), dtype=np.float64)
        grid[t_index, x_index] = values[:, 0]
        duplicate_endpoint = np.isclose(
            self.x[-1] - self.x[0], 2.0 * np.pi
        ) and np.allclose(grid[:, 0], grid[:, -1], rtol=1e-6, atol=1e-8)
        self.spectral_x = self.x[:-1] if duplicate_endpoint else self.x
        self.grid = grid[:, :-1] if duplicate_endpoint else grid
        self.spline = CubicSpline(self.t, self.grid, axis=0)

    def evaluate(self, horizon: float, nt: int, nx: int, alpha, beta, gamma):
        if horizon > self.t[-1] + 1e-12:
            raise ValueError(f"horizon {horizon} exceeds reference t_max={self.t[-1]}")
        times = np.linspace(0.0, horizon, nt, dtype=np.float64)
        native_u = self.spline(times)
        native_ut = self.spline(times, 1)
        native_nx = len(self.spectral_x)
        modes = 2.0 * np.pi * np.fft.fftfreq(
            native_nx, d=(self.spectral_x[-1] - self.spectral_x[0]) / (native_nx - 1)
        )
        coefficients = np.fft.fft(native_u, axis=1)
        native_ux = np.fft.ifft(1j * modes * coefficients, axis=1).real
        native_uxx = np.fft.ifft(-(modes**2) * coefficients, axis=1).real
        native_uxxxx = np.fft.ifft((modes**4) * coefficients, axis=1).real
        target_x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)

        def periodic_resample(field):
            extended_x = np.r_[self.spectral_x, 2.0 * np.pi]
            return np.vstack([
                np.interp(target_x, extended_x, np.r_[row, row[0]]) for row in field
            ])

        u = periodic_resample(native_u)
        u_t = periodic_resample(native_ut)
        u_x = periodic_resample(native_ux)
        u_xx = periodic_resample(native_uxx)
        u_xxxx = periodic_resample(native_uxxxx)
        terms = {
            "u": u,
            "term_t": u_t,
            "term_adv": alpha * u * u_x,
            "term_diff": beta * u_xx,
            "term_hyper": gamma * u_xxxx,
        }
        terms["residual"] = sum(terms[key] for key in TERM_KEYS[:-1])
        return target_x, times, terms


def build_network(args, horizon: float, device: torch.device) -> RWFMLP:
    dde.config.set_default_float(args.precision)
    layers = [2, *parse_hidden_layers(argparse.Namespace(hidden_layers=args.hidden_layers)), 1]
    network = RWFMLP(layers, mu=args.rwf_mu, sigma=args.rwf_sigma)
    network.apply_feature_transform(
        _normalization_transform((0.0, 0.0), (2.0 * np.pi, horizon))
    )
    return network.to(device=device, dtype=TORCH_DTYPES[args.precision])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_training_batch(args, horizon, rng, device):
    dtype = NUMPY_DTYPES[args.precision]
    # Keep the optimization budget identical across horizons.  Only the upper
    # time bound changes; no point count is scaled by T.
    n_domain = args.domain_points
    n_boundary = args.boundary_points
    domain = np.column_stack((
        rng.uniform(0.0, 2.0 * np.pi, n_domain),
        rng.uniform(0.0, horizon, n_domain),
    )).astype(dtype)
    ic_x = rng.uniform(0.0, 2.0 * np.pi, (args.ic_points, 1)).astype(dtype)
    boundary_t = rng.uniform(0.0, horizon, (n_boundary, 1)).astype(dtype)
    return (
        torch.as_tensor(domain, device=device).requires_grad_(True),
        torch.as_tensor(np.c_[ic_x, np.zeros_like(ic_x)], device=device),
        torch.as_tensor(np.cos(ic_x) * (1.0 + np.sin(ic_x)), device=device),
        torch.as_tensor(boundary_t, device=device),
    )


def loss_components(network, batch, args, backward_graph: bool):
    domain, ic_points, ic_exact, boundary_t = batch
    residual = ks_terms(
        network, domain, args.alpha, args.beta, args.gamma,
        create_graph_for_backward=backward_graph,
    )["residual"]
    pde = residual.square().mean()
    ic = (network(ic_points) - ic_exact).square().mean()
    jumps = periodic_boundary_errors(
        network, boundary_t, 0.0, 2.0 * np.pi,
        create_graph_for_backward=backward_graph,
    )
    periodic = sum(value.square().mean() for value in jumps.values())
    total = pde + args.ic_loss_weight * ic + args.periodic_loss_weight * periodic
    return {"total": total, "pde": pde, "ic": ic, "periodic": periodic}


def fixed_diagnostic_points(args, horizon, device):
    """Same x and normalized-time design for every horizon."""
    rng = np.random.default_rng(args.diagnostic_seed)
    dtype = NUMPY_DTYPES[args.precision]
    pde = np.column_stack((
        rng.uniform(0.0, 2.0 * np.pi, args.jacobian_pde_points),
        horizon * rng.uniform(0.0, 1.0, args.jacobian_pde_points),
    )).astype(dtype)
    ic_x = rng.uniform(0.0, 2.0 * np.pi, (args.jacobian_ic_points, 1)).astype(dtype)
    bt = horizon * rng.uniform(0.0, 1.0, (args.jacobian_boundary_points, 1)).astype(dtype)
    feature = np.column_stack((
        rng.uniform(0.0, 2.0 * np.pi, args.feature_points),
        horizon * rng.uniform(0.0, 1.0, args.feature_points),
    )).astype(dtype)
    chunk_points = []
    for index in range(args.gradient_chunks):
        tau = rng.uniform(index / args.gradient_chunks, (index + 1) / args.gradient_chunks,
                          args.gradient_points_per_chunk)
        chunk = np.column_stack((
            rng.uniform(0.0, 2.0 * np.pi, args.gradient_points_per_chunk),
            horizon * tau,
        )).astype(dtype)
        chunk_points.append(torch.as_tensor(chunk, device=device))
    return {
        "pde": torch.as_tensor(pde, device=device),
        "ic": torch.as_tensor(np.c_[ic_x, np.zeros_like(ic_x)], device=device),
        "ic_exact": torch.as_tensor(np.cos(ic_x) * (1.0 + np.sin(ic_x)), device=device),
        "boundary_t": torch.as_tensor(bt, device=device),
        "feature": torch.as_tensor(feature, device=device),
        "chunks": chunk_points,
    }


def flatten_gradients(loss, parameters, retain_graph=False):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def residual_vector(network, points, args):
    pde_points = points["pde"].detach().clone().requires_grad_(True)
    pde = ks_terms(
        network, pde_points, args.alpha, args.beta, args.gamma,
        create_graph_for_backward=True,
    )["residual"].reshape(-1)
    ic = (network(points["ic"]) - points["ic_exact"]).reshape(-1)
    jumps = periodic_boundary_errors(
        network, points["boundary_t"], 0.0, 2.0 * np.pi,
        create_graph_for_backward=True,
    )
    periodic = torch.cat([jumps[key].reshape(-1) for key in ("u", "u_x", "u_xx", "u_xxx")])
    return torch.cat((pde, ic, periodic))


def jacobian_diagnostic(network, points, args, output_dir: Path):
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    vector = residual_vector(network, points, args)
    rows = []
    for index in range(vector.numel()):
        rows.append(flatten_gradients(vector[index], parameters, index + 1 < vector.numel()).detach().cpu())
    jacobian = torch.stack(rows).double().numpy()
    finite = bool(np.isfinite(jacobian).all())
    singular_values = (
        np.linalg.svd(jacobian, compute_uv=False)
        if finite
        else np.full(min(jacobian.shape), np.nan, dtype=np.float64)
    )
    sigma_max = float(singular_values[0]) if len(singular_values) else 0.0
    threshold = args.jacobian_rank_epsilon * sigma_max
    resolved = singular_values[np.isfinite(singular_values) & (singular_values > threshold)]
    result = {
        "residual_count": int(jacobian.shape[0]),
        "parameter_count": int(jacobian.shape[1]),
        "sigma_max": sigma_max,
        "sigma_median": float(np.median(singular_values)) if len(singular_values) else 0.0,
        "sigma_min_resolved": float(resolved[-1]) if len(resolved) else None,
        "jacobian_condition_number": float(sigma_max / resolved[-1]) if len(resolved) else None,
        "jacobian_effective_rank": int(len(resolved)),
        "resolved_threshold": float(threshold),
        "finite": finite,
    }
    np.savez_compressed(output_dir / "jacobian_spectrum.npz", singular_values=singular_values,
                        resolved_threshold=threshold)
    return result


def gradient_diagnostic(network, points, args, output_dir: Path):
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    gradients = []
    for chunk in points["chunks"]:
        chunk = chunk.detach().clone().requires_grad_(True)
        residual = ks_terms(network, chunk, args.alpha, args.beta, args.gamma,
                            create_graph_for_backward=True)["residual"]
        gradients.append(flatten_gradients(residual.square().mean(), parameters).detach())
    matrix = torch.stack(gradients)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    denominator = torch.outer(norms, norms).clamp_min(args.cosine_epsilon)
    cosine = ((matrix @ matrix.T) / denominator).cpu().numpy()
    mask = ~np.eye(len(cosine), dtype=bool)
    off_diagonal = cosine[mask]
    ic_loss = (network(points["ic"]) - points["ic_exact"]).square().mean()
    ic_gradient = flatten_gradients(ic_loss, parameters).detach()
    ic_norm = torch.linalg.vector_norm(ic_gradient)
    ic_cosines = []
    for gradient, norm in zip(gradients, norms):
        value = torch.dot(ic_gradient, gradient) / torch.clamp(ic_norm * norm, min=args.cosine_epsilon)
        ic_cosines.append(float(value.cpu()))
    np.save(output_dir / "gradient_cosine_matrix.npy", cosine)
    write_csv(output_dir / "gradient_cosine_matrix.csv", [
        {"chunk": i, **{f"chunk_{j}": float(value) for j, value in enumerate(row)}}
        for i, row in enumerate(cosine)
    ])
    write_csv(output_dir / "ic_vs_time_gradient_cosine.csv", [
        {"chunk": index, "tau_start": index / args.gradient_chunks,
         "tau_end": (index + 1) / args.gradient_chunks, "cosine": value}
        for index, value in enumerate(ic_cosines)
    ])
    return {
        "gradient_conflict_fraction": float(np.mean(off_diagonal < 0.0)),
        "gradient_cosine_mean": float(np.mean(off_diagonal)),
        "gradient_cosine_min": float(np.min(off_diagonal)),
        "mean_gradient_norm": float(norms.mean().cpu()),
    }


def feature_activations(network, inputs):
    x = network._input_transform(inputs) if network._input_transform is not None else inputs
    first = torch.tanh(network.linears[0](x))
    current = first
    for layer in network.linears[1:-1]:
        current = torch.tanh(layer(current))
    return first, current


def gram_diagnostic(features, epsilon):
    gram = (features.double().T @ features.double()) / features.shape[0]
    finite = bool(torch.isfinite(gram).all())
    eigenvalues = (
        torch.linalg.eigvalsh(gram).clamp_min(0.0).flip(0).cpu().numpy()
        if finite
        else np.full(gram.shape[0], np.nan, dtype=np.float64)
    )
    largest = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    resolved = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > epsilon * largest)]
    return eigenvalues, {
        "e_rank": int(len(resolved)),
        "condition": float(largest / resolved[-1]) if len(resolved) else None,
        "lambda_max": largest,
        "resolved_threshold": epsilon * largest,
        "finite": finite,
    }


def feature_diagnostic(network, points, args, output_dir: Path):
    with torch.no_grad():
        first, last = feature_activations(network, points["feature"])
        first_spectrum, first_result = gram_diagnostic(first, args.feature_rank_epsilon)
        last_spectrum, last_result = gram_diagnostic(last, args.feature_rank_epsilon)
    np.savez_compressed(output_dir / "feature_spectra.npz",
                        first_layer=first_spectrum, last_layer=last_spectrum)
    return {
        "first_layer_e_rank": first_result["e_rank"],
        "first_layer_condition": first_result["condition"],
        "last_layer_e_rank": last_result["e_rank"],
        "last_layer_condition": last_result["condition"],
        "first_layer": first_result,
        "last_layer": last_result,
    }


def save_heatmap(matrix_path: Path, output_path: Path, title: str):
    matrix = np.load(matrix_path)
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xlabel("temporal chunk")
    axis.set_ylabel("temporal chunk")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="gradient cosine")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_checkpoint_diagnostics(network, points, args, run_dir, label, step):
    output_dir = run_dir / f"diagnostics_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    network.eval()
    result = {
        "label": label,
        "iteration": step,
        "jacobian": jacobian_diagnostic(network, points, args, output_dir),
        "gradient": gradient_diagnostic(network, points, args, output_dir),
        "features": feature_diagnostic(network, points, args, output_dir),
    }
    save_heatmap(output_dir / "gradient_cosine_matrix.npy",
                 output_dir / "gradient_cosine_heatmap.png",
                 f"PDE gradient cosine ({label})")
    save_json(output_dir / "diagnostics.json", result)
    network.zero_grad(set_to_none=True)
    return result


def evaluate_prediction(network, x, times, reference, args, device):
    xx, tt = np.meshgrid(x, times, indexing="xy")
    points = np.column_stack((xx.ravel(), tt.ravel())).astype(NUMPY_DTYPES[args.precision])
    predictions = []
    predicted_terms = {key: [] for key in TERM_KEYS}
    for start in range(0, len(points), args.eval_batch_size):
        batch = torch.as_tensor(points[start:start + args.eval_batch_size], device=device).requires_grad_(True)
        terms = ks_terms(network, batch, args.alpha, args.beta, args.gamma)
        predictions.append(terms["u"].detach().cpu().numpy().reshape(-1))
        for key in TERM_KEYS:
            predicted_terms[key].append(terms[key].detach().cpu().numpy().reshape(-1))
    prediction = np.concatenate(predictions).reshape(len(times), len(x)).astype(np.float64)
    fields = {key: np.concatenate(parts).reshape(len(times), len(x)).astype(np.float64)
              for key, parts in predicted_terms.items()}
    exact = reference["u"]
    error = prediction - exact
    squared_error = float(np.sum(error**2))
    metrics = {
        "relative_l2": math.sqrt(squared_error / float(np.sum(exact**2))),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
    }
    error_rows = []
    physics_rows = []
    for index, current_time in enumerate(times):
        denominator = np.linalg.norm(exact[index]) + args.metric_epsilon
        rel = float(np.linalg.norm(error[index]) / denominator)
        error_rows.append({"t": float(current_time), "relative_l2": rel})
        row = {"t": float(current_time), "solution_rel_l2": rel}
        for key, label in (("residual", "residual"), ("term_t", "u_t"),
                           ("term_adv", "adv"), ("term_diff", "diff"),
                           ("term_hyper", "hyper")):
            row[f"pred_{label}_rms"] = float(np.sqrt(np.mean(fields[key][index] ** 2)))
            row[f"ref_{label}_rms"] = float(np.sqrt(np.mean(reference[key][index] ** 2)))
        physics_rows.append(row)
    term_summary = {}
    for key, label in (("term_t", "u_t"), ("term_adv", "adv"),
                       ("term_diff", "diff"), ("term_hyper", "hyper"),
                       ("residual", "residual")):
        pred = fields[key]
        ref = reference[key]
        term_summary[label] = {
            "pred_rms": float(np.sqrt(np.mean(pred**2))),
            "pred_max_abs": float(np.max(np.abs(pred))),
            "pred_mean_abs": float(np.mean(np.abs(pred))),
            "ref_rms": float(np.sqrt(np.mean(ref**2))),
            "ref_max_abs": float(np.max(np.abs(ref))),
            "ref_mean_abs": float(np.mean(np.abs(ref))),
            "relative_error": float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + args.metric_epsilon)),
            "cosine": float(np.sum(pred * ref) / (np.linalg.norm(pred) * np.linalg.norm(ref) + args.metric_epsilon)),
            "correlation": float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1]),
        }
    return prediction, fields, metrics, error_rows, physics_rows, term_summary


def evaluate_losses(network, args, horizon, points, device):
    rng = np.random.default_rng(args.diagnostic_seed + 101)
    dtype = NUMPY_DTYPES[args.precision]
    domain = np.column_stack((
        rng.uniform(0.0, 2.0 * np.pi, args.eval_pde_points),
        rng.uniform(0.0, horizon, args.eval_pde_points),
    )).astype(dtype)
    domain_tensor = torch.as_tensor(domain, device=device).requires_grad_(True)
    residual = ks_terms(network, domain_tensor, args.alpha, args.beta, args.gamma)["residual"]
    with torch.no_grad():
        ic_mse = (network(points["ic"]) - points["ic_exact"]).square().mean()
    jumps = periodic_boundary_errors(network, points["boundary_t"], 0.0, 2.0 * np.pi)
    return {
        "pde_mse": float(residual.detach().double().square().mean().cpu()),
        "ic_mse": float(ic_mse.detach().double().cpu()),
        "periodic_mse": float(sum(value.detach().double().square().mean() for value in jumps.values()).cpu()),
    }


def checkpoint_name(step, iterations):
    if step == 0:
        return "initial"
    if step == iterations:
        return "final"
    return "mid" if step == iterations // 2 else f"step_{step}"


def run_one(args, horizon, seed, reference_solution, root_dir, device):
    run_dir = root_dir / f"seed_{seed}" / f"T_{horizon:.1f}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    network = build_network(args, horizon, device)
    optimizer = build_optimizer(network, args)
    rng = np.random.default_rng(seed + 17)
    diagnostic_points = fixed_diagnostic_points(args, horizon, device)
    checkpoint_steps = sorted({int(round(fraction * args.iterations)) for fraction in args.diagnostic_steps})
    diagnostics = {}
    history = []
    if 0 in checkpoint_steps:
        diagnostics["initial"] = run_checkpoint_diagnostics(
            network, diagnostic_points, args, run_dir, "initial", 0
        )
    started = time.time()
    for iteration in range(1, args.iterations + 1):
        network.train()
        batch = sample_training_batch(args, horizon, rng, device)
        optimizer.zero_grad(set_to_none=True)
        losses = loss_components(network, batch, args, backward_graph=True)
        losses["total"].backward()
        optimizer.step()
        if iteration == 1 or iteration % args.log_every == 0 or iteration == args.iterations:
            row = {"iteration": iteration, "lr": optimizer.param_groups[0]["lr"]}
            row.update({name: float(value.detach().cpu()) for name, value in losses.items()})
            history.append(row)
            print(f"seed={seed} T={horizon:.1f} optimizer={args.optimizer} "
                  f"iter={iteration}/{args.iterations} "
                  f"loss={row['total']:.4e} pde={row['pde']:.4e} ic={row['ic']:.4e}")
        if iteration in checkpoint_steps:
            label = checkpoint_name(iteration, args.iterations)
            diagnostics[label] = run_checkpoint_diagnostics(
                network, diagnostic_points, args, run_dir, label, iteration
            )
    write_csv(run_dir / "training_history.csv", history)
    x, eval_times, reference = reference_solution.evaluate(
        horizon, args.eval_nt, args.eval_nx, args.alpha, args.beta, args.gamma
    )
    prediction, predicted_terms, quality, error_rows, physics_rows, term_summary = evaluate_prediction(
        network, x, eval_times, reference, args, device
    )
    loss_metrics = evaluate_losses(network, args, horizon, diagnostic_points, device)
    write_csv(run_dir / "error_vs_time.csv", error_rows)
    write_csv(run_dir / "physics_vs_time.csv", physics_rows)
    np.savez_compressed(run_dir / "evaluation_fields.npz", x=x, t=eval_times,
                        prediction=prediction, reference=reference["u"],
                        **{f"pred_{key}": value for key, value in predicted_terms.items()},
                        **{f"ref_{key}": value for key, value in reference.items() if key != "u"})
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
                "horizon": horizon, "seed": seed, "args": vars(args)}, run_dir / "weights_final.pt")
    initial = diagnostics["initial"]
    middle = min(
        diagnostics.values(),
        key=lambda item: abs(item["iteration"] / args.iterations - 0.5),
    )
    final = diagnostics["final"]
    summary = {
        "T": horizon, "seed": seed,
        "final_relative_l2": quality["relative_l2"],
        "final_mse": quality["mse"], "final_mae": quality["mae"],
        "final_pde_mse": loss_metrics["pde_mse"],
        "final_ic_mse": loss_metrics["ic_mse"],
        "final_periodic_mse": loss_metrics["periodic_mse"],
        "jacobian_condition_initial": initial["jacobian"]["jacobian_condition_number"],
        "jacobian_condition_mid": middle["jacobian"]["jacobian_condition_number"],
        "jacobian_condition_final": final["jacobian"]["jacobian_condition_number"],
        "jacobian_rank_initial": initial["jacobian"]["jacobian_effective_rank"],
        "jacobian_rank_final": final["jacobian"]["jacobian_effective_rank"],
        "gradient_conflict_fraction_initial": initial["gradient"]["gradient_conflict_fraction"],
        "gradient_conflict_fraction_final": final["gradient"]["gradient_conflict_fraction"],
        "gradient_cosine_mean_final": final["gradient"]["gradient_cosine_mean"],
        "first_layer_e_rank_initial": initial["features"]["first_layer_e_rank"],
        "first_layer_e_rank_final": final["features"]["first_layer_e_rank"],
        "last_layer_e_rank_initial": initial["features"]["last_layer_e_rank"],
        "last_layer_e_rank_final": final["features"]["last_layer_e_rank"],
        "term_error_u_t": term_summary["u_t"]["relative_error"],
        "term_error_adv": term_summary["adv"]["relative_error"],
        "term_error_diff": term_summary["diff"]["relative_error"],
        "term_error_hyper": term_summary["hyper"]["relative_error"],
    }
    save_json(run_dir / "summary.json", {
        **summary, "quality": quality, "losses": loss_metrics,
        "physics_terms": term_summary, "diagnostics": diagnostics,
        "elapsed_seconds": time.time() - started,
    })
    return summary


def aggregate_rows(rows):
    aggregate = []
    numeric_fields = [field for field in SUMMARY_FIELDS if field not in ("T", "seed")]
    for horizon in sorted({row["T"] for row in rows}):
        selected = [row for row in rows if row["T"] == horizon]
        result = {"T": horizon, "n_seeds": len(selected)}
        for field in numeric_fields:
            values = np.asarray([
                np.nan if row[field] is None else row[field] for row in selected
            ], dtype=np.float64)
            result[f"{field}_mean"] = float(np.nanmean(values))
            result[f"{field}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(result)
    return aggregate


def plot_metric(aggregate, output, field, ylabel, log_scale=False, second_field=None):
    horizons = np.asarray([row["T"] for row in aggregate])
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for current_field, label in ((field, ylabel), (second_field, second_field)):
        if current_field is None:
            continue
        mean = np.asarray([row[f"{current_field}_mean"] for row in aggregate])
        std = np.asarray([row[f"{current_field}_std"] for row in aggregate])
        axis.plot(horizons, mean, marker="o", label=label)
        axis.fill_between(horizons, mean - std, mean + std, alpha=0.2)
    axis.set_xlabel("time horizon T")
    axis.set_ylabel(ylabel)
    if log_scale:
        axis.set_yscale("log")
    if second_field:
        axis.legend()
    axis.grid(alpha=0.3)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_physics_term_errors(aggregate, output):
    horizons = np.asarray([row["T"] for row in aggregate])
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for field, label in (("term_error_u_t", r"$u_t$"),
                         ("term_error_adv", r"$\alpha u u_x$"),
                         ("term_error_diff", r"$\beta u_{xx}$"),
                         ("term_error_hyper", r"$\gamma u_{xxxx}$")):
        mean = np.asarray([row[f"{field}_mean"] for row in aggregate])
        std = np.asarray([row[f"{field}_std"] for row in aggregate])
        axis.plot(horizons, mean, marker="o", label=label)
        axis.fill_between(horizons, np.maximum(mean - std, np.finfo(float).tiny),
                          mean + std, alpha=0.15)
    axis.set_xlabel("time horizon T")
    axis.set_ylabel("relative physics-term error")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_aggregate_outputs(rows, root_dir, args):
    write_csv(root_dir / "horizon_ablation.csv", rows, SUMMARY_FIELDS)
    aggregate = aggregate_rows(rows)
    write_csv(root_dir / "horizon_ablation_aggregate.csv", aggregate)
    plot_metric(aggregate, root_dir / "relative_l2_vs_T.png", "final_relative_l2", "relative L2", True)
    plot_metric(aggregate, root_dir / "pde_mse_vs_T.png", "final_pde_mse", "PDE MSE", True)
    plot_metric(aggregate, root_dir / "jacobian_condition_vs_T.png", "jacobian_condition_final", "Jacobian condition", True)
    plot_metric(aggregate, root_dir / "jacobian_effective_rank_vs_T.png", "jacobian_rank_final", "Jacobian effective rank")
    plot_metric(aggregate, root_dir / "gradient_conflict_fraction_vs_T.png", "gradient_conflict_fraction_final", "gradient conflict fraction")
    plot_metric(aggregate, root_dir / "gradient_cosine_mean_vs_T.png", "gradient_cosine_mean_final", "mean gradient cosine")
    plot_metric(aggregate, root_dir / "first_last_layer_e_rank_vs_T.png", "first_layer_e_rank_final", "feature effective rank", second_field="last_layer_e_rank_final")
    for field, label in (("term_error_u_t", "u_t"), ("term_error_adv", "advection"),
                         ("term_error_diff", "diffusion"), ("term_error_hyper", "hyperdiffusion")):
        plot_metric(aggregate, root_dir / f"physics_term_error_{label}_vs_T.png", field, f"relative error: {label}", True)
    plot_physics_term_errors(aggregate, root_dir / "physics_term_errors_vs_T.png")
    critical = next((row for row in aggregate
                     if row["final_relative_l2_mean"] >= args.critical_relative_l2), None)
    report = {
        "critical_relative_l2_threshold": args.critical_relative_l2,
        "empirical_critical_horizon": critical["T"] if critical else None,
        "interpretation": "Concurrent diagnostic changes are associations, not evidence of causality.",
    }
    if critical:
        report["diagnostics_at_critical_horizon"] = {
            key: critical[f"{key}_mean"] for key in (
                "final_relative_l2", "jacobian_condition_final", "jacobian_rank_final",
                "gradient_conflict_fraction_final", "gradient_cosine_mean_final",
                "first_layer_e_rank_final", "last_layer_e_rank_final",
            )
        }
    save_json(root_dir / "diagnostic_report.json", report)


def validate_args(args):
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("iterations and log-every must be positive")
    if args.lr <= 0.0 or not math.isfinite(args.lr):
        raise ValueError("lr must be positive and finite")
    if args.weight_decay < 0.0 or not math.isfinite(args.weight_decay):
        raise ValueError("weight-decay must be finite and non-negative")
    if args.adam_epsilon <= 0.0 or not math.isfinite(args.adam_epsilon):
        raise ValueError("adam-epsilon must be positive and finite")
    if not 0.0 <= args.soap_beta1 < 1.0 or not 0.0 <= args.soap_beta2 < 1.0:
        raise ValueError("SOAP beta values must be in [0, 1)")
    if args.soap_shampoo_beta is not None and not 0.0 <= args.soap_shampoo_beta < 1.0:
        raise ValueError("soap-shampoo-beta must be in [0, 1)")
    if args.soap_epsilon <= 0.0 or args.soap_precondition_frequency < 1:
        raise ValueError("SOAP epsilon and precondition frequency must be positive")
    if args.soap_max_precondition_dim < 1:
        raise ValueError("soap-max-precondition-dim must be positive")
    if not 0.0 <= args.muon_momentum < 1.0 or args.muon_ns_steps < 1:
        raise ValueError("Muon momentum must be in [0, 1) and ns-steps positive")
    if args.muon_adam_lr <= 0.0 or args.muon_adam_epsilon <= 0.0:
        raise ValueError("Muon auxiliary Adam lr and epsilon must be positive")
    if not 0.0 <= args.muon_adam_beta1 < 1.0 or not 0.0 <= args.muon_adam_beta2 < 1.0:
        raise ValueError("Muon auxiliary Adam beta values must be in [0, 1)")
    if args.muon_weight_decay < 0.0 or args.muon_adam_weight_decay < 0.0:
        raise ValueError("Muon weight decays must be non-negative")
    if any(horizon <= 0.0 or horizon > 1.0 for horizon in args.horizons):
        raise ValueError("all horizons must satisfy 0 < T <= 1")
    if len(set(args.horizons)) != len(args.horizons):
        raise ValueError("horizons must be unique")
    if len(args.seeds) != 3 and not args.allow_nonstandard_seed_count:
        raise ValueError("this ablation requires exactly three seeds")
    if args.diagnostic_steps[0] != 0.0 or args.diagnostic_steps[-1] != 1.0:
        raise ValueError("diagnostic-steps must include 0 and 1")
    if any(value < 0.0 or value > 1.0 for value in args.diagnostic_steps):
        raise ValueError("diagnostic step fractions must be in [0, 1]")
    positive = ("domain_points", "ic_points", "boundary_points",
                "jacobian_pde_points", "jacobian_ic_points", "jacobian_boundary_points",
                "gradient_chunks", "gradient_points_per_chunk", "feature_points",
                "eval_nx", "eval_nt", "eval_batch_size", "eval_pde_points")
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError("all point counts and grid sizes must be positive")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs_ks_time_horizon_ablation"))
    parser.add_argument("--horizons", type=lambda value: parse_number_list(value, float),
                        default=[round(index / 10, 1) for index in range(1, 11)])
    parser.add_argument("--seeds", type=lambda value: parse_number_list(value, int),
                        default=[1234, 2345, 3456])
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--precision", choices=sorted(TORCH_DTYPES), default="float32")
    parser.add_argument(
        "--optimizer", choices=("adam", "rmsprop", "soap", "muon"), default="adam",
        help="One optimizer is held fixed across every seed and time horizon.",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-epsilon", type=float, default=1e-8)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)
    parser.add_argument(
        "--soap-bias-correction", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument(
        "--muon-nesterov", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-adam-epsilon", type=float, default=1e-10)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=KS_ALPHA)
    parser.add_argument("--beta", type=float, default=KS_BETA)
    parser.add_argument("--gamma", type=float, default=KS_GAMMA)
    parser.add_argument("--ic-loss-weight", type=float, default=100.0)
    parser.add_argument("--periodic-loss-weight", type=float, default=100.0)
    parser.add_argument(
        "--domain-points", type=int, default=8000,
        help="Fixed PDE collocation count at every horizon (not scaled by T).",
    )
    parser.add_argument(
        "--boundary-points", type=int, default=1000,
        help="Fixed periodic-boundary time samples at every horizon.",
    )
    parser.add_argument("--ic-points", type=int, default=1000)
    parser.add_argument("--diagnostic-steps", type=lambda value: parse_number_list(value, float),
                        default=[0.0, 0.5, 1.0])
    parser.add_argument("--diagnostic-seed", type=int, default=90210)
    parser.add_argument("--jacobian-pde-points", type=int, default=2048)
    parser.add_argument("--jacobian-ic-points", type=int, default=256)
    parser.add_argument("--jacobian-boundary-points", type=int, default=256)
    parser.add_argument("--jacobian-rank-epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient-chunks", type=int, default=8)
    parser.add_argument("--gradient-points-per-chunk", type=int, default=32)
    parser.add_argument("--cosine-epsilon", type=float, default=1e-20)
    parser.add_argument("--feature-points", type=int, default=512)
    parser.add_argument("--feature-rank-epsilon", type=float, default=1e-8)
    parser.add_argument("--eval-nx", type=int, default=128)
    parser.add_argument("--eval-nt", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-pde-points", type=int, default=2048)
    parser.add_argument("--metric-epsilon", type=float, default=1e-12)
    parser.add_argument("--critical-relative-l2", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-nonstandard-seed-count", action="store_true",
                        help="Only for smoke tests; production defaults to exactly three seeds.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    root_dir = Path(args.out).expanduser().resolve() / timestamp
    root_dir.mkdir(parents=True, exist_ok=False)
    save_json(root_dir / "run_config.json", {**vars(args), "device_resolved": str(device)})
    reference = ReferenceSolution(Path(args.data).expanduser().resolve())
    rows = []
    for seed in args.seeds:
        for horizon in args.horizons:
            rows.append(run_one(args, float(horizon), int(seed), reference, root_dir, device))
            save_aggregate_outputs(rows, root_dir, args)
    print(f"Completed {len(rows)} runs. Results: {root_dir}")
    return root_dir


if __name__ == "__main__":
    main()
