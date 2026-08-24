"""Diagnose whether a local parameter step can improve a pretrained KS RWF PINN.

The script solves, around the loaded parameters theta, the linearized problem

    min_delta ||r + J_r delta||^2
              + lambda ||e_u + J_u delta||^2
              + mu ||delta||^2,

and then evaluates scaled versions of delta in the original nonlinear network.
"""

from __future__ import annotations

import argparse
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

from experiments.Chaotic.run_data_ks import (
    KS_ALPHA,
    KS_BETA,
    KS_GAMMA,
    load_checkpoint,
    load_data,
    prediction_metrics,
    save_checkpoint,
    save_solution_plot,
    ks_terms,
)


def comma_separated_floats(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not values or not all(math.isfinite(item) and item >= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be finite and non-negative")
    return values


def resolve_model_path(path: str) -> Path:
    model_path = Path(path).expanduser().resolve()
    if model_path.is_dir():
        model_path = model_path / "weights_best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return model_path


def resolve_data_path(explicit_path: str | None, source_dir: Path) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    config_path = source_dir / "run_config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as file_obj:
            configured = json.load(file_obj).get("data")
        if configured:
            return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"


def trainable_parameters(network) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The loaded network has no trainable parameters")
    return parameters


def flatten_tensors(tensors, parameters) -> torch.Tensor:
    pieces = []
    for tensor, parameter in zip(tensors, parameters):
        pieces.append(
            torch.zeros_like(parameter).reshape(-1)
            if tensor is None
            else tensor.reshape(-1)
        )
    return torch.cat(pieces)


def output_parameter_jacobian(
    outputs: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    label: str,
) -> torch.Tensor:
    """Build a row Jacobian on CPU without constructing a parameter-square matrix."""

    outputs = outputs.reshape(-1)
    rows = []
    for index in range(outputs.numel()):
        gradients = torch.autograd.grad(
            outputs[index],
            parameters,
            retain_graph=index + 1 < outputs.numel(),
            allow_unused=True,
        )
        rows.append(flatten_tensors(gradients, parameters).detach().double().cpu())
        if (index + 1) % 8 == 0 or index + 1 == outputs.numel():
            print(f"Jacobian {label}: row {index + 1}/{outputs.numel()}")
    return torch.stack(rows)


def residual_vector(network, points, alpha, beta, gamma, backward: bool):
    return ks_terms(
        network,
        points,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        create_graph_for_backward=backward,
    )["residual"].reshape(-1)


def residual_mse(network, points_numpy, alpha, beta, gamma, batch_size, device):
    square_sum = 0.0
    count_total = 0
    network.eval()
    for start in range(0, len(points_numpy), batch_size):
        points = torch.as_tensor(
            points_numpy[start : start + batch_size], device=device
        ).requires_grad_(True)
        residual = residual_vector(network, points, alpha, beta, gamma, backward=False)
        square_sum += float(torch.sum(residual.detach().double().square()).cpu())
        count_total += residual.numel()
    return square_sum / count_total


def parameter_state(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def set_parameter_step(parameters, initial_state, delta, scale):
    offset = 0
    with torch.no_grad():
        for parameter, initial in zip(parameters, initial_state):
            count = parameter.numel()
            update = delta[offset : offset + count].to(
                device=parameter.device, dtype=parameter.dtype
            ).reshape_as(parameter)
            parameter.copy_(initial + scale * update)
            offset += count


def cpu_state_dict(network):
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def gradient_cosine(jacobian_a, error_a, jacobian_b, error_b):
    gradient_a = jacobian_a.T @ error_a
    gradient_b = jacobian_b.T @ error_b
    denominator = torch.linalg.vector_norm(gradient_a) * torch.linalg.vector_norm(
        gradient_b
    )
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(gradient_a, gradient_b) / denominator)


def solve_dual_step(j_residual, residual, j_data, data_error, data_weight, damping):
    residual_scale = 1.0 / math.sqrt(residual.numel())
    data_scale = math.sqrt(data_weight / data_error.numel())
    blocks = [residual_scale * j_residual]
    targets = [residual_scale * residual]
    if data_weight > 0:
        blocks.append(data_scale * j_data)
        targets.append(data_scale * data_error)
    design = torch.cat(blocks, dim=0)
    target = torch.cat(targets, dim=0)
    gram = design @ design.T
    mean_diagonal = float(torch.mean(torch.diagonal(gram)))
    effective_damping = damping * max(mean_diagonal, torch.finfo(gram.dtype).eps)
    system = gram + effective_damping * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    try:
        dual = torch.linalg.solve(system, target)
    except RuntimeError:
        dual = torch.linalg.lstsq(system, target.unsqueeze(1)).solution[:, 0]
    delta = -(design.T @ dual)
    diagnostics = {
        "data_weight": data_weight,
        "relative_damping": damping,
        "effective_damping": effective_damping,
        "design_rows": design.shape[0],
        "design_columns": design.shape[1],
        "delta_norm": float(torch.linalg.vector_norm(delta)),
        "objective_uses_mse_blocks": True,
    }
    return delta, diagnostics


def save_pareto_plot(path: Path, rows: list[dict], initial_l2: float, initial_pde: float):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    groups = sorted({(row["damping"], row["data_weight"]) for row in rows})
    for damping, weight in groups:
        selected = [
            row
            for row in rows
            if row["damping"] == damping and row["data_weight"] == weight
        ]
        selected = sorted(selected, key=lambda row: row["step_scale"])
        axis.plot(
            [row["full_data_relative_l2"] for row in selected],
            [row["validation_pde_mse"] for row in selected],
            marker="o",
            label=f"damping={damping:g}, lambda={weight:g}",
        )
    axis.scatter([initial_l2], [initial_pde], color="black", marker="*", s=160, label="initial")
    axis.set_yscale("log")
    axis.set_xlabel("Full-data relative L2")
    axis.set_ylabel("Validation PDE MSE")
    axis.set_title("Nonlinear validation of local Gauss-Newton directions")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args) -> Path:
    if args.jacobian_domain_points <= 0 or args.jacobian_data_points <= 0:
        raise ValueError("Jacobian point counts must be positive")
    if (
        args.validation_domain_points <= 0
        or args.validation_batch_size <= 0
        or args.eval_batch_size <= 0
    ):
        raise ValueError("Validation point counts must be positive")
    if args.max_l2_growth < 0:
        raise ValueError("max_l2_growth must be non-negative")
    dampings = args.dampings if args.dampings is not None else [args.damping]
    if not all(math.isfinite(damping) and damping >= 0 for damping in dampings):
        raise ValueError("dampings must be finite and non-negative")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    model_path = resolve_model_path(args.model)
    source_dir = model_path.parent
    network, metadata = load_checkpoint(model_path, device=device)
    torch_dtype = torch.float64 if args.precision == "float64" else torch.float32
    numpy_dtype = np.float64 if args.precision == "float64" else np.float32
    network = network.to(device=device, dtype=torch_dtype)
    dde.config.set_default_float(args.precision)
    metadata = dict(metadata)
    metadata["precision"] = args.precision
    alpha = float(metadata.get("alpha", KS_ALPHA))
    beta = float(metadata.get("beta", KS_BETA))
    gamma = float(metadata.get("gamma", KS_GAMMA))

    data_path = resolve_data_path(args.data, source_dir)
    all_points, all_values = load_data(data_path, precision=args.precision)
    if args.jacobian_data_points > len(all_points):
        raise ValueError(
            "jacobian_data_points cannot exceed the number of observations"
        )
    lower = np.asarray(metadata["input_min"], dtype=numpy_dtype)
    upper = lower + np.asarray(metadata["input_scale"], dtype=numpy_dtype)
    jacobian_domain_rng = np.random.default_rng(args.seed)
    validation_domain_rng = np.random.default_rng(args.seed + 1)
    jacobian_data_rng = np.random.default_rng(args.seed + 2)
    jacobian_domain_numpy = jacobian_domain_rng.uniform(
        lower, upper, size=(args.jacobian_domain_points, 2)
    ).astype(numpy_dtype)
    validation_domain_numpy = validation_domain_rng.uniform(
        lower, upper, size=(args.validation_domain_points, 2)
    ).astype(numpy_dtype)
    data_indices = jacobian_data_rng.permutation(len(all_points))[
        : args.jacobian_data_points
    ]
    jacobian_data_numpy = all_points[data_indices]
    jacobian_values_numpy = all_values[data_indices]

    parameters = trainable_parameters(network)
    initial_parameter_state = parameter_state(parameters)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    parameter_norm = math.sqrt(
        sum(float(torch.sum(value.detach().double().square()).cpu()) for value in parameters)
    )
    initial_data_metric = prediction_metrics(
        network, all_points, all_values, args.eval_batch_size, device
    )
    initial_validation_pde = residual_mse(
        network,
        validation_domain_numpy,
        alpha,
        beta,
        gamma,
        args.validation_batch_size,
        device,
    )

    network.eval()
    domain_tensor = torch.as_tensor(
        jacobian_domain_numpy, device=device
    ).requires_grad_(True)
    residual = residual_vector(
        network, domain_tensor, alpha, beta, gamma, backward=True
    )
    residual_initial = residual.detach().double().cpu()
    j_residual = output_parameter_jacobian(residual, parameters, "PDE")
    del residual, domain_tensor

    data_tensor = torch.as_tensor(jacobian_data_numpy, device=device)
    data_target = torch.as_tensor(jacobian_values_numpy, device=device)
    data_error = (network(data_tensor) - data_target).reshape(-1)
    data_error_initial = data_error.detach().double().cpu()
    j_data = output_parameter_jacobian(data_error, parameters, "data")
    del data_error, data_tensor, data_target

    cosine = gradient_cosine(
        j_residual,
        residual_initial,
        j_data,
        data_error_initial,
    )
    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-ks-local-basin-{args.precision}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    direction_diagnostics = []
    best_row = None
    best_state = None
    initial_l2_limit = initial_data_metric["relative_l2"] * (1.0 + args.max_l2_growth)
    for damping in dampings:
        for data_weight in args.data_weights:
            delta, direction_metric = solve_dual_step(
                j_residual,
                residual_initial,
                j_data,
                data_error_initial,
                data_weight,
                damping,
            )
            relative_direction_norm = direction_metric["delta_norm"] / max(
                parameter_norm, np.finfo(np.float64).eps
            )
            direction_metric["relative_delta_norm"] = relative_direction_norm
            direction_diagnostics.append(direction_metric)
            print(
                f"Direction damping={damping:g}, lambda={data_weight:g}: "
                f"||delta||/||theta||={relative_direction_norm:.6e}"
            )

            for step_scale in args.step_scales:
                set_parameter_step(parameters, initial_parameter_state, delta, step_scale)
                with torch.no_grad():
                    sample_prediction = network(
                        torch.as_tensor(jacobian_data_numpy, device=device)
                    ).detach().double().cpu().reshape(-1)
                actual_data_error = sample_prediction - torch.as_tensor(
                    jacobian_values_numpy.reshape(-1),
                    dtype=torch.float64,
                    device="cpu",
                )
                actual_domain_pde = residual_mse(
                    network,
                    jacobian_domain_numpy,
                    alpha,
                    beta,
                    gamma,
                    args.validation_batch_size,
                    device,
                )
                validation_pde = residual_mse(
                    network,
                    validation_domain_numpy,
                    alpha,
                    beta,
                    gamma,
                    args.validation_batch_size,
                    device,
                )
                full_data_metric = prediction_metrics(
                    network, all_points, all_values, args.eval_batch_size, device
                )
                scaled_delta = step_scale * delta
                linear_residual = residual_initial + j_residual @ scaled_delta
                linear_data_error = data_error_initial + j_data @ scaled_delta
                row = {
                    "damping": damping,
                    "data_weight": data_weight,
                    "step_scale": step_scale,
                    "relative_parameter_step": step_scale * relative_direction_norm,
                    "linearized_pde_mse": float(torch.mean(linear_residual.square())),
                    "actual_jacobian_domain_pde_mse": actual_domain_pde,
                    "validation_pde_mse": validation_pde,
                    "linearized_data_mse": float(torch.mean(linear_data_error.square())),
                    "actual_jacobian_data_mse": float(torch.mean(actual_data_error.square())),
                    "full_data_mse": full_data_metric["mse"],
                    "full_data_relative_l2": full_data_metric["relative_l2"],
                    "l2_feasible": full_data_metric["relative_l2"] <= initial_l2_limit,
                }
                rows.append(row)
                print(
                    f"  scale={step_scale:g} linear_pde={row['linearized_pde_mse']:.6e} "
                    f"actual_pde={validation_pde:.6e} "
                    f"L2={full_data_metric['relative_l2']:.6e}"
                )
                if row["l2_feasible"] and (
                    best_row is None
                    or row["validation_pde_mse"] < best_row["validation_pde_mse"]
                ):
                    best_row = dict(row)
                    best_state = cpu_state_dict(network)

    set_parameter_step(
        parameters,
        initial_parameter_state,
        torch.zeros(parameter_count, dtype=torch.float64, device="cpu"),
        0.0,
    )
    save_checkpoint(run_dir / "weights_initial.pt", network, metadata)
    if best_state is not None:
        network.load_state_dict(best_state, strict=True)
        best_metadata = dict(metadata)
        best_metadata["local_basin"] = best_row
        save_checkpoint(run_dir / "weights_local_best.pt", network, best_metadata)
        predictions = []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(all_points), args.eval_batch_size):
                predictions.append(
                    network(
                        torch.as_tensor(
                            all_points[start : start + args.eval_batch_size],
                            device=device,
                        )
                    ).cpu().numpy()
                )
        prediction = np.vstack(predictions)[:, 0]
        save_solution_plot(
            run_dir / "solution_local_best.png",
            all_points,
            all_values[:, 0],
            prediction,
            f"Local basin best, relative L2={best_row['full_data_relative_l2']:.3e}",
        )

    fieldnames = list(rows[0])
    with (run_dir / "local_steps.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_pareto_plot(
        run_dir / "local_basin_pareto.png",
        rows,
        initial_data_metric["relative_l2"],
        initial_validation_pde,
    )
    result = {
        "configuration": {
            **vars(args),
            "resolved_dampings": dampings,
            "model": str(model_path),
            "data": str(data_path),
            "device": str(device),
            "parameter_count": parameter_count,
            "parameter_norm": parameter_norm,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "jacobian_domain_seed": args.seed,
            "validation_domain_seed": args.seed + 1,
            "jacobian_data_seed": args.seed + 2,
        },
        "initial": {
            "data": initial_data_metric,
            "validation_pde_mse": initial_validation_pde,
            "jacobian_domain_pde_mse": float(torch.mean(residual_initial.square())),
            "jacobian_data_mse": float(torch.mean(data_error_initial.square())),
        },
        "pde_data_gradient_cosine": cosine,
        "directions": direction_diagnostics,
        "l2_feasibility_limit": initial_l2_limit,
        "best_feasible": best_row,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, indent=2, sort_keys=True)
    print(
        f"Finished local-basin diagnostic; best={best_row}; artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=r"C:\Users\Рустам\Documents\GitHub\PINNacle\runs_data_ks_local_basin\08.25-00.32.07-08.24-23.53.08-08.24-07.14.29-08.24-01.58.59-ks-data-rwf-soap-float32-lr-cosine-spectral-k40-150-layerall-float64-local-basin-float64-local-basin-float64\weights_local_best.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs_data_ks_local_basin"))
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--jacobian-domain-points", type=int, default=2048)
    parser.add_argument("--jacobian-data-points", type=int, default=2048)
    parser.add_argument("--validation-domain-points", type=int, default=10000)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument(
        "--data-weights",
        type=comma_separated_floats,
        default=comma_separated_floats("3000,30000,300000,3000000"),
        metavar="LAMBDA,...",
    )
    parser.add_argument(
        "--step-scales",
        type=comma_separated_floats,
        default=comma_separated_floats("0.2,0.25,0.3,0.35,0.4"),
        metavar="SCALE,...",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=1e-5,
        help=(
            "Single damping value relative to the mean diagonal of the dual Gram "
            "matrix. Ignored when --dampings is provided."
        ),
    )
    parser.add_argument(
        "--dampings",
        type=comma_separated_floats,
        default="1e-6,1e-5,1e-4",
        metavar="DAMPING,...",
        help="Comma-separated damping sweep values.",
    )
    parser.add_argument(
        "--max-l2-growth",
        type=float,
        default=0.0,
        help="Maximum relative growth of full-data L2RE for a feasible step.",
    )
    parser.add_argument("--seed", type=int, default=1234567)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
