"""Spectral refinement of one pretrained KS RWF MLP.

The pretrained network is kept frozen.  A zero-initialized additive correction
``delta_weight`` (and ``delta_bias``) is attached to selected RWF layers and
is trained on complete spatial slices of the reference data.  The correction is
then merged into the original RWF parameters, so every saved model is again a
plain, single ``RWFMLP`` checkpoint that can be loaded by ``run_data_ks.py``.
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as functional
from deepxde.optimizers.pytorch.muon import MuonWithAuxAdam
from deepxde.optimizers.pytorch.soap import SOAP

from experiments.Chaotic.run_data_ks import (
    KS_ALPHA,
    KS_BETA,
    KS_GAMMA,
    evaluate_derivative_grid,
    evaluate_pinn_loss,
    load_checkpoint,
    load_data,
    save_checkpoint,
    save_solution_plot,
)
from src.model.rwf import RWFLinear


TORCH_DTYPES = {"float32": torch.float32, "float64": torch.float64}
NUMPY_DTYPES = {"float32": np.float32, "float64": np.float64}


class AdditiveRWFLinear(torch.nn.Module):
    """A frozen RWF layer plus a trainable effective-weight correction."""

    def __init__(self, base: RWFLinear):
        super().__init__()
        if not isinstance(base, RWFLinear):
            raise TypeError("Additive refinement requires an RWFLinear layer")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.delta_weight = torch.nn.Parameter(torch.zeros_like(base.weight))
        self.delta_bias = (
            torch.nn.Parameter(torch.zeros_like(base.bias))
            if base.bias is not None
            else None
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight + self.delta_weight

    @property
    def bias(self) -> torch.Tensor | None:
        if self.base.bias is None:
            return None
        return self.base.bias + self.delta_bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional.linear(inputs, self.weight, self.bias)

    def merged_layer(self) -> RWFLinear:
        """Return a regular RWF layer containing the effective correction."""

        merged = copy.deepcopy(self.base)
        with torch.no_grad():
            effective_weight = self.weight.detach()
            merged.V.copy_(effective_weight / torch.exp(merged.s).unsqueeze(1))
            if merged.bias is not None:
                merged.bias.copy_(self.bias.detach())
        for parameter in merged.parameters():
            parameter.requires_grad_(True)
        return merged


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
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file_obj:
            configured = json.load(file_obj).get("data")
        if configured:
            return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"


def rectangular_grid(points: np.ndarray, values: np.ndarray):
    """Restore a complete ``[nt, nx]`` reference grid from flat data."""

    x = np.unique(points[:, 0])
    t = np.unique(points[:, 1])
    if len(points) != len(x) * len(t):
        raise ValueError("Spectral refinement requires a complete rectangular x,t grid")
    ix = np.searchsorted(x, points[:, 0])
    it = np.searchsorted(t, points[:, 1])
    exact = np.full((len(t), len(x)), np.nan, dtype=values.dtype)
    exact[it, ix] = values[:, 0]
    if not np.isfinite(exact).all():
        raise ValueError("Reference x,t grid contains missing or non-finite values")

    duplicate_periodic_endpoint = bool(
        len(x) > 2 and np.allclose(exact[:, 0], exact[:, -1], rtol=1e-5, atol=1e-6)
    )
    spectral_nx = len(x) - 1 if duplicate_periodic_endpoint else len(x)
    return x, t, exact, spectral_nx, duplicate_periodic_endpoint


def resolve_refined_layers(network, specification: str) -> list[int]:
    """Resolve ``all`` or one one-based layer number."""

    if str(specification).lower() == "all":
        return list(range(1, len(network.linears) + 1))
    try:
        layer = int(specification)
    except (TypeError, ValueError) as error:
        raise ValueError("refine_layer must be 'all' or a one-based layer number") from error
    if not 1 <= layer <= len(network.linears):
        raise ValueError(f"refine_layer must be in [1, {len(network.linears)}], got {layer}")
    return [layer]


def attach_additive_layers(
    network, one_based_layers: list[int]
) -> dict[int, AdditiveRWFLinear]:
    """Freeze the model and attach zero corrections to selected layers."""

    for parameter in network.parameters():
        parameter.requires_grad_(False)
    additives = {}
    for layer in one_based_layers:
        index = layer - 1
        target = network.linears[index]
        if not isinstance(target, RWFLinear):
            raise TypeError(
                f"Layer {layer} is not a regular RWFLinear layer; "
                "Gaussian feature layers are not supported"
            )
        additive = AdditiveRWFLinear(target)
        network.linears[index] = additive
        additives[layer] = additive
    return additives


def materialize_merged_network(network, one_based_layers: list[int]):
    """Copy the model and merge every selected additive wrapper into RWFLinear."""

    merged_network = copy.deepcopy(network)
    for layer in one_based_layers:
        index = layer - 1
        additive = merged_network.linears[index]
        if not isinstance(additive, AdditiveRWFLinear):
            raise TypeError(f"Selected layer {layer} does not contain a correction")
        merged_network.linears[index] = additive.merged_layer()
    return merged_network


def frequency_weights(
    nx: int,
    min_mode: int,
    max_mode: int,
    power: float,
    boost: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return one-sided FFT weights restricted to a trusted mode band."""

    modes = torch.arange(nx // 2 + 1, device=device, dtype=dtype)
    normalized = modes / max(1, max_mode)
    weights = 1.0 + boost * normalized.pow(power)
    band_mask = (modes >= min_mode) & (modes <= max_mode)
    weights = weights * band_mask
    # Restore the energy of omitted negative frequencies for a real FFT.
    multiplicity = torch.full_like(weights, 2.0)
    multiplicity[0] = 1.0
    if nx % 2 == 0:
        multiplicity[-1] = 1.0
    return weights * multiplicity


def torch_spectral_loss(
    error: torch.Tensor,
    spectral_nx: int,
    min_mode: int,
    max_mode: int,
    power: float,
    boost: float,
) -> torch.Tensor:
    """Weighted spatial spectral error averaged over complete time slices."""

    spectral_error = error[:, :spectral_nx]
    coefficients = torch.fft.rfft(spectral_error, dim=1, norm="ortho")
    weights = frequency_weights(
        spectral_nx,
        min_mode,
        max_mode,
        power,
        boost,
        coefficients.device,
        coefficients.real.dtype,
    )
    energy = coefficients.real.square() + coefficients.imag.square()
    return torch.sum(energy * weights.unsqueeze(0)) / (
        error.shape[0] * torch.sum(weights)
    )


def predict_grid(network, x, t, batch_size: int, device, numpy_dtype):
    xx, tt = np.meshgrid(x, t, indexing="xy")
    points = np.column_stack((xx.reshape(-1), tt.reshape(-1))).astype(numpy_dtype)
    predictions = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.as_tensor(points[start : start + batch_size], device=device)
            predictions.append(network(batch).detach().cpu().numpy())
    return np.vstack(predictions)[:, 0].reshape(len(t), len(x))


def numpy_spectral_metrics(
    error: np.ndarray,
    spectral_nx: int,
    min_mode: int,
    max_mode: int,
    power: float,
    boost: float,
) -> dict:
    """Return the weighted spectral loss and interpretable band energies."""

    spectral_error = np.asarray(error[:, :spectral_nx], dtype=np.float64)
    coefficients = np.fft.rfft(spectral_error, axis=1, norm="ortho")
    energy = np.mean(np.abs(coefficients) ** 2, axis=0)
    multiplicity = np.full(len(energy), 2.0)
    multiplicity[0] = 1.0
    if spectral_nx % 2 == 0:
        multiplicity[-1] = 1.0
    energy *= multiplicity
    modes = np.arange(len(energy), dtype=np.float64)
    normalized = modes / max(1, max_mode)
    band_mask = (modes >= min_mode) & (modes <= max_mode)
    weights = (1.0 + boost * normalized**power) * multiplicity * band_mask
    weighted_loss = float(np.sum(np.mean(np.abs(coefficients) ** 2, axis=0) * weights) / np.sum(weights))
    total_energy = float(np.sum(energy))

    edges = (0, 11, 31, 61, 101, 151, len(energy))
    bands = {}
    for left, right in zip(edges[:-1], edges[1:]):
        if left >= len(energy):
            continue
        stop = min(right, len(energy))
        band_energy = float(np.sum(energy[left:stop]))
        bands[f"modes_{left}_{stop - 1}"] = {
            "energy": band_energy,
            "fraction": band_energy / total_energy if total_energy > 0 else 0.0,
        }
    return {
        "weighted_loss": weighted_loss,
        "weighted_mode_range": [int(min_mode), int(max_mode)],
        "total_error_energy": total_energy,
        "bands": bands,
    }


def flat_prediction_metrics(prediction: np.ndarray, exact: np.ndarray) -> dict:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    mse = float(np.mean(error**2))
    denominator = float(np.sum(np.asarray(exact, dtype=np.float64) ** 2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": math.sqrt(float(np.sum(error**2)) / denominator)
        if denominator > 0
        else None,
    }


def build_optimizer(name: str, parameters, args):
    parameters = list(parameters)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if name == "soap":
        return SOAP(
            parameters,
            lr=args.lr,
            betas=(args.soap_beta1, args.soap_beta2),
            shampoo_beta=(
                args.soap_beta2
                if args.soap_shampoo_beta is None
                else args.soap_shampoo_beta
            ),
            eps=args.soap_epsilon,
            weight_decay=args.weight_decay,
            precondition_frequency=args.soap_precondition_frequency,
            max_precondition_dim=args.soap_max_precondition_dim,
            bias_correction=args.soap_bias_correction,
        )
    if name == "muon":
        matrix_parameters = [parameter for parameter in parameters if parameter.ndim >= 2]
        auxiliary_parameters = [parameter for parameter in parameters if parameter.ndim < 2]
        parameter_groups = []
        if matrix_parameters:
            parameter_groups.append(
                {
                    "params": matrix_parameters,
                    "use_muon": True,
                    "lr": args.lr,
                    "momentum": args.muon_momentum,
                    "nesterov": args.muon_nesterov,
                    "ns_steps": args.muon_ns_steps,
                    "weight_decay": args.muon_weight_decay,
                }
            )
        if auxiliary_parameters:
            parameter_groups.append(
                {
                    "params": auxiliary_parameters,
                    "use_muon": False,
                    "lr": args.muon_adam_lr,
                    "betas": (args.muon_adam_beta1, args.muon_adam_beta2),
                    "eps": args.muon_adam_epsilon,
                    "weight_decay": args.muon_adam_weight_decay,
                }
            )
        return MuonWithAuxAdam(parameter_groups)
    raise ValueError(f"Unsupported optimizer: {name}")


def capture_corrections(additives: dict[int, AdditiveRWFLinear]) -> dict[str, torch.Tensor]:
    state = {}
    for layer, additive in additives.items():
        state[f"layer_{layer}_delta_weight"] = additive.delta_weight.detach().cpu().clone()
        if additive.delta_bias is not None:
            state[f"layer_{layer}_delta_bias"] = additive.delta_bias.detach().cpu().clone()
    return state


def restore_corrections(
    additives: dict[int, AdditiveRWFLinear],
    state: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    with torch.no_grad():
        for layer, additive in additives.items():
            additive.delta_weight.copy_(
                state[f"layer_{layer}_delta_weight"].to(device=device, dtype=dtype)
            )
            if additive.delta_bias is not None:
                additive.delta_bias.copy_(
                    state[f"layer_{layer}_delta_bias"].to(device=device, dtype=dtype)
                )


def save_corrections(path: Path, additives: dict[int, AdditiveRWFLinear]) -> None:
    """Save every effective-layer correction separately for diagnostics."""

    arrays = {}
    for layer, additive in additives.items():
        prefix = f"layer_{layer}_"
        arrays.update(
            {
                prefix + "base_weight": additive.base.weight.detach().cpu().numpy(),
                prefix + "delta_weight": additive.delta_weight.detach().cpu().numpy(),
                prefix + "effective_weight": additive.weight.detach().cpu().numpy(),
            }
        )
        if additive.base.bias is not None:
            arrays.update(
                {
                    prefix + "base_bias": additive.base.bias.detach().cpu().numpy(),
                    prefix + "delta_bias": additive.delta_bias.detach().cpu().numpy(),
                    prefix + "effective_bias": additive.bias.detach().cpu().numpy(),
                }
            )
    np.savez_compressed(path, **arrays)


def run(args) -> Path:
    if args.iterations <= 0 or args.time_batch_size <= 0:
        raise ValueError("iterations and time_batch_size must be positive")
    if args.log_every <= 0 or args.eval_batch_size <= 0:
        raise ValueError("log_every and eval_batch_size must be positive")
    if args.pinn_points <= 0 or args.pinn_ic_points <= 0 or args.pinn_batch_size <= 0:
        raise ValueError("PINN diagnostic sample and batch sizes must be positive")
    if args.lr <= 0 or args.spectral_weight < 0 or args.data_weight < 0:
        raise ValueError("lr must be positive and loss weights must be non-negative")
    if args.data_weight == 0 and args.spectral_weight == 0:
        raise ValueError("At least one of data_weight and spectral_weight must be positive")
    if args.weight_decay < 0 or args.delta_l2_weight < 0:
        raise ValueError("weight_decay and delta_l2_weight must be non-negative")
    if args.spectral_power <= 0 or args.spectral_boost < 0:
        raise ValueError("spectral_power must be positive and spectral_boost non-negative")
    if args.spectral_min_mode < 0 or args.spectral_max_mode < args.spectral_min_mode:
        raise ValueError(
            "spectral mode range must satisfy 0 <= spectral_min_mode <= spectral_max_mode"
        )
    if args.lr_scheduler == "cosine" and not 0 <= args.lr_min <= args.lr:
        raise ValueError("For cosine decay, lr_min must be between zero and lr")
    if args.lr_scheduler == "exponential" and args.lr_decay_rate <= 0:
        raise ValueError("lr_decay_rate must be positive")
    if not 0 <= args.soap_beta1 < 1 or not 0 <= args.soap_beta2 < 1:
        raise ValueError("SOAP beta values must be in [0, 1)")
    if args.soap_shampoo_beta is not None and not 0 <= args.soap_shampoo_beta < 1:
        raise ValueError("soap_shampoo_beta must be in [0, 1)")
    if args.soap_epsilon <= 0 or not math.isfinite(args.soap_epsilon):
        raise ValueError("soap_epsilon must be positive and finite")
    if args.soap_precondition_frequency <= 0 or args.soap_max_precondition_dim <= 0:
        raise ValueError("SOAP precondition settings must be positive")
    if not 0 <= args.muon_momentum < 1 or args.muon_ns_steps <= 0:
        raise ValueError("Muon momentum must be in [0, 1) and ns_steps must be positive")
    if args.muon_adam_lr <= 0 or args.muon_adam_epsilon <= 0:
        raise ValueError("Muon auxiliary Adam lr and epsilon must be positive")
    if not 0 <= args.muon_adam_beta1 < 1 or not 0 <= args.muon_adam_beta2 < 1:
        raise ValueError("Muon auxiliary Adam beta values must be in [0, 1)")
    if args.muon_weight_decay < 0 or args.muon_adam_weight_decay < 0:
        raise ValueError("Muon weight decay values must be non-negative")
    if (
        args.optimizer == "muon"
        and args.lr_scheduler == "cosine"
        and args.lr_min > args.muon_adam_lr
    ):
        raise ValueError("lr_min cannot exceed the Muon auxiliary Adam lr")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = resolve_model_path(args.model)
    source_dir = model_path.parent
    network, source_metadata = load_checkpoint(model_path, device=device)
    source_precision = source_metadata.get("precision", "float32")
    dtype = TORCH_DTYPES[args.precision]
    numpy_dtype = NUMPY_DTYPES[args.precision]
    network = network.to(device=device, dtype=dtype)
    refined_layers = resolve_refined_layers(network, args.refine_layer)
    layer_label = "all" if len(refined_layers) == len(network.linears) else str(refined_layers[0])
    dde.config.set_default_float(args.precision)
    metadata = dict(source_metadata)
    metadata["precision"] = args.precision
    args.alpha = float(metadata.get("alpha", KS_ALPHA))
    args.beta = float(metadata.get("beta", KS_BETA))
    args.gamma = float(metadata.get("gamma", KS_GAMMA))
    # Required by the shared PINN diagnostic helper.
    args.ic_loss_weight = float(args.ic_loss_weight)

    data_path = resolve_data_path(args.data, source_dir)
    points, values = load_data(data_path, precision=args.precision)
    x, t, exact_grid, spectral_nx, duplicate_endpoint = rectangular_grid(points, values)
    highest_available_mode = spectral_nx // 2
    if args.spectral_max_mode > highest_available_mode:
        raise ValueError(
            f"spectral_max_mode={args.spectral_max_mode} exceeds the available "
            f"Nyquist mode {highest_available_mode}"
        )
    lower = np.asarray(metadata["input_min"], dtype=numpy_dtype)
    upper = lower + np.asarray(metadata["input_scale"], dtype=numpy_dtype)

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-{source_dir.name}-spectral-k{args.spectral_min_mode}-"
        f"{args.spectral_max_mode}-layer{layer_label}-{args.precision}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    save_checkpoint(run_dir / "weights_initial.pt", network, metadata)

    initial_grid = predict_grid(
        network, x, t, args.eval_batch_size, device, numpy_dtype
    )
    initial_data = flat_prediction_metrics(initial_grid, exact_grid)
    initial_spectral = numpy_spectral_metrics(
        initial_grid - exact_grid,
        spectral_nx,
        args.spectral_min_mode,
        args.spectral_max_mode,
        args.spectral_power,
        args.spectral_boost,
    )

    additives = attach_additive_layers(network, refined_layers)
    trainable_parameters = [
        parameter
        for additive in additives.values()
        for parameter in additive.parameters()
        if parameter.requires_grad
    ]
    optimizer = build_optimizer(args.optimizer, trainable_parameters, args)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay_rate
        )

    resolved = vars(args).copy()
    resolved.update(
        model=str(model_path),
        source_run=str(source_dir),
        source_model_precision=source_precision,
        data=str(data_path),
        device=str(device),
        nx=int(len(x)),
        nt=int(len(t)),
        spectral_nx=int(spectral_nx),
        duplicate_periodic_endpoint=duplicate_endpoint,
        refined_layers=refined_layers,
        trainable_parameters=int(sum(p.numel() for p in trainable_parameters)),
        model_metadata=metadata,
    )
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(resolved, file_obj, indent=2, sort_keys=True)

    rng = np.random.default_rng(args.seed + 1)
    initial_delta_state = capture_corrections(additives)
    best_delta_state = initial_delta_state
    best_iteration = 0
    best_score = (
        args.data_weight * initial_data["mse"]
        + args.spectral_weight * initial_spectral["weighted_loss"]
    )
    history = []
    print(
        f"Spectral refinement: optimizer={args.optimizer}; layers={layer_label}; trainable="
        f"{sum(p.numel() for p in trainable_parameters)}; precision={args.precision}; "
        f"modes={args.spectral_min_mode}-{args.spectral_max_mode}; "
        f"initial relative L2={initial_data['relative_l2']:.6e}; "
        f"artifacts={run_dir}"
    )

    for iteration in range(1, args.iterations + 1):
        network.train()
        count = min(args.time_batch_size, len(t))
        time_indices = rng.choice(len(t), size=count, replace=False)
        selected_t = t[time_indices]
        xx, tt = np.meshgrid(x, selected_t, indexing="xy")
        batch_points = np.column_stack((xx.reshape(-1), tt.reshape(-1))).astype(
            numpy_dtype
        )
        batch_exact = exact_grid[time_indices]
        input_tensor = torch.as_tensor(batch_points, device=device)
        exact_tensor = torch.as_tensor(batch_exact, device=device)
        prediction = network(input_tensor).reshape(count, len(x))
        error = prediction - exact_tensor
        data_loss = torch.mean(error.square())
        spectral_loss = torch_spectral_loss(
            error,
            spectral_nx,
            args.spectral_min_mode,
            args.spectral_max_mode,
            args.spectral_power,
            args.spectral_boost,
        )
        delta_square_sum = torch.zeros((), device=device, dtype=dtype)
        delta_parameter_count = 0
        for additive in additives.values():
            delta_square_sum = delta_square_sum + torch.sum(additive.delta_weight.square())
            delta_parameter_count += additive.delta_weight.numel()
            if additive.delta_bias is not None:
                delta_square_sum = delta_square_sum + torch.sum(additive.delta_bias.square())
                delta_parameter_count += additive.delta_bias.numel()
        delta_l2 = delta_square_sum / delta_parameter_count
        total_loss = (
            args.data_weight * data_loss
            + args.spectral_weight * spectral_loss
            + args.delta_l2_weight * delta_l2
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        should_log = (
            iteration == 1
            or iteration % args.log_every == 0
            or iteration == args.iterations
        )
        if should_log:
            full_prediction = predict_grid(
                network, x, t, args.eval_batch_size, device, numpy_dtype
            )
            full_data = flat_prediction_metrics(full_prediction, exact_grid)
            full_spectral = numpy_spectral_metrics(
                full_prediction - exact_grid,
                spectral_nx,
                args.spectral_min_mode,
                args.spectral_max_mode,
                args.spectral_power,
                args.spectral_boost,
            )
            logged_delta_square_sum = torch.zeros((), device=device, dtype=dtype)
            logged_delta_parameter_count = 0
            for additive in additives.values():
                logged_delta_square_sum = logged_delta_square_sum + torch.sum(
                    additive.delta_weight.detach().square()
                )
                logged_delta_parameter_count += additive.delta_weight.numel()
                if additive.delta_bias is not None:
                    logged_delta_square_sum = logged_delta_square_sum + torch.sum(
                        additive.delta_bias.detach().square()
                    )
                    logged_delta_parameter_count += additive.delta_bias.numel()
            logged_delta_l2 = logged_delta_square_sum / logged_delta_parameter_count
            delta_norm = math.sqrt(float(logged_delta_square_sum.double().cpu()))
            score = (
                args.data_weight * full_data["mse"]
                + args.spectral_weight * full_spectral["weighted_loss"]
                + args.delta_l2_weight * float(logged_delta_l2.cpu())
            )
            if math.isfinite(score) and score < best_score:
                best_score = score
                best_iteration = iteration
                best_delta_state = capture_corrections(additives)
            history.append(
                [
                    iteration,
                    float(data_loss.detach().cpu()),
                    float(spectral_loss.detach().cpu()),
                    float(total_loss.detach().cpu()),
                    full_data["mse"],
                    full_data["relative_l2"],
                    full_spectral["weighted_loss"],
                    delta_norm,
                    optimizer.param_groups[0]["lr"],
                ]
            )
            print(
                f"step={iteration:7d} batch_data={history[-1][1]:.6e} "
                f"batch_spectral={history[-1][2]:.6e} "
                f"full_l2={full_data['relative_l2']:.6e} "
                f"full_spectral={full_spectral['weighted_loss']:.6e} "
                f"delta_norm={delta_norm:.6e} lr={history[-1][8]:.6e}"
            )

    last_network = materialize_merged_network(network, refined_layers)
    save_corrections(run_dir / "layer_correction_last.npz", additives)
    save_checkpoint(run_dir / "weights_refined_last.pt", last_network, metadata)
    save_checkpoint(run_dir / "weights_last.pt", last_network, metadata)

    restore_corrections(additives, best_delta_state, device, dtype)
    save_corrections(run_dir / "layer_correction_best.npz", additives)
    best_network = materialize_merged_network(network, refined_layers).to(device)
    best_network.eval()
    save_checkpoint(run_dir / "weights_refined_best.pt", best_network, metadata)
    save_checkpoint(run_dir / "weights_best.pt", best_network, metadata)

    final_grid = predict_grid(
        best_network, x, t, args.eval_batch_size, device, numpy_dtype
    )
    final_data = flat_prediction_metrics(final_grid, exact_grid)
    final_spectral = numpy_spectral_metrics(
        final_grid - exact_grid,
        spectral_nx,
        args.spectral_min_mode,
        args.spectral_max_mode,
        args.spectral_power,
        args.spectral_boost,
    )
    final_pinn = evaluate_pinn_loss(best_network, (lower, upper), args, device)
    derivative_metric = None
    if args.derivative_plots:
        derivative_metric = evaluate_derivative_grid(
            best_network,
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

    np.savez_compressed(
        run_dir / "predictions.npz",
        x=np.broadcast_to(x[None, :], exact_grid.shape).reshape(-1),
        t=np.broadcast_to(t[:, None], exact_grid.shape).reshape(-1),
        exact=exact_grid.reshape(-1),
        prediction=final_grid.reshape(-1),
    )
    save_solution_plot(
        run_dir / "solution.png",
        np.column_stack(
            (
                np.broadcast_to(x[None, :], exact_grid.shape).reshape(-1),
                np.broadcast_to(t[:, None], exact_grid.shape).reshape(-1),
            )
        ),
        exact_grid.reshape(-1),
        final_grid.reshape(-1),
        f"KS spectral layers-{layer_label} refinement, relative L2={final_data['relative_l2']:.3e}",
    )
    np.savetxt(
        run_dir / "history.csv",
        np.asarray(history, dtype=np.float64),
        delimiter=",",
        header=(
            "iteration,batch_data_mse,batch_spectral_loss,batch_total_loss,"
            "full_data_mse,full_relative_l2,full_spectral_loss,delta_norm,lr"
        ),
        comments="",
    )
    metrics = {
        "source_model": str(model_path),
        "refined_layer": args.refine_layer,
        "refined_layers": refined_layers,
        "best_iteration": best_iteration,
        "selection_score": best_score,
        "initial_data": initial_data,
        "initial_spectral": initial_spectral,
        "final_data": final_data,
        "final_spectral": final_spectral,
        "final_pinn_loss": final_pinn,
        "derivative_grid": derivative_metric,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)
    print(
        f"Finished: best step={best_iteration}; relative L2 "
        f"{initial_data['relative_l2']:.6e} -> {final_data['relative_l2']:.6e}; "
        f"weighted PINN loss={final_pinn['pinn_loss_weighted']:.6e}; "
        f"artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Refine one frozen pretrained KS RWF MLP through a trainable additive "
            "correction in all or one layer and a spatial spectral data loss."
        )
    )
    parser.add_argument("--model", default=r"C:\Users\Рустам\Documents\GitHub\PINNacle\runs_data_ks\08.24-01.58.59-ks-data-rwf-soap-float32-lr-cosine\weights_best.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "runs_data_ks_spectral_refine")
    )
    parser.add_argument(
        "--refine-layer",
        default="all",
        help="Use 'all' (default) or one one-based RWF layer index.",
    )
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--time-batch-size", type=int, default=8)
    parser.add_argument(
        "--optimizer", choices=["adam", "rmsprop", "soap", "muon"], default="soap"
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-epsilon", type=float, default=1e-8)
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
    parser.add_argument("--muon-adam-lr", type=float, default=1e-4)
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-adam-epsilon", type=float, default=1e-10)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "exponential"], default="cosine")
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-decay-rate", type=float, default=0.999)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--spectral-weight", type=float, default=1000.0)
    parser.add_argument(
        "--spectral-min-mode",
        type=int,
        default=40,
        help="Lowest spatial Fourier mode included in the extra spectral loss.",
    )
    parser.add_argument(
        "--spectral-max-mode",
        type=int,
        default=150,
        help="Highest trusted spatial Fourier mode included in the spectral loss.",
    )
    parser.add_argument("--spectral-power", type=float, default=8.0)
    parser.add_argument(
        "--spectral-boost",
        type=float,
        default=100.0,
        help="Maximum extra weight assigned smoothly to the highest spatial mode.",
    )
    parser.add_argument("--delta-l2-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--pinn-points", type=int, default=8192)
    parser.add_argument("--pinn-ic-points", type=int, default=2048)
    parser.add_argument("--pinn-batch-size", type=int, default=512)
    parser.add_argument("--ic-loss-weight", type=float, default=100.0)
    parser.add_argument("--derivative-grid-nx", type=int, default=128)
    parser.add_argument("--derivative-grid-nt", type=int, default=64)
    parser.add_argument("--derivative-batch-size", type=int, default=512)
    parser.add_argument("--no-derivative-plots", dest="derivative_plots", action="store_false")
    parser.set_defaults(derivative_plots=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
