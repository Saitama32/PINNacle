"""Reproduce the JAX-PI chaotic Kuramoto--Sivashinsky experiment in PyTorch.

The ablation mechanisms are deliberately independent. Run ``--help``
for the paired feature flags and preset overrides.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import deepxde as dde
import numpy as np
import torch

from src.losses.front_integral import (
    FrontIntegralLoss,
    attach_front_integral_loss_train_step,
)
from src.model import JaxpiKSNetwork, PinnacleKSFNN, SFLIConfig
from src.utils.callbacks import (
    FrontIntegralDiagnosticsCallback,
    LossCallback,
    TesterCallback,
)
from src.utils.grad_norm import AdaptiveLossWeights, GradNormCallback


NETWORK_CHOICES = (
    "mlp",
    "rwf-mlp",
    "modified-mlp",
    "rwf-modified-mlp",
    "repnn",
    "repnn-rwf",
)
INITIALIZATION_CHOICES = (
    "none",
    "stfli_cos",
    "stfli_gauss",
    "stfli_tanh",
)
INITIALIZATION_TO_SFLI_TYPE = {
    "stfli_cos": "cosine",
    "stfli_gauss": "gaussian",
    "stfli_tanh": "tanh",
}


PRESETS = {
    "ablation": {
        "time_fraction": 0.1,
        "steps_per_window": 100_000,
        "num_time_windows": 1,
        "batch_size": 10000,
        "num_layers": 5,
        "hidden_dim": 100,
        "learning_rate": 1e-3,
        "decay_rate": 0.9,
        "decay_steps": 1_000,
        "initial_condition_weight": 100.0,
        "save_every": 0,
        "modified_mlp": False,
        "fourier_features": False,
        "grad_norm": True,
        "causal": False,
        "jaxpi_network": True,
        "periodic_encoding": False,
        "periodic_bc": False,
        "net": "mlp",
    },
    "sota": {
        "time_fraction": 1.0,
        "steps_per_window": 10_000,
        "num_time_windows": 10,
        "batch_size": 10000,
        "num_layers": 5,
        "hidden_dim": 100,
        "learning_rate": 1e-3,
        "decay_rate": 0.9,
        "decay_steps": 2_000,
        "initial_condition_weight": 1_000.0,
        "save_every": 10_000,
        "modified_mlp": False,
        "fourier_features": False,
        "grad_norm": True,
        "causal": False,
        "jaxpi_network": True,
        "periodic_encoding": False,
        "periodic_bc": False,
        "net": "mlp",
    },
}


@dataclass(frozen=True)
class KSReference:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray  # [time, space]


@dataclass(frozen=True)
class TimeWindow:
    index: int
    start_index: int
    stop_index: int
    global_t: np.ndarray
    local_t: np.ndarray
    time_scale: float
    train_t_max: float
    transfer_t: float


class CausalHistoryCallback(dde.callbacks.Callback):
    def __init__(self, path: Path, every: int):
        super().__init__()
        self.path = path
        self.every = int(every)

    def on_epoch_end(self):
        step = int(self.model.train_state.step)
        if step % self.every:
            return
        details = getattr(self.model, "causal_loss_details", None)
        if not details:
            return
        record = {"step": step}
        for key, value in details.items():
            if torch.is_tensor(value):
                array = value.detach().cpu().numpy()
                record[key] = array.tolist() if array.ndim else float(array)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, sort_keys=True) + "\n")


class PresetFeatureAction(argparse.Action):
    """Apply a boolean feature override and remember that it was explicit."""

    def __init__(self, option_strings, dest, enabled, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
        self.enabled = bool(enabled)

    def __call__(self, parser, namespace, values, option_string=None):
        explicit = set(getattr(namespace, "_preset_overrides", ()))
        explicit.add(self.dest)
        setattr(namespace, "_preset_overrides", explicit)
        setattr(namespace, self.dest, self.enabled)


def add_feature_flag(parser: argparse.ArgumentParser, name: str, destination: str):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}",
        dest=destination,
        action=PresetFeatureAction,
        enabled=True,
    )
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action=PresetFeatureAction,
        enabled=False,
    )
    parser.set_defaults(**{destination: PRESETS["ablation"][destination]})


class PresetOverrideAction(argparse.Action):
    """Record that a preset-controlled value was explicitly passed on the CLI."""

    def __call__(self, parser, namespace, values, option_string=None):
        explicit = set(getattr(namespace, "_preset_overrides", ()))
        explicit.add(self.dest)
        setattr(namespace, "_preset_overrides", explicit)
        setattr(namespace, self.dest, values)


def str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JAX-PI chaotic KS reproduction using PINNacle's PyTorch backend."
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="ablation")
    add_feature_flag(parser, "modified-mlp", "modified_mlp")
    add_feature_flag(parser, "fourier-features", "fourier_features")
    add_feature_flag(parser, "grad-norm", "grad_norm")
    add_feature_flag(parser, "causal", "causal")
    add_feature_flag(parser, "jaxpi-network", "jaxpi_network")
    add_feature_flag(parser, "periodic-encoding", "periodic_encoding")

    parser.set_defaults(_preset_overrides=set())
    parser.add_argument(
        "--periodic-bc",
        type=str2bool,
        default=PRESETS["ablation"]["periodic_bc"],
        action=PresetOverrideAction,
        metavar="{true,false}",
        help="Enable or disable the soft higher-order periodic boundary loss.",
    )
    parser.add_argument(
        "--net",
        choices=NETWORK_CHOICES,
        default="mlp",
        action=PresetOverrideAction,
        help="Network architecture. This is the canonical architecture selector.",
    )
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default="none",
        help=(
            "First-layer initialization for mlp or rwf-mlp: none, "
            "stfli_cos, stfli_gauss, or stfli_tanh."
        ),
    )
    parser.add_argument(
        "--steps-per-window",
        type=int,
        default=10_000,
        action=PresetOverrideAction,
    )
    parser.add_argument(
        "--num-time-windows",
        type=int,
        default=5,
        action=PresetOverrideAction,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_024,
        action=PresetOverrideAction,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        action=PresetOverrideAction,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=100,
        action=PresetOverrideAction,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        action=PresetOverrideAction,
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.9,
        action=PresetOverrideAction,
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=1_000,
        action=PresetOverrideAction,
    )
    parser.add_argument("--fourier-dim", type=int, default=256)
    parser.add_argument("--fourier-scale", type=float, default=1.0)
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument(
        "--repnn-nu-s",
        type=float,
        default=10.0,
        help="Standard deviation of the RepNN first-layer effective weights.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "muon", "soap", "pcgrad"),
        default="adam",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pcgrad-base-optimizer", choices=("adam",), default="adam")
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-epsilon", type=float, default=1e-8)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4_096)
    parser.add_argument(
        "--soap-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument(
        "--muon-nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-adam-epsilon", type=float, default=1e-10)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-norm-momentum", type=float, default=0.9)
    parser.add_argument("--grad-norm-update-every", type=int, default=1_000)
    parser.add_argument("--causal-tol", type=float, default=1.0)
    parser.add_argument("--causal-num-chunks", type=int, default=16)
    parser.add_argument(
        "--periodic-bc-weight",
        type=float,
        default=100.0,
        help="Initial weight of the combined u/u_x/u_xx/u_xxx periodic loss.",
    )
    parser.add_argument(
        "--periodic-bc-points",
        type=int,
        default=256,
        help="Number of base boundary samples per time window before periodic pairing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-front-integral-loss", action="store_true", default=False)
    parser.add_argument("--front-integral-weight", type=float, default=10.00)
    parser.add_argument("--front-integral-num-intervals", type=int, default=10)
    parser.add_argument("--front-integral-num-x-points", type=int, default=100)
    parser.add_argument("--front-integral-quadrature-order", type=int, default=6)
    parser.add_argument("--front-integral-x-batch-size", type=int, default=250)
    parser.add_argument(
        "--front-integral-sampling",
        choices=("fixed", "pseudo"),
        default="pseudo",
    )
    parser.add_argument("--out", type=Path, default=Path("runs_jaxpi_ks"))
    parser.add_argument("--reference", type=Path, default=PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        action=PresetOverrideAction,
    )
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    config = dict(PRESETS[args.preset])
    preset_overrides = set(getattr(args, "_preset_overrides", ()))
    for name in (
        "steps_per_window",
        "num_time_windows",
        "batch_size",
        "num_layers",
        "hidden_dim",
        "learning_rate",
        "decay_rate",
        "decay_steps",
        "save_every",
        "modified_mlp",
        "fourier_features",
        "grad_norm",
        "causal",
        "jaxpi_network",
        "periodic_encoding",
        "periodic_bc",
        "net",
    ):
        if name in preset_overrides:
            config[name] = getattr(args, name)
    config.update(
        {
            "preset": args.preset,
            "fourier_dim": args.fourier_dim,
            "fourier_scale": args.fourier_scale,
            "rwf_mu": args.rwf_mu,
            "rwf_sigma": args.rwf_sigma,
            "repnn_nu_s": args.repnn_nu_s,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "pcgrad_base_optimizer": args.pcgrad_base_optimizer,
            "soap_beta1": args.soap_beta1,
            "soap_beta2": args.soap_beta2,
            "soap_shampoo_beta": args.soap_shampoo_beta,
            "soap_epsilon": args.soap_epsilon,
            "soap_precondition_frequency": args.soap_precondition_frequency,
            "soap_max_precondition_dim": args.soap_max_precondition_dim,
            "soap_bias_correction": args.soap_bias_correction,
            "muon_momentum": args.muon_momentum,
            "muon_nesterov": args.muon_nesterov,
            "muon_ns_steps": args.muon_ns_steps,
            "muon_adam_lr": args.muon_adam_lr,
            "muon_adam_beta1": args.muon_adam_beta1,
            "muon_adam_beta2": args.muon_adam_beta2,
            "muon_adam_epsilon": args.muon_adam_epsilon,
            "muon_weight_decay": args.muon_weight_decay,
            "muon_adam_weight_decay": args.muon_adam_weight_decay,
            "grad_norm_momentum": args.grad_norm_momentum,
            "grad_norm_update_every": args.grad_norm_update_every,
            "causal_tol": args.causal_tol,
            "causal_num_chunks": args.causal_num_chunks,
            "periodic_bc_weight": args.periodic_bc_weight,
            "periodic_bc_points": args.periodic_bc_points,
            "initialization": args.initialization,
            "seed": args.seed,
            "use_front_integral_loss": args.use_front_integral_loss,
            "front_integral_weight": args.front_integral_weight,
            "front_integral_num_intervals": args.front_integral_num_intervals,
            "front_integral_num_x_points": args.front_integral_num_x_points,
            "front_integral_quadrature_order": args.front_integral_quadrature_order,
            "front_integral_x_batch_size": args.front_integral_x_batch_size,
            "front_integral_sampling": args.front_integral_sampling,
            "out": str(args.out),
            "reference": str(args.reference),
            "log_every": args.log_every,
        }
    )
    # Keep the old boolean architecture flags usable by existing notebooks.
    # An explicit --net is canonical and takes precedence over them.
    if "net" not in preset_overrides and (
        "modified_mlp" in preset_overrides or "jaxpi_network" in preset_overrides
    ):
        config["net"] = "modified-mlp" if config["modified_mlp"] else "mlp"
    config["modified_mlp"] = config["net"] in {
        "modified-mlp",
        "rwf-modified-mlp",
    }
    config["jaxpi_network"] = config["modified_mlp"]
    config["use_rwf"] = config["net"] in {
        "rwf-mlp",
        "rwf-modified-mlp",
        "repnn-rwf",
    }
    validate_config(config)
    return config


def validate_config(config: dict):
    if config.get("initialization", "none") not in INITIALIZATION_CHOICES:
        raise ValueError(
            f"initialization must be one of: {', '.join(INITIALIZATION_CHOICES)}"
        )
    positive_ints = (
        "steps_per_window",
        "num_time_windows",
        "batch_size",
        "num_layers",
        "hidden_dim",
        "decay_steps",
        "grad_norm_update_every",
        "causal_num_chunks",
        "log_every",
        "soap_precondition_frequency",
        "soap_max_precondition_dim",
        "muon_ns_steps",
        "periodic_bc_points",
        "front_integral_num_intervals",
        "front_integral_num_x_points",
        "front_integral_quadrature_order",
        "front_integral_x_batch_size",
    )
    for name in positive_ints:
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if config["save_every"] < 0:
        raise ValueError("save_every must be non-negative")
    if (
        not math.isfinite(config["front_integral_weight"])
        or config["front_integral_weight"] < 0
    ):
        raise ValueError("front_integral_weight must be non-negative and finite")
    if config["use_front_integral_loss"] and config["optimizer"] == "pcgrad":
        raise ValueError("front integral loss currently supports adam, soap, and muon only")
    if config["use_front_integral_loss"] and config["causal"]:
        raise ValueError("front integral loss cannot be combined with causal loss yet")
    if not math.isfinite(config["rwf_mu"]):
        raise ValueError("rwf_mu must be finite")
    if not math.isfinite(config["rwf_sigma"]) or config["rwf_sigma"] < 0:
        raise ValueError("rwf_sigma must be non-negative and finite")
    if not math.isfinite(config["repnn_nu_s"]) or config["repnn_nu_s"] <= 0:
        raise ValueError("repnn_nu_s must be positive and finite")
    for name in (
        "learning_rate",
        "decay_rate",
        "fourier_scale",
        "causal_tol",
        "soap_epsilon",
        "muon_adam_lr",
        "muon_adam_epsilon",
        "periodic_bc_weight",
    ):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in (
        "weight_decay",
        "muon_weight_decay",
        "muon_adam_weight_decay",
    ):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    for name in ("soap_beta1", "soap_beta2", "muon_momentum", "muon_adam_beta1", "muon_adam_beta2"):
        if not math.isfinite(config[name]) or not 0 <= config[name] < 1:
            raise ValueError(f"{name} must satisfy 0 <= value < 1")
    if config["soap_shampoo_beta"] is not None and (
        not math.isfinite(config["soap_shampoo_beta"])
        or not 0 <= config["soap_shampoo_beta"] < 1
    ):
        raise ValueError("soap_shampoo_beta must satisfy 0 <= value < 1")
    if config["fourier_dim"] <= 0 or config["fourier_dim"] % 2:
        raise ValueError("fourier_dim must be a positive even integer")
    if not 0 <= config["grad_norm_momentum"] < 1:
        raise ValueError("grad_norm_momentum must satisfy 0 <= value < 1")
    if config["causal"] and config["batch_size"] % config["causal_num_chunks"]:
        raise ValueError(
            "batch_size must be divisible by causal_num_chunks for JAX-PI equal-count chunking"
        )
    if config.get("initialization", "none") != "none" and config["net"] not in {
        "mlp",
        "rwf-mlp",
    }:
        raise ValueError(
            "STFLI initialization supports only --net mlp and --net rwf-mlp"
        )


def load_reference(path: os.PathLike) -> KSReference:
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("KS reference must have columns x, t, u")
    x = np.unique(raw[:, 0])
    t = np.unique(raw[:, 1])
    if len(raw) != len(x) * len(t):
        raise ValueError("KS reference must contain a complete rectangular x/t grid")
    x_indices = np.searchsorted(x, raw[:, 0])
    t_indices = np.searchsorted(t, raw[:, 1])
    u = np.empty((len(t), len(x)), dtype=np.float64)
    u[t_indices, x_indices] = raw[:, 2]
    return KSReference(x=x, t=t, u=u)


def build_time_windows(reference: KSReference, time_fraction: float, count: int):
    num_times = int(float(time_fraction) * len(reference.t))
    if num_times < 2:
        raise ValueError("time_fraction leaves fewer than two reference time levels")
    times_per_window = num_times // int(count)
    if times_per_window < 2:
        raise ValueError("num_time_windows leaves fewer than two time levels per window")
    dt = float(reference.t[1] - reference.t[0])
    windows = []
    for index in range(int(count)):
        start = index * times_per_window
        stop = start + times_per_window
        global_t = reference.t[start:stop].copy()
        local_t = global_t - global_t[0]
        time_scale = float(local_t[-1])
        transfer_t = float(reference.t[stop] - global_t[0]) if stop < len(reference.t) else time_scale + dt
        windows.append(
            TimeWindow(
                index=index,
                start_index=start,
                stop_index=stop,
                global_t=global_t,
                local_t=local_t,
                time_scale=time_scale,
                train_t_max=time_scale + 2.0 * dt,
                transfer_t=transfer_t,
            )
        )
    return windows


def _ks_spatial_operator(points, values):
    u_x = dde.grad.jacobian(values, points, i=0, j=0)
    u_xx = dde.grad.hessian(values, points, i=0, j=0)
    u_xxxx = dde.grad.hessian(u_xx, points, i=0, j=0)
    return (
        (100.0 / 16.0) * values * u_x
        + (100.0 / (16.0**2)) * u_xx
        + (100.0 / (16.0**4)) * u_xxxx
    )


def _ks_residual(points, values):
    u_t = dde.grad.jacobian(values, points, i=0, j=1)
    return u_t + _ks_spatial_operator(points, values)


class InterpolatedInitialCondition:
    """Piecewise-linear window IC evaluated directly on the active device."""

    def __init__(self, x, values):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if x.size < 2 or values.shape != x.shape:
            raise ValueError("window initial condition requires matching x/value arrays")
        if np.any(np.diff(x) <= 0):
            raise ValueError("window initial-condition coordinates must be strictly increasing")
        self.x = x
        self.values = values
        self._tensor_cache = {}

    def _tensors(self, device, dtype):
        key = (str(device), str(dtype))
        if key not in self._tensor_cache:
            self._tensor_cache[key] = (
                torch.as_tensor(self.x, device=device, dtype=dtype),
                torch.as_tensor(self.values, device=device, dtype=dtype),
            )
        return self._tensor_cache[key]

    def __call__(self, x):
        knots, values = self._tensors(x.device, x.dtype)
        flat_x = x.reshape(-1).clamp(min=knots[0], max=knots[-1])
        right = torch.searchsorted(knots, flat_x, right=True).clamp(1, knots.numel() - 1)
        left = right - 1
        fraction = (flat_x - knots[left]) / (knots[right] - knots[left])
        interpolated = values[left] + fraction * (values[right] - values[left])
        return interpolated.reshape_as(x)


def sfli_input_bounds(config: dict, reference: KSReference, window: TimeWindow):
    """Return box bounds in the coordinates received by the inner MLP."""

    if config["fourier_features"]:
        return tuple((-1.0, 1.0) for _ in range(int(config["fourier_dim"])))

    normalized_time_max = float(window.train_t_max / window.time_scale)
    if config["periodic_encoding"]:
        return (
            (0.0, normalized_time_max),
            (-1.0, 1.0),
            (-1.0, 1.0),
        )
    return (
        (float(reference.x[0]), float(reference.x[-1])),
        (0.0, normalized_time_max),
    )


def build_model(config: dict, reference: KSReference, window: TimeWindow, initial_values):
    geometry = dde.geometry.Interval(float(reference.x[0]), float(reference.x[-1]))
    time_domain = dde.geometry.TimeDomain(0.0, window.train_t_max)
    geometry_time = dde.geometry.GeometryXTime(geometry, time_domain)
    initial_points = np.column_stack(
        (reference.x, np.zeros_like(reference.x))
    ).astype(np.float32)
    initial_values = np.asarray(initial_values, dtype=np.float32).reshape(-1, 1)
    initial_condition = dde.icbc.PointSetBC(initial_points, initial_values, component=0)
    boundary_conditions = [initial_condition]
    if config["periodic_bc"]:
        boundary_conditions.append(
            dde.HigherOrderPeriodicBC(
                geometry_time,
                component_x=0,
                on_boundary=lambda _, on_boundary: on_boundary,
                derivative_orders=(0, 1, 2, 3),
                component=0,
            )
        )
    data = dde.data.TimePDE(
        geometry_time,
        _ks_residual,
        boundary_conditions,
        num_domain=config["batch_size"],
        num_boundary=config["periodic_bc_points"] if config["periodic_bc"] else 0,
        num_initial=0,
        num_test=config["batch_size"],
        train_distribution="pseudo",
    )
    network_options = {
        "time_scale": window.time_scale,
        "num_layers": config["num_layers"],
        "hidden_dim": config["hidden_dim"],
        "fourier_features": config["fourier_features"],
        "periodic_encoding": config["periodic_encoding"],
        "fourier_dim": config["fourier_dim"],
        "fourier_scale": config["fourier_scale"],
        "use_rwf": config["use_rwf"],
        "rwf_mu": config["rwf_mu"],
        "rwf_sigma": config["rwf_sigma"],
    }
    if config["modified_mlp"]:
        network = JaxpiKSNetwork(
            modified_mlp=True,
            **network_options,
        ).float()
    else:
        sfli = None
        initialization = config.get("initialization", "none")
        if initialization != "none":
            sfli = SFLIConfig(
                bounds=sfli_input_bounds(config, reference, window),
                type=INITIALIZATION_TO_SFLI_TYPE[initialization],
            )
        network = PinnacleKSFNN(
            sfli=sfli,
            network_type=config["net"],
            repnn_nu_s=config["repnn_nu_s"],
            input_bounds=sfli_input_bounds(config, reference, window),
            **network_options,
        ).float()
    trainable_parameters = sum(
        parameter.numel() for parameter in network.parameters() if parameter.requires_grad
    )
    print(
        f"Network: {config['net']} | hidden: {config['hidden_dim']}x{config['num_layers']} | "
        f"gating: {'on' if config['modified_mlp'] else 'off'} | "
        f"RWF: {'on' if config['use_rwf'] else 'off'} | "
        f"initialization: {config.get('initialization', 'none')} | "
        f"trainable parameters: {trainable_parameters}"
    )
    if config["use_rwf"]:
        print(f"RWF mu: {config['rwf_mu']} | RWF sigma: {config['rwf_sigma']}")
    if config["net"] in {"repnn", "repnn-rwf"}:
        print(f"RepNN nu_s: {config['repnn_nu_s']}")
    model = dde.Model(data, network)
    reference_points = prediction_grid(reference.x, window.local_t)
    reference_values = reference.u[
        window.start_index : window.stop_index
    ].reshape(-1, 1)
    model.pde = SimpleNamespace(
        bbox=[float(reference.x[0]), float(reference.x[-1]), 0.0, window.train_t_max],
        geom=geometry,
        input_dim=2,
        output_dim=1,
        ref_sol=None,
        ref_data=np.hstack((reference_points, reference_values)).astype(np.float32),
        ks_spatial_operator=_ks_spatial_operator,
        loss_config=[
            {"name": "res", "type": "pde"},
            {"name": "ics", "type": "ic"},
        ]
        + (
            [{"name": "periodic", "type": "boundary"}]
            if config["periodic_bc"]
            else []
        ),
        num_loss=3 if config["periodic_bc"] else 2,
    )
    if config["causal"]:
        model.causal_loss_options = {
            "enabled": True,
            "num_chunks": config["causal_num_chunks"],
            "tol": config["causal_tol"],
            "time_index": -1,
            "include_ic_in_weights": False,
            "ic_weight_in_causal": 0.0,
            "t_min": 0.0,
            "t_max": window.train_t_max,
            "chunking": "equal_count",
        }

    initial_weights = np.array(
        [1.0, config["initial_condition_weight"]]
        + ([config["periodic_bc_weight"]] if config["periodic_bc"] else []),
        dtype=np.float32,
    )
    adapter = AdaptiveLossWeights(initial_weights) if config["grad_norm"] else None
    if config["weight_decay"] > 0:
        network.regularizer = ("l2", config["weight_decay"])
    if config["optimizer"] == "soap":
        dde.optimizers.set_SOAP_options(
            beta1=config["soap_beta1"],
            beta2=config["soap_beta2"],
            shampoo_beta=config["soap_shampoo_beta"],
            epsilon=config["soap_epsilon"],
            precondition_frequency=config["soap_precondition_frequency"],
            max_precondition_dim=config["soap_max_precondition_dim"],
            bias_correction=config["soap_bias_correction"],
        )
    elif config["optimizer"] == "muon":
        dde.optimizers.set_MUON_options(
            momentum=config["muon_momentum"],
            nesterov=config["muon_nesterov"],
            ns_steps=config["muon_ns_steps"],
            adam_lr=config["muon_adam_lr"],
            adam_betas=(config["muon_adam_beta1"], config["muon_adam_beta2"]),
            adam_eps=config["muon_adam_epsilon"],
            muon_weight_decay=config["muon_weight_decay"],
            adam_weight_decay=config["muon_adam_weight_decay"],
        )
    elif config["optimizer"] == "pcgrad":
        dde.optimizers.set_PCGRAD_options(
            base_optimizer=config["pcgrad_base_optimizer"],
        )
    model.compile(
        config["optimizer"],
        lr=config["learning_rate"],
        decay=("exponential", config["decay_steps"], config["decay_rate"]),
        loss_weights=adapter if adapter is not None else initial_weights,
    )
    if config["use_front_integral_loss"]:
        front_integral_loss = FrontIntegralLoss(
            model=model,
            pde=model.pde,
            num_intervals=config["front_integral_num_intervals"],
            num_x_points=config["front_integral_num_x_points"],
            quadrature_order=config["front_integral_quadrature_order"],
            x_batch_size=config["front_integral_x_batch_size"],
            weight=config["front_integral_weight"],
            sampling=config["front_integral_sampling"],
            initial_condition_fn=InterpolatedInitialCondition(
                reference.x,
                initial_values,
            ),
        )
        attach_front_integral_loss_train_step(model, front_integral_loss)
    else:
        model.front_integral_loss = None
        model.front_integral_loss_diagnostics = None
    # Keep the mutable adapter in the compiled loss closure, while exposing a
    # numerical snapshot to LossCallback and the loss-history plotting code.
    if adapter is not None:
        model.losshistory.set_loss_weights(initial_weights.copy())
    return model, adapter


def prediction_grid(x: np.ndarray, local_t: np.ndarray):
    tt, xx = np.meshgrid(local_t, x, indexing="ij")
    return np.column_stack((xx.reshape(-1), tt.reshape(-1))).astype(np.float32)


def relative_l2(prediction, exact):
    denominator = np.linalg.norm(exact.reshape(-1))
    return float(np.linalg.norm((prediction - exact).reshape(-1)) / denominator)


def long_horizon_metrics(prediction, exact, late_fraction: float = 0.5):
    """Return phase-insensitive metrics over the late-time part of a field.

    Arrays must have shape ``[time, space]``. Energy and spectra are computed
    after removing the spatial mean at every time level. Wasserstein-1 uses
    the empirical distributions of the original field values and is
    normalized by the reference standard deviation.
    """

    prediction = np.asarray(prediction, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if prediction.shape != exact.shape or prediction.ndim != 2:
        raise ValueError("prediction and exact must have the same [time, space] shape")
    if not 0.0 < float(late_fraction) <= 1.0:
        raise ValueError("late_fraction must satisfy 0 < late_fraction <= 1")
    if prediction.size == 0:
        raise ValueError("prediction and exact must not be empty")
    if not np.isfinite(prediction).all() or not np.isfinite(exact).all():
        return {
            "late_energy_agreement": float("nan"),
            "late_spectral_overlap": float("nan"),
            "late_normalized_wasserstein": float("nan"),
        }

    late_count = max(1, int(math.ceil(prediction.shape[0] * float(late_fraction))))
    pred_late = prediction[-late_count:]
    exact_late = exact[-late_count:]
    pred_centered = pred_late - pred_late.mean(axis=1, keepdims=True)
    exact_centered = exact_late - exact_late.mean(axis=1, keepdims=True)

    pred_energy = float(np.median(np.mean(pred_centered**2, axis=1)))
    exact_energy = float(np.median(np.mean(exact_centered**2, axis=1)))
    energy_floor = np.finfo(np.float64).eps * max(pred_energy, exact_energy, 1.0)
    if pred_energy <= energy_floor and exact_energy <= energy_floor:
        energy_agreement = 1.0
    else:
        energy_agreement = (
            2.0
            * pred_energy
            * exact_energy
            / (pred_energy**2 + exact_energy**2 + energy_floor**2)
        )

    pred_power = np.mean(np.abs(np.fft.rfft(pred_centered, axis=1)) ** 2, axis=0)
    exact_power = np.mean(np.abs(np.fft.rfft(exact_centered, axis=1)) ** 2, axis=0)
    spectral_denominator = float(pred_power.sum() + exact_power.sum())
    if spectral_denominator <= energy_floor:
        spectral_overlap = 1.0
    else:
        spectral_overlap = float(
            2.0 * np.minimum(pred_power, exact_power).sum() / spectral_denominator
        )

    pred_sorted = np.sort(pred_late.reshape(-1))
    exact_sorted = np.sort(exact_late.reshape(-1))
    wasserstein = float(np.mean(np.abs(pred_sorted - exact_sorted)))
    exact_scale = float(np.std(exact_late))
    normalized_wasserstein = wasserstein / max(exact_scale, np.finfo(np.float64).eps)

    return {
        "late_energy_agreement": float(np.clip(energy_agreement, 0.0, 1.0)),
        "late_spectral_overlap": float(np.clip(spectral_overlap, 0.0, 1.0)),
        "late_normalized_wasserstein": normalized_wasserstein,
    }


def save_loss_history(model, path: Path):
    history = model.losshistory
    np.savez_compressed(
        path,
        steps=np.asarray(history.steps),
        loss_train=np.asarray(history.loss_train),
        loss_test=np.asarray(history.loss_test),
    )


def save_solution_plot(path: Path, exact, prediction, title: str):
    import matplotlib.pyplot as plt

    error = np.abs(prediction - exact)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, values, label in zip(
        axes,
        (exact, prediction, error),
        ("Exact", "Prediction", "Absolute error"),
    ):
        image = axis.imshow(values.T, origin="lower", aspect="auto", cmap="jet")
        axis.set_title(label)
        axis.set_xlabel("time index")
        axis.set_ylabel("space index")
        figure.colorbar(image, ax=axis)
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def feature_tag(config: dict):
    parts = [
        f"net-{config['net']}",
        f"init-{config.get('initialization', 'none')}",
        f"ff-{'on' if config['fourier_features'] else 'off'}",
        f"gn-{'on' if config['grad_norm'] else 'off'}",
        f"causal-{'on' if config['causal'] else 'off'}",
        f"penc-{'on' if config['periodic_encoding'] else 'off'}",
        f"pbc-{'on' if config['periodic_bc'] else 'off'}",
    ]
    if config["use_front_integral_loss"]:
        parts.append(f"front-{config['front_integral_sampling']}")
    return "_".join(parts)


def run(config: dict) -> Path:
    dde.config.set_random_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    reference = load_reference(config["reference"])
    windows = build_time_windows(
        reference,
        time_fraction=config["time_fraction"],
        count=config["num_time_windows"],
    )
    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(config["out"]) / (
        f"{timestamp}-{config['preset']}-opt-{config['optimizer']}-{feature_tag(config)}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = dict(config)
    resolved["windows"] = [
        {**asdict(window), "global_t": window.global_t.tolist(), "local_t": window.local_t.tolist()}
        for window in windows
    ]
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(resolved, file_obj, indent=2, sort_keys=True)

    initial_values = reference.u[0].copy()
    stitched_predictions = []
    stitched_exact = []
    stitched_times = []
    window_metrics = []

    for window in windows:
        window_dir = run_dir / f"window_{window.index + 1:02d}"
        window_dir.mkdir(parents=True, exist_ok=False)
        (window_dir / "final").mkdir()
        if config["save_every"]:
            (window_dir / "checkpoint").mkdir()
        print(
            f"Training KS window {window.index + 1}/{len(windows)} "
            f"for {config['steps_per_window']} steps ({feature_tag(config)})."
        )
        model, adapter = build_model(config, reference, window, initial_values)
        callbacks = [
            TesterCallback(
                log_every=config["log_every"],
                verbose=True,
                fRMSE_param={"enable": False},
                additional_metrics_fn=lambda predicted, expected, shape=(
                    len(window.local_t),
                    len(reference.x),
                ): long_horizon_metrics(
                    np.asarray(predicted).reshape(shape),
                    np.asarray(expected).reshape(shape),
                ),
            ),
            LossCallback(verbose=True),
        ]
        if config["grad_norm"]:
            callbacks.append(
                GradNormCallback(
                    adapter,
                    loss_names=tuple(item["name"] for item in model.pde.loss_config),
                    momentum=config["grad_norm_momentum"],
                    update_every=config["grad_norm_update_every"],
                    log_path=window_dir / "grad_norm.jsonl",
                )
            )
        if config["causal"]:
            callbacks.append(
                CausalHistoryCallback(window_dir / "causal_weights.jsonl", config["log_every"])
            )
        if config["use_front_integral_loss"]:
            callbacks.append(
                FrontIntegralDiagnosticsCallback(
                    log_every=config["log_every"],
                    verbose=True,
                )
            )
        callbacks.append(dde.callbacks.PDEPointResampler(period=1, pde_points=True, bc_points=False))
        if config["save_every"]:
            callbacks.append(
                dde.callbacks.ModelCheckpoint(
                    str(window_dir / "checkpoint"),
                    period=config["save_every"],
                    save_better_only=False,
                )
            )

        model.train(
            iterations=config["steps_per_window"],
            display_every=config["log_every"],
            callbacks=callbacks,
            model_save_path=str(window_dir / "final"),
            save_model=True,
        )
        local_points = prediction_grid(reference.x, window.local_t)
        prediction = model.predict(local_points).reshape(len(window.local_t), len(reference.x))
        exact = reference.u[window.start_index : window.stop_index]
        error = relative_l2(prediction, exact)
        metrics = {
            "window": window.index + 1,
            "start_index": window.start_index,
            "stop_index": window.stop_index,
            "relative_l2": error,
            **long_horizon_metrics(prediction, exact),
        }
        window_metrics.append(metrics)
        with (window_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, indent=2, sort_keys=True)
        np.savez_compressed(
            window_dir / "prediction.npz",
            x=reference.x,
            global_t=window.global_t,
            local_t=window.local_t,
            prediction=prediction,
            exact=exact,
            absolute_error=np.abs(prediction - exact),
        )
        save_loss_history(model, window_dir / "loss_history.npz")
        save_solution_plot(
            window_dir / "solution.png",
            exact,
            prediction,
            f"KS window {window.index + 1}, relative L2={error:.3e}",
        )
        stitched_predictions.append(prediction)
        stitched_exact.append(exact)
        stitched_times.append(window.global_t)

        transfer_points = prediction_grid(reference.x, np.array([window.transfer_t]))
        initial_values = model.predict(transfer_points).reshape(-1)
        for callback in callbacks:
            callback.model = None
        del callbacks, adapter, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_prediction = np.concatenate(stitched_predictions, axis=0)
    global_exact = np.concatenate(stitched_exact, axis=0)
    global_t = np.concatenate(stitched_times)
    global_error = relative_l2(global_prediction, global_exact)
    metrics = {
        "relative_l2": global_error,
        **long_horizon_metrics(global_prediction, global_exact),
        "windows": window_metrics,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)
    np.savez_compressed(
        run_dir / "prediction.npz",
        x=reference.x,
        t=global_t,
        prediction=global_prediction,
        exact=global_exact,
        absolute_error=np.abs(global_prediction - global_exact),
    )
    save_solution_plot(
        run_dir / "solution.png",
        global_exact,
        global_prediction,
        f"JAX-PI KS {config['preset']}, relative L2={global_error:.3e}",
    )
    print(
        f"Finished. relative L2={global_error:.6e}; "
        f"late energy agreement={metrics['late_energy_agreement']:.6e}; "
        f"late spectral overlap={metrics['late_spectral_overlap']:.6e}; "
        f"late normalized W1={metrics['late_normalized_wasserstein']:.6e}; "
        f"artifacts: {run_dir}"
    )
    return run_dir


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(resolve_config(args))


if __name__ == "__main__":
    main()
