"""Reproduce the JAX-PI chaotic Kuramoto--Sivashinsky experiment in PyTorch.

The four ablation mechanisms are deliberately independent.  Run ``--help``
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

from src.model import JaxpiKSNetwork, PinnacleKSFNN
from src.utils.callbacks import LossCallback, TesterCallback
from src.utils.grad_norm import AdaptiveLossWeights, GradNormCallback


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
        "periodic_encoding": True,
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
        "periodic_encoding": True,
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
    parser.add_argument("--optimizer", choices=("adam", "muon", "soap"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
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
    parser.add_argument("--seed", type=int, default=42)
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
    ):
        if name in preset_overrides:
            config[name] = getattr(args, name)
    config.update(
        {
            "preset": args.preset,
            "fourier_dim": args.fourier_dim,
            "fourier_scale": args.fourier_scale,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
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
            "seed": args.seed,
            "out": str(args.out),
            "reference": str(args.reference),
            "log_every": args.log_every,
        }
    )
    validate_config(config)
    return config


def validate_config(config: dict):
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
    )
    for name in positive_ints:
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if config["save_every"] < 0:
        raise ValueError("save_every must be non-negative")
    for name in (
        "learning_rate",
        "decay_rate",
        "fourier_scale",
        "causal_tol",
        "soap_epsilon",
        "muon_adam_lr",
        "muon_adam_epsilon",
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
    if config["modified_mlp"] and not config["jaxpi_network"]:
        raise ValueError("modified_mlp requires jaxpi_network")


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


def _ks_residual(points, values):
    u_t = dde.grad.jacobian(values, points, i=0, j=1)
    u_x = dde.grad.jacobian(values, points, i=0, j=0)
    u_xx = dde.grad.hessian(values, points, i=0, j=0)
    u_xxxx = dde.grad.hessian(u_xx, points, i=0, j=0)
    return (
        u_t
        + (100.0 / 16.0) * values * u_x
        + (100.0 / (16.0**2)) * u_xx
        + (100.0 / (16.0**4)) * u_xxxx
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
    data = dde.data.TimePDE(
        geometry_time,
        _ks_residual,
        [initial_condition],
        num_domain=config["batch_size"],
        num_boundary=0,
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
    }
    if config["jaxpi_network"]:
        network = JaxpiKSNetwork(
            modified_mlp=config["modified_mlp"],
            **network_options,
        ).float()
    else:
        network = PinnacleKSFNN(**network_options).float()
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
        loss_config=[
            {"name": "res", "type": "pde"},
            {"name": "ics", "type": "ic"},
        ],
        num_loss=2,
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
        [1.0, config["initial_condition_weight"]], dtype=np.float32
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
    model.compile(
        config["optimizer"],
        lr=config["learning_rate"],
        decay=("exponential", config["decay_steps"], config["decay_rate"]),
        loss_weights=adapter if adapter is not None else initial_weights,
    )
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
    return "_".join(
        (
            f"mmlp-{'on' if config['modified_mlp'] else 'off'}",
            f"ff-{'on' if config['fourier_features'] else 'off'}",
            f"gn-{'on' if config['grad_norm'] else 'off'}",
            f"causal-{'on' if config['causal'] else 'off'}",
            f"jaxnet-{'on' if config['jaxpi_network'] else 'off'}",
            f"periodic-{'on' if config['periodic_encoding'] else 'off'}",
        )
    )


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
            ),
            LossCallback(verbose=True),
        ]
        if config["grad_norm"]:
            callbacks.append(
                GradNormCallback(
                    adapter,
                    loss_names=("res", "ics"),
                    momentum=config["grad_norm_momentum"],
                    update_every=config["grad_norm_update_every"],
                    log_path=window_dir / "grad_norm.jsonl",
                )
            )
        if config["causal"]:
            callbacks.append(
                CausalHistoryCallback(window_dir / "causal_weights.jsonl", config["log_every"])
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
    metrics = {"relative_l2": global_error, "windows": window_metrics}
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
    print(f"Finished. relative L2={global_error:.6e}; artifacts: {run_dir}")
    return run_dir


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(resolve_config(args))


if __name__ == "__main__":
    main()
