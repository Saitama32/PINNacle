"""Train Wave1D with the spatial weak-form VPINN objective."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import deepxde as dde
import numpy as np
import torch

from src.losses.weak_form import WeakFormConfig, WeakFormLoss, attach_weak_form_loss
from src.pde.wave import Wave1D
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import LossCallback, PlotCallback, TesterCallback


def json_value(value):
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
        return float(array) if array.ndim == 0 else array.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


class WeakDiagnosticsCallback(dde.callbacks.Callback):
    def __init__(self, weak_loss, save_path, log_every):
        super().__init__()
        self.weak_loss = weak_loss
        self.save_path = save_path
        self.log_every = int(log_every)
        self.records = []

    def record(self):
        step = int(self.model.train_state.step)
        if self.records and self.records[-1]["step"] == step:
            return
        self.records.append(
            {
                "step": step,
                "train": json_value(self.weak_loss.last_train_diagnostics),
                "test": json_value(self.weak_loss.last_test_diagnostics),
            }
        )

    def on_train_begin(self):
        self.record()

    def on_epoch_end(self):
        if int(self.model.train_state.step) % self.log_every == 0:
            self.record()

    def on_train_end(self):
        self.record()
        filename = os.path.join(self.save_path, "weak_diagnostics.json")
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(self.records, stream, indent=2, ensure_ascii=False)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="wave1d_weak")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="cpu, cuda, cuda:N, or a bare CUDA index N",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--out", default="runs_single/weak")
    parser.add_argument("--boundary-points", type=int, default=None)
    parser.add_argument("--spatial-cells", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=10)
    parser.add_argument("--test-function-count", type=int, default=8)
    parser.add_argument("--time-samples", type=int, default=256)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--constraint-weight", type=float, default=100.0)
    parser.add_argument("--no-cell-volume-normalization", action="store_true")
    parser.add_argument("--no-tester", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--C", type=float, default=2.0, help="Wave speed")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--a", type=int, default=4, help="High-frequency mode multiplier")
    return parser


def configure_device(device_arg, seed):
    value = str(device_arg).strip().lower()
    if value == "cpu" or not torch.cuda.is_available():
        if value != "cpu":
            print("CUDA is unavailable; falling back to CPU.")
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        if value.isdigit():
            value = f"cuda:{value}"
        if value == "cuda":
            value = "cuda:0"
        device = torch.device(value)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    dde.config.set_default_float("float32")
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = make_parser().parse_args()
    if args.iterations <= 0 or args.log_every <= 0 or args.plot_every <= 0:
        raise ValueError("iterations, log-every and plot-every must be positive")
    if args.boundary_points is not None and args.boundary_points <= 0:
        raise ValueError("boundary-points must be positive")

    configure_device(args.device, args.seed)
    pde = Wave1D(C=args.C, scale=args.scale, a=args.a)
    if args.boundary_points is not None:
        pde.training_points(domain=1, boundary=args.boundary_points, test=1)

    layers = [pde.input_dim] + parse_hidden_layers(args) + [pde.output_dim]
    net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
    model = pde.create_model(net)

    weak_config = WeakFormConfig(
        spatial_cells=args.spatial_cells,
        quadrature_order=args.quadrature_order,
        test_function_count=args.test_function_count,
        time_samples=args.time_samples,
        seed=args.seed,
        normalize_by_cell_volume=not args.no_cell_volume_normalization,
    )
    weak_loss = WeakFormLoss(pde.weak_form_adapter(), weak_config)
    attach_weak_form_loss(model, weak_loss)

    loss_weights = np.array(
        [
            args.pde_weight if item.get("type") == "pde" else args.constraint_weight
            for item in pde.loss_config
        ],
        dtype=float,
    )
    model.compile("adam", lr=args.lr, loss_weights=loss_weights)

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.abspath(os.path.join(args.out, f"{date_str}-{args.name}"))
    os.makedirs(save_path, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update(
        {
            "resolved_device": str(next(model.net.parameters()).device),
            "pde_class": type(pde).__name__,
            "loss_config": pde.loss_config,
            "loss_weights": loss_weights.tolist(),
            "weak_form": asdict(weak_config),
        }
    )
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as stream:
        json.dump(json_value(run_config), stream, indent=2, ensure_ascii=False)

    callbacks = [
        WeakDiagnosticsCallback(weak_loss, save_path, args.log_every),
        LossCallback(verbose=True),
    ]
    if not args.no_tester:
        callbacks.insert(0, TesterCallback(log_every=args.log_every))
    if not args.no_plot:
        callbacks.append(PlotCallback(log_every=args.plot_every, fast=True))

    print(f"Weak-form run directory: {save_path}")
    print(f"Weak-form config: {weak_config}")
    model.train(
        iterations=args.iterations,
        display_every=args.log_every,
        callbacks=callbacks,
        model_save_path=save_path,
    )


if __name__ == "__main__":
    main()
