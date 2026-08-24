"""Construct an exact cubic-ReLU coefficient MLP for the KS data solution."""

from __future__ import annotations

import argparse
import io
import json
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
from scipy.interpolate import CubicSpline

from experiments.Chaotic.run_data_ks import (
    KS_ALPHA,
    KS_BETA,
    KS_GAMMA,
    evaluate_derivative_grid,
    evaluate_pinn_loss,
    load_data,
    prediction_metrics,
    save_solution_plot,
)


class CubicReLUCoefficientMLP(torch.nn.Module):
    """One-hidden-layer MLP representing a vector-valued cubic spline exactly."""

    def __init__(self, feature_knots, output_weights):
        super().__init__()
        feature_knots = torch.as_tensor(feature_knots, dtype=torch.float64)
        output_weights = torch.as_tensor(output_weights, dtype=torch.float64)
        self.hidden = torch.nn.Linear(1, len(feature_knots), bias=True, dtype=torch.float64)
        self.output = torch.nn.Linear(
            len(feature_knots), output_weights.shape[0], bias=False, dtype=torch.float64
        )
        with torch.no_grad():
            self.hidden.weight.fill_(1.0)
            self.hidden.bias.copy_(-feature_knots)
            self.output.weight.copy_(output_weights)

    def forward(self, t):
        return self.output(torch.relu(self.hidden(t)).pow(3))


class ExactFourierMLP(torch.nn.Module):
    """Cubic-ReLU MLP in time followed by exact Fourier synthesis in x."""

    def __init__(self, modes, feature_knots, output_weights):
        super().__init__()
        self.coefficient_mlp = CubicReLUCoefficientMLP(
            feature_knots, output_weights
        )
        self.register_buffer("modes", torch.as_tensor(modes, dtype=torch.float64))

    def forward(self, points):
        vector = self.coefficient_mlp(points[:, 1:2])
        count = self.modes.numel()
        real = vector[:, :count]
        imag = vector[:, count:]
        angle = points[:, 0:1] * self.modes.unsqueeze(0)
        multiplicity = torch.ones_like(self.modes)
        if multiplicity.numel() > 1:
            multiplicity[1:] = 2.0
        values = torch.sum(
            multiplicity.unsqueeze(0)
            * (real * torch.cos(angle) - imag * torch.sin(angle)),
            dim=1,
        )
        return values.unsqueeze(1)


def rectangular_field(points, values):
    x = np.unique(points[:, 0])
    t = np.unique(points[:, 1])
    field = values[:, 0].reshape(len(x), len(t)).T
    duplicate = np.allclose(field[:, 0], field[:, -1], rtol=0, atol=1e-12)
    if duplicate:
        x = x[:-1]
        field = field[:, :-1]
    return x, t, field, duplicate


def spline_mlp_weights(t, coefficient_values):
    real_spline = CubicSpline(t, coefficient_values.real, axis=0)
    imag_spline = CubicSpline(t, coefficient_values.imag, axis=0)

    outside_knots = np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float64)
    polynomial_matrix = np.stack(
        (
            -(outside_knots**3),
            3.0 * outside_knots**2,
            -3.0 * outside_knots,
            np.ones_like(outside_knots),
        ),
        axis=0,
    )

    def convert(spline):
        base_polynomial = np.stack(
            (spline.c[3, 0], spline.c[2, 0], spline.c[1, 0], spline.c[0, 0]),
            axis=0,
        )
        outside_weights = np.linalg.solve(polynomial_matrix, base_polynomial)
        hinge_weights = spline.c[0, 1:] - spline.c[0, :-1]
        return np.concatenate((outside_weights, hinge_weights), axis=0)

    feature_knots = np.concatenate((outside_knots, t[1:-1]))
    real_weights = convert(real_spline)
    imag_weights = convert(imag_spline)
    output_weights = np.concatenate((real_weights.T, imag_weights.T), axis=0)
    return feature_knots, output_weights


def save_checkpoint(path, model, configuration):
    torch.save(
        {
            "model": "ExactFourierMLP",
            "configuration": configuration,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        path,
    )


def run(args):
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    dde.config.set_default_float("float64")
    points, values = load_data(args.data, precision="float64")
    x, t, field, duplicate = rectangular_field(points, values)
    modes = np.arange(len(x) // 2 + 1, dtype=np.float64)
    coefficients = np.fft.rfft(field, axis=1) / len(x)
    feature_knots, output_weights = spline_mlp_weights(t, coefficients)
    model = ExactFourierMLP(modes, feature_knots, output_weights).to(device)

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(args.out).expanduser().resolve() / f"{timestamp}-ks-exact-fourier-mlp"
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        **vars(args),
        "data": str(Path(args.data).resolve()),
        "device": str(device),
        "hidden_width": len(feature_knots),
        "output_width": output_weights.shape[0],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "duplicate_periodic_endpoint": duplicate,
    }
    save_checkpoint(run_dir / "weights.pt", model, configuration)
    scripted_buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(model), scripted_buffer)
    (run_dir / "model_scripted.pt").write_bytes(scripted_buffer.getvalue())

    data_metric = prediction_metrics(model, points, values, args.eval_batch_size, device)

    class EvaluationArgs:
        precision = "float64"
        seed = args.seed
        pinn_points = args.pinn_points
        pinn_ic_points = args.pinn_ic_points
        pinn_batch_size = args.pinn_batch_size
        alpha = KS_ALPHA
        beta = KS_BETA
        gamma = KS_GAMMA
        ic_loss_weight = 100.0

    pinn_metric = evaluate_pinn_loss(
        model,
        (np.min(points, axis=0), np.max(points, axis=0)),
        EvaluationArgs,
        device,
    )
    derivative_grid = evaluate_derivative_grid(
        model,
        (np.min(points, axis=0), np.max(points, axis=0)),
        KS_ALPHA, KS_BETA, KS_GAMMA,
        args.derivative_grid_nx, args.derivative_grid_nt,
        args.derivative_batch_size, run_dir, device,
    )
    predictions = []
    with torch.no_grad():
        for start in range(0, len(points), args.eval_batch_size):
            predictions.append(
                model(torch.as_tensor(points[start:start + args.eval_batch_size], device=device)).cpu().numpy()
            )
    prediction = np.vstack(predictions)[:, 0]
    np.savez_compressed(
        run_dir / "predictions.npz",
        x=points[:, 0], t=points[:, 1], exact=values[:, 0], prediction=prediction,
    )
    save_solution_plot(
        run_dir / "solution.png", points, values[:, 0], prediction,
        f"Exact Fourier coefficient MLP, relative L2={data_metric['relative_l2']:.3e}",
    )
    metrics = {
        "data": data_metric,
        "pinn": pinn_metric,
        "derivative_grid": derivative_grid,
        "configuration": configuration,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)
    print(
        f"Exact Fourier MLP: L2={data_metric['relative_l2']:.6e}; "
        f"PDE={pinn_metric['pde_mse']:.6e}; parameters={configuration['parameters']}; "
        f"artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs_data_ks_exact_mlp"))
    parser.add_argument("--pinn-points", type=int, default=20000)
    parser.add_argument("--pinn-ic-points", type=int, default=2048)
    parser.add_argument("--pinn-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--derivative-grid-nx", type=int, default=128)
    parser.add_argument("--derivative-grid-nt", type=int, default=64)
    parser.add_argument("--derivative-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
