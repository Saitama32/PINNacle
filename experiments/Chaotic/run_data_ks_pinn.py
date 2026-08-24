"""PINN fine-tuning of a data-pretrained Kuramoto--Sivashinsky RWF MLP."""

from __future__ import annotations

import argparse
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

import numpy as np
import torch
import deepxde as dde

from experiments.Chaotic.run_data_ks import (
    KS_ALPHA,
    KS_BETA,
    KS_GAMMA,
    evaluate_derivative_grid,
    evaluate_pinn_loss,
    load_checkpoint,
    load_data,
    prediction_metrics,
    save_checkpoint,
    save_solution_plot,
    train_pinn_stage,
)


def resolve_model_path(path: str) -> Path:
    model_path = Path(path).expanduser().resolve()
    if model_path.is_dir():
        model_path = model_path / "weights_best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return model_path


def resolve_data_path(explicit_path, source_dir: Path) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    config_path = source_dir / "run_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file_obj:
            configured = json.load(file_obj).get("data")
        if configured:
            return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "ref" / "Kuramoto_Sivashinsky.dat"


def validation_subset(points, values, source_dir: Path):
    split_path = source_dir / "split_indices.npz"
    if split_path.exists():
        with np.load(split_path) as split:
            indices = np.asarray(split["test"], dtype=np.int64)
        if len(indices) and int(np.max(indices)) < len(points):
            return points[indices], values[indices]
    return points, values


def run(args) -> Path:
    if args.n_iter_pinn <= 0:
        raise ValueError("n_iter_pinn must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path = resolve_model_path(args.model)
    source_dir = model_path.parent
    network, metadata = load_checkpoint(model_path, device=device)
    source_precision = metadata.get("precision", "float32")
    args.precision = args.pinn_precision
    torch_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.precision]
    network = network.to(device=device, dtype=torch_dtype)
    dde.config.set_default_float(args.precision)
    metadata = dict(metadata)
    metadata["precision"] = args.precision
    args.alpha = float(metadata.get("alpha", KS_ALPHA))
    args.beta = float(metadata.get("beta", KS_BETA))
    args.gamma = float(metadata.get("gamma", KS_GAMMA))
    args.adam_epsilon = 1e-4 if args.precision == "float16" else 1e-8
    args.soap_epsilon = 1e-4 if args.precision == "float16" else 1e-8
    args.muon_adam_epsilon = 1e-4 if args.precision == "float16" else 1e-10

    data_path = resolve_data_path(args.data, source_dir)
    points, values = load_data(data_path, precision=args.precision)
    validation_points, validation_values = validation_subset(points, values, source_dir)
    numpy_dtype = np.float64 if args.precision == "float64" else np.float32
    lower = np.asarray(metadata["input_min"], dtype=numpy_dtype)
    upper = lower + np.asarray(metadata["input_scale"], dtype=numpy_dtype)

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_dir = Path(args.out).expanduser().resolve() / (
        f"{timestamp}-{source_dir.name}-pinn-{args.pinn_optimizer}-{args.precision}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    save_checkpoint(run_dir / "weights_initial.pt", network, metadata)
    resolved = vars(args).copy()
    resolved.update(
        model=str(model_path),
        source_run=str(source_dir),
        data=str(data_path),
        device=str(device),
        source_model_precision=source_precision,
        model_metadata=metadata,
    )
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(resolved, file_obj, indent=2, sort_keys=True)

    initial_data = prediction_metrics(
        network, points, values, args.eval_batch_size, device
    )
    initial_pinn = evaluate_pinn_loss(network, (lower, upper), args, device)
    fine_tune = train_pinn_stage(
        network,
        (lower, upper),
        validation_points,
        validation_values,
        args,
        device,
        metadata,
        run_dir,
    )
    final_data = prediction_metrics(network, points, values, args.eval_batch_size, device)
    final_pinn = evaluate_pinn_loss(network, (lower, upper), args, device)
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

    predictions = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(points), args.eval_batch_size):
            batch = torch.as_tensor(points[start : start + args.eval_batch_size], device=device)
            predictions.append(network(batch).cpu().numpy())
    prediction = np.vstack(predictions)[:, 0]
    np.savez_compressed(
        run_dir / "predictions.npz",
        x=points[:, 0],
        t=points[:, 1],
        exact=values[:, 0],
        prediction=prediction,
    )
    save_solution_plot(
        run_dir / "solution.png",
        points,
        values[:, 0],
        prediction,
        f"Data-pretrained + PINN KS, relative L2={final_data['relative_l2']:.3e}",
    )
    metrics = {
        "source_model": str(model_path),
        "initial_data": initial_data,
        "initial_pinn_loss": initial_pinn,
        "final_data": final_data,
        "final_pinn_loss": final_pinn,
        "pinn_finetune": fine_tune,
        "derivative_grid": derivative_metric,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)
    print(
        f"Finished PINN fine-tuning: best step={fine_tune['best_iteration']}, "
        f"relative L2={final_data['relative_l2']:.6e}, "
        f"weighted PINN loss={final_pinn['pinn_loss_weighted']:.6e}; "
        f"artifacts={run_dir}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune a data-pretrained KS RWF model using PDE and IC losses."
    )
    parser.add_argument("--model", default=r"C:\Users\Рустам\Documents\GitHub\PINNacle\runs_data_ks\08.24-01.58.59-ks-data-rwf-soap-float32-lr-cosine\weights_best.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs_data_ks_pinn"))
    parser.add_argument(
        "--pinn-precision",
        choices=["float32", "float64"],
        default="float64",
        help="Precision used after loading the pretrained model and during PINN fine-tuning.",
    )
    parser.add_argument("--n-iter-pinn", type=int, default=8000)
    parser.add_argument(
        "--pinn-optimizer", choices=["adam", "rmsprop", "muon", "soap"], default="soap"
    )
    parser.add_argument("--pinn-lr", type=float, default=1e-4)
    parser.add_argument("--pinn-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--pinn-lr-scheduler",
        choices=["none", "exponential", "cosine", "step"],
        default="cosine",
    )
    parser.add_argument("--pinn-lr-decay-steps", type=int, default=1000)
    parser.add_argument("--pinn-lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--pinn-lr-min", type=float, default=5e-4)
    parser.add_argument("--pinn-train-domain-points", type=int, default=256)
    parser.add_argument("--pinn-train-ic-points", type=int, default=256)
    parser.add_argument("--pinn-train-log-every", type=int, default=100)
    parser.add_argument("--pinn-grad-clip", type=float, default=1.0)
    parser.add_argument("--ic-loss-weight", type=float, default=1.0)
    parser.add_argument("--pinn-points", type=int, default=20000)
    parser.add_argument("--pinn-ic-points", type=int, default=2048)
    parser.add_argument("--pinn-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--derivative-grid-nx", type=int, default=128)
    parser.add_argument("--derivative-grid-nt", type=int, default=64)
    parser.add_argument("--derivative-batch-size", type=int, default=512)
    parser.add_argument("--no-derivative-plots", dest="derivative_plots", action="store_false")
    parser.set_defaults(derivative_plots=True)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)
    parser.add_argument("--soap-bias-correction", action="store_true", default=True)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", action="store_true", default=True)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-adam-weight-decay", type=float, default=0.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
