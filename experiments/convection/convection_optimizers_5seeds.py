import argparse
import csv
import gc
import json
import os
import sys
import time
from contextlib import contextmanager

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import numpy as np
import torch

import deepxde as dde
from src.pde.convection import Convection1D
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import (
    LossCallback,
    ModelSaverCallback,
    PlotCallback,
    TesterCallback,
)


class Tee:
    def __init__(self, filename, stream):
        self.stream = stream
        self.file = open(filename, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


@contextmanager
def hooked_output(save_path):
    stdout = sys.stdout
    stderr = sys.stderr
    tee_out = Tee(os.path.join(save_path, "log.txt"), stdout)
    tee_err = Tee(os.path.join(save_path, "logerr.txt"), stderr)
    sys.stdout = tee_out
    sys.stderr = tee_err
    try:
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
        tee_out.close()
        tee_err.close()


def add_args(parser):
    parser.add_argument("--name", type=str, default="convection_beta50_optimizers_5seeds")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seeds", type=str, default="123,234,345,456,567,678,789,890,901,1012")
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--loss-weight", type=str, default="")

    parser.add_argument("--iter", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--metric-points", type=int, default=2500)
    parser.add_argument("--out", type=str, default="runs")

    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=2 * np.pi)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=1.0)

    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--soap-lr", type=float, default=3e-4)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)

    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--muon-adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)


def parse_seeds(value):
    seeds = [int(seed.strip()) for seed in value.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def setup_runtime(device, seed):
    dde.config.set_default_float("float32")
    torch.set_default_dtype(torch.float32)
    dde.config.set_random_seed(seed)

    if device == "cpu" or not torch.cuda.is_available():
        torch.set_default_tensor_type(torch.FloatTensor)
        return "cpu"

    cuda_device = f"cuda:{device}"
    torch.cuda.set_device(cuda_device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    return cuda_device


def default_loss_weights(pde, args):
    loss_weights_cli = parse_loss_weight(args)
    if loss_weights_cli is not None:
        return np.array(loss_weights_cli, dtype=float)

    loss_weights = np.ones(pde.num_loss, dtype=float)
    for i, config in enumerate(pde.loss_config):
        loss_type = config.get("type", "")
        if loss_type in ("boundary", "initial", "ic"):
            loss_weights[i] = 100.0
        elif loss_type == "pde":
            loss_weights[i] = 1.0
    return loss_weights


def build_model(args):
    pde = Convection1D(
        beta=args.beta,
        geom=(args.x_min, args.x_max),
        time=(args.t_min, args.t_max),
    )
    layers = [pde.input_dim, *parse_hidden_layers(args), pde.output_dim]
    net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
    loss_weights = default_loss_weights(pde, args)
    model = pde.create_model(net)
    return model, pde, loss_weights


def compile_model(model, optimizer_name, args, loss_weights):
    name = optimizer_name.lower()
    if name == "adam":
        model.compile("adam", lr=args.adam_lr, loss_weights=loss_weights)
        return {"lr": args.adam_lr}

    if name == "soap":
        dde.optimizers.set_SOAP_options(
            beta1=args.soap_beta1,
            beta2=args.soap_beta2,
            precondition_frequency=args.soap_precondition_frequency,
            max_precondition_dim=args.soap_max_precondition_dim,
        )
        model.compile("soap", lr=args.soap_lr, loss_weights=loss_weights)
        return {
            "lr": args.soap_lr,
            "beta1": args.soap_beta1,
            "beta2": args.soap_beta2,
            "precondition_frequency": args.soap_precondition_frequency,
            "max_precondition_dim": args.soap_max_precondition_dim,
        }

    if name == "muon":
        dde.optimizers.set_MUON_options(
            momentum=args.muon_momentum,
            ns_steps=args.muon_ns_steps,
            adam_lr=args.muon_adam_lr,
        )
        model.compile("muon", lr=args.muon_lr, loss_weights=loss_weights)
        return {
            "lr": args.muon_lr,
            "adam_lr": args.muon_adam_lr,
            "momentum": args.muon_momentum,
            "ns_steps": args.muon_ns_steps,
        }

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def final_losses(model):
    loss_train = getattr(model.losshistory, "loss_train", [])
    loss_test = getattr(model.losshistory, "loss_test", [])
    train = np.asarray(loss_train[-1], dtype=float).tolist() if loss_train else []
    test = np.asarray(loss_test[-1], dtype=float).tolist() if loss_test else []
    return {
        "loss_train": train,
        "loss_test": test,
        "loss_train_sum": float(np.sum(train)) if train else float("nan"),
        "loss_test_sum": float(np.sum(test)) if test else float("nan"),
    }


def reference_metrics(model, pde, metric_points):
    if pde.ref_sol is None:
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "max_error": float("nan"),
            "l2re": float("nan"),
        }

    sample_func = getattr(model.data.geom, "uniform_points", None)
    if sample_func is None:
        sample_func = model.data.geom.random_points
    x = sample_func(metric_points, boundary=True)
    y_true = pde.ref_sol(x)
    y_pred = model.predict(x)
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    max_error = float(np.max(np.abs(error)))
    denominator = float(np.sqrt(np.mean(y_true**2)))
    l2re = float(np.sqrt(mse) / denominator) if denominator > 0 else float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "l2re": l2re,
    }


def train_one(optimizer_name, seed, args, root_save_path):
    save_path = os.path.join(root_save_path, optimizer_name.lower(), f"seed_{seed}")
    os.makedirs(save_path, exist_ok=True)

    with hooked_output(save_path):
        active_device = setup_runtime(args.device, seed)
        print(f"optimizer = {optimizer_name}")
        print(f"seed = {seed}")
        print(f"device = {active_device}")

        model, pde, loss_weights = build_model(args)
        print("num_loss =", pde.num_loss)
        print("num_pde  =", pde.num_pde)
        for i, config in enumerate(pde.loss_config):
            print(i, config["type"], config["name"])

        optimizer_params = compile_model(model, optimizer_name, args, loss_weights)
        json.dump(
            {
                "optimizer": optimizer_name,
                "optimizer_params": optimizer_params,
                "seed": seed,
                "iterations": args.iter,
                "hidden_layers": args.hidden_layers,
                "loss_weights": loss_weights.tolist(),
                "beta": args.beta,
                "geom": [args.x_min, args.x_max],
                "time": [args.t_min, args.t_max],
            },
            open(os.path.join(save_path, "config.json"), "w", encoding="utf-8"),
            indent=4,
        )

        start = time.time()
        model.train(
            iterations=args.iter,
            display_every=args.log_every,
            callbacks=[
                TesterCallback(log_every=args.log_every),
                PlotCallback(log_every=args.plot_every, fast=True),
                LossCallback(verbose=True),
                ModelSaverCallback(
                    total_iterations=args.iter,
                    n_save_models=args.n_save_models,
                ),
            ],
            model_save_path=save_path,
        )
        elapsed = time.time() - start
        metrics = reference_metrics(model, pde, args.metric_points)
        result = {
            "optimizer": optimizer_name,
            "seed": seed,
            "save_path": save_path,
            "elapsed_seconds": elapsed,
            **metrics,
            **final_losses(model),
        }
        json.dump(
            result,
            open(os.path.join(save_path, "result.json"), "w", encoding="utf-8"),
            indent=4,
        )
        print("result =", result)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def t_critical_95(n):
    by_df = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        24: 2.064,
        29: 2.045,
    }
    if n <= 1:
        return float("nan")
    df = n - 1
    if df in by_df:
        return by_df[df]
    return 1.96


def summarize_metric(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_half_width": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    t_value = t_critical_95(n)
    ci95_half_width = float(t_value * std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_half_width": ci95_half_width,
        "ci95_low": mean - ci95_half_width,
        "ci95_high": mean + ci95_half_width,
    }


def aggregate_results(results):
    metrics = [
        "l2re",
        "mse",
        "mae",
        "max_error",
        "loss_train_sum",
        "loss_test_sum",
        "elapsed_seconds",
    ]
    optimizers = []
    for result in results:
        if result["optimizer"] not in optimizers:
            optimizers.append(result["optimizer"])

    rows = []
    for optimizer_name in optimizers:
        optimizer_results = [
            result for result in results if result["optimizer"] == optimizer_name
        ]
        row = {"optimizer": optimizer_name, "n": len(optimizer_results)}
        for metric in metrics:
            summary = summarize_metric(
                [result.get(metric, float("nan")) for result in optimizer_results]
            )
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
            row[f"{metric}_ci95_half_width"] = summary["ci95_half_width"]
            row[f"{metric}_ci95_low"] = summary["ci95_low"]
            row[f"{metric}_ci95_high"] = summary["ci95_high"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_method_table(rows):
    print("\nMethod summary, 95% confidence interval:")
    print("optimizer,n,l2re_mean,l2re_std,l2re_ci95_low,l2re_ci95_high")
    for row in rows:
        print(
            f"{row['optimizer']},{row['n']},"
            f"{row['l2re_mean']:.10e},{row['l2re_std']:.10e},"
            f"{row['l2re_ci95_low']:.10e},{row['l2re_ci95_high']:.10e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convection1D benchmark with Adam, SOAP, and Muon over five seeds."
    )
    add_args(parser)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    optimizers = ["Adam", "SOAP", "Muon"]
    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    root_save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(root_save_path, exist_ok=True)

    json.dump(
        {
            "script": __file__,
            "optimizers": optimizers,
            "seeds": seeds,
            "args": vars(args),
        },
        open(os.path.join(root_save_path, "config.json"), "w", encoding="utf-8"),
        indent=4,
    )

    results = []
    for optimizer_name in optimizers:
        for seed in seeds:
            results.append(train_one(optimizer_name, seed, args, root_save_path))

    json.dump(
        results,
        open(os.path.join(root_save_path, "summary.json"), "w", encoding="utf-8"),
        indent=4,
    )
    write_csv(os.path.join(root_save_path, "runs.csv"), results)

    method_summary = aggregate_results(results)
    json.dump(
        method_summary,
        open(os.path.join(root_save_path, "methods_summary.json"), "w", encoding="utf-8"),
        indent=4,
    )
    write_csv(os.path.join(root_save_path, "methods_summary.csv"), method_summary)
    print_method_table(method_summary)
    print(f"Saved summary to {os.path.join(root_save_path, 'summary.json')}")
    print(f"Saved run table to {os.path.join(root_save_path, 'runs.csv')}")
    print(f"Saved method table to {os.path.join(root_save_path, 'methods_summary.csv')}")


if __name__ == "__main__":
    main()
