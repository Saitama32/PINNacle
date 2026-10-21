import os
import sys

os.environ["DDEBACKEND"] = "pytorch"

from comet_ml import start
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
api_key = os.getenv("COMET_API_KEY")

experiment = start(
    api_key=api_key,
    project_name="optuna-poisson-2d-classic-fixed-chain_compare",
    workspace="saitama32",
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(PROJECT_ROOT)

import argparse
import random
import time

import deepxde as dde
import numpy as np
import torch
from deepxde.optimizers.config import set_PSO_options

from src.pde.poisson import Poisson2D_Classic
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import LossCallback, PlotCallback, TesterCallback


dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)


OPTUNA_CHAIN = [
    {"optimizer": "PSO", "epochs": 300, "lr": 0.0},
    {"optimizer": "PSO", "epochs": 300, "lr": 1e-3},
    {"optimizer": "PSO", "epochs": 300, "lr": 0.0},
    {"optimizer": "Adam", "epochs": 2500, "lr": 1e-3},
    {"optimizer": "LBFGS", "epochs": 1500, "lr": 1.0},
]

experiment.log_parameters(
    {
        "param": "v_1",
        "description": "poisson2d_classic_fixed_optuna_chain_comparison",
        "optimizer_chain": str(OPTUNA_CHAIN),
        "total_epochs": sum(stage["epochs"] for stage in OPTUNA_CHAIN),
    }
)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dde.config.set_random_seed(seed)


def build_get_model_poisson2d_classic(hidden_layers):
    def get_model():
        pde = Poisson2D_Classic()
        layers = (
            [pde.input_dim]
            + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers))
            + [pde.output_dim]
        )
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, config in enumerate(pde.loss_config):
            loss_type = config.get("type", "")
            if loss_type in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0
            else:
                loss_weights[i] = 1.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model


def build_optimizer(stage, model):
    optimizer_name = stage["optimizer"].lower()
    learning_rate = float(stage["lr"])

    if optimizer_name == "adam":
        return torch.optim.Adam(model.net.parameters(), lr=learning_rate)
    if optimizer_name == "lbfgs":
        return torch.optim.LBFGS(
            model.net.parameters(),
            lr=learning_rate,
            line_search_fn="strong_wolfe",
            max_iter=10,
        )
    if optimizer_name == "pso":
        set_PSO_options(lr=learning_rate)
        return "PSO"
    raise ValueError(f"Unknown optimizer: {stage['optimizer']}")


def run_fixed_chain(seed, args):
    set_global_seed(seed)

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}-seed-{seed}")
    os.makedirs(save_path, exist_ok=True)

    get_model = build_get_model_poisson2d_classic(args.hidden_layers)
    model, loss_weights = get_model()

    tester_callback = TesterCallback(
        log_every=args.log_every,
        reset_on_train_end=False,
    )
    callbacks = [
        tester_callback,
        PlotCallback(log_every=args.plot_every, fast=True),
        LossCallback(verbose=True),
    ]

    for stage_index, stage in enumerate(OPTUNA_CHAIN, start=1):
        optimizer = build_optimizer(stage, model)
        epochs = int(stage["epochs"])

        print(
            f"\nSeed {seed}, chain stage {stage_index}/{len(OPTUNA_CHAIN)}: "
            f"{stage['optimizer']} for {epochs} epochs, lr={stage['lr']}"
        )

        model.compile(optimizer, loss_weights=loss_weights)
        model.optimizer = optimizer
        model.train(
            iterations=epochs,
            display_every=args.log_every,
            callbacks=callbacks,
            model_save_path=save_path,
            save_model=False,
        )

    best_metrics = tester_callback.get_best_metrics()
    if best_metrics:
        print(f"Best metrics for seed {seed}: {best_metrics}")
        experiment.log_metrics(best_metrics, step=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="poisson2d_classic_optuna_fixed")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--out", type=str, default="runs_single")
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\nStarting fixed Optuna chain with seed = {seed}")
        run_fixed_chain(seed, args)


if __name__ == "__main__":
    main()
