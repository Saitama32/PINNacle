import argparse
import os
import sys
import time

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import torch

import deepxde as dde
from src.pde.burgers import Burgers1D
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import (
    LossCallback,
    ModelSaverCallback,
    PlotCallback,
    TesterCallback,
)


def add_args(parser):
    parser.add_argument("--name", type=str, default="burgers1d_muon")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--loss-weight", type=str, default="")
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--iter", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs")


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


def setup_runtime(device, seed):
    dde.config.set_default_float("float32")
    torch.set_default_dtype(torch.float32)
    if seed is not None:
        dde.config.set_random_seed(seed)

    if device == "cpu" or not torch.cuda.is_available():
        torch.set_default_tensor_type(torch.FloatTensor)
        return "cpu"

    cuda_device = f"cuda:{device}"
    torch.cuda.set_device(cuda_device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    return cuda_device


def main():
    parser = argparse.ArgumentParser(description="Burgers1D PINN with Muon.")
    add_args(parser)
    args = parser.parse_args()

    active_device = setup_runtime(args.device, args.seed)
    print(f"device = {active_device}")

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    pde = Burgers1D()
    print("num_loss =", pde.num_loss)
    print("num_pde  =", pde.num_pde)
    for i, config in enumerate(pde.loss_config):
        print(i, config["type"], config["name"])

    layers = [pde.input_dim, *parse_hidden_layers(args), pde.output_dim]
    net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
    loss_weights = default_loss_weights(pde, args)

    dde.optimizers.set_MUON_options(
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        adam_lr=args.adam_lr,
    )

    model = pde.create_model(net)
    model.compile("muon", lr=args.lr, loss_weights=loss_weights)
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


if __name__ == "__main__":
    main()
