import os, sys

os.environ["DDEBACKEND"] = "pytorch"
from comet_ml import start
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
api_key = os.getenv("COMET_API_KEY")

experiment = start(
    api_key=api_key,
    project_name="pinn-reaction-soap-muon-comparison",
    workspace="saitama32",
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(PROJECT_ROOT)
import argparse
import gc
import json
import random
import time

import deepxde as dde
import numpy as np
import torch

from src.pde.reaction import Reaction1D
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import LossCallback, PlotCallback, TesterCallback


SNAPSHOT_EPOCHS = (7000, 20000)
DEFAULT_LR = {"soap": 1e-3, "muon": 1e-2}

experiment.log_parameters({
    "param": "v_1",
    "description": "classic_reaction_rho_soap_muon_comparison",
    "total_epochs": 20000,
    "snapshot_epochs": list(SNAPSHOT_EPOCHS),
})


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dde.config.set_random_seed(seed)


def build_get_model_reaction(hidden_layers: str, rho: float):
    def get_model():
        pde = Reaction1D(rho=rho)
        layers = [
            pde.input_dim,
            *parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)),
            pde.output_dim,
        ]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()

        loss_weights = np.ones(pde.num_loss, dtype=np.float32)
        for i, config in enumerate(pde.loss_config):
            if config.get("type", "") in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model


class SnapshotCallback(dde.callbacks.Callback):
    def __init__(self, save_path, tester_callback, seed, optimizer, rho, upload):
        super().__init__()
        self.save_path = save_path
        self.tester_callback = tester_callback
        self.seed = seed
        self.optimizer = optimizer
        self.rho = rho
        self.upload = upload
        self.snapshots = {}

    def on_epoch_end(self):
        epoch = int(self.model.train_state.epoch)
        if epoch not in SNAPSHOT_EPOCHS:
            return

        metrics = {
            "train_loss_total": float(np.sum(self.model.train_state.loss_train)),
            "test_loss_total": float(np.sum(self.model.train_state.loss_test)),
        }
        tester = self.tester_callback
        if tester.indexes and int(tester.indexes[-1]) == epoch:
            metrics.update({
                "mse": float(tester.mses[-1]),
                "mae": float(tester.maes[-1]),
                "l1re": float(tester.l1res[-1]),
                "l2re": float(tester.l2res[-1]),
                "bc_mse": float(tester.bc_mses[-1]),
                "ic_mse": float(tester.ic_mses[-1]),
            })

        checkpoint_path = os.path.join(self.save_path, f"model_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "seed": self.seed,
            "optimizer": self.optimizer,
            "rho": self.rho,
            "metrics": metrics,
            "model_state_dict": self.model.net.state_dict(),
        }, checkpoint_path)
        self.snapshots[str(epoch)] = {
            "checkpoint": os.path.basename(checkpoint_path),
            "metrics": metrics,
        }

        experiment.log_metrics(
            {f"epoch_{epoch}_{name}": value for name, value in metrics.items()},
            step=self.seed,
        )
        if self.upload:
            experiment.log_asset(
                checkpoint_path,
                file_name=f"reaction_rho{self.rho:g}_{self.optimizer}_seed{self.seed}_epoch{epoch}.pt",
                step=self.seed,
            )


def main(seed_override=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="reaction_soap_muon")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--optimizer", choices=["soap", "muon"], required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=7000)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--upload-checkpoints", action="store_true")
    parser.add_argument("--out", type=str, default="runs_single/soap_muon")
    args = parser.parse_args()
    if seed_override is not None:
        args.seed = int(seed_override)
    if args.rho <= 0:
        parser.error("--rho must be positive")
    if args.epochs != 20000:
        parser.error("--epochs must be 20000")

    args.lr = args.lr if args.lr is not None else DEFAULT_LR[args.optimizer]
    set_global_seed(args.seed)
    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(
        args.out,
        f"{date_str}-{args.name}-{args.optimizer}-rho{args.rho:g}-seed{args.seed}",
    )
    os.makedirs(save_path, exist_ok=True)

    get_model = build_get_model_reaction(args.hidden_layers, args.rho)
    model, loss_weights = get_model()
    model.compile(args.optimizer, lr=args.lr, loss_weights=loss_weights)

    tester_callback = TesterCallback(log_every=args.log_every)
    snapshot_callback = SnapshotCallback(
        save_path, tester_callback, args.seed, args.optimizer, args.rho,
        args.upload_checkpoints,
    )
    callbacks = [
        tester_callback,
        snapshot_callback,
        PlotCallback(log_every=args.plot_every, fast=True),
        LossCallback(verbose=True),
    ]

    experiment.set_name(f"reaction-rho{args.rho:g}-{args.optimizer}")
    experiment.log_parameters({
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "rho": args.rho,
        "hidden_layers": args.hidden_layers,
    })
    model.train(
        iterations=args.epochs,
        display_every=args.log_every,
        callbacks=callbacks,
        model_save_path=save_path,
        save_model=False,
    )

    best_metrics = tester_callback.get_best_metrics()
    if best_metrics:
        experiment.log_metrics(best_metrics, step=args.seed)
    result = {
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "rho": args.rho,
        "snapshots": snapshot_callback.snapshots,
        "best_metrics": best_metrics,
    }
    result_path = os.path.join(save_path, "summary.json")
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    experiment.log_asset(result_path, step=args.seed)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]

    for seed in seeds:
        print(f"\nStarting experiment with seed = {seed}")
        main(seed_override=seed)

    experiment.end()
