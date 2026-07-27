import os
import sys

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv
from comet_ml import start

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import argparse
import dill
import time

import deepxde as dde
import numpy as np
import torch

from RL.rl_utils.load_buffer.load_exps_from_comet import (
    rebuild_single_comet_experiment_transitions,
)
from rl_trainer import train_process_rl
from src.pde.chaotic import KuramotoSivashinskyEquation
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import LossCallback, PlotCallback, TesterCallback


dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)

TARGET_PROJECT_NAME = "rlpinn_ks_rebuild_buffer_3_dim"
DEFAULT_SOURCE_PROJECT_NAME = "rlpinn-kuramoto-sivashinsky-tolerance"
path_to_trans = os.path.join("transitions_rebuilt", "kuramoto_sivashinsky")


def build_get_model_kuramoto_sivashinsky(hidden_layers: str):
    def get_model():
        pde = KuramotoSivashinskyEquation()

        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, c in enumerate(pde.loss_config):
            loss_type = c.get("type", "")
            if loss_type in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0
            elif loss_type == "pde":
                loss_weights[i] = 1.0
            else:
                loss_weights[i] = 1.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="kuramoto_sivashinsky_rebuild")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--n-trajectories", type=int, default=1000)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs_single")
    parser.add_argument("--rebuild-buffer-only", action="store_true", default=True)
    parser.add_argument("--source-project-name", type=str, default=DEFAULT_SOURCE_PROJECT_NAME)
    parser.add_argument("--source-experiment-key", type=str, default=None)

    args = parser.parse_args()
    if args.rebuild_buffer_only and not args.source_experiment_key:
        parser.error("--source-experiment-key is required with --rebuild-buffer-only")

    experiment = start(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=TARGET_PROJECT_NAME,
        workspace=os.getenv("COMET_WORKSPACE", "saitama32"),
    )

    experiment.log_parameters({
        "param": "v_1",
        "reward_function": "v_2",
        "description": (
            "rebuild_kuramoto_sivashinsky_buffer"
            if args.rebuild_buffer_only
            else "optimization_kuramoto_sivashinsky_rl_optimizer"
        ),
        "rebuild_buffer_only": args.rebuild_buffer_only,
        "source_project_name": args.source_project_name,
        "source_experiment_key": args.source_experiment_key,
        "target_project_name": TARGET_PROJECT_NAME,
    })

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    get_model = build_get_model_kuramoto_sivashinsky(args.hidden_layers)
    get_model_rec = build_get_model_kuramoto_sivashinsky(args.hidden_layers)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_args = {
        "iterations": 1,
        "display_every": args.log_every,
        "callbacks": [
            TesterCallback(log_every=args.log_every),
            PlotCallback(log_every=args.plot_every, fast=True),
            LossCallback(verbose=True),
        ],
        "n_trajectories": args.n_trajectories,
        "n_save_models": args.n_save_models,
        "operator_coeff": 1,
        "bnd_coeff": 1,
    }

    optimizers = {
        "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
        "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1500]},
        "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
    }

    latent_dim = 3

    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [991, 125, 15],
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "every_nth": 1,
        "grid_step": 0.1,
        "d_max_latent": 2,
        "anchor_mode": "circle",
        "rec_weight": 10000.0,
        "anchor_weight": 0.0,
        "lastzero_weight": 0.0,
        "polars_weight": 0.0,
        "wellspacedtrajectory_weight": 0.0,
        "gridscaling_weight": 0.0,
        "latent_dim": latent_dim,
        "device": device,
    }

    AE_train_params = {
        "first_RL_epoch_AE_params": {
            "epochs": 10000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "other_RL_epoch_AE_params": {
            "epochs": 20000,
            "patience_scheduler": 4000,
            "cosine_scheduler_patience": 1200,
        },
        "batch_size": 32,
        "every_epoch": 100,
        "learning_rate": 5e-4,
        "resume": True,
        "finetune_AE_model": False,
        "log_key": True,
    }

    loss_surface_params = {
        "loss_types": ["loss_total", "loss_oper", "loss_bnd"],
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [991, 125, 15],
        "batch_size": 32,
        "num_models": None,
        "from_last": False,
        "prefix": "model-",
        "loss_name": "loss_total",
        "x_range": [-1.25, 1.25, 25],
        "vmax": -1.0,
        "vmin": -1.0,
        "vlevel": 30.0,
        "key_models": None,
        "key_modelnames": None,
        "density_type": "CKA",
        "density_p": 2,
        "density_vmax": -1,
        "density_vmin": -1,
        "colorFromGridOnly": True,
        "img_dir": "",
        "dde_pde_model": get_model_rec,
        "latent_dim": latent_dim,
    }

    if args.rebuild_buffer_only:
        rebuild_single_comet_experiment_transitions(
            source_project_name=args.source_project_name,
            source_experiment_key=args.source_experiment_key,
            target_experiment=experiment,
            output_dir=path_to_trans,
            AE_model_params=AE_model_params,
            AE_train_params=AE_train_params,
            loss_surface_params=loss_surface_params,
        )
        return

    rl_agent_params = {
        "n_save_models": args.n_save_models,
        "n_trajectories": args.n_trajectories,
        "tolerance": 1.67,
        "prev_tol": 0,
        "use_tol": False,
        "stuck_threshold": 10,
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit": 2000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "agent_min_buffer": 32,
        "agent_update_iters": 5,
        "lr": args.lr,
        "exp": experiment,
        "log_key": True,
        "proj_name": args.source_project_name,
    }

    experiment.log_parameters(rl_agent_params)

    data = dill.dumps((get_model, train_args, optimizers, AE_model_params, AE_train_params, loss_surface_params))
    train_process_rl(data=data, save_path=save_path, device=args.device, seed=args.seed, rl_agent_params=rl_agent_params)


if __name__ == "__main__":
    main()
