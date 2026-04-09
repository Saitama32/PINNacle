# run_ns2d_liddriven_rl.py
import os, sys
os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_poissonnd_comparison",
  workspace="saitama32"
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
import time
import argparse
import dill
import numpy as np


import torch
import deepxde as dde

from src.pde.poisson import PoissonND
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from rl_trainer import train_process_rl



experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "comparison_poissonnd_rl_optimizer",
})

def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"true", "True", "1", "yes", "y", "on"}:
        return True
    if val in {"false", "False","0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def build_get_model_poissonnd(hidden_layers: str, **pde_kwargs):
    def get_model():
        pde = PoissonND(**pde_kwargs)

        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, c in enumerate(pde.loss_config):
            t = c.get("type", "")
            if t in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0
            elif t == "pde":
                loss_weights[i] = 1.0
            else:
                loss_weights[i] = 1.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model


def main(seed_override=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="poissonnd_rl")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    # RL config
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--n-steps-max", type=int, default=1000)
    parser.add_argument("--state-h", type=int, default=26)
    parser.add_argument("--state-w", type=int, default=26)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--log_key_for_new_state", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--exp_key", type=str, default="7f7a91cef55d4aeba0e509024977456b")

    parser.add_argument("--out", type=str, default="runs_single")

    parser.add_argument("--dim", type=int, default=5, help="Problem dimensionality")
    parser.add_argument("--length", type=float, default=1.0, help="Hypercube side length")

    args = parser.parse_args()
    if seed_override is not None:
        args.seed = int(seed_override)

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    pde_kwargs = dict(
        dim=args.dim,
        len=args.length
    )

    get_model = build_get_model_poissonnd(args.hidden_layers, **pde_kwargs)
    get_model_rec = build_get_model_poissonnd(args.hidden_layers, **pde_kwargs)

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
        "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1000]},
        "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
    }

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
        "log_key": args.log_key_for_new_state,
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
    }

    rl_agent_params = {
        "n_save_models": args.n_save_models,
        "n_trajectories": args.n_trajectories,
        "tolerance": 0.000433647801401093,
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
        "lr": 1e-3,
        "exp": experiment,
        "log_key": False
    }

    comparison_params = {
        "seed": args.seed,
        "total_epochs": 7000,
        "experiment_key": args.exp_key,
        # "multi_pde_comparison": True,
    }

    experiment.log_parameters(rl_agent_params)
    experiment.log_parameters(comparison_params)

    data = dill.dumps((get_model, train_args, optimizers, AE_model_params, AE_train_params, loss_surface_params, comparison_params))
    train_process_rl(data=data, save_path=save_path, device=args.device, seed=args.seed, rl_agent_params=rl_agent_params, comparison_params=comparison_params,)
    # --- вызов train_process_rl ---



if __name__ == "__main__":
    # список сидов для экспериментов
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]   # можно расширить список
    # seeds = [456, 567, 678, 789, 890, 901, 1012]   # можно расширить список
    # seeds = [789, 890, 901, 1012]   # можно расширить список
    # seeds = [901, 1012]   # можно расширить список


    for seed in seeds:
        print(f"\n🔹 Запуск эксперимента с seed = {seed}")
        main(seed_override=seed)
