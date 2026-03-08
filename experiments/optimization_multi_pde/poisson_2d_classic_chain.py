# run_burgers1d_rl.py
import os, sys
os.environ["DDEBACKEND"] = "pytorch"
from comet_ml import start
from comet_ml.integration.pytorch import log_model

proj_name = "rlpinn-poisson-2d-classic-optimization"
experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name=proj_name,
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


from src.pde.poisson import Poisson2D_Classic
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback, ModelSaverCallback
from rl_trainer import train_process_rl

experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "optimization_poisson_2d_classic_basic_RL_optimizer"
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



def build_get_model_poisson2d_classic(hidden_layers: str):
    """
    Возвращает функцию get_model() как в benchmark_xxx.py, но только для Burgers1D. :contentReference[oaicite:1]{index=1}
    """

    def get_model():
        pde = Poisson2D_Classic()

        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")

        net = net.float()

                # loss weights
        loss_weights = np.ones(pde.num_loss, dtype=float)

        for i, c in enumerate(pde.loss_config):
            t = c.get("type", "")
            if t in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0
            elif t == "pde":
                loss_weights[i] = 1.0
            else:
                # на всякий случай: оставляем 1 для прочих типов (например, gepinn/data/regularization)
                loss_weights[i] = 1.0


        model = pde.create_model(net)
        # model.compile(opt, loss_weights=loss_weights)

        # ВАЖНО: ModelSaverCallback здесь нужен именно для RL, чтобы после каждого chunk получать список моделей
        # RL-тренер добавит свой saver на каждый шаг, но базовый можно оставить для “обычных” логов, если хочешь.

        return model, loss_weights

    return get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="poisson2d_classic_rl")
    parser.add_argument("--device", type=str, default="0")  # "cpu" or cuda index
    parser.add_argument("--seed", type=int, default=1234)

    # модель/обычный train
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
    parser.add_argument("--log_key", type=str2bool, nargs="?", const=True, default=False)

    # куда писать
    parser.add_argument("--out", type=str, default="runs_single")

    args = parser.parse_args()

    # --- папка эксперимента ---
    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    # --- get_model / train_args как в benchmark_xxx.py :contentReference[oaicite:5]{index=5} ---
    get_model = build_get_model_poisson2d_classic(args.hidden_layers)
    get_model_rec = build_get_model_poisson2d_classic(args.hidden_layers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_args = {
        # В RL-режиме iterations тут не главный (чанки задаёт action["epochs"]),
        # но display_every/callbacks используются.
        "iterations": 1,
        "display_every": args.log_every,
        "callbacks": [
            TesterCallback(log_every=args.log_every),
            PlotCallback(log_every=args.plot_every, fast=True),
            LossCallback(verbose=True),
        ],
        "n_trajectories": 1000,
        "n_save_models": 10,
        "operator_coeff": 1,
        "bnd_coeff": 1,

    }
    optimizers = {
        'Adam':{
            'lr':[1e-2, 1e-3, 1e-4],
            'epochs':[100, 1000, 2500]
            # 'epochs':[500, 500, 500]
        },
        'LBFGS':{
            'lr':[1, 5e-1, 1e-1],
            'epochs':[100, 500, 1500]
        },
        'PSO':{
            'lr':[0.0, 1e-3, 1e-4],
            'epochs':[100, 200, 300]
        },
    }

    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
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
        "device": device
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
        "log_key": args.log_key
    }

    loss_surface_params = {
        "loss_types": ["loss_total", "loss_oper", "loss_bnd"],
        "every_nth": 1,
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15
        ],
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
        "img_dir": '',
        "dde_pde_model": get_model_rec
    }

    rl_agent_params = {
        "n_save_models": 10,
        "n_trajectories": 1000,
        "tolerance": 0.000063, 
        "prev_tol": 0,
        "stuck_threshold": 10,  # Число эпох без значительного изменения прогресса
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit" : 2000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "agent_min_buffer": 32,
        "agent_update_iters": 5,
        "lr": 1e-3,
        "exp": experiment,
        "log_key": args.log_key,
        "proj_name": "rlpinn-poisson-2d-classic-farm-transitions"
    }
    print(args.log_key)
    # backup_params = {
    #     "experiment_key" : "b0dae86c42924e4484b8bd194e2d58d9",
    # }
    backup_params = None

    experiment.log_parameters(rl_agent_params)
    # experiment.log_parameters(backup_params)
    # --- вызов train_process_rl ---

    # --- вызов train_process_rl ---

    data = dill.dumps((get_model, train_args, optimizers, AE_model_params, AE_train_params, loss_surface_params))
    train_process_rl(data=data, save_path=save_path, device=0, seed=args.seed, rl_agent_params=rl_agent_params)

if __name__ == "__main__":
    main()

