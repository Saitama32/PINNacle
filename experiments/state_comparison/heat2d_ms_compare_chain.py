import os
os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from comet_ml import start
from dotenv import load_dotenv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
api_key = os.getenv("COMET_API_KEY")

_state_type_for_comet = "unknown"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--state-type" and _i + 1 < len(sys.argv):
        _state_type_for_comet = sys.argv[_i + 1]
        break
    if _arg.startswith("--state-type="):
        _state_type_for_comet = _arg.split("=", 1)[1]
        break

experiment = start(
    api_key=api_key,
    project_name=f"rlpinn_heat2d_multiscale_{_state_type_for_comet}_comparison",
    workspace="saitama32",
)

sys.path.append(PROJECT_ROOT)
import time
import argparse
import dill
import numpy as np
import torch
import deepxde as dde

from src.pde.heat import Heat2D_Multiscale
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from rl_trainer import train_process_rl


dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)


def build_get_model_heat2d_multiscale(hidden_layers: str, **pde_kwargs):
    def get_model():
        pde = Heat2D_Multiscale(**pde_kwargs)

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
    parser.add_argument("--name", type=str, default="heat2d_multiscale_rl")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--state-type", type=str, choices=("raw", "log_raw", "1d", "2d", "3d"), required=True)
    parser.add_argument("--exp_key", type=str, required=True)
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
    parser.add_argument("--out", type=str, default="runs_single")

    parser.add_argument("--pde-coef-x", type=float, default=1 / (500 * np.pi) ** 2, help="PDE coefficient for x diffusion")
    parser.add_argument("--pde-coef-y", type=float, default=1 / (np.pi ** 2), help="PDE coefficient for y diffusion")
    parser.add_argument("--init-coef-x", type=float, default=20 * np.pi, help="Initial condition frequency in x")
    parser.add_argument("--init-coef-y", type=float, default=np.pi, help="Initial condition frequency in y")

    args = parser.parse_args()
    if seed_override is not None:
        args.seed = int(seed_override)

    experiment.log_parameters({
        "param": "v_1",
        "reward_function": "v_2",
        "description": "comparison_heat2d_multiscale_loaded_dqn_final_eval",
        "state_type": args.state_type,
    })

    if args.state_type in ("raw", "log_raw"):
        state_type = "raw_loss"
        raw_loss_state_len = args.n_save_models
        raw_loss_log_state = args.state_type == "log_raw"
        latent_dim = 1
        log_key_for_new_state = False
    else:
        state_type = "loss_surface"
        raw_loss_state_len = None
        raw_loss_log_state = False
        latent_dim = int(args.state_type[0])
        log_key_for_new_state = True

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    pde_kwargs = dict(
        pde_coef=(args.pde_coef_x, args.pde_coef_y),
        init_coef=(args.init_coef_x, args.init_coef_y)
    )

    get_model = build_get_model_heat2d_multiscale(args.hidden_layers, **pde_kwargs)
    get_model_rec = build_get_model_heat2d_multiscale(args.hidden_layers, **pde_kwargs)

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
        "log_key": log_key_for_new_state,
    }

    loss_surface_params = {
        "state_type": state_type,
        "raw_loss_state_len": raw_loss_state_len,
        "raw_loss_log_state": raw_loss_log_state,
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

    rl_agent_params = {
        "n_save_models": args.n_save_models,
        "n_trajectories": args.n_trajectories,
        "tolerance": 0.00910732802003622,
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
        "state_type": args.state_type,
    }

    experiment.log_parameters(rl_agent_params)
    experiment.log_parameters(comparison_params)

    data = dill.dumps((get_model, train_args, optimizers, AE_model_params, AE_train_params, loss_surface_params, comparison_params))
    train_process_rl(data=data, save_path=save_path, device=args.device, seed=args.seed, rl_agent_params=rl_agent_params, comparison_params=comparison_params)


if __name__ == "__main__":
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]

    for seed in seeds:
        print(f"\nStarting comparison experiment with seed = {seed}")
        main(seed_override=seed)
