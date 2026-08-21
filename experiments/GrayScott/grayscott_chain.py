import argparse
import os
import sys
import time

os.environ["DDEBACKEND"] = "pytorch"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import dill
import numpy as np
import torch

from comet_config import start_comet_experiment
from rl_trainer import train_process_rl
from src.model import RWFMLP, SFLIConfig, materialize_effective_mlp
from src.pde.chaotic import GrayScottEquation
from src.utils.args import parse_hidden_layers


experiment = start_comet_experiment(
    project_name="rlpinn-grayscott-expand-set-rwf-tolerance",
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

experiment.log_parameters(
    {
        "param": "v_1",
        "reward_function": "v_2",
        "description": "grayscott_rwf_sfli_chain_reward_expanded_optimizers",
    }
)


def build_get_model_grayscott(
    hidden_layers: str,
    rwf_mu: float,
    rwf_sigma: float,
    sfli_gamma: float | None,
    sfli_c: float,
    sfli_seed: int,
    *,
    effective_dense: bool = False,
):
    def get_model():
        pde = GrayScottEquation()
        layers = [
            pde.input_dim,
            *parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)),
            pde.output_dim,
        ]
        sfli = SFLIConfig(
            bounds=(
                (pde.bbox[0], pde.bbox[1]),
                (pde.bbox[2], pde.bbox[3]),
                (pde.bbox[4], pde.bbox[5]),
            ),
            gamma=sfli_gamma,
            C=sfli_c,
            seed=sfli_seed,
            type="tanh",
        )
        net = RWFMLP(
            layers,
            mu=rwf_mu,
            sigma=rwf_sigma,
            sfli=sfli,
        ).float()
        if effective_dense:
            net = materialize_effective_mlp(net)

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for index, config in enumerate(pde.loss_config):
            if config.get("type", "") in ("boundary", "initial", "ic"):
                loss_weights[index] = 100.0

        return pde.create_model(net), loss_weights

    return get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="grayscott_rl")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--state-h", type=int, default=26)
    parser.add_argument("--state-w", type=int, default=26)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs_single")
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--sfli-gamma", type=float, default=None)
    parser.add_argument("--sfli-c", type=float, default=1.0)
    parser.add_argument("--sfli-seed", type=int, default=0)
    args = parser.parse_args()

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", args.device).replace(",", "-")
    process_tag = f"gpu_{visible_device}-pid_{os.getpid()}"
    save_path = os.path.join(args.out, f"{date_str}-{args.name}-{process_tag}")
    os.makedirs(save_path, exist_ok=True)

    model_kwargs = dict(
        hidden_layers=args.hidden_layers,
        rwf_mu=args.rwf_mu,
        rwf_sigma=args.rwf_sigma,
        sfli_gamma=args.sfli_gamma,
        sfli_c=args.sfli_c,
        sfli_seed=args.sfli_seed,
    )
    get_model = build_get_model_grayscott(**model_kwargs)
    get_model_rec = build_get_model_grayscott(
        **model_kwargs,
        effective_dense=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_args = {
        "iterations": 1,
        "display_every": args.log_every,
        "callbacks": [],
        "n_trajectories": args.n_trajectories,
        "n_save_models": args.n_save_models,
        "save_solution_images": False,
        "log_long_horizon_metrics": True,
        "image_log_every": args.log_every,
        "operator_coeff": 1,
        "bnd_coeff": 1,
    }
    optimizers = {
        "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
        "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1000]},
        "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
        "SOAP": {"lr": [1e-2, 1e-3, 3e-4], "epochs": [100, 1000, 2500]},
        "Muon": {"lr": [2e-2, 1e-2, 5e-3], "epochs": [100, 1000, 2500]},
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
    }
    rl_agent_params = {
        "n_save_models": args.n_save_models,
        "n_trajectories": args.n_trajectories,
        "tolerance": 0.0,
        "prev_tol": 0,
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
    }

    experiment.log_parameters(rl_agent_params)
    data = dill.dumps(
        (
            get_model,
            train_args,
            optimizers,
            AE_model_params,
            AE_train_params,
            loss_surface_params,
        )
    )
    train_process_rl(
        data=data,
        save_path=save_path,
        device=0,
        seed=args.seed,
        rl_agent_params=rl_agent_params,
    )


if __name__ == "__main__":
    main()
