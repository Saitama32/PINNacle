# run_ks_rl.py
import os, sys

os.environ["DDEBACKEND"] = "pytorch"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from comet_config import start_comet_experiment
from comet_ml.integration.pytorch import log_model

experiment = start_comet_experiment(
    project_name="rlpinn-ks-expand_set_and_imtegral_loss_testing",
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import argparse
import dill
import numpy as np
import torch
import deepxde as dde

from src.pde.chaotic import KuramotoSivashinskyEquation
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import (
    IntegralDiagnosticsCallback,
    LossCallback,
    PlotCallback,
    TesterCallback,
)
from rl_trainer import train_process_rl

experiment.log_parameters(
    {
        "param": "v_1",
        "reward_function": "v_2",
        "description": "farm_transitions_ks_basic_RL_optimizer",
    }
)


def build_get_model_ks(hidden_layers: str):
    def get_model():
        pde = KuramotoSivashinskyEquation()

        layers = [
            pde.input_dim,
            *parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)),
            pde.output_dim,
        ]
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
    parser.add_argument("--name", type=str, default="ks_rl")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--hidden-layers", type=str, default="190*5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=2000)
    parser.add_argument("--use-integral-loss", action="store_true", default=True)
    parser.add_argument("--integral-only", action="store_true", default=True)
    parser.add_argument("--integral-batch-size", type=int, default=256)
    parser.add_argument("--integral-quadrature-order", type=int, default=4)
    parser.add_argument("--integral-local-enabled", action="store_true", default=True)
    parser.add_argument("--no-integral-local", dest="integral_local_enabled", action="store_false")
    parser.add_argument("--integral-local-weight", type=float, default=1.0)
    parser.add_argument("--integral-local-quadrature-order", type=int, default=4)
    parser.add_argument("--integral-local-hmax", type=float, default=0.05)
    parser.add_argument("--integral-local-segment-batch-size", type=int, default=256)
    parser.add_argument("--integral-local-normalize-by-length", action="store_true", default=True)
    parser.add_argument("--integral-local-contiguous-chain", action="store_true", default=True)
    parser.add_argument("--integral-t0-fraction", type=float, default=0.2)
    parser.add_argument("--integral-t-min", type=float, default=0.0)
    parser.add_argument("--integral-resample-every", type=int, default=5)
    parser.add_argument("--integral-seed", type=int, default=None)
    parser.add_argument("--integral-ic-enabled", action="store_true", default=True)
    parser.add_argument("--integral-ic-weight", type=float, default=100.0)

    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--n-steps-max", type=int, default=1000)
    parser.add_argument("--state-h", type=int, default=26)
    parser.add_argument("--state-w", type=int, default=26)
    parser.add_argument("--n-save-models", type=int, default=10)

    parser.add_argument("--out", type=str, default="runs_single")

    args = parser.parse_args()
    if args.integral_only and not args.use_integral_loss:
        raise ValueError("--integral-only requires --use-integral-loss.")
    if args.integral_ic_enabled and not args.integral_only:
        raise ValueError("--integral-ic-enabled requires --integral-only.")

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{date_str}-{args.name}")
    os.makedirs(save_path, exist_ok=True)

    get_model = build_get_model_ks(args.hidden_layers)
    get_model_rec = build_get_model_ks(args.hidden_layers)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    callbacks = [
        TesterCallback(log_every=args.log_every),
        PlotCallback(log_every=args.plot_every, fast=True),
        LossCallback(verbose=True),
    ]
    if args.use_integral_loss:
        callbacks.append(IntegralDiagnosticsCallback(log_every=args.log_every, verbose=True))

    train_args = {
        "iterations": 1,
        "display_every": args.log_every,
        "callbacks": callbacks,
        "n_trajectories": 1000,
        "n_save_models": 10,
        "operator_coeff": 1,
        "bnd_coeff": 1,
        "integral_loss": {
            "enabled": args.use_integral_loss,
            "integral_only": args.integral_only,
            "batch_size": args.integral_batch_size,
            "weight": 1.0,
            "warmup_steps": 0,
            "start_step": 0,
            "quadrature_order": args.integral_quadrature_order,
            "local_enabled": args.integral_local_enabled,
            "local_weight": args.integral_local_weight,
            "local_quadrature_order": args.integral_local_quadrature_order,
            "local_hmax": args.integral_local_hmax,
            "local_segment_batch_size": args.integral_local_segment_batch_size,
            "local_normalize_by_length": args.integral_local_normalize_by_length,
            "local_contiguous_chain": args.integral_local_contiguous_chain,
            "t0_fraction": args.integral_t0_fraction,
            "t_min": args.integral_t_min,
            "resample_every": args.integral_resample_every,
            "seed": args.integral_seed if args.integral_seed is not None else args.seed,
            "ic_enabled": args.integral_ic_enabled,
            "ic_weight": args.integral_ic_weight,
        },
    }
    optimizers = {
        # "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
        "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1000]},
        "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
        "SOAP": {"lr": [1e-2, 1e-3, 3e-4], "epochs": [100, 1000, 2500],},
        "Muon": {"lr": [2e-2, 1e-2, 5e-3], "epochs": [100, 1000, 2500]},
        "PCGrad": {
            "lr": [1e-2, 1e-3, 1e-4],
            "epochs": [100, 1000, 2000],
        },
    }

    AE_model_params = {
        "mode": "NN",
        "num_of_layers": 3,
        "layers_AE": [
            991,
            125,
            15,
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
        "layers_AE": [
            991,
            125,
            15,
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
        "img_dir": "",
        "dde_pde_model": get_model_rec,
    }

    rl_agent_params = {
        "n_save_models": 10,
        "n_trajectories": 1000,
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
        "lr": 1e-3,
        "seed": args.seed,
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
