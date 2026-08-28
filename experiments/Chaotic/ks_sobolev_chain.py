# run_ks_sobolev_rl.py
import os, sys
import kagglehub

os.environ["DDEBACKEND"] = "pytorch"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from comet_config import start_comet_experiment
from comet_ml.integration.pytorch import log_model

experiment = start_comet_experiment(
    project_name="rlpinn-ks-sobolev-expand-set-rwf",
)


kaggle_account = kagglehub.whoami()["username"]
print("KAGGLE_ACCOUNT:", kaggle_account)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import argparse
import functools
import dill
import numpy as np
import torch
import deepxde as dde

from src.pde.chaotic import KuramotoSivashinskyEquation
from src.model import RWFMLP, SFLIConfig, materialize_effective_mlp
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import TesterCallback
from rl_trainer import train_process_rl
from experiments.Chaotic.run_data_ks import load_data
from experiments.Chaotic.run_data_ks_exact_mlp import reference_derivative_targets

experiment.log_parameters(
    {
        "param": "v_1",
        "reward_function": "v_2",
        "description": "farm_transitions_ks_supervised_sobolev_RL_optimizer",
    }
)


DERIVATIVE_KEYS = ("u_t", "u_x", "u_xx", "u_xxxx")


class BatchedPointSetOperatorBC(dde.icbc.PointSetOperatorBC):
    """PyTorch PointSetOperatorBC with PointSetBC-style minibatching."""

    def __init__(self, points, values, func, batch_size, shuffle=True):
        super().__init__(points, values, func)
        if batch_size <= 0:
            raise ValueError("Sobolev batch size must be positive")
        self.batch_size = min(int(batch_size), len(self.points))
        self.batch_sampler = dde.data.sampler.BatchSampler(
            len(self.points), shuffle=shuffle
        )
        self.batch_indices = None

    def collocation_points(self, _):
        self.batch_indices = self.batch_sampler.get_next(self.batch_size)
        return self.points[self.batch_indices]

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        if self.batch_indices is None:
            raise RuntimeError("Sobolev operator batch has not been sampled")
        return self.func(inputs, outputs, X)[beg:end] - self.values[self.batch_indices]


@functools.lru_cache(maxsize=None)
def _load_sobolev_reference(data_path):
    points, values = load_data(data_path, precision="float64")
    targets, scales = reference_derivative_targets(points, values)
    return points, targets, scales


def _operator_for(name):
    if name == "u_t":
        return lambda x, u, _: dde.grad.jacobian(u, x, i=0, j=1)
    if name == "u_x":
        return lambda x, u, _: dde.grad.jacobian(u, x, i=0, j=0)
    if name == "u_xx":
        return lambda x, u, _: dde.grad.hessian(u, x, i=0, j=0)
    if name == "u_xxxx":
        return lambda x, u, _: dde.grad.hessian(
            dde.grad.hessian(u, x, i=0, j=0), x, i=0, j=0
        )
    raise ValueError(f"Unknown Sobolev target: {name}")


def build_get_model_ks_sobolev(
    hidden_layers,
    rwf_mu,
    rwf_sigma,
    sfli_gamma,
    sfli_c,
    sfli_seed,
    data_path,
    sobolev_batch_size,
    data_weight,
    ut_weight,
    ux_weight,
    uxx_weight,
    uxxxx_weight,
    pde_weight,
    ic_weight,
    domain_points,
    ic_points,
    *,
    effective_dense=False,
):
    component_weights = {
        "u": data_weight,
        "u_t": ut_weight,
        "u_x": ux_weight,
        "u_xx": uxx_weight,
        "u_xxxx": uxxxx_weight,
    }

    def get_model():
        points, targets, scales = _load_sobolev_reference(data_path)
        pde = KuramotoSivashinskyEquation(datapath=data_path)
        analytic_ic = pde.bcs[0]

        # Preserve one operator slot for the RL code.  In pure Sobolev mode it
        # is a cheap identically-zero residual instead of an unnecessary
        # fourth-order KS evaluation.
        if pde_weight == 0.0:
            pde.pde = lambda _, u: 0.0 * u

        pde.bcs = []
        pde.loss_config = [{"name": "ks_pde", "type": "pde"}]
        pde.training_points(
            domain=max(1, int(domain_points)),
            boundary=0,
            initial=int(ic_points) if ic_weight > 0.0 else 0,
            test=max(1, int(domain_points)),
        )

        value_bc = dde.PointSetBC(
            points,
            targets["u"],
            component=0,
            batch_size=min(int(sobolev_batch_size), len(points)),
            shuffle=True,
        )
        pde.bcs.append(value_bc)
        pde.loss_config.append({"name": "sobolev_u", "type": "boundary"})

        for name in DERIVATIVE_KEYS:
            pde.bcs.append(
                BatchedPointSetOperatorBC(
                    points,
                    targets[name],
                    _operator_for(name),
                    batch_size=sobolev_batch_size,
                    shuffle=True,
                )
            )
            pde.loss_config.append(
                {"name": f"sobolev_{name}", "type": "boundary"}
            )

        if ic_weight > 0.0:
            initial_time = dde.config.real(np)(pde.bbox[2])
            analytic_ic.on_initial = (
                lambda initial_points, _: initial_points[:, -1] == initial_time
            )
            pde.bcs.append(analytic_ic)
            pde.loss_config.append({"name": "analytic_ic", "type": "boundary"})

        layers = [
            pde.input_dim,
            *parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)),
            pde.output_dim,
        ]
        sfli = SFLIConfig(
            bounds=((pde.bbox[0], pde.bbox[1]), (pde.bbox[2], pde.bbox[3])),
            gamma=sfli_gamma,
            C=sfli_c,
            seed=sfli_seed,
            type="tanh",
        )
        net = RWFMLP(layers, mu=rwf_mu, sigma=rwf_sigma, sfli=sfli).float()
        if effective_dense:
            net = materialize_effective_mlp(net)

        loss_weights = [float(pde_weight)]
        loss_weights.extend(
            float(component_weights[name]) / float(scales[name] ** 2)
            for name in ("u", *DERIVATIVE_KEYS)
        )
        if ic_weight > 0.0:
            loss_weights.append(float(ic_weight))
        loss_weights = np.asarray(loss_weights, dtype=float)

        model = pde.create_model(net)
        model.data.train_distribution = "pseudo"
        model.data.resample_train_points(pde_points=True, bc_points=True)
        return model, loss_weights

    return get_model


def main():
    parser = argparse.ArgumentParser(
        description="Run the KS RL optimizer chain with supervised Sobolev loss."
    )
    parser.add_argument("--name", default="ks_sobolev_rl")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data", default=os.path.join(project_root, "ref", "Kuramoto_Sivashinsky.dat"))
    parser.add_argument("--out", default="runs_ks_sobolev")
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--n-save-models", type=int, default=10)
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--sfli-gamma", type=float, default=None)
    parser.add_argument("--sfli-c", type=float, default=1.0)
    parser.add_argument("--sfli-seed", type=int, default=0)
    parser.add_argument("--sobolev-batch-size", type=int, default=8)
    parser.add_argument("--domain-points", type=int, default=8192)
    parser.add_argument("--ic-points", type=int, default=2048)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--ut-weight", type=float, default=1.0)
    parser.add_argument("--ux-weight", type=float, default=1.0)
    parser.add_argument("--uxx-weight", type=float, default=1.0)
    parser.add_argument("--uxxxx-weight", type=float, default=1.0)
    parser.add_argument("--pde-weight", type=float, default=0.0)
    parser.add_argument("--ic-weight", type=float, default=0.0)
    args = parser.parse_args()

    weights = (
        args.data_weight,
        args.ut_weight,
        args.ux_weight,
        args.uxx_weight,
        args.uxxxx_weight,
        args.pde_weight,
        args.ic_weight,
    )
    if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
        parser.error("All loss weights must be finite and non-negative")
    if not any(weight > 0.0 for weight in weights):
        parser.error("At least one loss weight must be positive")
    if args.sobolev_batch_size <= 0:
        parser.error("--sobolev-batch-size must be positive")
    if args.domain_points <= 0:
        parser.error("--domain-points must be positive")
    if args.ic_points <= 0:
        parser.error("--ic-points must be positive")
    data_path = os.path.abspath(os.path.expanduser(args.data))
    if not os.path.isfile(data_path):
        parser.error(f"KS reference data not found: {data_path}")

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
        data_path=data_path,
        sobolev_batch_size=args.sobolev_batch_size,
        data_weight=args.data_weight,
        ut_weight=args.ut_weight,
        ux_weight=args.ux_weight,
        uxx_weight=args.uxx_weight,
        uxxxx_weight=args.uxxxx_weight,
        pde_weight=args.pde_weight,
        ic_weight=args.ic_weight,
        domain_points=args.domain_points,
        ic_points=args.ic_points,
    )
    get_model = build_get_model_ks_sobolev(**model_kwargs)
    get_model_rec = build_get_model_ks_sobolev(**model_kwargs, effective_dense=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_args = {
        "iterations": 1,
        "display_every": args.log_every,
        "callbacks": [
            dde.callbacks.PDEPointResampler(
                period=1,
                pde_points=args.pde_weight > 0.0,
                bc_points=True,
            ),
            TesterCallback(
                log_every=args.log_every,
                verbose=True,
                fRMSE_param={"enable": False},
            ),
        ],
        "n_trajectories": args.n_trajectories,
        "n_save_models": args.n_save_models,
        "save_solution_images": True,
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
    ae_model_params = {
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
    ae_train_params = {
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
        "lr": 1e-3,
        "exp": experiment,
    }
    experiment.log_parameters(
        {
            **rl_agent_params,
            "objective": "normalized_supervised_sobolev",
            "data_path": data_path,
            "sobolev_batch_size": args.sobolev_batch_size,
            "loss_weights_cli": dict(zip(
                ("u", "u_t", "u_x", "u_xx", "u_xxxx", "pde", "ic"),
                weights,
            )),
        }
    )

    serialized = dill.dumps(
        (
            get_model,
            train_args,
            optimizers,
            ae_model_params,
            ae_train_params,
            loss_surface_params,
        )
    )
    train_process_rl(
        data=serialized,
        save_path=save_path,
        device=0,
        seed=args.seed,
        rl_agent_params=rl_agent_params,
    )


if __name__ == "__main__":
    main()
