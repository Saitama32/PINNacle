from comet_ml import start
from comet_ml.integration.pytorch import log_model

experiment = start(
  api_key="aP71fQTYPNqfsYWvudPPmoBl5",
  project_name="rlpinn_poisson_2d_ms_comparison",
  workspace="saitama32"
)

import torch
import os
import sys
import numpy as np
import time
import random
import tempfile
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.utils import exact_solution_data
from tedeous.error_calc_utils import boundary_report

experiment.log_parameters({
    "param": "v_1",
    "reward_function": "v_2",
    "description": "comparison_poisson_2d_ms_RL_optimizer"
})

device = "cuda" if torch.cuda.is_available() else "cpu"
solver_device(device)

datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_manyarea.npy"))
datapath_a_cof = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_a_coef.npy"))
datapath_f_cof = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../PINNacle_data/poisson_f_coef.npy"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_key",
        type=str,
        default=None,
        help="Comet experiment key for backup / resume"
    )
    parser.add_argument(
        "--exp_key",
        type=str,
        default=None,
        help="Comet experiment key for comparison runs"
    )
    return parser.parse_args()

def poisson_2d_many_subdomains_experiment(grid_res, log_key=None, exp_key=None):
    if log_key == "True":
        log_key = True
    elif log_key == "False":
        log_key = False    

    exp_dict_list = []

    x_min, x_max = -10, 10
    y_min, y_max = -10, 10

    pde_dim_in = 2
    pde_dim_out = 1

    domain = Domain()
    domain.variable('x', [x_min, x_max], grid_res)
    domain.variable('y', [y_min, y_max], grid_res)

    domain_test = Domain()
    grid_test_res = 80
    domain_test.variable('x', [x_min, x_max], grid_test_res)
    domain_test.variable('y', [y_min, y_max], grid_test_res)

    def exact_fn(pts):
        return exact_solution_data(
            pts,
            datapath,
            pde_dim_in,
            pde_dim_out,
            t_dim_flag=("t" in list(domain.variable_dict.keys())),
        )


    split = (5, 5)
    freq = 2
    block_size = np.array([(x_max - x_min + 2e-5) / split[0], (y_max - y_min + 2e-5) / split[1]])

    a_cof = np.load(datapath_a_cof)
    f_cof = np.load(datapath_f_cof).reshape(split[0], split[1], freq, freq)

    # --- torch tensors for coefficients (keep on GPU) ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    a_t = torch.as_tensor(a_cof, dtype=torch.float32, device=device)  # shape зависит от файла
    f_t = torch.as_tensor(f_cof, dtype=torch.float32, device=device)  # (sx, sy, freq, freq)

    dx = (x_max - x_min) / split[0]
    dy = (y_max - y_min) / split[1]

    sx, sy = split

    def dom_and_local(grid: torch.Tensor):
        """
        grid: (N,2) on GPU
        returns:
        ix, iy: (N,) long indices of block
        rx, ry: (N,1) local coords in [0,1] inside the block
        """
        x = grid[:, 0]
        y = grid[:, 1]

        ix = torch.floor((x - x_min) / dx).long()
        iy = torch.floor((y - y_min) / dy).long()

        # clamp to be safe on boundary
        ix = ix.clamp(0, sx - 1)
        iy = iy.clamp(0, sy - 1)

        x0 = x_min + ix.float() * dx
        y0 = y_min + iy.float() * dy

        rx = ((x - x0) / dx).unsqueeze(1)  # (N,1)
        ry = ((y - y0) / dy).unsqueeze(1)  # (N,1)

        return ix, iy, rx, ry
    
    def a_and_grads(grid: torch.Tensor):
        """
        Returns:
        a  (N,1)
        ax (N,1) = ∂a/∂x
        ay (N,1) = ∂a/∂y
        """
        ix, iy, rx, ry = dom_and_local(grid)

        # Case A: a = a0 + ax_hat*rx + ay_hat*ry  (linear in local coords)
        # a_t shape: (sx,sy,3) where [:,:,0]=a0, [:,:,1]=ax_hat, [:,:,2]=ay_hat
        if a_t.ndim == 3 and a_t.shape[-1] == 3:
            p = a_t[ix, iy]  # (N,3)
            a0 = p[:, 0:1]
            ax_hat = p[:, 1:2]
            ay_hat = p[:, 2:3]

            a = a0 + ax_hat * rx + ay_hat * ry
            ax = ax_hat / dx
            ay = ay_hat / dy
            return a, ax, ay

        # Case B (optional): bilinear on local square:
        # a = a00 + a10*rx + a01*ry + a11*rx*ry
        if a_t.ndim == 3 and a_t.shape[-1] == 4:
            p = a_t[ix, iy]  # (N,4)
            a00 = p[:, 0:1]
            a10 = p[:, 1:2]
            a01 = p[:, 2:3]
            a11 = p[:, 3:4]

            a = a00 + a10*rx + a01*ry + a11*rx*ry
            ax = (a10 + a11*ry) / dx
            ay = (a01 + a11*rx) / dy
            return a, ax, ay

        # Fallback: piecewise-constant
        # a_t shape (sx,sy)
        a = a_t[ix, iy].unsqueeze(1)
        ax = torch.zeros_like(a)
        ay = torch.zeros_like(a)
        return a, ax, ay


    def get_a(grid):
        a, _, _ = a_and_grads(grid.to(device))
        return a


    def get_ax(grid):
        _, ax, _ = a_and_grads(grid.to(device))
        return ax


    def get_ay(grid):
        _, _, ay = a_and_grads(grid.to(device))
        return ay
    
    i_idx = torch.arange(freq, device=device, dtype=torch.float32).view(1, freq, 1)
    j_idx = torch.arange(freq, device=device, dtype=torch.float32).view(1, 1, freq)

    def get_f(grid: torch.Tensor):
        grid = grid.to(device)
        ix, iy, rx, ry = dom_and_local(grid)

        coef = f_t[ix, iy]  # (N,freq,freq)

        # sin(pi*i*rx)*sin(pi*j*ry)
        sinx = torch.sin(torch.pi * i_idx * rx.view(-1,1,1))  # (N,freq,1)
        siny = torch.sin(torch.pi * j_idx * ry.view(-1,1,1))  # (N,1,freq)
        basis = sinx * siny                                   # (N,freq,freq)

        fval = torch.sum(coef * basis, dim=(1,2), keepdim=True)  # (N,1)
        return fval




    boundaries = Conditions()

    # Operator: u + du/dn = 0

    def bop_generation(func_coeff, deriv_coeff, deriv_dim):
        bop = {
            'u':
                {
                    'coeff': func_coeff,
                    'term': [None],
                    'pow': 1
                },
            'du/dn':
                {
                    'coeff': deriv_coeff,
                    'term': [deriv_dim],
                    'pow': 1
                }
        }
        return bop

    bop_x_min = bop_generation(1, -1, 0)
    boundaries.robin({'x': x_min, 'y': [y_min, y_max]}, operator=bop_x_min, value=0.0)

    bop_x_max = bop_generation(1, 1, 0)
    boundaries.robin({'x': x_max, 'y': [y_min, y_max]}, operator=bop_x_max, value=0.0)

    bop_y_min = bop_generation(1, -1, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_min}, operator=bop_y_min, value=0.0)

    bop_y_max = bop_generation(1, 1, 1)
    boundaries.robin({'x': [x_min, x_max], 'y': y_max}, operator=bop_y_max, value=0.0)


    equation = Equation()

    # Operator: −∇(a(x)∇u) = f(x, y)

    poisson = {
    '-a * d2u/dx2': {'coeff': lambda g: -get_a(g),  'term': [0, 0], 'pow': 1, 'var': 0},
    '-ax * du/dx':  {'coeff': lambda g: -get_ax(g), 'term': [0],    'pow': 1, 'var': 0},

    '-a * d2u/dy2': {'coeff': lambda g: -get_a(g),  'term': [1, 1], 'pow': 1, 'var': 0},
    '-ay * du/dy':  {'coeff': lambda g: -get_ay(g), 'term': [1],    'pow': 1, 'var': 0},

    '-f(x,y)':      {'coeff': lambda g: -get_f(g),  'term': [None], 'pow': 0, 'var': 0},
    }



    equation.add(poisson)

    neurons = 100

    net = torch.nn.Sequential(
        torch.nn.Linear(pde_dim_in, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, pde_dim_out)
    )

    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model_layers = [pde_dim_in, neurons, neurons, neurons, neurons, neurons, pde_dim_out]

    grid = domain.build('NN').to(device)
    grid_test = domain_test.build('NN').to(device)

    model = Model(net, domain, equation, boundaries)

    model.compile('autograd', lambda_operator=1, lambda_bound=100)

    img_dir = os.path.join(os.path.dirname(__file__), 'poisson_2d_many_subdomains_img')

    u_exact_test = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out).reshape(-1)
    
    equation_params = [u_exact_test, grid_test, grid, domain, equation, boundaries, model_layers]

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=100,
                                         no_improvement_patience=1000,
                                         patience=10,
                                         info_string_every=10,
                                         randomize_parameter=1e-4)

    optimizer = {
        'Adam':{
            'lr':[1e-2, 1e-3, 1e-4],
            'epochs':[100, 1000, 2500]
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
        "log_key": log_key
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
        "img_dir": img_dir
    }

    rl_agent_params = {
        "n_save_models": 10,
        "n_trajectories": 1000,
        "tolerance": 3.79691889513976,
        "prev_tol": 0.0,
        "stuck_threshold": 10,  # Число эпох без значительного изменения прогресса
        "min_loss_change": 1e-7,
        "min_grad_norm": 1e-5,
        "rl_buffer_size": 10000,
        "rl_batch_size": 32,
        "n_transitions_reinit" : 1000,
        "gamma": 0.9,
        "rl_reward_method": "absolute",
        "exact_solution": datapath,
        "reward_operator_coeff": 1,
        "reward_boundary_coeff": 1,
        "lr": 5e-4,
        "exp": experiment,
        "log_key": log_key,
    }

    comparison_params = {
        "seed": seed, 
        "total_epochs": 7000,
        "experiment_key": exp_key,
        "multi_pde_comparison": True,
    }

    experiment.log_parameters(rl_agent_params)
    experiment.log_parameters(comparison_params)

    model.train(optimizer,
                5e5,
                save_model=True,
                callbacks=[cb_es],
                rl_agent_params=rl_agent_params,
                models_concat_flag=False,
                model_name='rl_optimization_agent',
                equation_params=equation_params,
                AE_model_params=AE_model_params,
                AE_train_params=AE_train_params,
                loss_surface_params=loss_surface_params,
                comparison_param=comparison_params)

    net = model.net.to(device)
    grid_test = grid_test.to(device)
    u_exact = exact_solution_data(grid, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape(-1, 1)
    u_pred = net(grid)
    diff = u_exact - u_pred
    error_op_mse_train = torch.mean(diff ** 2)
    error_op_rmse_train = torch.sqrt(torch.mean(diff ** 2))
    variable_dict = domain.variable_dict
    bconds = boundaries.build(variable_dict)
    report = boundary_report(
        net=net,
        grid_for_dtype=grid,
        bconds=bconds,
        exact_solution_fn=exact_fn,
        mode="autograd",
        derivative_points=2,
    )
    error_bnd_mse_train = report["exact_u_mse"]
    error_bnd_rmse_train = report["exact_u_rmse"]
    print({k: float(v) for k, v in report.items()})
    error_rmse_train_full = error_op_rmse_train + error_bnd_rmse_train

    error_l2re_train = torch.sqrt(torch.sum(
    (u_exact - net(grid)) ** 2) / torch.sum(u_exact ** 2))
    print(f"Train full RMSE: {error_rmse_train_full}, Train op RMSE: {error_op_rmse_train}, Train bnd RMSE: {error_bnd_rmse_train}, L2RE op: {error_l2re_train}")


    # Test errors
    variable_dict = domain_test.variable_dict
    bconds = boundaries.build(variable_dict)
    u_exact_test = exact_solution_data(grid_test, datapath, pde_dim_in, pde_dim_out, t_dim_flag='t' in list(domain.variable_dict.keys())).to(device).reshape(-1, 1)
    error_op_mse_test = torch.mean((u_exact_test - net(grid_test)) ** 2)
    error_op_rmse_test = torch.sqrt(torch.mean((u_exact_test - net(grid_test)) ** 2))
    report = boundary_report(
        net=net,
        grid_for_dtype=grid_test,
        bconds=bconds,
        exact_solution_fn=exact_fn,
        mode="autograd",
        derivative_points=2,
    )
    error_bnd_mse_test = report["exact_u_mse"]
    error_bnd_rmse_test = report["exact_u_rmse"]
    error_rmse_test_full = error_op_rmse_test + error_bnd_rmse_test
    error_l2re_test = torch.sqrt(torch.sum(
        (u_exact_test - net(grid_test)) ** 2) / torch.sum(u_exact_test ** 2))
    print(f"Train full RMSE: {error_rmse_test_full}, Train op RMSE: {error_op_rmse_test}, Train bnd RMSE: {error_bnd_rmse_test}, L2RE op: {error_l2re_test}")

    
    experiment.log_metrics({
        # RMSE
        "error_op_rmse_train": error_op_rmse_train.item(),
        "error_bnd_rmse_train": error_bnd_rmse_train.item(),
        "error_rmse_train_full": error_rmse_train_full.item(),

        "error_op_rmse_test": error_op_rmse_test.item(),
        "error_bnd_rmse_test": error_bnd_rmse_test.item(),
        "error_rmse_test_full": error_rmse_test_full.item(),

        # MSE (новое)
        "error_op_mse_train": error_op_mse_train.item(),
        "error_bnd_mse_train": error_bnd_mse_train.item(),

        "error_op_mse_test": error_op_mse_test.item(),
        "error_bnd_mse_test": error_bnd_mse_test.item(),

        # остальное
        "error_l2re_train": error_l2re_train.item(),
        "error_l2re_test": error_l2re_test.item(),
    }, step=seed)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_params:
        torch.save(net.state_dict(), tmp_params.name)
        params_path = tmp_params.name

    # --- логируем как модельные файлы ---
    experiment.log_model(
        name="PINN_optim",
        file_or_folder=params_path,
        file_name=f"model_PINN_{seed}_.pt",
        overwrite=True,
        metadata={"type": "optimizer_state"}
            )

    return exp_dict_list



if __name__ == "__main__":
    args = parse_args()
    grid_res = 100
    seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901, 1012]   # можно расширить список

    for seed in seeds:
        print(f"\n🔹 Запуск эксперимента с seed = {seed}")

        # установка детерминированности
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        exp_dict_list = poisson_2d_many_subdomains_experiment(grid_res, log_key=args.log_key, exp_key=args.exp_key)
