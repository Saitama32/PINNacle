import argparse
import os
import sys
import time

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import torch
import deepxde as dde

from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.model.features import PeriodicFourierFeatures
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import (
    CausalDiagnosticsCallback,
    LossCallback,
    PlotCallback,
    TesterCallback,
)


EQUATIONS = {
    "gs": GrayScottEquation,
    "grayscott": GrayScottEquation,
    "gray-scott": GrayScottEquation,
    "ks": KuramotoSivashinskyEquation,
    "kuramoto-sivashinsky": KuramotoSivashinskyEquation,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def loss_weights_for(pde, bc_loss_weight):
    weights = np.ones(pde.num_loss, dtype=float)
    for i, config in enumerate(pde.loss_config):
        if config.get("type") in ("boundary", "initial", "ic"):
            weights[i] = bc_loss_weight
    return weights


def build_state_grid(pde, state_grid_size):
    spatial_dim = pde.input_dim - 1
    if spatial_dim < 1:
        raise ValueError("window training requires at least one spatial dimension.")

    axes = []
    for i in range(spatial_dim):
        lo = pde.bbox[2 * i]
        hi = pde.bbox[2 * i + 1]
        axes.append(np.linspace(lo, hi, state_grid_size, dtype=np.float32))

    if spatial_dim == 1:
        return axes[0][:, None]

    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1).astype(np.float32)


def build_model_from_pde(pde, args):
    feature_encoder = None
    input_dim = pde.input_dim
    if args.use_fourier_features:
        if not isinstance(pde, KuramotoSivashinskyEquation):
            raise ValueError("Fourier features are enabled only for KS in this runner.")
        x_period = pde.bbox[1] - pde.bbox[0]
        feature_encoder = PeriodicFourierFeatures(
            x_period=x_period,
            num_modes_x=args.fourier_num_modes_x,
            include_t=True,
            include_raw_x=args.fourier_include_raw_x,
            include_bias=args.fourier_include_bias,
        )
        input_dim = feature_encoder.out_dim

    layers = [
        input_dim,
        *parse_hidden_layers(argparse.Namespace(hidden_layers=args.hidden_layers)),
        pde.output_dim,
    ]
    net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
    if feature_encoder is not None:
        net.apply_feature_transform(feature_encoder)
    return pde.create_model(net), loss_weights_for(pde, args.bc_loss_weight)


def build_model(equation_name, args):
    return build_model_from_pde(EQUATIONS[equation_name](), args)


def make_window_initial_state(x):
    return np.cos(x) * (1.0 + np.sin(x))


def build_window_ks_pde(window_length, x_state, y_state):
    base = KuramotoSivashinskyEquation(bbox=[0, 2 * np.pi, 0, window_length])
    x_ic = np.hstack(
        [
            x_state.reshape(-1, 1).astype(np.float32),
            np.zeros((x_state.size, 1), dtype=np.float32),
        ]
    )
    y_ic = y_state.reshape(-1, 1).astype(np.float32)
    base.bcs = [dde.PointSetBC(x_ic, y_ic, component=0)]
    base.loss_config = base.loss_config[: base.num_pde] + [
        {"name": "window_ic", "type": "boundary"}
    ]
    return base


def configure_optimizer(args):
    wrap_windows = args.use_windows and args.window_model_mode == "reuse_model"
    optimizer = "Causal" if wrap_windows else args.optimizer
    if wrap_windows:
        supported = {"adam", "soap", "L-BFGS", "L-BFGS-B", "PSO"}
        if args.optimizer not in supported:
            raise ValueError(
                "--use-windows wraps the selected optimizer with CausalOptimizer; "
                f"choose one of {sorted(supported)} via --optimizer."
            )
        if args.optimizer == "PSO":
            raise ValueError("cyclic window training does not support PSO.")
        dde.optimizers.set_CAUSAL_options(
            base_optimizer=args.optimizer,
            n_time_bins=args.num_windows,
            start_bins=1,
            time_index=args.causal_time_index,
            causal_strategy="cyclic_windows",
            steps_per_window=args.window_steps_per_window,
            x_state=args.window_x_state,
            window_ic_weight=args.window_ic_weight,
            verbose=args.window_verbose,
        )

    if args.optimizer == "PSO":
        dde.optimizers.set_PSO_options(
            pop_size=args.pso_pop_size,
            b=args.pso_b,
            c1=args.pso_c1,
            c2=args.pso_c2,
            lr=args.pso_lr,
            betas=(args.pso_beta1, args.pso_beta2),
            c_decrease=args.pso_c_decrease,
            variance=args.pso_variance,
            epsilon=args.pso_epsilon,
            n_iter=args.pso_n_iter,
        )
    elif args.optimizer in {"L-BFGS", "L-BFGS-B"}:
        dde.optimizers.set_LBFGS_options(
            lr=args.lbfgs_lr,
            maxls=50,
        )
        dde.optimizers.LBFGS_options["iter_per_step"] = 10
        dde.optimizers.LBFGS_options["fun_per_step"] = None
    elif args.optimizer == "soap":
        dde.optimizers.set_SOAP_options(
            beta1=args.soap_beta1,
            beta2=args.soap_beta2,
            shampoo_beta=args.soap_shampoo_beta,
            epsilon=args.soap_epsilon,
            precondition_frequency=args.soap_precondition_frequency,
            max_precondition_dim=args.soap_max_precondition_dim,
            bias_correction=args.soap_bias_correction,
        )
    elif args.optimizer == "pcgrad":
        dde.optimizers.set_PCGRAD_options(
            base_optimizer=args.pcgrad_base_optimizer,
        )
    elif args.optimizer in {"SSBroyden", "ssbroyden"}:
        dde.optimizers.set_SSBROYDEN_options(
            lr=args.ssbroyden_lr,
            tolerance_grad=args.ssbroyden_tolerance_grad,
            debug=args.ssbroyden_debug,
            debug_every=args.ssbroyden_debug_every,
        )
    return optimizer


def make_callbacks(args):
    callbacks = []
    if not args.no_callbacks:
        callbacks.extend(
            [
                TesterCallback(log_every=args.log_every),
                LossCallback(verbose=args.loss_verbose),
            ]
        )
        if args.use_causal_loss:
            callbacks.append(
                CausalDiagnosticsCallback(
                    log_every=args.log_every,
                    verbose=args.causal_diagnostics_verbose,
                )
            )
        if args.plot_every > 0:
            callbacks.append(PlotCallback(log_every=args.plot_every, fast=args.fast_plot))
    if args.resample_collocation:
        callbacks.append(
            dde.callbacks.PDEPointResampler(
                period=args.resample_every,
                pde_points=True,
                bc_points=False,
            )
        )
    if not callbacks:
        return None
    return callbacks


def make_window_callbacks(args):
    callbacks = []
    if not args.no_callbacks:
        callbacks.append(LossCallback(verbose=args.loss_verbose))
        if args.use_causal_loss:
            callbacks.append(
                CausalDiagnosticsCallback(
                    log_every=args.log_every,
                    verbose=args.causal_diagnostics_verbose,
                )
            )
    if args.resample_collocation:
        callbacks.append(
            dde.callbacks.PDEPointResampler(
                period=args.resample_every,
                pde_points=True,
                bc_points=False,
            )
        )
    return callbacks or None


def apply_causal_loss_options(model, args):
    if args.use_causal_loss:
        model.causal_loss_options = {
            "enabled": True,
            "num_chunks": args.causal_num_chunks,
            "tol": args.causal_tol,
            "time_index": args.causal_time_index,
            "include_ic_in_weights": args.causal_include_ic,
            "ic_weight_in_causal": args.causal_ic_weight,
        }


def run_ks_new_model_windows(args):
    if args.seed is not None:
        dde.config.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    reference_pde = KuramotoSivashinskyEquation()
    t_min, t_max = reference_pde.bbox[2], reference_pde.bbox[3]
    window_length = (t_max - t_min) / args.num_windows
    x_state = np.linspace(
        reference_pde.bbox[0],
        reference_pde.bbox[1],
        args.window_state_grid_size,
        dtype=np.float32,
    )
    y_state = make_window_initial_state(x_state).astype(np.float32)

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    run_name = args.name or "ks_new_model_windows"
    base_save_path = os.path.join(args.out, f"{timestamp}-{run_name}")
    os.makedirs(base_save_path, exist_ok=True)

    np.savetxt(
        os.path.join(base_save_path, "window_state_x.txt"),
        x_state.reshape(-1, 1),
        header="x coordinates used to pass predicted ICs between window models",
    )

    for window_idx in range(args.num_windows):
        print(
            f"Training KS window {window_idx + 1}/{args.num_windows} "
            f"with a fresh neural network."
        )
        pde = build_window_ks_pde(window_length, x_state, y_state)
        model, loss_weights = build_model_from_pde(pde, args)
        apply_causal_loss_options(model, args)
        if args.weight_decay > 0:
            model.net.regularizer = ("l2", args.weight_decay)

        optimizer = configure_optimizer(args)
        model.compile(optimizer, lr=args.lr, loss_weights=loss_weights)

        window_save_path = os.path.join(
            base_save_path,
            f"window_{window_idx + 1:03d}_t_{t_min + window_idx * window_length:.6f}_"
            f"{t_min + (window_idx + 1) * window_length:.6f}",
        )
        os.makedirs(window_save_path, exist_ok=True)
        model.train(
            iterations=args.iterations,
            display_every=args.log_every,
            callbacks=make_window_callbacks(args),
            model_save_path=window_save_path,
            save_model=args.save_model,
        )

        x_right = np.hstack(
            [
                x_state.reshape(-1, 1),
                np.full((x_state.size, 1), window_length, dtype=np.float32),
            ]
        )
        y_state = model.predict(x_right).reshape(-1, 1).astype(np.float32)
        np.savetxt(
            os.path.join(base_save_path, f"window_{window_idx + 1:03d}_right_state.txt"),
            np.hstack([x_state.reshape(-1, 1), y_state]),
            header="x, predicted_u_at_right_window_boundary",
        )


def run_one(equation_name, args):
    if args.use_windows and args.window_model_mode == "new_model":
        if equation_name not in {"ks", "kuramoto-sivashinsky"}:
            raise ValueError("--window-model-mode new_model is implemented only for KS.")
        run_ks_new_model_windows(args)
        return None, None

    if args.seed is not None:
        dde.config.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    model, loss_weights = build_model(equation_name, args)
    apply_causal_loss_options(model, args)
    if args.weight_decay > 0:
        model.net.regularizer = ("l2", args.weight_decay)
    if args.use_windows and args.window_model_mode == "reuse_model":
        args.window_x_state = build_state_grid(model.pde, args.window_state_grid_size)
    else:
        args.window_x_state = None
    optimizer = configure_optimizer(args)
    model.compile(optimizer, lr=args.lr, loss_weights=loss_weights)

    run_name = args.name or equation_name.replace("-", "_")
    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(
        args.out,
        f"{timestamp}-{run_name}-pinn-{optimizer.lower()}",
    )
    os.makedirs(save_path, exist_ok=True)

    print(
        f"Training {equation_name} with plain PINN optimizer={optimizer} "
        f"for {args.iterations} iterations."
    )
    losshistory, train_state = model.train(
        iterations=args.iterations,
        display_every=args.log_every,
        callbacks=make_callbacks(args),
        model_save_path=save_path,
        save_model=args.save_model,
    )

    return losshistory, train_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Gray-Scott and Kuramoto-Sivashinsky PINNs without causal wrapper."
    )
    parser.add_argument(
        "--equation",
        choices=["gs", "grayscott", "gray-scott", "ks", "kuramoto-sivashinsky", "both"],
        default="kuramoto-sivashinsky",
    )
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bc-loss-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default="runs_plain")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=0)
    parser.add_argument("--fast-plot", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--loss-verbose", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-callbacks", action="store_true")
    parser.add_argument("--save-model", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--use-causal-loss", action="store_true")
    parser.add_argument("--causal-num-chunks", type=int, default=16)
    parser.add_argument("--causal-tol", type=float, default=0.1)
    parser.add_argument("--causal-time-index", type=int, default=-1)
    parser.add_argument("--causal-include-ic", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--causal-ic-weight", type=float, default=0.0)
    parser.add_argument(
        "--causal-diagnostics-verbose",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )

    parser.add_argument("--use-fourier-features", action="store_true")
    parser.add_argument("--fourier-num-modes-x", type=int, default=16)
    parser.add_argument("--fourier-include-raw-x", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--fourier-include-bias", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--resample-collocation", action="store_true")
    parser.add_argument("--resample-every", type=int, default=1)

    parser.add_argument("--use-windows", action="store_true")
    parser.add_argument("--num-windows", type=int, default=1)
    parser.add_argument(
        "--window-model-mode",
        choices=["reuse_model", "new_model"],
        default="reuse_model",
    )
    parser.add_argument("--window-steps-per-window", type=int, default=200)
    parser.add_argument("--window-state-grid-size", type=int, default=128)
    parser.add_argument("--window-ic-weight", type=float, default=100.0)
    parser.add_argument("--window-verbose", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument(
        "--optimizer",
        choices=[
            "adam",
            "pcgrad",
            "soap",
            "L-BFGS",
            "L-BFGS-B",
            "PSO",
            "sgd",
            "rmsprop",
            "adamw",
            "SSBroyden",
            "ssbroyden",
        ],
        default="ssbroyden",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--pso-pop-size", type=int, default=10)
    parser.add_argument("--pso-b", type=float, default=0.9)
    parser.add_argument("--pso-c1", type=float, default=8e-2)
    parser.add_argument("--pso-c2", type=float, default=5e-1)
    parser.add_argument("--pso-lr", type=float, default=0.0)
    parser.add_argument("--pso-beta1", type=float, default=0.99)
    parser.add_argument("--pso-beta2", type=float, default=0.999)
    parser.add_argument("--pso-c-decrease", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--pso-variance", type=float, default=1.0)
    parser.add_argument("--pso-epsilon", type=float, default=1e-8)
    parser.add_argument("--pso-n-iter", type=int, default=2000)

    parser.add_argument("--lbfgs-lr", type=float, default=1)
    parser.add_argument("--pcgrad-base-optimizer", choices=["adam"], default="adam")
    parser.add_argument("--ssbroyden-lr", type=float, default=1.0)
    parser.add_argument("--ssbroyden-tolerance-grad", type=float, default=1e-10)
    parser.add_argument("--ssbroyden-debug", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--ssbroyden-debug-every", type=int, default=100)
    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-epsilon", type=float, default=1e-8)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)
    parser.add_argument(
        "--soap-bias-correction",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.equation == "both":
        for equation_name in ("gs", "ks"):
            run_one(equation_name, args)
    else:
        run_one(args.equation, args)


if __name__ == "__main__":
    main()
