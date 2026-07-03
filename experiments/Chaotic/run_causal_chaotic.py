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
from deepxde.callbacks import Callback

from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import LossCallback, PlotCallback, TesterCallback


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


def loss_weights_for(pde):
    weights = np.ones(pde.num_loss, dtype=float)
    for i, config in enumerate(pde.loss_config):
        if config.get("type") in ("boundary", "initial", "ic"):
            weights[i] = 10000.0
    return weights


def build_model(equation_name, hidden_layers):
    pde = EQUATIONS[equation_name]()
    layers = [
        pde.input_dim,
        *parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)),
        pde.output_dim,
    ]
    net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
    return pde.create_model(net), loss_weights_for(pde)


def build_state_grid(pde, state_grid_size):
    spatial_dim = pde.input_dim - 1
    if spatial_dim < 1:
        raise ValueError("cyclic_windows requires at least one spatial dimension.")

    axes = []
    for i in range(spatial_dim):
        lo = pde.bbox[2 * i]
        hi = pde.bbox[2 * i + 1]
        axes.append(np.linspace(lo, hi, state_grid_size, dtype=np.float32))

    if spatial_dim == 1:
        return axes[0][:, None]

    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1).astype(np.float32)


class CausalValidationCallback(Callback):
    def __init__(self, log_every=100, time_index=-1, verbose=True):
        super().__init__()
        self.log_every = log_every
        self.time_index = time_index
        self.verbose = verbose
        self.epochs_since_last_log = 0
        self.rows = []
        self.disabled = False

    def on_train_begin(self):
        pde = self.model.pde
        if pde.ref_data is None:
            self.disabled = True
            return

        ref_data = pde.ref_data
        nan_mask = np.isnan(ref_data).any(axis=1)
        ref_data = ref_data[~nan_mask]
        self.x_ref = ref_data[:, : pde.input_dim]
        self.y_ref = ref_data[:, pde.input_dim :]
        self.t_ref = self.x_ref[:, self.time_index]
        self.t_min = float(np.min(self.t_ref))
        self.save_path = os.path.join(self.model.model_save_path, "causal_validation.txt")

    def _threshold(self):
        threshold_fn = getattr(self.model.opt, "current_t_threshold", None)
        threshold = threshold_fn() if callable(threshold_fn) else None
        if threshold is None:
            return self.t_min
        return float(threshold)

    def _window_info(self):
        bounds_fn = getattr(self.model.opt, "current_window_bounds", None)
        if callable(bounds_fn):
            t_left, t_right = bounds_fn()
        else:
            t_left, t_right = None, None

        if t_left is None or t_right is None:
            t_left = self.t_min
            t_right = self._threshold()

        cycle = getattr(self.model.opt, "current_cycle", 0)
        window = getattr(self.model.opt, "current_window", 0)
        strategy = getattr(self.model.opt, "causal_strategy", "prefix")
        return strategy, int(cycle), int(window), float(t_left), float(t_right)

    def _metrics(self, mask):
        count = int(np.sum(mask))
        if count == 0:
            return count, np.nan, np.nan, np.nan

        x = self.x_ref[mask]
        y_true = self.y_ref[mask]
        y_pred = self.model.predict(x)
        mse = float(np.mean((y_pred - y_true) ** 2))
        rmse = float(np.sqrt(mse))
        y_norm = float(np.sqrt(np.mean(y_true**2)))
        l2re = rmse / (y_norm + 1e-12)
        return count, mse, rmse, l2re

    def on_epoch_end(self):
        if self.disabled:
            return

        self.epochs_since_last_log += 1
        if self.log_every is None or self.epochs_since_last_log < self.log_every:
            return
        self.epochs_since_last_log = 0

        strategy, cycle, window, t_left, t_right = self._window_info()
        if strategy == "cyclic_windows":
            active_mask = (self.t_ref >= t_left) & (self.t_ref <= t_right)
            future_mask = ~active_mask
        else:
            active_mask = self.t_ref <= t_right
            future_mask = self.t_ref > t_right
        full_mask = np.ones_like(self.t_ref, dtype=bool)

        active_count, active_mse, active_rmse, active_l2re = self._metrics(active_mask)
        future_count, future_mse, future_rmse, future_l2re = self._metrics(future_mask)
        full_count, full_mse, full_rmse, full_l2re = self._metrics(full_mask)

        epoch = self.model.train_state.step
        self.rows.append(
            [
                epoch,
                cycle,
                window,
                t_left,
                t_right,
                active_count,
                future_count,
                active_mse,
                active_rmse,
                active_l2re,
                future_mse,
                future_rmse,
                future_l2re,
                full_mse,
                full_rmse,
                full_l2re,
            ]
        )

        if self.verbose:
            print(
                "CausalValidation: "
                f"epoch {epoch} cycle {cycle} window {window + 1} "
                f"t=[{t_left:.10e}, {t_right:.10e}] "
                f"active_count {active_count} active_MSE {active_mse:.10e} "
                f"active_RMSE {active_rmse:.10e} active_L2RE {active_l2re:.10e} "
                f"future_count {future_count} future_MSE {future_mse:.10e} "
                f"future_L2RE {future_l2re:.10e} "
                f"full_count {full_count} full_MSE {full_mse:.10e} "
                f"full_L2RE {full_l2re:.10e}"
            )

    def on_train_end(self):
        if self.disabled or not self.rows:
            return

        np.savetxt(
            self.save_path,
            np.asarray(self.rows, dtype=float),
            header=(
                "epoch, cycle, window, t_left, t_right, active_count, future_count, "
                "active_mse, active_rmse, active_l2re, "
                "future_mse, future_rmse, future_l2re, "
                "full_mse, full_rmse, full_l2re"
            ),
        )


def configure_causal_optimizer(args, pde):
    if args.base_optimizer == "PSO" and args.causal_strategy == "cyclic_windows":
        raise ValueError(
            "cyclic_windows does not support PSO yet. Use adam, soap, L-BFGS, or L-BFGS-B."
        )

    x_state = None
    if args.causal_strategy == "cyclic_windows":
        x_state = build_state_grid(pde, args.state_grid_size)

    dde.optimizers.set_CAUSAL_options(
        base_optimizer=args.base_optimizer,
        n_time_bins=args.n_time_bins,
        start_bins=args.start_bins,
        time_index=args.time_index,
        unlock_every=args.unlock_every,
        unlock_tol=args.unlock_tol,
        min_steps_per_bin=args.min_steps_per_bin,
        bc_mode=args.bc_mode,
        min_points_per_bc=args.min_points_per_bc,
        causal_strategy=args.causal_strategy,
        steps_per_window=args.steps_per_window,
        state_alpha=args.state_alpha,
        x_state=x_state,
        window_ic_weight=args.window_ic_weight,
        verbose=args.causal_verbose,
    )

    if args.base_optimizer == "PSO":
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
    elif args.base_optimizer in {"L-BFGS", "L-BFGS-B"}:
        dde.optimizers.set_LBFGS_options(
            lr=args.lbfgs_lr,
            maxls=50,
        )
        dde.optimizers.LBFGS_options["iter_per_step"] = 10
        dde.optimizers.LBFGS_options["fun_per_step"] = None
    elif args.base_optimizer == "soap":
        dde.optimizers.set_SOAP_options(
            beta1=args.soap_beta1,
            beta2=args.soap_beta2,
            shampoo_beta=args.soap_shampoo_beta,
            epsilon=args.soap_epsilon,
            precondition_frequency=args.soap_precondition_frequency,
            max_precondition_dim=args.soap_max_precondition_dim,
            bias_correction=args.soap_bias_correction,
        )


def make_callbacks(args):
    if args.no_callbacks:
        return None

    callbacks = [
        TesterCallback(log_every=args.log_every),
        LossCallback(verbose=args.loss_verbose),
    ]
    if args.causal_val:
        causal_val_every = args.causal_val_every or args.log_every
        causal_val_time_index = (
            args.causal_val_time_index
            if args.causal_val_time_index is not None
            else args.time_index
        )
        callbacks.append(
            CausalValidationCallback(
                log_every=causal_val_every,
                time_index=causal_val_time_index,
                verbose=True,
            )
        )
    if args.plot_every > 0:
        callbacks.append(PlotCallback(log_every=args.plot_every, fast=args.fast_plot))
    return callbacks


def run_one(equation_name, args):
    if args.seed is not None:
        dde.config.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    model, loss_weights = build_model(equation_name, args.hidden_layers)
    configure_causal_optimizer(args, model.pde)

    model.compile("Causal", lr=args.lr, loss_weights=loss_weights)

    run_name = equation_name.replace("-", "_")
    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(
        args.out,
        f"{timestamp}-{run_name}-causal-{args.base_optimizer.lower()}",
    )
    os.makedirs(save_path, exist_ok=True)

    print(
        f"Training {equation_name} with Causal(base_optimizer={args.base_optimizer}) "
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
        description="Train Gray-Scott and Kuramoto-Sivashinsky PINNs with Causal optimizer."
    )
    parser.add_argument(
        "--equation",
        choices=["gs", "grayscott", "gray-scott", "ks", "kuramoto-sivashinsky", "both"],
        default="kuramoto-sivashinsky",
    )
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--iterations", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default="runs_causal")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=0)
    parser.add_argument("--fast-plot", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--loss-verbose", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-callbacks", action="store_true")
    parser.add_argument("--save-model", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--causal-val", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--causal-val-every", type=int, default=None)
    parser.add_argument("--causal-val-time-index", type=int, default=None)

    parser.add_argument(
        "--base-optimizer",
        choices=["adam", "soap", "L-BFGS", "L-BFGS-B", "PSO"],
        default="soap",
    )
    parser.add_argument("--n-time-bins", type=int, default=20)
    parser.add_argument("--start-bins", type=int, default=1)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--unlock-every", type=int, default=2000)
    parser.add_argument("--unlock-tol", type=float, default=None)
    parser.add_argument("--min-steps-per-bin", type=int, default=20)
    parser.add_argument("--bc-mode", choices=["all", "causal"], default="causal")
    parser.add_argument("--min-points-per-bc", type=int, default=1)
    parser.add_argument(
        "--causal-strategy",
        choices=["prefix", "cyclic_windows"],
        default="prefix",
    )
    parser.add_argument("--steps-per-window", type=int, default=200)
    parser.add_argument("--state-alpha", type=float, default=0.8)
    parser.add_argument("--window-ic-weight", type=float, default=100.0)
    parser.add_argument("--state-grid-size", type=int, default=128)
    parser.add_argument("--causal-verbose", type=str2bool, nargs="?", const=True, default=True)

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
