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


def loss_weights_for(pde, bc_loss_weight):
    weights = np.ones(pde.num_loss, dtype=float)
    for i, config in enumerate(pde.loss_config):
        if config.get("type") in ("boundary", "initial", "ic"):
            weights[i] = bc_loss_weight
    return weights


def build_model(equation_name, hidden_layers, bc_loss_weight):
    pde = EQUATIONS[equation_name]()
    layers = [
        pde.input_dim,
        *parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)),
        pde.output_dim,
    ]
    net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
    return pde.create_model(net), loss_weights_for(pde, bc_loss_weight)


def configure_optimizer(args):
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


def make_callbacks(args):
    if args.no_callbacks:
        return None

    callbacks = [
        TesterCallback(log_every=args.log_every),
        LossCallback(verbose=args.loss_verbose),
    ]
    if args.plot_every > 0:
        callbacks.append(PlotCallback(log_every=args.plot_every, fast=args.fast_plot))
    return callbacks


def run_one(equation_name, args):
    if args.seed is not None:
        dde.config.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    model, loss_weights = build_model(
        equation_name,
        args.hidden_layers,
        args.bc_loss_weight,
    )
    if args.weight_decay > 0:
        model.net.regularizer = ("l2", args.weight_decay)
    configure_optimizer(args)
    model.compile(args.optimizer, lr=args.lr, loss_weights=loss_weights)

    run_name = equation_name.replace("-", "_")
    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(
        args.out,
        f"{timestamp}-{run_name}-pinn-{args.optimizer.lower()}",
    )
    os.makedirs(save_path, exist_ok=True)

    print(
        f"Training {equation_name} with plain PINN optimizer={args.optimizer} "
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
