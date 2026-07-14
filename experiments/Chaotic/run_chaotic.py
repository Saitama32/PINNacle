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

from src.model import PeriodicFourierFeatures, ResNet
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.utils.args import parse_hidden_layers
from src.utils.callbacks import (
    CausalDiagnosticsCallback,
    LossCallback,
    PlotCallback,
    TesterCallback,
)
from src.utils.fam import FAMTrainConfig, FAMTrainer, LossWeightAdapter


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


def parse_resnet_shape(hidden_layers):
    layers = parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers))
    if not layers:
        raise ValueError("ResNet requires at least one hidden layer specification.")

    width = layers[0]
    if any(layer != width for layer in layers):
        raise ValueError(
            "ResNet mode only supports uniform hidden layers, for example '50*5' or '64,64,64'."
        )
    return width, len(layers)


def build_network(
    pde,
    hidden_layers,
    net_type,
    fourier_features=None,
    fourier_sigma=None,
    fourier_include_raw_x=False,
    fourier_include_bias=True,
):
    hidden = parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers))
    if net_type == "mlp":
        layers = [
            pde.input_dim,
            *hidden,
            pde.output_dim,
        ]
        return dde.nn.FNN(layers, "tanh", "Glorot normal").float()

    if net_type == "resnet":
        width, num_blocks = parse_resnet_shape(hidden_layers)
        return ResNet(
            input_size=pde.input_dim,
            output_size=pde.output_dim,
            width=width,
            num_blocks=num_blocks,
            activation="tanh",
            kernel_initializer="Glorot normal",
        ).float()

    if net_type == "fourier-mlp":
        if isinstance(pde, KuramotoSivashinskyEquation):
            feature_encoder = PeriodicFourierFeatures(
                x_period=pde.bbox[1] - pde.bbox[0],
                num_modes_x=fourier_features,
                include_t=True,
                include_raw_x=fourier_include_raw_x,
                include_bias=fourier_include_bias,
            )
        elif isinstance(pde, GrayScottEquation):
            feature_encoder = PeriodicFourierFeatures(
                x_period=pde.bbox[1] - pde.bbox[0],
                y_period=pde.bbox[3] - pde.bbox[2],
                num_modes_x=fourier_features,
                include_t=True,
                include_raw_x=fourier_include_raw_x,
                include_bias=fourier_include_bias,
            )
        else:
            raise ValueError("fourier-mlp is currently supported only for KS and Gray-Scott.")
        layers = [
            feature_encoder.out_dim,
            *hidden,
            pde.output_dim,
        ]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal").float()
        net.apply_feature_transform(feature_encoder)
        return net

    raise ValueError(f"Unsupported network type: {net_type}")


def build_model(
    equation_name,
    hidden_layers,
    bc_loss_weight,
    net_type,
    fourier_features=None,
    fourier_sigma=None,
    fourier_include_raw_x=False,
    fourier_include_bias=True,
):
    pde = EQUATIONS[equation_name]()
    net = build_network(
        pde,
        hidden_layers,
        net_type,
        fourier_features=fourier_features,
        fourier_sigma=fourier_sigma,
        fourier_include_raw_x=fourier_include_raw_x,
        fourier_include_bias=fourier_include_bias,
    )
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
    elif args.optimizer == "ZOCGE":
        dde.optimizers.set_ZOCGE_options(
            mu=args.zo_step_size,
            sparsity=args.zo_sparsity,
            prune_method=args.zo_prune_method,
            remask_interval=args.zo_remask_interval,
            feature_reuse=args.zo_feature_reuse,
            grasp_sample_size=args.zo_grasp_sample_size,
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


def validate_args(args):
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")

    if args.optimizer == "adamw" and args.weight_decay == 0:
        raise ValueError(
            "AdamW requires --weight-decay > 0. "
            "Either pass a positive value, for example --weight-decay 1e-4, "
            "or switch to --optimizer adam."
        )

    if args.famaw_causal_window and args.sampling_method not in {"fam-w", "famaw-w"}:
        raise ValueError("--famaw-causal-window is only supported with --sampling-method fam-w or famaw-w.")
    if args.famaw_causal_sigma is not None and args.famaw_causal_sigma <= 0:
        raise ValueError("--famaw-causal-sigma must be positive.")
    if args.famaw_causal_w0 <= 0:
        raise ValueError("--famaw-causal-w0 must be positive.")
    if not np.isfinite(args.famaw_causal_threshold):
        raise ValueError("--famaw-causal-threshold must be finite.")
    if args.fam_pde_point_weighting and args.sampling_method not in {"fam-w", "famaw-w"}:
        raise ValueError("--fam-pde-point-weighting is only supported with --sampling-method fam-w or famaw-w.")
    if args.fam_pde_point_weight_coeff < 0 or not np.isfinite(args.fam_pde_point_weight_coeff):
        raise ValueError("--fam-pde-point-weight-coeff must be finite and non-negative.")
    if args.sampling_refresh_every <= 0:
        raise ValueError("--sampling-refresh-every must be positive.")
    if args.fourier_features <= 0:
        raise ValueError("--fourier-features must be positive.")
    if args.fourier_sigma <= 0 or not np.isfinite(args.fourier_sigma):
        raise ValueError("--fourier-sigma must be positive and finite.")
    if args.causal_num_chunks <= 0:
        raise ValueError("--causal-num-chunks must be positive.")


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


def make_callbacks(args):
    if args.no_callbacks:
        return None

    callbacks = [
        TesterCallback(log_every=args.log_every),
        LossCallback(verbose=args.loss_verbose),
        PlotCallback(log_every=args.plot_every, fast=args.fast_plot),
    ]
    if args.use_causal_loss:
        callbacks.append(
            CausalDiagnosticsCallback(
                log_every=args.log_every,
                verbose=args.causal_diagnostics_verbose,
            )
        )
    return callbacks


def run_one(equation_name, args):
    if args.seed is not None:
        dde.config.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.net == "fourier-mlp" and not np.isclose(args.fourier_sigma, 3.0):
        print(
            "Warning: --fourier-sigma is kept only for CLI backward compatibility and is ignored "
            "by the current periodic Fourier features."
        )

    model, loss_weights = build_model(
        equation_name,
        args.hidden_layers,
        args.bc_loss_weight,
        args.net,
        fourier_features=args.fourier_features,
        fourier_sigma=args.fourier_sigma,
        fourier_include_raw_x=args.fourier_include_raw_x,
        fourier_include_bias=args.fourier_include_bias,
    )
    apply_causal_loss_options(model, args)
    if args.weight_decay > 0:
        model.net.regularizer = ("l2", args.weight_decay)
    configure_optimizer(args)
    if args.sampling_method == "none":
        model.compile(args.optimizer, lr=args.lr, loss_weights=loss_weights)
    else:
        loss_weight_adapter = LossWeightAdapter(np.ones_like(loss_weights))
        model.compile(args.optimizer, lr=args.lr, loss_weights=loss_weight_adapter)

    run_name = equation_name.replace("-", "_")
    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(
        args.out,
        f"{timestamp}-{run_name}-pinn-{args.net}-{args.optimizer.lower()}-{args.sampling_method}",
    )
    os.makedirs(save_path, exist_ok=True)

    print(
        f"Training {equation_name} with {args.net} PINN optimizer={args.optimizer} "
        f"sampling={args.sampling_method} for {args.iterations} iterations."
    )
    callbacks = make_callbacks(args)
    if args.sampling_method == "none":
        losshistory, train_state = model.train(
            iterations=args.iterations,
            display_every=args.log_every,
            callbacks=callbacks,
            model_save_path=save_path,
            save_model=args.save_model,
        )
    else:
        if args.fam_fixed_points is None or args.fam_movable_points is None:
            raise ValueError(
                "--fam-fixed-points and --fam-movable-points are required when sampling-method is fam-w or famaw-w."
            )
        fam_config = FAMTrainConfig(
            mode=args.sampling_method,
            iterations=args.iterations,
            refresh_every=args.sampling_refresh_every,
            weight_lr=args.faw_lr,
            alpha=args.fam_alpha,
            beta=args.fam_beta,
            gamma=args.fam_gamma,
            num_fixed_points=args.fam_fixed_points,
            num_movable_points=args.fam_movable_points,
            display_every=args.log_every,
            save_model=args.save_model,
            save_diagnostics=args.fam_save_diagnostics,
            save_point_plots=args.fam_save_point_plots,
            point_plot_every=args.plot_every,
            causal_window_enabled=args.famaw_causal_window,
            causal_sigma=args.famaw_causal_sigma,
            causal_w0=args.famaw_causal_w0,
            causal_threshold=args.famaw_causal_threshold,
            causal_log_brightness=args.famaw_causal_log_brightness,
            pde_point_weighting_enabled=args.fam_pde_point_weighting,
            pde_point_weight_coeff=args.fam_pde_point_weight_coeff,
        )
        trainer = FAMTrainer(
            model,
            fam_config,
            loss_weight_adapter=loss_weight_adapter,
            callbacks=callbacks,
            model_save_path=save_path,
            seed=args.seed,
            static_loss_weights=loss_weights,
        )
        losshistory, train_state = trainer.train()

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
    parser.add_argument("--net", choices=["mlp", "resnet", "fourier-mlp"], default="mlp")
    parser.add_argument("--fourier-features", type=int, default=10)
    parser.add_argument("--fourier-sigma", type=float, default=5.0)
    parser.add_argument("--fourier-include-raw-x", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--fourier-include-bias", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bc-loss-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default="runs_plain")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--plot-every", type=int, default=100)
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
    parser.add_argument(
        "--sampling-method",
        choices=["none", "fam-w", "famaw-w"],
        default="fam-w",
    )
    parser.add_argument("--sampling-refresh-every", type=int, default=100)
    parser.add_argument("--fam-alpha", type=float, default=0.6)
    parser.add_argument("--fam-beta", type=float, default=1.0)
    parser.add_argument("--fam-gamma", type=float, default=0.8)
    parser.add_argument("--faw-lr", type=float, default=1e-3)
    parser.add_argument("--fam-fixed-points", type=int, default=2000)
    parser.add_argument("--fam-movable-points", type=int, default=1500)
    parser.add_argument("--fam-save-diagnostics", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--fam-save-point-plots", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--famaw-causal-window", action="store_true", default=False)
    parser.add_argument("--famaw-causal-sigma", type=float, default=0.1)
    parser.add_argument("--famaw-causal-w0", type=float, default=1)
    parser.add_argument("--famaw-causal-threshold", type=float, default=1.05)
    parser.add_argument("--famaw-causal-log-brightness", action="store_true", default=False)
    parser.add_argument("--fam-pde-point-weighting", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--fam-pde-point-weight-coeff", type=float, default=10000.0)

    parser.add_argument(
        "--optimizer",
        choices=[
            "adam",
            "pcgrad",
            "soap",
            "L-BFGS",
            "L-BFGS-B",
            "PSO",
            "ZOCGE",
            "sgd",
            "rmsprop",
            "adamw",
            "SSBroyden",
            "adam",
        ],
        default="sgd",
    )
    parser.add_argument("--weight-decay", type=float, default=0)

    parser.add_argument("--pso-pop-size", type=int, default=30)
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
    parser.add_argument("--zo-step-size", type=float, default=1e-3)
    parser.add_argument("--zo-sparsity", type=float, default=0.9)
    parser.add_argument(
        "--zo-prune-method",
        choices=["random", "zo_grasp"],
        default="zo_grasp",
    )
    parser.add_argument("--zo-remask-interval", type=int, default=10)
    parser.add_argument("--zo-feature-reuse", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--zo-grasp-sample-size", type=int, default=32)

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
    validate_args(args)
    if args.equation == "both":
        for equation_name in ("gs", "ks"):
            run_one(equation_name, args)
    else:
        run_one(args.equation, args)


if __name__ == "__main__":
    main()
