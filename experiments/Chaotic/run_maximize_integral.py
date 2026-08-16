"""Maximize the KS global/local residual objective inside a parameter box.

The global and weighted local residual terms use gradient ascent. Initial and
periodic constraints remain minimization penalties. Model selection is based
on the largest mean residual objective on reproducible fixed batches.
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time

os.environ["DDEBACKEND"] = "pytorch"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import deepxde as dde
import numpy as np
import torch

from experiments.Chaotic import run_chaotic as chaotic
from src.losses.global_integral import GlobalIntegralLoss, attach_integral_loss_train_step


TRAJECTORY_COLUMNS = [
    "step",
    "total",
    "residual",
    "constraint",
    "global",
    "local",
    "ic",
    "periodic",
    "optimization_objective",
    "parameter_min",
    "parameter_max",
    "bound_fraction",
    "is_best",
]


def cpu_state_dict(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def project_parameters(module, lower, upper):
    """Project every trainable tensor, including one-dimensional biases."""
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.clamp_(min=float(lower), max=float(upper))


def parameter_box_stats(module, lower, upper):
    trainable = [parameter.detach().reshape(-1) for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        return float("nan"), float("nan"), 0.0
    values = torch.cat(trainable)
    tolerance = max(1e-7, 1e-6 * max(1.0, abs(float(lower)), abs(float(upper))))
    at_bound = (values <= float(lower) + tolerance) | (values >= float(upper) - tolerance)
    return (
        float(values.min().cpu()),
        float(values.max().cpu()),
        float(at_bound.float().mean().cpu()),
    )


def install_box_projection(optimizer, module, lower, upper):
    """Apply the projection after each completed optimizer step."""
    original_step = optimizer.step

    def projected_step(closure=None):
        result = original_step(closure)
        project_parameters(module, lower, upper)
        return result

    optimizer.step = projected_step
    optimizer._integral_box_projection = (float(lower), float(upper))
    return optimizer


def build_integral_loss(model, args, seed):
    return GlobalIntegralLoss(
        model=model,
        pde=model.pde,
        batch_size=args.integral_batch_size,
        weight=args.integral_loss_weight,
        warmup_steps=args.integral_warmup_steps,
        start_step=args.integral_start_step,
        quadrature_order=args.integral_quadrature_order,
        local_enabled=args.integral_local_enabled,
        local_weight=args.integral_local_weight,
        local_quadrature_order=args.integral_local_quadrature_order,
        local_hmax=args.integral_local_hmax,
        local_segment_batch_size=args.integral_local_segment_batch_size,
        local_normalize_by_length=args.integral_local_normalize_by_length,
        local_contiguous_chain=args.integral_local_contiguous_chain,
        t0_fraction=args.integral_t0_fraction,
        t_min=args.integral_t_min,
        seed=seed,
        resample_every=args.integral_resample_every,
        initial_condition_enabled=args.integral_ic_enabled,
        initial_condition_weight=args.integral_ic_weight,
        periodic_enabled=args.integral_periodic_enabled,
        periodic_weight=args.integral_periodic_weight,
    )


class FixedEvaluationMaximum(dde.callbacks.Callback):
    """Select the largest objective on deterministic integral batches."""

    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.rows = []
        self.fixed_batches = None
        self.evaluation_loss = None
        self.best_loss = -math.inf
        self.best_step = None
        self.best_state = None
        self.last_evaluated_step = None

    @staticmethod
    def _activate_fixed_batch(loss, batch, step):
        x, t, segments = batch
        loss._set_cached_endpoints(x, t, step)
        loss.cached_local_segments = segments
        loss.last_local_segment_step = step
        loss._cached_local_endpoint_ptrs = loss._endpoint_ptrs(x, t)

    def _make_fixed_batches(self):
        loss = self.evaluation_loss
        batches = []
        loss.seed = self.args.reverse_eval_seed
        loss._generator = None
        for batch_id in range(self.args.reverse_eval_batches):
            x, t = loss.sample_endpoints(step=batch_id, force=True)
            x, t = x.detach().clone(), t.detach().clone()
            loss._set_cached_endpoints(x, t, batch_id)
            segments = copy.deepcopy(loss._get_local_segments(x, t, step=batch_id))
            batches.append((x, t, segments))
        loss._generator = None
        return batches

    @staticmethod
    def _components(loss):
        diagnostics = loss.last_diagnostics
        weighted_components = [
            0.0 if component is None else float(component.detach().cpu())
            for component in loss.last_components
        ]
        residual = weighted_components[0] + weighted_components[1]
        constraint = weighted_components[2] + weighted_components[3]
        return {
            "total": residual + constraint,
            "residual": residual,
            "constraint": constraint,
            "global": float(diagnostics["global_integral_loss"].detach().cpu()),
            "local": float(diagnostics["local_integral_loss"].detach().cpu()),
            "ic": float(diagnostics["initial_condition_loss"].detach().cpu()),
            "periodic": float(diagnostics["periodic_loss"].detach().cpu()),
        }

    def evaluate(self):
        totals = {
            key: 0.0
            for key in ("total", "residual", "constraint", "global", "local", "ic", "periodic")
        }
        for batch_id, batch in enumerate(self.fixed_batches):
            self._activate_fixed_batch(self.evaluation_loss, batch, batch_id)
            try:
                self.evaluation_loss.compute_raw_loss(
                    step=batch_id,
                    endpoints=(batch[0], batch[1]),
                )
                values = self._components(self.evaluation_loss)
            finally:
                dde.grad.clear()
            for key in totals:
                totals[key] += values[key]
        self.model.net.zero_grad(set_to_none=True)
        averaged = {key: value / len(self.fixed_batches) for key, value in totals.items()}
        if not all(math.isfinite(value) for value in averaged.values()):
            raise FloatingPointError("Fixed evaluation produced a non-finite integral loss.")
        return averaged

    def _save_checkpoint(self, filename, state, step, loss):
        if not self.args.save_model:
            return
        torch.save(
            {
                "model_state_dict": state,
                "global_step": int(step),
                "fixed_evaluation_loss": float(loss),
            },
            os.path.join(self.save_path, filename),
        )

    def _write_csv(self):
        with open(
            os.path.join(self.save_path, "reverse_integral_trajectory.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=TRAJECTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

    def record(self, step):
        values = self.evaluate()
        parameter_min, parameter_max, bound_fraction = parameter_box_stats(
            self.model.net,
            self.args.parameter_lower,
            self.args.parameter_upper,
        )
        is_best = values["residual"] > self.best_loss
        if is_best:
            self.best_loss = values["residual"]
            self.best_step = int(step)
            self.best_state = cpu_state_dict(self.model.net)
            self._save_checkpoint(
                "reverse_integral_best.pt",
                self.best_state,
                step,
                self.best_loss,
            )
        self.rows.append({
            "step": int(step),
            **values,
            "optimization_objective": -values["residual"] + values["constraint"],
            "parameter_min": parameter_min,
            "parameter_max": parameter_max,
            "bound_fraction": bound_fraction,
            "is_best": is_best,
        })
        self.last_evaluated_step = int(step)
        self._write_csv()
        print(
            f"[Fixed reverse eval] step={step} residual={values['residual']:.6e} "
            f"constraints={values['constraint']:.6e} best={self.best_loss:.6e} "
            f"bounds={bound_fraction:.3%}"
        )
        return values

    def on_train_begin(self):
        self.evaluation_loss = build_integral_loss(
            self.model,
            self.args,
            self.args.reverse_eval_seed,
        )
        self.fixed_batches = self._make_fixed_batches()
        initial_state = cpu_state_dict(self.model.net)
        values = self.record(self.model.train_state.step)
        self._save_checkpoint(
            "reverse_integral_initial.pt",
            initial_state,
            self.model.train_state.step,
            values["residual"],
        )

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.args.reverse_eval_every == 0:
            self.record(step)

    def on_train_end(self):
        step = self.model.train_state.step
        if self.last_evaluated_step != step:
            self.record(step)
        final_state = cpu_state_dict(self.model.net)
        final_loss = self.rows[-1]["residual"]
        self._save_checkpoint("reverse_integral_final.pt", final_state, step, final_loss)
        if self.best_state is None:
            raise RuntimeError("No finite fixed-evaluation checkpoint was produced.")
        self.model.net.load_state_dict(self.best_state)
        self._save_checkpoint(
            "reverse_integral_selected.pt",
            self.best_state,
            self.best_step,
            self.best_loss,
        )
        print(
            f"[Fixed reverse eval] restored maximum from step={self.best_step}, "
            f"loss={self.best_loss:.6e}."
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Maximize KS global/local residual integral losses.")
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument("--net", choices=["mlp", "resnet", "fourier-mlp"], default="mlp")
    parser.add_argument("--fourier-features", type=int, default=10)
    parser.add_argument("--fourier-sigma", type=float, default=5.0)
    parser.add_argument("--fourier-include-raw-x", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--fourier-include-bias", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bc-loss-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default="runs_maximize_integral")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=100)
    parser.add_argument("--fast-plot", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--loss-verbose", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--ks-diagnostics-verbose", type=chaotic.str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--ks-diagnostics-chunk-every", type=int, default=1000)
    parser.add_argument("--no-callbacks", action="store_true")
    parser.add_argument("--save-model", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "muon", "soap"], default="muon")

    parser.add_argument("--parameter-lower", type=float, default=-1.0)
    parser.add_argument("--parameter-upper", type=float, default=1.0)
    parser.add_argument("--reverse-eval-every", type=int, default=100)
    parser.add_argument("--reverse-eval-batches", type=int, default=4)
    parser.add_argument("--reverse-eval-seed", type=int, default=98765)

    parser.add_argument("--use-integral-loss", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--integral-only", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--integral-loss-weight", type=float, default=1.0)
    parser.add_argument("--integral-batch-size", type=int, default=256)
    parser.add_argument("--integral-warmup-steps", type=int, default=0)
    parser.add_argument("--integral-start-step", type=int, default=0)
    parser.add_argument("--integral-quadrature-order", type=int, default=4)
    parser.add_argument("--integral-local-enabled", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--integral-local-weight", type=float, default=10.0)
    parser.add_argument("--integral-local-quadrature-order", type=int, default=4)
    parser.add_argument("--integral-local-hmax", type=float, default=0.05)
    parser.add_argument("--integral-local-segment-batch-size", type=int, default=256)
    parser.add_argument("--integral-local-normalize-by-length", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--integral-local-contiguous-chain", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--integral-t0-fraction", type=float, default=0.1)
    parser.add_argument("--integral-t-min", type=float, default=0.0)
    parser.add_argument("--integral-resample-every", type=int, default=1)
    parser.add_argument("--integral-seed", type=int, default=None)
    parser.add_argument("--integral-ic-enabled", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--integral-ic-weight", type=float, default=10.0)
    parser.add_argument("--integral-periodic-enabled", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--integral-periodic-weight", type=float, default=100.0)

    parser.add_argument("--soap-beta1", type=float, default=0.99)
    parser.add_argument("--soap-beta2", type=float, default=0.999)
    parser.add_argument("--soap-shampoo-beta", type=float, default=None)
    parser.add_argument("--soap-epsilon", type=float, default=1e-8)
    parser.add_argument("--soap-precondition-frequency", type=int, default=10)
    parser.add_argument("--soap-max-precondition-dim", type=int, default=4096)
    parser.add_argument("--soap-bias-correction", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adam-lr", type=float, default=3e-4)
    parser.add_argument("--muon-adam-beta1", type=float, default=0.9)
    parser.add_argument("--muon-adam-beta2", type=float, default=0.95)
    parser.add_argument("--muon-adam-epsilon", type=float, default=1e-10)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-adam-weight-decay", type=float, default=0.0)
    return parser


def validate_args(args):
    if args.iterations < 1:
        raise ValueError("--iterations must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive.")
    if not math.isfinite(args.parameter_lower) or not math.isfinite(args.parameter_upper):
        raise ValueError("Parameter bounds must be finite.")
    if args.parameter_lower >= args.parameter_upper:
        raise ValueError("--parameter-lower must be smaller than --parameter-upper.")
    if args.reverse_eval_every < 1 or args.reverse_eval_batches < 1:
        raise ValueError("Reverse evaluation frequency and batch count must be positive.")
    if not args.use_integral_loss or not args.integral_only:
        raise ValueError("This runner requires --use-integral-loss true and --integral-only true.")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError("--weight-decay must be finite and non-negative.")
    for name in (
        "integral_loss_weight",
        "integral_local_weight",
        "integral_ic_weight",
        "integral_periodic_weight",
    ):
        value = getattr(args, name)
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")
    if args.integral_batch_size < 1 or args.integral_quadrature_order < 1:
        raise ValueError("Integral batch size and quadrature order must be positive.")
    if args.integral_local_quadrature_order < 1 or args.integral_local_segment_batch_size < 1:
        raise ValueError("Local quadrature and segment batch sizes must be positive.")
    if not math.isfinite(args.integral_local_hmax) or args.integral_local_hmax <= 0:
        raise ValueError("--integral-local-hmax must be finite and positive.")
    if args.log_every < 1 or args.plot_every < 1:
        raise ValueError("Logging and plotting frequencies must be positive.")


def configure_custom_optimizers(args):
    dde.optimizers.set_SOAP_options(
        beta1=args.soap_beta1,
        beta2=args.soap_beta2,
        shampoo_beta=args.soap_shampoo_beta,
        epsilon=args.soap_epsilon,
        precondition_frequency=args.soap_precondition_frequency,
        max_precondition_dim=args.soap_max_precondition_dim,
        bias_correction=args.soap_bias_correction,
    )
    dde.optimizers.set_MUON_options(
        momentum=args.muon_momentum,
        nesterov=args.muon_nesterov,
        ns_steps=args.muon_ns_steps,
        adam_lr=args.muon_adam_lr,
        adam_betas=(args.muon_adam_beta1, args.muon_adam_beta2),
        adam_eps=args.muon_adam_epsilon,
        muon_weight_decay=args.muon_weight_decay,
        adam_weight_decay=args.muon_adam_weight_decay,
    )


def run(args):
    validate_args(args)
    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, loss_weights = chaotic.build_model(
        "ks",
        args.hidden_layers,
        args.bc_loss_weight,
        args.net,
        fourier_features=args.fourier_features,
        fourier_sigma=args.fourier_sigma,
        fourier_include_raw_x=args.fourier_include_raw_x,
        fourier_include_bias=args.fourier_include_bias,
    )
    configure_custom_optimizers(args)
    if args.weight_decay > 0:
        model.net.regularizer = ("l2", args.weight_decay)
    model.compile(
        args.optimizer,
        lr=args.lr,
        loss_weights=loss_weights,
        decay=None,
    )
    project_parameters(model.net, args.parameter_lower, args.parameter_upper)
    install_box_projection(
        model.opt,
        model.net,
        args.parameter_lower,
        args.parameter_upper,
    )
    integral_loss = build_integral_loss(
        model,
        args,
        args.integral_seed if args.integral_seed is not None else args.seed,
    )
    attach_integral_loss_train_step(
        model,
        integral_loss,
        integral_only=True,
        maximize=True,
        maximize_residual_only=True,
    )

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{timestamp}-ks-integral-maximize-{args.optimizer}")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "run_config.json"), "w", encoding="utf-8") as file_obj:
        json.dump(vars(args), file_obj, indent=2, sort_keys=True)

    callbacks = []
    if not args.no_callbacks:
        callbacks.extend(chaotic.make_callbacks(
            argparse.Namespace(
                no_callbacks=False,
                log_every=args.log_every,
                loss_verbose=args.loss_verbose,
                plot_every=args.plot_every,
                fast_plot=args.fast_plot,
                ks_diagnostics_chunk_every=args.ks_diagnostics_chunk_every,
                ks_diagnostics_verbose=args.ks_diagnostics_verbose,
                use_causal_loss=False,
                causal_diagnostics_verbose=False,
                use_integral_loss=True,
            ),
            "ks",
        ))
    maximum_callback = FixedEvaluationMaximum(args, save_path)
    callbacks.append(maximum_callback)
    print(
        f"Maximizing KS global + weighted-local residual objective with {args.optimizer} for "
        f"{args.iterations} iterations; parameter box="
        f"[{args.parameter_lower:g}, {args.parameter_upper:g}]."
    )
    result = model.train(
        iterations=args.iterations,
        display_every=args.log_every,
        callbacks=callbacks,
        model_save_path=save_path,
        save_model=False,
    )
    return result, model, maximum_callback


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
