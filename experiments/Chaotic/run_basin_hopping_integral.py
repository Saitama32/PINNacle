"""Residual deoptimization followed by one-shot basin hopping for KS.

The first phase maximizes the global plus weighted-local residual objective
while minimizing IC/periodic penalties. At the switching step, perturbed
candidates are processed sequentially on one network and normally minimize
the full positive integral loss. The winning minimum then continues ordinary
descent, so the experiment never keeps N model replicas on the GPU.
"""

import argparse
import copy
import csv
import json
import os
import random
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
from experiments.Chaotic.run_maximize_integral import (
    install_box_projection,
    project_parameters,
)
from src.losses.global_integral import GlobalIntegralLoss, attach_integral_loss_train_step


CANDIDATE_COLUMNS = [
    "candidate_id", "is_control", "sigma", "seed",
    "relative_parameter_displacement", "pre_total", "pre_global",
    "pre_local", "pre_ic", "pre_periodic", "post_total", "post_global", "post_local",
    "post_ic", "post_periodic", "delta_vs_control_total", "delta_vs_original_total", "selected",
]
TRAJECTORY_COLUMNS = [
    "candidate_id", "local_step", "train_total", "global", "local", "ic", "periodic",
]


def parse_scales(value):
    try:
        scales = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Basin-hopping scales must be comma-separated floats.") from exc
    if not scales or any(not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("Basin-hopping scales must be finite and positive.")
    return scales


def cpu_state_dict(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def perturb_parameters(module, sigma, seed, epsilon=1e-12):
    """Apply theta_l <- theta_l + sigma * s_l * N(0, I), layer-wise."""
    if sigma == 0:
        return
    generators = {}
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        device_key = str(parameter.device)
        if device_key not in generators:
            generators[device_key] = torch.Generator(device=parameter.device)
            generators[device_key].manual_seed(int(seed))
        generator = generators[device_key]
        if parameter.ndim <= 1 or parameter.numel() < 2:
            layer_scale = torch.sqrt(torch.mean(parameter.detach().square()) + epsilon)
        else:
            layer_scale = parameter.detach().std(unbiased=False).clamp_min(epsilon)
        noise = torch.randn(
            parameter.shape,
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )
        with torch.no_grad():
            parameter.add_(noise * (float(sigma) * layer_scale))


def relative_parameter_displacement(module, base_state, epsilon=1e-12):
    numerator = 0.0
    denominator = 0.0
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        base = base_state[name].to(device=parameter.device, dtype=parameter.dtype)
        numerator += float(torch.sum((parameter.detach() - base).square()).cpu())
        denominator += float(torch.sum(base.square()).cpu())
    return numerator ** 0.5 / (denominator ** 0.5 + epsilon)


def trajectory_steps(local_steps):
    anchors = {0, local_steps}
    anchors.update(step for step in (10, 50, 100, 250, 500, 1000) if step <= local_steps)
    return sorted(anchors)


class OneShotBasinHopper(dde.callbacks.Callback):
    def __init__(self, args, save_path, loss_weights):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.loss_weights = loss_weights
        self.completed = False

    def on_train_begin(self):
        self._maybe_run()

    def on_batch_end(self):
        self._maybe_run()

    def _maybe_run(self):
        if self.completed or self.model.train_state.step != self.args.basin_hopping_step:
            return
        self.completed = True
        self.run_event()

    def _new_integral_loss(self, seed):
        args = self.args
        return GlobalIntegralLoss(
            model=self.model,
            pde=self.model.pde,
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

    @staticmethod
    def _components(loss):
        diagnostics = loss.last_diagnostics
        return {
            "total": float(diagnostics["integral_loss_raw"].detach().cpu()),
            "global": float(diagnostics["global_integral_loss"].detach().cpu()),
            "local": float(diagnostics["local_integral_loss"].detach().cpu()),
            "ic": float(diagnostics["initial_condition_loss"].detach().cpu()),
            "periodic": float(diagnostics["periodic_loss"].detach().cpu()),
        }

    def _fixed_evaluation_batches(self, loss):
        batches = []
        original_seed = loss.seed
        loss.seed = self.args.basin_hopping_eval_seed
        loss._generator = None
        for batch_id in range(self.args.basin_hopping_eval_batches):
            endpoints = loss.sample_endpoints(step=batch_id, force=True)
            x, t = (tensor.detach().clone() for tensor in endpoints)
            loss._set_cached_endpoints(x, t, batch_id)
            segments = copy.deepcopy(loss._get_local_segments(x, t, step=batch_id))
            batches.append((x, t, segments))
        loss.seed = original_seed
        loss._generator = None
        return batches

    @staticmethod
    def _activate_fixed_batch(loss, batch, step):
        x, t, segments = batch
        loss._set_cached_endpoints(x, t, step)
        loss.cached_local_segments = segments
        loss.last_local_segment_step = step
        loss._cached_local_endpoint_ptrs = loss._endpoint_ptrs(x, t)

    def _evaluate(self, loss, batches):
        totals = {key: 0.0 for key in ("total", "global", "local", "ic", "periodic")}
        for batch_id, batch in enumerate(batches):
            self._activate_fixed_batch(loss, batch, batch_id)
            try:
                loss.compute_raw_loss(step=batch_id, endpoints=(batch[0], batch[1]))
                values = self._components(loss)
            finally:
                # KS uses derivatives up to u_xxxx. DeepXDE caches the
                # intermediate Jacobians/Hessians globally, so direct loss
                # calls outside Model.train_step must clear that cache too.
                dde.grad.clear()
            for key in totals:
                totals[key] += values[key]
        self.model.net.zero_grad(set_to_none=True)
        return {key: value / len(batches) for key, value in totals.items()}

    def _fresh_optimizer(self, name, lr):
        optimizer, _ = dde.optimizers.get(
            list(self.model.net.parameters()),
            name,
            learning_rate=lr,
            weight_decay=self.args.weight_decay,
            model=self.model.net,
        )
        return optimizer

    def _relax(self, loss, optimizer, candidate_id, fixed_batches, trajectory):
        checkpoints = set(trajectory_steps(self.args.basin_hopping_local_steps))
        for local_step in range(1, self.args.basin_hopping_local_steps + 1):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                try:
                    total = loss.compute_weighted_loss(self.args.basin_hopping_step + local_step)
                    total.backward()
                    return total
                finally:
                    dde.grad.clear()

            optimizer.step(closure)
            if hasattr(optimizer, "after_train_step"):
                optimizer.after_train_step()
            # The update has already consumed the gradients; retaining them
            # until the next step only increases the peak at fixed evaluation.
            optimizer.zero_grad(set_to_none=True)
            if local_step in checkpoints:
                values = self._evaluate(loss, fixed_batches)
                trajectory.append(self._trajectory_row(candidate_id, local_step, values))

    @staticmethod
    def _trajectory_row(candidate_id, local_step, values):
        return {
            "candidate_id": candidate_id,
            "local_step": local_step,
            "train_total": values["total"],
            "global": values["global"],
            "local": values["local"],
            "ic": values["ic"],
            "periodic": values["periodic"],
        }

    def run_event(self):
        args = self.args
        print(
            f"\n[Basin hopping] ending residual deoptimization and starting "
            f"candidate descent at global step {self.model.train_state.step}"
        )
        main_integral_loss = self.model.integral_loss
        base_state = cpu_state_dict(self.model.net)

        rng_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        evaluation_loss = self._new_integral_loss(args.basin_hopping_eval_seed)
        fixed_batches = self._fixed_evaluation_batches(evaluation_loss)
        original = self._evaluate(evaluation_loss, fixed_batches)

        rows = []
        trajectory = []
        final_states = []
        scales = args.basin_hopping_scales
        for candidate_id in range(args.basin_hopping_candidates):
            seed = args.basin_hopping_seed + candidate_id
            sigma = 0.0 if candidate_id == 0 else scales[(candidate_id - 1) % len(scales)]
            self.model.net.load_state_dict(base_state)
            perturb_parameters(self.model.net, sigma, seed)
            displacement = relative_parameter_displacement(self.model.net, base_state)

            candidate_loss = self._new_integral_loss(seed)
            pre = self._evaluate(candidate_loss, fixed_batches)
            trajectory.append(self._trajectory_row(candidate_id, 0, pre))
            optimizer = self._fresh_optimizer(args.basin_hopping_local_optimizer, args.basin_hopping_local_lr)
            self._relax(candidate_loss, optimizer, candidate_id, fixed_batches, trajectory)
            post = self._evaluate(candidate_loss, fixed_batches)
            state = cpu_state_dict(self.model.net)
            final_states.append(state)
            rows.append({
                "candidate_id": candidate_id,
                "is_control": candidate_id == 0,
                "sigma": sigma,
                "seed": seed,
                "relative_parameter_displacement": displacement,
                "pre_total": pre["total"], "pre_global": pre["global"],
                "pre_local": pre["local"], "pre_ic": pre["ic"], "pre_periodic": pre["periodic"],
                "post_total": post["total"], "post_global": post["global"],
                "post_local": post["local"], "post_ic": post["ic"], "post_periodic": post["periodic"],
            })
            print(
                f"[Basin hopping] candidate={candidate_id} sigma={sigma:g} "
                f"displacement={displacement:.3e} pre={pre['total']:.6e} post={post['total']:.6e}"
            )

        winner = min(range(len(rows)), key=lambda index: rows[index]["post_total"])
        control_total = rows[0]["post_total"]
        for index, row in enumerate(rows):
            row["delta_vs_control_total"] = row["post_total"] - control_total
            row["delta_vs_original_total"] = row["post_total"] - original["total"]
            row["selected"] = index == winner

        self.model.net.load_state_dict(final_states[winner])
        self._write_csv("basin_hopping_candidates.csv", CANDIDATE_COLUMNS, rows)
        self._write_csv("basin_hopping_trajectory.csv", TRAJECTORY_COLUMNS, trajectory)

        # The winner continues ordinary descent with a fresh optimizer. The
        # ascent optimizer's box projection is deliberately not carried over.
        self.model.opt = self._fresh_optimizer(args.optimizer, args.lr)
        self.model.lr_scheduler = None
        attach_integral_loss_train_step(self.model, main_integral_loss, integral_only=True)
        random.setstate(rng_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        print(
            f"[Basin hopping] selected candidate={winner}; "
            f"fitness={rows[winner]['post_total']:.6e}, relaxed_control={control_total:.6e}\n"
        )

    def _write_csv(self, filename, columns, rows):
        with open(os.path.join(self.save_path, filename), "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description="KS residual deoptimization followed by integral-loss basin hopping."
    )
    parser.add_argument("--hidden-layers", default="100*5")
    parser.add_argument("--net", choices=["mlp", "resnet", "fourier-mlp"], default="mlp")
    parser.add_argument("--fourier-features", type=int, default=10)
    parser.add_argument("--fourier-sigma", type=float, default=5.0)
    parser.add_argument("--fourier-include-raw-x", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--fourier-include-bias", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bc-loss-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default="runs_basin_hopping")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=100)
    parser.add_argument("--fast-plot", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--loss-verbose", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--ks-diagnostics-verbose", type=chaotic.str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--ks-diagnostics-chunk-every", type=int, default=1000)
    parser.add_argument("--no-callbacks", action="store_true")
    parser.add_argument(
        "--save-model",
        type=chaotic.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Deprecated compatibility option; .pt checkpoint saving is always disabled.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "muon", "soap"], default="adam")
    parser.add_argument("--parameter-lower", type=float, default=-1.0)
    parser.add_argument("--parameter-upper", type=float, default=1.0)

    parser.add_argument("--use-integral-loss", action="store_true", default=True)
    parser.add_argument("--integral-only", action="store_true", default=True)
    parser.add_argument("--integral-loss-weight", type=float, default=1.0)
    parser.add_argument("--integral-batch-size", type=int, default=512)
    parser.add_argument("--integral-warmup-steps", type=int, default=0)
    parser.add_argument("--integral-start-step", type=int, default=0)
    parser.add_argument("--integral-quadrature-order", type=int, default=6)
    parser.add_argument("--integral-local-enabled", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--integral-local-weight", type=float, default=10.0)
    parser.add_argument("--integral-local-quadrature-order", type=int, default=4)
    parser.add_argument("--integral-local-hmax", type=float, default=0.05)
    parser.add_argument("--integral-local-segment-batch-size", type=int, default=512)
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

    parser.add_argument("--basin-hopping", type=chaotic.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--basin-hopping-step", type=int, default=1000)
    parser.add_argument("--basin-hopping-candidates", type=int, default=12)
    parser.add_argument("--basin-hopping-scales", type=parse_scales, default=parse_scales("0.005,0.01,0.02,0.05,0.1"))
    parser.add_argument("--basin-hopping-local-optimizer", choices=["adam", "muon", "soap"], default="adam")
    parser.add_argument("--basin-hopping-local-lr", type=float, default=5e-4)
    parser.add_argument("--basin-hopping-local-steps", type=int, default=1000)
    parser.add_argument("--basin-hopping-eval-batches", type=int, default=4)
    parser.add_argument("--basin-hopping-eval-seed", type=int, default=98765)
    parser.add_argument("--basin-hopping-seed", type=int, default=12345)
    parser.add_argument(
        "--basin-hopping-save-all-candidates",
        type=chaotic.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Deprecated compatibility option; candidate .pt saving is always disabled.",
    )

    # Existing optimizer implementations are configured with these values.
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
    if args.iterations < 0 or not 0 <= args.basin_hopping_step <= args.iterations:
        raise ValueError("--basin-hopping-step must be between 0 and --iterations.")
    if args.basin_hopping_candidates < 1:
        raise ValueError("--basin-hopping-candidates must be positive.")
    if args.basin_hopping_local_steps < 0:
        raise ValueError("--basin-hopping-local-steps must be non-negative.")
    if args.basin_hopping_local_lr <= 0 or args.basin_hopping_eval_batches < 1:
        raise ValueError("Local learning rate and evaluation batch count must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if (
        not np.isfinite(args.parameter_lower)
        or not np.isfinite(args.parameter_upper)
        or args.parameter_lower >= args.parameter_upper
    ):
        raise ValueError("Finite parameter bounds must satisfy lower < upper.")
    if args.integral_periodic_weight < 0 or not np.isfinite(args.integral_periodic_weight):
        raise ValueError("--integral-periodic-weight must be finite and non-negative.")


def run(args):
    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, loss_weights = chaotic.build_model(
        "ks", args.hidden_layers, args.bc_loss_weight, args.net,
        fourier_features=args.fourier_features,
        fourier_sigma=args.fourier_sigma,
        fourier_include_raw_x=args.fourier_include_raw_x,
        fourier_include_bias=args.fourier_include_bias,
    )
    # Configure both possible main/local custom optimizers.  Their constructors
    # are then shared by ordinary training and candidate relaxation.
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
    model.compile(args.optimizer, lr=args.lr, loss_weights=loss_weights)
    loss_builder = OneShotBasinHopper(args, "", loss_weights)
    loss_builder.set_model(model)
    main_loss = loss_builder._new_integral_loss(
        args.integral_seed if args.integral_seed is not None else args.seed
    )
    if args.basin_hopping:
        project_parameters(model.net, args.parameter_lower, args.parameter_upper)
        install_box_projection(
            model.opt,
            model.net,
            args.parameter_lower,
            args.parameter_upper,
        )
        attach_integral_loss_train_step(
            model,
            main_loss,
            integral_only=True,
            maximize=True,
            maximize_residual_only=True,
        )
    else:
        attach_integral_loss_train_step(model, main_loss, integral_only=True)

    timestamp = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    save_path = os.path.join(args.out, f"{timestamp}-ks-integral-basin-{args.optimizer}")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "run_config.json"), "w", encoding="utf-8") as file_obj:
        json.dump(vars(args), file_obj, indent=2, sort_keys=True)
    callbacks = [] if args.no_callbacks else chaotic.make_callbacks(
        argparse.Namespace(
            no_callbacks=False, log_every=args.log_every, loss_verbose=args.loss_verbose,
            plot_every=args.plot_every, fast_plot=args.fast_plot,
            ks_diagnostics_chunk_every=args.ks_diagnostics_chunk_every,
            ks_diagnostics_verbose=args.ks_diagnostics_verbose,
            use_causal_loss=False, causal_diagnostics_verbose=False, use_integral_loss=True,
        ),
        "ks",
    )
    if args.basin_hopping:
        callbacks.append(OneShotBasinHopper(args, save_path, loss_weights))
    print(
        f"Training KS integral-only PINN with {args.optimizer} for {args.iterations} iterations; "
        f"residual_ascent_steps={args.basin_hopping_step if args.basin_hopping else 0}, "
        f"basin_hopping={args.basin_hopping}."
    )
    return model.train(
        iterations=args.iterations,
        display_every=args.log_every,
        callbacks=callbacks,
        model_save_path=save_path,
        save_model=False,
    )


def main():
    args = build_parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
