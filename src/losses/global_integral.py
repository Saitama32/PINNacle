import math

import numpy as np
import torch
import deepxde as dde
from torch.nn.utils import parameters_to_vector, vector_to_parameters


GL4_NODES = (
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
)

GL4_WEIGHTS = (
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
)


def ks_initial_condition_torch(x):
    return torch.cos(x) * (1.0 + torch.sin(x))


class GlobalIntegralLoss:
    def __init__(
        self,
        model,
        pde,
        batch_size,
        weight,
        warmup_steps,
        t_min=None,
        seed=None,
        resample_every=1,
        initial_condition_fn=None,
    ):
        self.model = model
        self.pde = pde
        self.batch_size = int(batch_size)
        self.max_weight = float(weight)
        self.warmup_steps = int(warmup_steps)
        self.resample_every = int(resample_every)
        self.initial_condition_fn = initial_condition_fn or ks_initial_condition_torch

        if self.batch_size <= 0:
            raise ValueError("integral batch_size must be positive.")
        if self.max_weight < 0 or not math.isfinite(self.max_weight):
            raise ValueError("integral loss weight must be finite and non-negative.")
        if self.warmup_steps < 0:
            raise ValueError("integral warmup_steps must be non-negative.")
        if self.resample_every <= 0:
            raise ValueError("integral resample_every must be positive.")
        if getattr(pde, "input_dim", 2) != 2:
            raise ValueError("GlobalIntegralLoss currently supports only 2D inputs ordered as (x, t).")
        if not hasattr(pde, "ks_spatial_operator"):
            raise ValueError("GlobalIntegralLoss requires pde.ks_spatial_operator(x, u).")

        bbox = np.asarray(pde.bbox, dtype=np.float64)
        if bbox.shape[0] != 4:
            raise ValueError("GlobalIntegralLoss expects a KS bbox [x_min, x_max, t_min, t_max].")
        self.x_min = float(bbox[0])
        self.x_max = float(bbox[1])
        self.domain_t_min = float(bbox[2])
        self.t_max = float(bbox[3])
        self.t_min = self.domain_t_min if t_min is None else float(t_min)
        if not math.isfinite(self.t_min) or not self.domain_t_min <= self.t_min < self.t_max:
            raise ValueError("integral t_min must be finite and satisfy domain_t_min <= t_min < t_max.")

        self.seed = seed
        self._generator = None
        self._generator_device = None
        self._gl_nodes = None
        self._gl_weights = None
        self.cached_x = None
        self.cached_t = None
        self.last_sample_step = None
        self.last_diagnostics = {}

    def _device_dtype(self):
        param = next(self.model.net.parameters(), None)
        if param is None:
            return torch.device("cpu"), torch.get_default_dtype()
        return param.device, param.dtype

    def _rand_generator(self, device):
        if self.seed is None:
            return None
        if self._generator is not None and self._generator_device == device:
            return self._generator
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(self.seed))
        self._generator_device = device
        return self._generator

    def _quadrature_constants(self, device, dtype):
        if (
            self._gl_nodes is None
            or self._gl_nodes.device != device
            or self._gl_nodes.dtype != dtype
        ):
            self._gl_nodes = torch.tensor(GL4_NODES, dtype=dtype, device=device)
            self._gl_weights = torch.tensor(GL4_WEIGHTS, dtype=dtype, device=device)
        return self._gl_nodes, self._gl_weights

    def current_weight(self, step):
        if self.warmup_steps <= 0:
            return self.max_weight
        progress = min(max(float(step), 0.0) / float(self.warmup_steps), 1.0)
        return self.max_weight * progress

    def sample_endpoints(self, step=None, force=False):
        current_step = None if step is None else int(step)
        if not force and self.cached_x is not None and current_step is not None:
            if current_step == self.last_sample_step:
                return self.cached_x, self.cached_t
            if current_step % self.resample_every != 0:
                return self.cached_x, self.cached_t

        device, dtype = self._device_dtype()
        generator = self._rand_generator(device)
        x = self.x_min + (self.x_max - self.x_min) * torch.rand(
            self.batch_size,
            1,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        t = self.t_min + (self.t_max - self.t_min) * torch.rand(
            self.batch_size,
            1,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        self.cached_x = x
        self.cached_t = t
        self.last_sample_step = current_step
        return x, t

    def quadrature_times(self, t):
        nodes, _ = self._quadrature_constants(t.device, t.dtype)
        return 0.5 * t * (1.0 + nodes.view(1, 4))

    def quadrature_points(self, x, t):
        times = self.quadrature_times(t)
        x_quad = x.expand(-1, 4)
        points = torch.stack((x_quad, times), dim=-1).reshape(-1, 2)
        return points.requires_grad_(True)

    def integral_residual(self, step=None, endpoints=None):
        if endpoints is None:
            x, t = self.sample_endpoints(step=step)
        else:
            x, t = endpoints
            self.cached_x = x
            self.cached_t = t

        endpoints_tensor = torch.cat((x, t), dim=1)
        u_end = self.model.net(endpoints_tensor)
        u0 = self.initial_condition_fn(x)

        quad_points = self.quadrature_points(x, t)
        u_quad = self.model.net(quad_points)
        g_quad = self.pde.ks_spatial_operator(quad_points, u_quad).reshape(x.shape[0], 4)
        _, weights = self._quadrature_constants(quad_points.device, quad_points.dtype)
        weighted_sum = torch.sum(weights.view(1, 4) * g_quad, dim=1, keepdim=True)
        integral = 0.5 * t * weighted_sum
        dde.grad.clear()
        return u_end - u0 + integral

    def compute_raw_loss(self, step=None, endpoints=None):
        residual = self.integral_residual(step=step, endpoints=endpoints)
        raw_loss = torch.mean(torch.square(residual))
        self.last_diagnostics = self._build_diagnostics(raw_loss, residual)
        return raw_loss

    def compute_weighted_loss(self, step):
        raw_loss = self.compute_raw_loss(step=step)
        weight = self.current_weight(step)
        weighted_loss = raw_loss * weight
        self.last_diagnostics["integral_weight"] = weight
        self.last_diagnostics["integral_loss_weighted"] = weighted_loss.detach()
        self.model.integral_loss_diagnostics = self.last_diagnostics
        return weighted_loss

    def _build_diagnostics(self, raw_loss, residual):
        residual_detached = residual.detach()
        abs_residual = torch.abs(residual_detached)
        t = self.cached_t.detach() if self.cached_t is not None else None
        diagnostics = {
            "integral_loss_raw": raw_loss.detach(),
            "integral_weight": 0.0,
            "integral_loss_weighted": raw_loss.detach() * 0.0,
            "integral_residual_rms": torch.sqrt(torch.mean(residual_detached**2)),
            "integral_residual_abs_mean": torch.mean(abs_residual),
            "integral_residual_abs_max": torch.max(abs_residual),
            "integral_loss_early": self._bucket_loss(residual_detached, t, 0),
            "integral_loss_middle": self._bucket_loss(residual_detached, t, 1),
            "integral_loss_late": self._bucket_loss(residual_detached, t, 2),
        }
        return diagnostics

    def _bucket_loss(self, residual, t, bucket):
        if t is None:
            return torch.tensor(float("nan"), dtype=residual.dtype, device=residual.device)
        span = self.t_max - self.t_min
        left = self.t_min + bucket * span / 3.0
        right = self.t_min + (bucket + 1) * span / 3.0
        if bucket == 2:
            mask = t.reshape(-1) >= left
        else:
            mask = (t.reshape(-1) >= left) & (t.reshape(-1) < right)
        if not torch.any(mask):
            return torch.tensor(float("nan"), dtype=residual.dtype, device=residual.device)
        return torch.mean(torch.square(residual.reshape(-1)[mask])).detach()


def attach_integral_loss_train_step(model, integral_loss):
    if dde.backend.backend_name != "pytorch":
        raise ValueError("Integral loss train-step attachment currently supports only the PyTorch backend.")

    model.integral_loss = integral_loss
    model.integral_loss_diagnostics = None

    def _compute_total_loss(active_inputs, active_targets, skip_backward=False):
        losses = model.outputs_losses_train(active_inputs, active_targets)[1]
        if hasattr(model.opt, "window_ic_loss"):
            ic_loss = model.opt.window_ic_loss()
            if ic_loss is not None:
                losses = torch.cat([losses, ic_loss.reshape(1)])
        model.opt.losses = losses
        total_loss = torch.sum(losses) + integral_loss.compute_weighted_loss(model.train_state.step)
        if not skip_backward:
            model.opt.zero_grad()
            total_loss.backward()
        return total_loss

    def _with_causal_context(inputs, targets, fn):
        if hasattr(model.opt, "causal_context"):
            with model.opt.causal_context(inputs, targets, model.data) as (active_inputs, active_targets):
                return fn(active_inputs, active_targets)
        return fn(inputs, targets)

    def train_step(inputs, targets):
        def closure(
            *,
            skip_backward=False,
            return_intermediates=False,
            cached_intermediates=None,
            starting_id=0,
        ):
            del cached_intermediates
            del starting_id

            def _run(active_inputs, active_targets):
                total_loss = _compute_total_loss(
                    active_inputs,
                    active_targets,
                    skip_backward=skip_backward,
                )
                if return_intermediates:
                    return total_loss, None
                return total_loss

            return _with_causal_context(inputs, targets, _run)

        loss = model.opt.step(closure)
        if hasattr(model.opt, "after_train_step"):
            model.opt.after_train_step()
        if model.lr_scheduler is not None:
            if model.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                model.lr_scheduler.step(loss)
            else:
                model.lr_scheduler.step()

    def train_step_pso(inputs, targets):
        params = list(model.opt.param_groups[0]["params"])

        def closure():
            loss_list = []
            grad_list = []
            use_grad = getattr(model.opt, "use_grad", True)

            def _run(active_inputs, active_targets):
                for i in range(model.opt.pop_size):
                    vector_to_parameters(model.opt.swarm[i], params)
                    total_loss = _compute_total_loss(
                        active_inputs,
                        active_targets,
                        skip_backward=True,
                    )
                    if use_grad:
                        loss_list.append(total_loss)
                        grads = torch.autograd.grad(total_loss, params)
                        grad_list.append(parameters_to_vector(grads))
                    else:
                        loss_list.append(total_loss.detach())

            _with_causal_context(inputs, targets, _run)
            grads_swarm = torch.stack(grad_list) if use_grad else None
            return torch.stack(loss_list), grads_swarm

        model.opt.step(closure)
        if model.lr_scheduler is not None:
            model.lr_scheduler.step()

    base_opt_name = getattr(model.opt, "base_optimizer_name", None)
    if model.opt_name == "PSO" or base_opt_name == "PSO":
        model.train_step = train_step_pso
    else:
        model.train_step = train_step
