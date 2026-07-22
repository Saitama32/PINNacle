import math

import numpy as np
import torch
import deepxde as dde
from torch.nn.utils import parameters_to_vector, vector_to_parameters


GAUSS_LEGENDRE_RULES = {
    4: {
        "nodes": (
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ),
        "weights": (
            0.3478548451374538,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374538,
        ),
    },
    10: {
        "nodes": (
            -0.9739065285171717,
            -0.8650633666889845,
            -0.6794095682990244,
            -0.4333953941292472,
            -0.1488743389816312,
            0.1488743389816312,
            0.4333953941292472,
            0.6794095682990244,
            0.8650633666889845,
            0.9739065285171717,
        ),
        "weights": (
            0.0666713443086881,
            0.1494513491505806,
            0.2190863625159820,
            0.2692667193099963,
            0.2955242247147529,
            0.2955242247147529,
            0.2692667193099963,
            0.2190863625159820,
            0.1494513491505806,
            0.0666713443086881,
        ),
    },
}


_GAUSS_LEGENDRE_TENSOR_CACHE = {}


def get_gauss_legendre(order, device, dtype):
    key = (int(order), str(device), str(dtype))
    if key not in _GAUSS_LEGENDRE_TENSOR_CACHE:
        rule = GAUSS_LEGENDRE_RULES.get(int(order))
        if rule is None:
            nodes_np, weights_np = np.polynomial.legendre.leggauss(int(order))
        else:
            nodes_np = np.asarray(rule["nodes"], dtype=np.float64)
            weights_np = np.asarray(rule["weights"], dtype=np.float64)
        _GAUSS_LEGENDRE_TENSOR_CACHE[key] = (
            torch.tensor(nodes_np, dtype=dtype, device=device),
            torch.tensor(weights_np, dtype=dtype, device=device),
        )
    return _GAUSS_LEGENDRE_TENSOR_CACHE[key]


def ks_initial_condition_torch(x):
    return torch.cos(x) * (1.0 + torch.sin(x))


def compute_ks_spatial_operator(model, inputs, equation):
    if not hasattr(equation, "ks_spatial_operator"):
        raise ValueError("Local/global integral loss requires equation.ks_spatial_operator(inputs, u).")
    if not inputs.requires_grad:
        inputs = inputs.requires_grad_(True)
    outputs = model(inputs)
    return equation.ks_spatial_operator(inputs, outputs)


def _uses_pcgrad_task_projection(opt_name=None, opt=None):
    names = []
    if opt_name is not None:
        names.append(str(opt_name).lower())
    if opt is not None:
        base_optimizer_name = getattr(opt, "base_optimizer_name", None)
        if base_optimizer_name is not None:
            names.append(str(base_optimizer_name).lower())
        class_name = getattr(opt.__class__, "__name__", None)
        if class_name is not None:
            names.append(str(class_name).lower())
    return any("pcgrad" in name for name in names)


def compose_optimizer_task_losses(base_losses, integral_weighted_loss, integral_only=False, opt_name=None, opt=None):
    integral_task_loss = integral_weighted_loss.reshape(1)
    if integral_only or base_losses is None:
        return integral_task_loss
    if base_losses.ndim == 0:
        base_losses = base_losses.reshape(1)
    if _uses_pcgrad_task_projection(opt_name=opt_name, opt=opt) and base_losses.numel() > 0:
        return torch.cat([base_losses, integral_task_loss])
    return base_losses


def _zero_local_diagnostics(device, dtype):
    zero = torch.zeros((), dtype=dtype, device=device)
    return {
        "num_local_intervals": 0,
        "mean_intervals_per_point": 0.0,
        "max_intervals_per_point": 0,
        "local_residual_rms": zero,
        "local_residual_mae": zero,
        "local_residual_max": zero,
        "mean_interval_length": zero,
        "max_interval_length": zero,
        "local_loss_early": zero,
        "local_loss_middle": zero,
        "local_loss_late": zero,
    }


def _bucketed_segment_loss(residual, midpoint_t, t_min, t_max, bucket):
    span = float(t_max - t_min)
    left = t_min + bucket * span / 3.0
    right = t_min + (bucket + 1) * span / 3.0
    flat_mid = midpoint_t.reshape(-1)
    if bucket == 2:
        mask = flat_mid >= left
    else:
        mask = (flat_mid >= left) & (flat_mid < right)
    if not torch.any(mask):
        return torch.tensor(float("nan"), dtype=residual.dtype, device=residual.device)
    return torch.mean(torch.square(residual.reshape(-1)[mask])).detach()


def compute_local_integral_loss(
    model,
    x,
    t,
    equation,
    hmax=0.05,
    quadrature_order=4,
    segment_batch_size=None,
    t_min=0.0,
    t_max=1.0,
    generator=None,
):
    if x.ndim == 1:
        x = x[:, None]
    if t.ndim == 1:
        t = t[:, None]

    eps = torch.finfo(t.dtype).eps
    active = t[:, 0] > eps
    x_active = x[active]
    t_active = t[active]

    if x_active.shape[0] == 0:
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        diagnostics = _zero_local_diagnostics(x.device, x.dtype)
        return zero, diagnostics

    num_sections = torch.ceil(t_active[:, 0] / float(hmax)).to(torch.long)
    num_sections = torch.clamp(num_sections, min=1)

    segment_x = []
    segment_a = []
    segment_b = []
    for i in range(x_active.shape[0]):
        k_i = int(num_sections[i].item())
        section_length = t_active[i : i + 1] / float(k_i)
        segment_x.append(x_active[i : i + 1].expand(k_i, -1))
        max_start = torch.clamp(
            torch.full_like(section_length, float(t_max)) - section_length,
            min=float(t_min),
        )
        if max_start.item() <= float(t_min):
            starts = torch.full((k_i, 1), float(t_min), dtype=t.dtype, device=t.device)
        else:
            starts = float(t_min) + (max_start - float(t_min)) * torch.rand(
                k_i,
                1,
                generator=generator,
                device=t.device,
                dtype=t.dtype,
            )
        segment_a.append(starts)
        segment_b.append(starts + section_length.expand(k_i, 1))

    segment_x = torch.cat(segment_x, dim=0)
    segment_a = torch.cat(segment_a, dim=0)
    segment_b = torch.cat(segment_b, dim=0)
    segment_mid = 0.5 * (segment_a + segment_b)

    if segment_batch_size is None or segment_batch_size <= 0:
        segment_batch_size = int(segment_x.shape[0])

    nodes, weights = get_gauss_legendre(quadrature_order, x.device, x.dtype)
    loss_sum = torch.zeros((), dtype=x.dtype, device=x.device)
    residual_chunks = []

    total_segments = int(segment_x.shape[0])
    for start in range(0, total_segments, int(segment_batch_size)):
        stop = min(start + int(segment_batch_size), total_segments)
        chunk_x = segment_x[start:stop]
        chunk_a = segment_a[start:stop]
        chunk_b = segment_b[start:stop]

        inputs_a = torch.cat([chunk_x, chunk_a], dim=1)
        inputs_b = torch.cat([chunk_x, chunk_b], dim=1)
        u_a = model(inputs_a)
        u_b = model(inputs_b)

        midpoint = 0.5 * (chunk_a + chunk_b)
        half_width = 0.5 * (chunk_b - chunk_a)
        quad_t = midpoint[:, None, :] + half_width[:, None, :] * nodes.view(1, quadrature_order, 1)
        quad_x = chunk_x[:, None, :].expand(-1, quadrature_order, -1)
        quad_inputs = torch.cat([quad_x, quad_t], dim=-1).reshape(-1, 2).requires_grad_(True)
        g_values = compute_ks_spatial_operator(
            model=model,
            inputs=quad_inputs,
            equation=equation,
        ).reshape(stop - start, quadrature_order, -1)
        integral = half_width * torch.sum(weights.view(1, quadrature_order, 1) * g_values, dim=1)
        local_residual = u_b - u_a + integral
        loss_sum = loss_sum + local_residual.square().sum()
        residual_chunks.append(local_residual.detach().reshape(-1))

    local_loss = loss_sum / max(total_segments, 1)
    residual_all = torch.cat(residual_chunks, dim=0)
    interval_lengths = segment_b - segment_a
    diagnostics = {
        "num_local_intervals": total_segments,
        "mean_intervals_per_point": float(num_sections.float().mean().detach().cpu().item()),
        "max_intervals_per_point": int(num_sections.max().detach().cpu().item()),
        "local_residual_rms": torch.sqrt(torch.mean(residual_all.square())),
        "local_residual_mae": torch.mean(torch.abs(residual_all)),
        "local_residual_max": torch.max(torch.abs(residual_all)),
        "mean_interval_length": torch.mean(interval_lengths).detach(),
        "max_interval_length": torch.max(interval_lengths).detach(),
        "local_loss_early": _bucketed_segment_loss(residual_all, segment_mid, t_min, t_max, 0),
        "local_loss_middle": _bucketed_segment_loss(residual_all, segment_mid, t_min, t_max, 1),
        "local_loss_late": _bucketed_segment_loss(residual_all, segment_mid, t_min, t_max, 2),
    }
    return local_loss, diagnostics


class GlobalIntegralLoss:
    def __init__(
        self,
        model,
        pde,
        batch_size,
        weight,
        warmup_steps,
        start_step=0,
        quadrature_order=4,
        local_enabled=True,
        local_weight=1.0,
        local_quadrature_order=4,
        local_hmax=0.05,
        local_segment_batch_size=256,
        t0_fraction=0.1,
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
        self.start_step = int(start_step)
        self.quadrature_order = int(quadrature_order)
        self.local_enabled = bool(local_enabled)
        self.local_weight = float(local_weight)
        self.local_quadrature_order = int(local_quadrature_order)
        self.local_hmax = float(local_hmax)
        self.local_segment_batch_size = (
            None if local_segment_batch_size is None else int(local_segment_batch_size)
        )
        self.t0_fraction = float(t0_fraction)
        self.resample_every = int(resample_every)
        self.initial_condition_fn = initial_condition_fn or ks_initial_condition_torch

        if self.batch_size <= 0:
            raise ValueError("integral batch_size must be positive.")
        if self.max_weight < 0 or not math.isfinite(self.max_weight):
            raise ValueError("integral loss weight must be finite and non-negative.")
        if self.warmup_steps < 0:
            raise ValueError("integral warmup_steps must be non-negative.")
        if self.start_step < 0:
            raise ValueError("integral start_step must be non-negative.")
        if self.quadrature_order <= 0:
            raise ValueError("integral quadrature_order must be positive.")
        if self.local_weight < 0 or not math.isfinite(self.local_weight):
            raise ValueError("integral local_weight must be finite and non-negative.")
        if self.local_quadrature_order <= 0:
            raise ValueError("integral local_quadrature_order must be positive.")
        if self.local_hmax <= 0 or not math.isfinite(self.local_hmax):
            raise ValueError("integral local_hmax must be positive and finite.")
        if self.local_segment_batch_size is not None and self.local_segment_batch_size <= 0:
            raise ValueError("integral local_segment_batch_size must be positive.")
        if not math.isfinite(self.t0_fraction) or not 0.0 <= self.t0_fraction <= 1.0:
            raise ValueError("integral t0_fraction must be finite and satisfy 0 <= t0_fraction <= 1.")
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

    def _quadrature_constants(self, device, dtype, order=None):
        if order is None:
            order = self.quadrature_order
        return get_gauss_legendre(order, device, dtype)

    def current_weight(self, step):
        if step is None:
            step = 0
        step = float(step)
        if step < float(self.start_step):
            return 0.0
        if self.warmup_steps <= 0:
            return self.max_weight
        progress = min(max(step - float(self.start_step), 0.0) / float(self.warmup_steps), 1.0)
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
        num_t0 = int(round(self.t0_fraction * self.batch_size))
        num_t0 = min(num_t0, self.batch_size)
        if num_t0 > 0:
            t[:num_t0] = self.domain_t_min
        self.cached_x = x
        self.cached_t = t
        self.last_sample_step = current_step
        return x, t

    def quadrature_times(self, t):
        nodes, _ = self._quadrature_constants(t.device, t.dtype)
        span = t - self.domain_t_min
        return 0.5 * (self.domain_t_min + t) + 0.5 * span * nodes.view(1, self.quadrature_order)

    def quadrature_points(self, x, t):
        times = self.quadrature_times(t)
        x_quad = x.expand(-1, self.quadrature_order)
        points = torch.stack((x_quad, times), dim=-1).reshape(-1, 2)
        return points.requires_grad_(True)

    def global_integral_residual(self, step=None, endpoints=None):
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
        g_quad = self.pde.ks_spatial_operator(quad_points, u_quad).reshape(
            x.shape[0], self.quadrature_order
        )
        _, weights = self._quadrature_constants(quad_points.device, quad_points.dtype)
        weighted_sum = torch.sum(
            weights.view(1, self.quadrature_order) * g_quad,
            dim=1,
            keepdim=True,
        )
        span = t - self.domain_t_min
        integral = 0.5 * span * weighted_sum
        return u_end - u0 + integral

    def integral_residual(self, step=None, endpoints=None):
        return self.global_integral_residual(step=step, endpoints=endpoints)

    def compute_local_raw_loss(self, step=None, endpoints=None):
        if endpoints is None:
            x, t = self.sample_endpoints(step=step)
        else:
            x, t = endpoints
            self.cached_x = x
            self.cached_t = t
        return compute_local_integral_loss(
            model=self.model.net,
            x=x,
            t=t,
            equation=self.pde,
            hmax=self.local_hmax,
            quadrature_order=self.local_quadrature_order,
            segment_batch_size=self.local_segment_batch_size,
            t_min=self.t_min,
            t_max=self.t_max,
            generator=self._rand_generator(x.device),
        )

    def compute_raw_loss(self, step=None, endpoints=None, deepxde_loss_sum=None):
        global_residual = self.global_integral_residual(step=step, endpoints=endpoints)
        global_raw_loss = torch.mean(torch.square(global_residual))
        local_raw_loss, local_diagnostics = self.compute_local_raw_loss(
            step=step,
            endpoints=(self.cached_x, self.cached_t),
        ) if self.local_enabled else (
            torch.zeros((), dtype=global_raw_loss.dtype, device=global_raw_loss.device),
            _zero_local_diagnostics(global_raw_loss.device, global_raw_loss.dtype),
        )
        raw_loss = global_raw_loss + self.local_weight * local_raw_loss
        self.last_diagnostics = self._build_diagnostics(
            raw_loss=raw_loss,
            global_raw_loss=global_raw_loss,
            global_residual=global_residual,
            local_raw_loss=local_raw_loss,
            local_diagnostics=local_diagnostics,
            deepxde_loss_sum=deepxde_loss_sum,
        )
        return raw_loss

    def compute_weighted_loss(self, step, deepxde_loss_sum=None):
        raw_loss = self.compute_raw_loss(step=step, deepxde_loss_sum=deepxde_loss_sum)
        weight = self.current_weight(step)
        weighted_loss = raw_loss * weight
        self.last_diagnostics["integral_weight"] = weight
        self.last_diagnostics["integral_loss_weighted"] = weighted_loss.detach()
        self.last_diagnostics["weighted_global_integral_loss"] = (
            self.last_diagnostics["global_integral_loss"] * weight
        )
        self.last_diagnostics["weighted_local_integral_loss"] = (
            self.last_diagnostics["local_integral_loss"] * self.local_weight * weight
        )
        if deepxde_loss_sum is not None:
            deepxde_loss_sum = deepxde_loss_sum.detach()
            self.last_diagnostics["deepxde_loss_sum"] = deepxde_loss_sum
            self.last_diagnostics["actual_total_loss"] = deepxde_loss_sum + weighted_loss.detach()
        self.model.integral_loss_diagnostics = self.last_diagnostics
        return weighted_loss

    def _build_diagnostics(
        self,
        raw_loss,
        global_raw_loss,
        global_residual,
        local_raw_loss,
        local_diagnostics,
        deepxde_loss_sum=None,
    ):
        residual_detached = global_residual.detach()
        abs_residual = torch.abs(residual_detached)
        t = self.cached_t.detach() if self.cached_t is not None else None
        diagnostics = {
            "integral_loss_raw": raw_loss.detach(),
            "global_integral_loss": global_raw_loss.detach(),
            "global_integral_rms": torch.sqrt(torch.mean(residual_detached**2)),
            "local_integral_loss": local_raw_loss.detach(),
            "local_integral_rms": local_diagnostics["local_residual_rms"].detach(),
            "local_integral_mae": local_diagnostics["local_residual_mae"].detach(),
            "local_integral_max": local_diagnostics["local_residual_max"].detach(),
            "local_num_segments": float(local_diagnostics["num_local_intervals"]),
            "local_mean_segments_per_point": float(local_diagnostics["mean_intervals_per_point"]),
            "local_max_segments_per_point": float(local_diagnostics["max_intervals_per_point"]),
            "local_mean_segment_length": local_diagnostics["mean_interval_length"],
            "local_max_segment_length": local_diagnostics["max_interval_length"],
            "weighted_global_integral_loss": global_raw_loss.detach() * 0.0,
            "weighted_local_integral_loss": local_raw_loss.detach() * 0.0,
            "integral_weight": 0.0,
            "integral_loss_weighted": raw_loss.detach() * 0.0,
            "integral_residual_rms": torch.sqrt(torch.mean(residual_detached**2)),
            "integral_residual_abs_mean": torch.mean(abs_residual),
            "integral_residual_abs_max": torch.max(abs_residual),
            "integral_loss_early": self._bucket_loss(residual_detached, t, 0),
            "integral_loss_middle": self._bucket_loss(residual_detached, t, 1),
            "integral_loss_late": self._bucket_loss(residual_detached, t, 2),
            "local_integral_loss_early": local_diagnostics["local_loss_early"],
            "local_integral_loss_middle": local_diagnostics["local_loss_middle"],
            "local_integral_loss_late": local_diagnostics["local_loss_late"],
        }
        if deepxde_loss_sum is not None:
            diagnostics["deepxde_loss_sum"] = deepxde_loss_sum.detach()
            diagnostics["actual_total_loss"] = deepxde_loss_sum.detach()
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


def attach_integral_loss_train_step(model, integral_loss, integral_only=False):
    if dde.backend.backend_name != "pytorch":
        raise ValueError("Integral loss train-step attachment currently supports only the PyTorch backend.")

    model.integral_loss = integral_loss
    model.integral_loss_diagnostics = None
    integral_only = bool(integral_only)

    def _compute_total_loss(active_inputs, active_targets, skip_backward=False):
        if integral_only:
            integral_weighted_loss = integral_loss.compute_weighted_loss(
                model.train_state.step,
                deepxde_loss_sum=None,
            )
            model.opt.losses = compose_optimizer_task_losses(
                base_losses=None,
                integral_weighted_loss=integral_weighted_loss,
                integral_only=True,
                opt_name=getattr(model, "opt_name", None),
                opt=model.opt,
            )
            integral_loss.last_diagnostics["actual_total_loss"] = integral_weighted_loss.detach()
            integral_loss.last_diagnostics["deepxde_loss_sum"] = torch.zeros_like(integral_weighted_loss.detach())
            model.integral_loss_diagnostics = integral_loss.last_diagnostics
            total_loss = integral_weighted_loss
        else:
            losses = model.outputs_losses_train(active_inputs, active_targets)[1]
            if hasattr(model.opt, "window_ic_loss"):
                ic_loss = model.opt.window_ic_loss()
                if ic_loss is not None:
                    losses = torch.cat([losses, ic_loss.reshape(1)])
            deepxde_loss_sum = torch.sum(losses)
            integral_weighted_loss = integral_loss.compute_weighted_loss(
                model.train_state.step,
                deepxde_loss_sum=deepxde_loss_sum,
            )
            model.opt.losses = compose_optimizer_task_losses(
                base_losses=losses,
                integral_weighted_loss=integral_weighted_loss,
                integral_only=False,
                opt_name=getattr(model, "opt_name", None),
                opt=model.opt,
            )
            total_loss = deepxde_loss_sum + integral_weighted_loss
            integral_loss.last_diagnostics["actual_total_loss"] = total_loss.detach()
            model.integral_loss_diagnostics = integral_loss.last_diagnostics
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
