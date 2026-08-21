import math

import deepxde as dde
import numpy as np
import torch

from src.losses.global_integral import get_gauss_legendre, ks_initial_condition_torch


class FrontIntegralLoss:
    """Fixed-time integral defects for the Kuramoto--Sivashinsky equation."""

    def __init__(
        self,
        model,
        pde,
        num_intervals=10,
        num_x_points=1000,
        quadrature_order=6,
        x_batch_size=250,
        weight=0.01,
        seed=None,
        initial_condition_fn=None,
    ):
        self.model = model
        self.pde = pde
        self.num_intervals = int(num_intervals)
        self.num_x_points = int(num_x_points)
        self.quadrature_order = int(quadrature_order)
        self.x_batch_size = int(x_batch_size)
        self.weight = float(weight)
        self.seed = seed
        self.initial_condition_fn = initial_condition_fn or ks_initial_condition_torch

        if self.num_intervals <= 0:
            raise ValueError("front integral num_intervals must be positive.")
        if self.num_x_points <= 0:
            raise ValueError("front integral num_x_points must be positive.")
        if self.quadrature_order <= 0:
            raise ValueError("front integral quadrature_order must be positive.")
        if self.x_batch_size <= 0:
            raise ValueError("front integral x_batch_size must be positive.")
        if self.weight < 0 or not math.isfinite(self.weight):
            raise ValueError("front integral weight must be finite and non-negative.")
        if getattr(pde, "input_dim", 2) != 2:
            raise ValueError("FrontIntegralLoss supports only inputs ordered as (x, t).")
        if not hasattr(pde, "ks_spatial_operator"):
            raise ValueError("FrontIntegralLoss requires pde.ks_spatial_operator(inputs, u).")

        bbox = np.asarray(pde.bbox, dtype=np.float64)
        if bbox.shape != (4,):
            raise ValueError("FrontIntegralLoss expects a KS bbox [x_min, x_max, t_min, t_max].")
        self.x_min, self.x_max, self.t_min, self.t_max = map(float, bbox)
        if not self.x_min < self.x_max or not self.t_min < self.t_max:
            raise ValueError("FrontIntegralLoss requires non-empty spatial and temporal domains.")

        generator = np.random.default_rng(seed)
        self.x_front_numpy = generator.uniform(
            self.x_min,
            self.x_max,
            size=(self.num_x_points, 1),
        )
        self.last_diagnostics = None

    def _device_and_dtype(self):
        parameter = next(self.model.net.parameters(), None)
        if parameter is None:
            return torch.device("cpu"), torch.float32
        return parameter.device, parameter.dtype

    def front_grid(self):
        device, dtype = self._device_and_dtype()
        x_front = torch.as_tensor(self.x_front_numpy, device=device, dtype=dtype)
        front_times = torch.linspace(
            self.t_min,
            self.t_max,
            self.num_intervals + 1,
            device=device,
            dtype=dtype,
        )
        return x_front, front_times

    def quadrature_points(self, x_front, front_times):
        nodes, _ = get_gauss_legendre(
            self.quadrature_order,
            x_front.device,
            x_front.dtype,
        )
        interval_left = front_times[:-1]
        interval_right = front_times[1:]
        midpoint = 0.5 * (interval_left + interval_right)
        half_width = 0.5 * (interval_right - interval_left)
        quadrature_times = midpoint[:, None] + half_width[:, None] * nodes[None, :]

        x = x_front[None, :, None, :].expand(
            self.num_intervals,
            self.num_x_points,
            self.quadrature_order,
            1,
        )
        t = quadrature_times[:, None, :, None].expand(
            self.num_intervals,
            self.num_x_points,
            self.quadrature_order,
            1,
        )
        return torch.cat((x, t), dim=-1).reshape(-1, 2).requires_grad_(True)

    def compute(self):
        x_front, front_times = self.front_grid()
        dt = front_times[1:] - front_times[:-1]

        endpoint_x = x_front[None, :, :].expand(
            self.num_intervals + 1,
            self.num_x_points,
            1,
        )
        endpoint_t = front_times[:, None, None].expand(
            self.num_intervals + 1,
            self.num_x_points,
            1,
        )
        endpoint_inputs = torch.cat((endpoint_x, endpoint_t), dim=-1).reshape(-1, 2)
        u_front = self.model.net(endpoint_inputs).reshape(
            self.num_intervals + 1,
            self.num_x_points,
            -1,
        )
        if u_front.shape[-1] != 1:
            raise ValueError("FrontIntegralLoss currently supports scalar KS outputs only.")

        quad_points = self.quadrature_points(x_front, front_times)
        u_quad = self.model.net(quad_points)
        spatial_quad = self.pde.ks_spatial_operator(quad_points, u_quad).reshape(
            self.num_intervals,
            self.num_x_points,
            self.quadrature_order,
            1,
        )
        _, weights = get_gauss_legendre(
            self.quadrature_order,
            quad_points.device,
            quad_points.dtype,
        )
        integral = 0.5 * dt[:, None, None] * torch.sum(
            weights[None, None, :, None] * spatial_quad,
            dim=2,
        )

        left_values = torch.cat(
            (self.initial_condition_fn(x_front)[None, :, :], u_front[1:-1]),
            dim=0,
        )
        defects = u_front[1:] - left_values + integral
        normalized_defects = defects / dt[:, None, None]
        interval_losses = torch.mean(normalized_defects.square(), dim=(1, 2))
        raw_loss = torch.mean(interval_losses)

        detached_defects = normalized_defects.detach()
        diagnostics = {
            "front_integral_loss": raw_loss.detach(),
            "front_integral_weight": self.weight,
            "weighted_front_integral_loss": raw_loss.detach() * self.weight,
            "front_defect_rms": torch.sqrt(torch.mean(detached_defects.square())),
            "front_defect_max": torch.max(torch.abs(detached_defects)),
        }
        for index, interval_loss in enumerate(interval_losses):
            diagnostics[f"front_{index}_loss"] = interval_loss.detach()
        for index, values in enumerate(u_front):
            diagnostics[f"u_front_{index}_rms"] = torch.sqrt(
                torch.mean(values.detach().square())
            )
        self.last_diagnostics = diagnostics
        self.model.front_integral_loss_diagnostics = diagnostics
        return raw_loss

    def set_zero_diagnostics(self, reference):
        zero = reference.detach() * 0.0
        diagnostics = {
            "front_integral_loss": zero,
            "front_integral_weight": self.weight,
            "weighted_front_integral_loss": zero,
            "front_defect_rms": zero,
            "front_defect_max": zero,
        }
        for index in range(self.num_intervals):
            diagnostics[f"front_{index}_loss"] = zero
        for index in range(self.num_intervals + 1):
            diagnostics[f"u_front_{index}_rms"] = torch.tensor(
                float("nan"), device=zero.device, dtype=zero.dtype
            )
        self.last_diagnostics = diagnostics
        self.model.front_integral_loss_diagnostics = diagnostics

    def _interval_defect(self, interval_index, x_front, front_times):
        left_time = front_times[interval_index]
        right_time = front_times[interval_index + 1]
        dt = right_time - left_time

        right_inputs = torch.cat(
            (x_front, torch.full_like(x_front, right_time)),
            dim=1,
        )
        u_right = self.model.net(right_inputs)
        if interval_index == 0:
            u_left = self.initial_condition_fn(x_front)
        else:
            left_inputs = torch.cat(
                (x_front, torch.full_like(x_front, left_time)),
                dim=1,
            )
            u_left = self.model.net(left_inputs)

        nodes, weights = get_gauss_legendre(
            self.quadrature_order,
            x_front.device,
            x_front.dtype,
        )
        quadrature_times = 0.5 * (left_time + right_time) + 0.5 * dt * nodes
        quad_x = x_front[:, None, :].expand(-1, self.quadrature_order, -1)
        quad_t = quadrature_times[None, :, None].expand(x_front.shape[0], -1, -1)
        quad_points = torch.cat((quad_x, quad_t), dim=-1).reshape(-1, 2)
        quad_points.requires_grad_(True)
        u_quad = self.model.net(quad_points)
        spatial_quad = self.pde.ks_spatial_operator(quad_points, u_quad).reshape(
            x_front.shape[0],
            self.quadrature_order,
            1,
        )
        integral = 0.5 * dt * torch.sum(
            weights[None, :, None] * spatial_quad,
            dim=1,
        )
        return (u_right - u_left + integral) / dt, u_right

    def backward_weighted(self, base_loss):
        """Backpropagate the exact weighted mean in memory-bounded chunks."""
        base_loss.backward()
        dde.grad.clear()

        x_front, front_times = self.front_grid()
        interval_square_sums = [
            torch.zeros((), device=x_front.device, dtype=x_front.dtype)
            for _ in range(self.num_intervals)
        ]
        right_front_square_sums = [
            torch.zeros((), device=x_front.device, dtype=x_front.dtype)
            for _ in range(self.num_intervals)
        ]
        defect_max = torch.zeros((), device=x_front.device, dtype=x_front.dtype)

        for interval_index in range(self.num_intervals):
            for start in range(0, self.num_x_points, self.x_batch_size):
                stop = min(start + self.x_batch_size, self.num_x_points)
                chunk_x = x_front[start:stop]
                normalized_defect, u_right = self._interval_defect(
                    interval_index,
                    chunk_x,
                    front_times,
                )
                chunk_square_sum = torch.sum(normalized_defect.square())
                weighted_contribution = (
                    self.weight
                    * chunk_square_sum
                    / float(self.num_intervals * self.num_x_points)
                )
                weighted_contribution.backward()
                dde.grad.clear()

                interval_square_sums[interval_index] += chunk_square_sum.detach()
                right_front_square_sums[interval_index] += torch.sum(
                    u_right.detach().square()
                )
                defect_max = torch.maximum(
                    defect_max,
                    torch.max(torch.abs(normalized_defect.detach())),
                )

        interval_losses = torch.stack(interval_square_sums) / float(self.num_x_points)
        raw_loss = torch.mean(interval_losses)
        with torch.no_grad():
            initial_inputs = torch.cat(
                (x_front, torch.full_like(x_front, front_times[0])),
                dim=1,
            )
            initial_front_square_sum = torch.sum(self.model.net(initial_inputs).square())
        front_square_sums = [initial_front_square_sum, *right_front_square_sums]

        diagnostics = {
            "front_integral_loss": raw_loss,
            "front_integral_weight": self.weight,
            "weighted_front_integral_loss": raw_loss * self.weight,
            "front_defect_rms": torch.sqrt(raw_loss),
            "front_defect_max": defect_max,
        }
        for index, interval_loss in enumerate(interval_losses):
            diagnostics[f"front_{index}_loss"] = interval_loss
        for index, square_sum in enumerate(front_square_sums):
            diagnostics[f"u_front_{index}_rms"] = torch.sqrt(
                square_sum / float(self.num_x_points)
            )
        self.last_diagnostics = diagnostics
        self.model.front_integral_loss_diagnostics = diagnostics
        return raw_loss


def attach_front_integral_loss_train_step(model, front_integral_loss):
    """Attach scalar PINN + front-loss training for Adam, SOAP, and Muon."""
    if dde.backend.backend_name != "pytorch":
        raise ValueError("Front integral loss currently supports only the PyTorch backend.")

    optimizer_name = str(getattr(model, "opt_name", "")).lower()
    if optimizer_name not in {"adam", "soap", "muon"}:
        raise ValueError("Front integral loss supports only Adam, SOAP, and Muon.")

    model.front_integral_loss = front_integral_loss
    model.front_integral_loss_diagnostics = None

    def train_step(inputs, targets):
        def closure():
            losses = model.outputs_losses_train(inputs, targets)[1]
            model.opt.losses = losses
            base_loss = torch.sum(losses)
            model.opt.zero_grad()

            if front_integral_loss.weight == 0.0:
                base_loss.backward()
                front_integral_loss.set_zero_diagnostics(base_loss)
                total_loss = base_loss
            else:
                front_raw_loss = front_integral_loss.backward_weighted(base_loss)
                total_loss = (
                    base_loss.detach()
                    + front_integral_loss.weight * front_raw_loss
                )

            diagnostics = front_integral_loss.last_diagnostics
            diagnostics["deepxde_loss_sum"] = base_loss.detach()
            diagnostics["actual_total_loss"] = total_loss.detach()
            model.front_integral_loss_diagnostics = diagnostics
            dde.grad.clear()
            return total_loss

        loss = model.opt.step(closure)
        if hasattr(model.opt, "after_train_step"):
            model.opt.after_train_step()
        if model.lr_scheduler is not None:
            if model.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                model.lr_scheduler.step(loss)
            else:
                model.lr_scheduler.step()

    model.train_step = train_step
