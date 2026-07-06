"""Self-scaled Broyden optimizer for the PyTorch backend.

This implementation follows scimba_torch.optimizers.ssbroyden.SSBroyden with
only the "ssbroyden" method exposed.
"""

import math
from typing import Callable, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func,
    x,
    t,
    d,
    f,
    g,
    gtd,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-9,
    max_ls=25,
):
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0

    while ls_iter < max_ls:
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break
        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break
        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]
        bracket_gtd = [gtd, gtd_new]

    insuf_progress = False
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        if len(bracket) == 1 or abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(float(t), device=g_new.device, dtype=g_new.dtype)
    return f_new, g_new, t, ls_func_evals


class SSBroyden(Optimizer):
    """Implements the self-scaled Broyden algorithm."""

    _INIT_MATRIX_MULTIPLIER = 1
    _UPDATE_MATRIX_MULTIPLIER = 6

    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1.0,
        tolerance_grad: float = 1e-10,
        debug: bool = False,
        debug_every: int = 100,
    ):
        if isinstance(lr, Tensor):
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
            lr_value = float(lr.item())
        else:
            lr_value = float(lr)
            lr = torch.tensor(lr_value)
        if not 0.0 < lr_value:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < tolerance_grad:
            raise ValueError(f"Invalid tolerance on gradient: {tolerance_grad}")
        if debug_every < 1:
            raise ValueError(f"Invalid debug_every value: {debug_every}")

        defaults = {
            "lr": lr,
            "tolerance_grad": tolerance_grad,
            "method": "ssbroyden",
            "debug": debug,
            "debug_every": debug_every,
        }
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "SS Broyden/BFGS doesn't support per-parameter options"
                " (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        nbparams = self._numel()
        self._check_cuda_dense_memory(
            nbparams,
            self._params[0].dtype,
            self._params[0].device,
            self._INIT_MATRIX_MULTIPLIER,
            "initializing Hk",
        )
        state = self.state[self._params[0]]
        state["k"] = 0
        state["h_updates"] = 0
        state["first_step"] = True
        state["Hk"] = torch.eye(
            nbparams,
            dtype=self._params[0].dtype,
            device=self._params[0].device,
        )

    def _reset_hessian(self, state):
        state["Hk"] = torch.eye(
            self._numel(),
            dtype=state["Hk"].dtype,
            device=state["Hk"].device,
        )
        state["first_step"] = True

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )
        return self._numel_cache

    @staticmethod
    def _format_bytes(num_bytes):
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        value = float(num_bytes)
        for unit in units:
            if abs(value) < 1024.0 or unit == units[-1]:
                return f"{value:.2f} {unit}"
            value /= 1024.0

    @classmethod
    def _dense_matrix_bytes(cls, num_params, dtype):
        element_size = torch.empty((), dtype=dtype).element_size()
        return int(num_params) * int(num_params) * element_size

    @classmethod
    def _check_cuda_dense_memory(cls, num_params, dtype, device, matrix_multiplier, stage):
        if device.type != "cuda":
            return

        one_matrix = cls._dense_matrix_bytes(num_params, dtype)
        required = matrix_multiplier * one_matrix
        free, total = torch.cuda.mem_get_info(device)
        if required <= free:
            return

        raise RuntimeError(
            "SSBroyden requires dense O(n_params^2) memory and cannot safely "
            f"continue while {stage}. "
            f"num_params={num_params}; one dense matrix="
            f"{cls._format_bytes(one_matrix)}; estimated peak for this stage="
            f"{cls._format_bytes(required)}; CUDA free={cls._format_bytes(free)}; "
            f"CUDA total={cls._format_bytes(total)}. Reduce --hidden-layers or use "
            "a first-order optimizer for this model size."
        )

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _flatten(self, params):
        views = []
        for p in params:
            if p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure: Callable) -> torch.Tensor:
        assert len(self.param_groups) == 1
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = float(group["lr"])
        tolerance_grad = group["tolerance_grad"]

        state = self.state[self._params[0]]
        state["k"] += 1
        x_init = self._clone_param()
        theta_k = self._flatten(x_init)

        def obj_func(x, t, d):
            return self._directional_evaluate(closure, x, t, d)

        orig_loss = closure()
        loss = float(orig_loss)
        grad_k = self._gather_flat_grad()
        opt_cond = grad_k.abs().max() <= tolerance_grad
        if opt_cond:
            self._print_debug(state, status="optimality", loss=loss, grad_max=grad_k.abs().max())
            return orig_loss

        prec_grad = state["Hk"] @ grad_k
        prec_grad = prec_grad.neg()
        direction_norm = prec_grad.norm()
        h_norm = state["Hk"].norm()
        gtd = grad_k.dot(prec_grad)

        loss, grad_kp1, alpha_k, ls_func_evals = _strong_wolfe(
            obj_func, x_init, lr, prec_grad, loss, grad_k, gtd
        )
        if (
            math.isnan(loss)
            or math.isinf(loss)
            or torch.isnan(alpha_k)
            or torch.isinf(alpha_k)
            or torch.any(torch.isnan(grad_kp1))
            or torch.any(torch.isinf(grad_kp1))
        ):
            orig_loss = closure()
            self._reset_hessian(state)
            self._print_debug(
                state,
                status="line_search_invalid_reset_hk",
                loss=loss,
                alpha_k=alpha_k,
                gtd=gtd,
                grad_max=grad_k.abs().max(),
                direction_norm=direction_norm,
                h_norm=h_norm,
            )
            return orig_loss

        self._add_grad(alpha_k, prec_grad)

        theta_kp1 = self._flatten(self._clone_param())
        s_k = theta_kp1 - theta_k
        y_k = grad_kp1 - grad_k

        Hkyk = state["Hk"] @ y_k
        yk_dot_Hkyk = y_k @ Hkyk
        yk_dot_sk = y_k @ s_k
        eps = torch.finfo(yk_dot_sk.dtype).eps
        if (
            yk_dot_sk <= eps
            or yk_dot_Hkyk <= eps
            or torch.isnan(yk_dot_sk)
            or torch.isnan(yk_dot_Hkyk)
            or torch.isinf(yk_dot_sk)
            or torch.isinf(yk_dot_Hkyk)
        ):
            state["first_step"] = False
            self._print_debug(
                state,
                status="bad_curvature",
                loss=loss,
                alpha_k=alpha_k,
                gtd=gtd,
                yk_dot_sk=yk_dot_sk,
                yk_dot_Hkyk=yk_dot_Hkyk,
                grad_max=grad_k.abs().max(),
                direction_norm=direction_norm,
                h_norm=h_norm,
            )
            return orig_loss

        v_k = torch.sqrt(yk_dot_Hkyk) * (s_k / (yk_dot_sk) - Hkyk / yk_dot_Hkyk)
        phi_k = 1.0

        b_k = -alpha_k * (s_k @ grad_k) / yk_dot_sk
        h_k = yk_dot_Hkyk / yk_dot_sk
        a_k = h_k * b_k - 1.0
        c_k = torch.sqrt(torch.abs(a_k) / (a_k + 1.0))
        rhom_k = min(1.0, h_k * (1 - c_k))
        thetam_k = (rhom_k - 1) / a_k
        thetap_k = 1.0 / rhom_k
        theta_k = max(thetam_k, min(thetap_k, (1.0 - b_k) / b_k))
        sigma_k = 1 + a_k * theta_k
        n = self._numel()
        if state.get("first_step", False):
            tau_k = h_k / (1.0 + a_k * theta_k)
        else:
            rhok_k = min(1.0, 1.0 / b_k)
            sigma_k_pow = torch.abs(sigma_k) ** (1.0 / (1.0 - n))
            if theta_k <= 0:
                tau_k = min(rhok_k * sigma_k_pow, sigma_k)
            else:
                tau_k = rhok_k * min(sigma_k_pow, 1.0 / theta_k)
        phi_k = (1 - theta_k) / (1.0 + a_k * theta_k)

        self._check_cuda_dense_memory(
            self._numel(),
            state["Hk"].dtype,
            state["Hk"].device,
            self._UPDATE_MATRIX_MULTIPLIER,
            "updating Hk",
        )
        temp1 = (Hkyk[:, None] @ Hkyk[None, :]) / yk_dot_Hkyk
        temp2 = phi_k * (v_k[:, None] @ v_k[None, :])
        temp3 = (s_k[:, None] @ s_k[None, :]) / yk_dot_sk
        H_kp1 = (1 / tau_k) * (state["Hk"] - temp1 + temp2) + temp3

        if torch.any(torch.isnan(H_kp1)):
            orig_loss = closure()
            self._reset_hessian(state)
            self._print_debug(
                state,
                status="h_update_nan_reset_hk",
                loss=loss,
                alpha_k=alpha_k,
                gtd=gtd,
                yk_dot_sk=yk_dot_sk,
                yk_dot_Hkyk=yk_dot_Hkyk,
                tau_k=tau_k,
                theta_k=theta_k,
                phi_k=phi_k,
                grad_max=grad_k.abs().max(),
                direction_norm=direction_norm,
                h_norm=h_norm,
            )
            return orig_loss

        state["Hk"] = H_kp1
        state["first_step"] = False
        state["h_updates"] += 1
        self._print_debug(
            state,
            status="updated",
            loss=loss,
            alpha_k=alpha_k,
            gtd=gtd,
            yk_dot_sk=yk_dot_sk,
            yk_dot_Hkyk=yk_dot_Hkyk,
            tau_k=tau_k,
            theta_k=theta_k,
            phi_k=phi_k,
            grad_max=grad_k.abs().max(),
            direction_norm=direction_norm,
            h_norm=h_norm,
        )
        return orig_loss

    def _print_debug(self, state, status, **values):
        group = self.param_groups[0]
        if not group["debug"]:
            return
        if state["k"] % group["debug_every"] != 0:
            return

        def scalar(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        fields = [
            f"step={state['k']}",
            f"h_updates={state['h_updates']}",
            f"status={status}",
        ]
        for name in (
            "loss",
            "alpha_k",
            "gtd",
            "yk_dot_sk",
            "yk_dot_Hkyk",
            "tau_k",
            "theta_k",
            "phi_k",
            "grad_max",
            "direction_norm",
            "h_norm",
        ):
            value = scalar(values.get(name))
            if value is not None:
                fields.append(f"{name}={value:.6e}")

        print("[SSBroyden debug] " + " ".join(fields))
