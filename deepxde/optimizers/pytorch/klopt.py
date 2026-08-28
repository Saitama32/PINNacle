"""KL-Shampoo and KL-SOAP optimizers for the PyTorch backend.

The KL preconditioning implementation is adapted from the authors' prototype:
https://github.com/yorkerlin/KL-Methods/blob/main/optim/kl_opt.py

``KLOpt`` retains the upstream algorithm for 2D/3D tensors. The DeepXDE
integration uses ``KLOptWithAuxAdam`` so biases, scalars, and tensors whose
shape is changed below two dimensions by the upstream ``squeeze`` operation
receive an AdamW update instead of failing in the KL preconditioner.
"""

from itertools import chain
from importlib.util import find_spec

import torch


# The upstream prototype decorates these kernels with ``torch.compile``. The
# default Inductor backend requires Triton and otherwise fails on the first
# optimizer step (notably on Windows), so retain compilation only where that
# runtime is actually installed.
_compile_update = (
    torch.compile
    if hasattr(torch, "compile") and find_spec("triton") is not None
    else (lambda function: function)
)


class KLOpt(torch.optim.Optimizer):
    """Prototype implementation of KL-Shampoo and KL-SOAP."""

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.98),
        shampoo_beta=-1,
        eps=1e-8,
        weight_decay=0.01,
        precondition_frequency=10,
        using_klsoap=False,
        normalize_grads=False,
        init_factor=0.1,
        using_damping=False,
        using_clamping=True,
        max_clamp_value=4000,
        cast_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "normalize_grads": normalize_grads,
        }
        self.cast_dtype = cast_dtype
        self.using_klsoap = using_klsoap
        self.using_clamping = using_clamping
        self.max_clamp_value = max_clamp_value
        self.init_factor = init_factor
        self.using_damping = using_damping
        self.damping = eps if using_damping else 0.0
        super().__init__(params, defaults)

    def init_preconditioner(
        self, grad, state, precondition_frequency=10, shampoo_beta=0.95
    ):
        state["GG"] = []
        if grad.dim() == 1:
            state["GG"].append([])
        else:
            for size in grad.shape:
                state["GG"].append(
                    torch.zeros(size, size, device=grad.device, dtype=grad.dtype)
                )
        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    @_compile_update
    def update_S(self, grad, state, mat, idx, beta, total_factor):
        factor = total_factor / grad.shape[idx]
        state["GG"][idx].mul_(beta).add_(mat, alpha=(1.0 - beta) / factor)

    @_compile_update
    def update_eigen_value(
        self, state, diag, idx, beta, traces, total_trace, damping
    ):
        if damping > 0:
            diag = diag + total_trace / traces[idx]
        inv_d = state["eigen_sqrt_inv"][idx] ** 2
        eigenvalue = torch.squeeze(1.0 / inv_d).nan_to_num_(
            nan=0.0, posinf=0.0, neginf=0.0
        )
        eigenvalue.lerp_(diag, 1.0 - beta)
        sqrt_inv = (1.0 / torch.sqrt(eigenvalue)).nan_to_num_(
            nan=0.0, posinf=0.0, neginf=0.0
        )
        if self.using_clamping:
            sqrt_inv = torch.clamp(
                sqrt_inv,
                max=max(10, min(eigenvalue.shape[0], self.max_clamp_value)),
            )
        state["eigen_sqrt_inv"][idx] = sqrt_inv

    @_compile_update
    def update_3d_preconditioner(
        self, grad, state, total_factor, traces, total_trace, damping
    ):
        assert grad.dim() == 3
        inv_s_half = []
        for idx in range(grad.dim()):
            inv_s_half.append(
                state["Q"][idx] * state["eigen_sqrt_inv"][idx].view(1, -1)
            )

        beta = state["shampoo_beta"]
        g_inv_s1 = torch.einsum("ija,ip->pja", grad, inv_s_half[0])
        g_inv_s1_q2 = torch.einsum("pja,jl->pla", g_inv_s1, state["Q"][1])
        g_inv_s1_q2_q3 = torch.einsum(
            "pqa,am->pqm", g_inv_s1_q2, state["Q"][2]
        )

        g_inv_s12 = g_inv_s1_q2 * state["eigen_sqrt_inv"][1].view(1, -1, 1)
        s3 = torch.tensordot(g_inv_s12, g_inv_s12, dims=[[0, 1], [0, 1]])
        self.update_S(grad, state, s3, 2, beta, total_factor)

        g_inv_s1_q3 = torch.einsum("pqb,bm->pqm", g_inv_s1, state["Q"][2])
        g_inv_s13 = g_inv_s1_q3 * state["eigen_sqrt_inv"][2].view(1, 1, -1)
        s2 = torch.tensordot(g_inv_s13, g_inv_s13, dims=[[0, 2], [0, 2]])
        self.update_S(grad, state, s2, 1, beta, total_factor)

        diag3 = torch.mean(
            (
                g_inv_s1_q2_q3
                * state["eigen_sqrt_inv"][1].view(1, -1, 1)
            )
            ** 2,
            dim=(0, 1),
        )
        self.update_eigen_value(
            state, diag3, 2, beta, traces, total_trace, damping
        )
        diag2 = torch.mean(
            (
                g_inv_s1_q2_q3
                * state["eigen_sqrt_inv"][2].view(1, 1, -1)
            )
            ** 2,
            dim=(0, 2),
        )
        self.update_eigen_value(
            state, diag2, 1, beta, traces, total_trace, damping
        )

        g_inv_s3 = torch.einsum("ijb,bm->ijm", grad, inv_s_half[2])
        g_inv_s3_q2 = torch.einsum("ijm,jq->iqm", g_inv_s3, state["Q"][1])
        g_inv_s3_q2_q1 = torch.einsum(
            "iqm,ip->pqm", g_inv_s3_q2, state["Q"][0]
        )
        g_inv_s32 = g_inv_s3_q2 * state["eigen_sqrt_inv"][1].view(1, -1, 1)
        s1 = torch.tensordot(g_inv_s32, g_inv_s32, dims=[[1, 2], [1, 2]])
        self.update_S(grad, state, s1, 0, beta, total_factor)
        diag1 = torch.mean(
            (
                g_inv_s3_q2_q1
                * state["eigen_sqrt_inv"][1].view(1, -1, 1)
            )
            ** 2,
            dim=(1, 2),
        )
        self.update_eigen_value(
            state, diag1, 0, beta, traces, total_trace, damping
        )

    @_compile_update
    def update_2d_preconditioner(
        self, grad, state, total_factor, traces, total_trace, damping
    ):
        assert grad.dim() == 2
        beta = state["shampoo_beta"]
        for idx, _ in enumerate(grad.shape):
            basis = state["Q"][abs(idx - 1)]
            sqrt_inv = state["eigen_sqrt_inv"][abs(idx - 1)]
            if idx == 0:
                step0 = basis.T @ grad.T
                half = step0 * sqrt_inv.view(-1, 1)
                mat = half.T @ half
            else:
                step1 = basis.T @ grad
                half = step1 * sqrt_inv.view(-1, 1)
                mat = half.T @ half
            self.update_S(grad, state, mat, idx, beta, total_factor)

        diag_half = step1 @ state["Q"][1]
        left_diag = torch.mean(
            (diag_half * state["eigen_sqrt_inv"][1].view(1, -1)) ** 2, 1
        )
        right_diag = torch.mean(
            (diag_half * state["eigen_sqrt_inv"][0].view(-1, 1)) ** 2, 0
        )
        self.update_eigen_value(
            state, left_diag, 0, beta, traces, total_trace, damping
        )
        self.update_eigen_value(
            state, right_diag, 1, beta, traces, total_trace, damping
        )

    @torch.no_grad()
    def update_preconditioner(self, grad, state):
        traces = []
        total_factor = torch.numel(grad)
        damping = self.damping if self.using_damping else 0.0
        total_trace = damping
        if damping > 0:
            for idx, _ in enumerate(grad.shape):
                if state["Q"] is None:
                    current_trace = 1.0
                else:
                    current_trace = torch.mean(
                        state["eigen_sqrt_inv"][idx] ** 2
                    )
                total_trace *= current_trace
                traces.append(current_trace)

        if state["Q"] is None:
            beta = state["shampoo_beta"]
            for idx, _ in enumerate(grad.shape):
                mat = torch.tensordot(
                    grad,
                    grad,
                    dims=[
                        [*chain(range(idx), range(idx + 1, len(grad.shape)))]
                    ]
                    * 2,
                )
                self.update_S(grad, state, mat, idx, beta, total_factor)
            state["Q"], state["eigen_sqrt_inv"] = self.get_orthogonal_matrix(
                state["GG"]
            )
        else:
            if (
                self.using_klsoap
                and state["step"] % state["precondition_frequency"] == 0
            ):
                state["exp_avg"] = self.project_back(state["exp_avg"], state)
            if len(grad.shape) == 2:
                self.update_2d_preconditioner(
                    grad, state, total_factor, traces, total_trace, damping
                )
            elif len(grad.shape) == 3:
                self.update_3d_preconditioner(
                    grad, state, total_factor, traces, total_trace, damping
                )
            else:
                raise AssertionError("KLOpt supports only 2D and 3D tensors")

        if (
            state["step"] > 0
            and state["step"] % state["precondition_frequency"] == 0
        ):
            state["Q"] = self.get_orthogonal_matrix_QR(state)
            if self.using_klsoap:
                state["exp_avg"] = self.project(state["exp_avg"], state)

    def _kl_step_parameter(self, p, group):
        grad = torch.squeeze(p.grad.to(dtype=self.cast_dtype))
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(grad)
            if self.using_klsoap:
                state["exp_avg_sq"] = torch.zeros_like(grad)
        if "Q" not in state:
            self.init_preconditioner(
                grad,
                state,
                precondition_frequency=group["precondition_frequency"],
                shampoo_beta=(
                    group["shampoo_beta"]
                    if group["shampoo_beta"] >= 0
                    else group["betas"][1]
                ),
            )
            self.update_preconditioner(grad, state)
            return

        state["step"] += 1
        if self.using_klsoap:
            update = self.klsoap_update(
                state, grad, group["betas"][0], group["betas"][1], group["eps"]
            )
        else:
            update = self.klshampoo_update(
                state, grad, group["betas"][0], group["eps"]
            )
        self.update_preconditioner(grad, state)
        if group["normalize_grads"]:
            update = update / (1e-30 + torch.mean(update**2) ** 0.5)
        p.add_(update.view(p.shape), alpha=-group["lr"])
        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=-group["lr"] * group["weight_decay"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._kl_step_parameter(p, group)
        return loss

    def klshampoo_update(self, state, grad, beta1, damping):
        exp_avg = state["exp_avg"]
        exp_avg.lerp_(grad, 1.0 - beta1)
        projected = self.project(exp_avg, state)
        if len(grad.shape) == 2:
            sqrt_inv = state["eigen_sqrt_inv"][0].view(-1, 1) * state[
                "eigen_sqrt_inv"
            ][1].view(1, -1)
        elif len(grad.shape) == 3:
            sqrt_inv = (
                state["eigen_sqrt_inv"][0].view(-1, 1, 1)
                * state["eigen_sqrt_inv"][1].view(1, -1, 1)
                * state["eigen_sqrt_inv"][2].view(1, 1, -1)
            )
        else:
            raise AssertionError("KLOpt supports only 2D and 3D tensors")
        sqrt_inv.div_(1.0 + sqrt_inv * damping)
        return self.project_back(projected * sqrt_inv, state)

    def klsoap_update(self, state, grad, beta1, beta2, damping):
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        projected = self.project(grad, state)
        exp_avg.lerp_(projected, 1.0 - beta1)
        exp_avg_sq.lerp_(projected.square(), 1.0 - beta2)
        return self.project_back(exp_avg / exp_avg_sq.sqrt().add_(damping), state)

    @staticmethod
    def project(grad, state):
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                grad = grad.permute(list(range(1, len(grad.shape))) + [0])
        return grad

    @staticmethod
    def project_back(grad, state):
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                grad = grad.permute(list(range(1, len(grad.shape))) + [0])
        return grad

    def get_orthogonal_matrix(self, matrices):
        final = []
        info = []
        for source in matrices:
            if len(source) == 0:
                final.append([])
                continue
            original_dtype = source.dtype
            matrix = source if source.dtype == torch.float else source.float()
            try:
                _, basis = torch.linalg.eigh(
                    matrix
                    + 1e-30
                    * torch.eye(matrix.shape[0], device=matrix.device)
                )
            except RuntimeError:
                _, basis = torch.linalg.eigh(
                    matrix.to(torch.float64)
                    + 1e-30
                    * torch.eye(matrix.shape[0], device=matrix.device)
                )
                basis = basis.to(matrix.dtype)
            basis = torch.flip(basis, [1])
            initial = torch.ones(
                basis.shape[0], device=basis.device, dtype=basis.dtype
            ) * self.init_factor
            sqrt_inv = (1.0 / torch.sqrt(initial)).nan_to_num_(
                nan=0.0, posinf=0.0, neginf=0.0
            )
            if original_dtype != torch.float:
                sqrt_inv = sqrt_inv.to(dtype=original_dtype)
                basis = basis.to(dtype=original_dtype)
            info.append(sqrt_inv)
            final.append(basis)
        return final, info

    @staticmethod
    def get_orthogonal_matrix_QR(state):
        final = []
        for matrix, basis in zip(state["GG"], state["Q"]):
            assert len(matrix) > 0
            original_dtype = matrix.dtype
            refreshed, _ = torch.linalg.qr(matrix.float() @ basis.float())
            final.append(refreshed.to(dtype=original_dtype))
        return final


class KLOptWithAuxAdam(KLOpt):
    """KLOpt on supported tensors and AdamW on auxiliary parameters."""

    @staticmethod
    def _adamw_step_parameter(p, group, state, cast_dtype):
        grad = p.grad.to(dtype=cast_dtype)
        if grad.is_sparse:
            raise RuntimeError("KLOpt auxiliary AdamW does not support sparse gradients")
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["step"] += 1
        beta1, beta2 = group["betas"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        correction1 = 1.0 - beta1 ** state["step"]
        correction2 = 1.0 - beta2 ** state["step"]
        step_size = group["lr"] * correction2**0.5 / correction1
        if group["weight_decay"] > 0.0:
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
        update = exp_avg / exp_avg_sq.sqrt().add_(group["eps"])
        p.add_(update.to(dtype=p.dtype), alpha=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            use_klopt = group.get("use_klopt", True)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if use_klopt:
                    self._kl_step_parameter(p, group)
                else:
                    self._adamw_step_parameter(
                        p, group, self.state[p], self.cast_dtype
                    )
        return loss


__all__ = ["KLOpt", "KLOptWithAuxAdam"]
