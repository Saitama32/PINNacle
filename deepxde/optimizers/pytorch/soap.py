"""SOAP optimizer for the PyTorch backend.

This implementation follows the public SOAP optimizer structure from
nikhilvyas/SOAP for the common PINN case: matrix parameters are optimized in
the Shampoo eigenbasis with Adam-style moments, while 1D tensors, scalars, and
oversized matrices use an AdamW-style fallback.
"""

import torch


class SOAP(torch.optim.Optimizer):
    """SOAP optimizer with 2D matrix preconditioning and AdamW fallback."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.99, 0.999),
        shampoo_beta=0.999,
        eps=1e-8,
        weight_decay=0,
        precondition_frequency=10,
        max_precondition_dim=4096,
        bias_correction=True,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0 <= shampoo_beta < 1:
            raise ValueError(f"Invalid shampoo_beta parameter: {shampoo_beta}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if precondition_frequency < 1:
            raise ValueError("precondition_frequency must be >= 1")
        if max_precondition_dim < 1:
            raise ValueError("max_precondition_dim must be >= 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precondition_dim=max_precondition_dim,
            bias_correction=bias_correction,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients.")
                if not torch.isfinite(grad).all():
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if self._use_preconditioner(p, group):
                        self._init_preconditioner(p, grad, state, group)

                state["step"] += 1

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                if self._use_preconditioner(p, group):
                    self._soap_step(p, grad, state, group, beta1, beta2)
                    self._update_preconditioner(grad, state, group)
                else:
                    self._adamw_step(p, grad, state, group, beta1, beta2)

        return loss

    def _use_preconditioner(self, p, group):
        return (
            p.ndim == 2
            and p.shape[0] <= group["max_precondition_dim"]
            and p.shape[1] <= group["max_precondition_dim"]
        )

    def _adamw_step(self, p, grad, state, group, beta1, beta2):
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if group["bias_correction"]:
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = group["lr"] * (bias_correction2**0.5) / bias_correction1
        else:
            step_size = group["lr"]

        denom = exp_avg_sq.sqrt().add_(group["eps"])
        update = exp_avg / denom
        if torch.isfinite(update).all():
            p.add_(update, alpha=-step_size)

    def _soap_step(self, p, grad, state, group, beta1, beta2):
        projected_grad = self._project(grad, state)
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(projected_grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            projected_grad, projected_grad, value=1 - beta2
        )

        if group["bias_correction"]:
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = group["lr"] * (bias_correction2**0.5) / bias_correction1
        else:
            step_size = group["lr"]

        denom = exp_avg_sq.sqrt().add_(group["eps"])
        projected_update = exp_avg / denom
        update = self._project_back(projected_update, state)
        if torch.isfinite(update).all():
            p.add_(update, alpha=-step_size)

    def _init_preconditioner(self, p, grad, state, group):
        rows, cols = p.shape
        state["ggt_left"] = torch.zeros(rows, rows, dtype=p.dtype, device=p.device)
        state["ggt_right"] = torch.zeros(cols, cols, dtype=p.dtype, device=p.device)
        state["q_left"] = torch.eye(rows, dtype=p.dtype, device=p.device)
        state["q_right"] = torch.eye(cols, dtype=p.dtype, device=p.device)
        self._update_preconditioner(grad, state, group, force=True)

    def _update_preconditioner(self, grad, state, group, force=False):
        shampoo_beta = group["shampoo_beta"]
        state["ggt_left"].mul_(shampoo_beta).addmm_(
            grad, grad.t(), beta=1, alpha=1 - shampoo_beta
        )
        state["ggt_right"].mul_(shampoo_beta).addmm_(
            grad.t(), grad, beta=1, alpha=1 - shampoo_beta
        )

        if force:
            state["q_left"] = self._orthogonal_matrix(state["ggt_left"])
            state["q_right"] = self._orthogonal_matrix(state["ggt_right"])
            return

        if state["step"] % group["precondition_frequency"]:
            return

        old_q_left = state["q_left"]
        old_q_right = state["q_right"]
        q_left, left_order = self._orthogonal_matrix_qr(
            state["ggt_left"], old_q_left
        )
        q_right, right_order = self._orthogonal_matrix_qr(
            state["ggt_right"], old_q_right
        )

        # The first moment is a vector expressed in the old eigenbasis, so it
        # can be projected back to parameter space and into the refreshed one.
        state["exp_avg"] = q_left.t().matmul(
            old_q_left.matmul(state["exp_avg"]).matmul(old_q_right.t())
        ).matmul(q_right)

        # Adam's second moment is elementwise and must not be matrix-rotated.
        # The QR refresh only reorders eigen-directions, exactly as in the
        # official SOAP implementation.
        state["exp_avg_sq"] = state["exp_avg_sq"].index_select(
            0, left_order
        ).index_select(1, right_order)
        state["q_left"] = q_left
        state["q_right"] = q_right

    @staticmethod
    def _project(grad, state):
        return state["q_left"].t().matmul(grad).matmul(state["q_right"])

    @staticmethod
    def _project_back(projected_update, state):
        return state["q_left"].matmul(projected_update).matmul(state["q_right"].t())

    @staticmethod
    def _orthogonal_matrix(matrix):
        matrix = (matrix + matrix.t()) * 0.5
        eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        try:
            eigvals, eigvecs = torch.linalg.eigh(matrix + 1e-30 * eye)
        except RuntimeError:
            return eye
        order = torch.argsort(eigvals, descending=True)
        return eigvecs[:, order]

    @staticmethod
    def _orthogonal_matrix_qr(matrix, basis):
        """Refresh one eigenbasis with power iteration and QR."""
        original_dtype = matrix.dtype
        work_matrix = matrix.float()
        work_basis = basis.float()
        estimated_eigenvalues = torch.diag(
            work_basis.t().matmul(work_matrix).matmul(work_basis)
        )
        order = torch.argsort(estimated_eigenvalues, descending=True)
        work_basis = work_basis[:, order]
        refreshed_basis, _ = torch.linalg.qr(work_matrix.matmul(work_basis))
        return refreshed_basis.to(original_dtype), order
