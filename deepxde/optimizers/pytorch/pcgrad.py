"""PCGrad optimizer wrapper for the PyTorch backend.

This follows the algorithm in tianheyu927/PCGrad's TensorFlow implementation:
shuffle task losses, compute one gradient vector per task, project conflicting
gradients away, sum the projected gradients, then step the wrapped optimizer.
"""

import random

import torch


class PCGrad(torch.optim.Optimizer):
    """Gradient surgery wrapper for multi-task objectives."""

    def __init__(self, optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer instance")
        self.optimizer = optimizer
        self._initializing = True
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self._initializing = False
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults
        self.losses = None

    def __getstate__(self):
        return {"optimizer": self.optimizer, "losses": self.losses}

    def __setstate__(self, state):
        self.optimizer = state["optimizer"]
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        self.losses = state.get("losses")

    def zero_grad(self, set_to_none=True):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        if getattr(self, "_initializing", False):
            return super().add_param_group(param_group)
        result = self.optimizer.add_param_group(param_group)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        return result

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("PCGrad requires a closure returning the total loss.")

        with torch.enable_grad():
            loss = closure(skip_backward=True)
            losses = self.losses
            if losses is None:
                raise RuntimeError("PCGrad requires per-task losses on optimizer.losses.")
            if isinstance(losses, torch.Tensor):
                losses = list(torch.unbind(losses))
            else:
                losses = list(losses)
            if len(losses) == 0:
                raise RuntimeError("PCGrad received no losses.")

            params = [
                p
                for group in self.param_groups
                for p in group["params"]
                if p.requires_grad
            ]
            grads_task = []
            random.shuffle(losses)
            for task_loss in losses:
                grads = torch.autograd.grad(
                    task_loss,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                grads_task.append(self._flatten_grads(grads, params))

        pc_grad = self._project_conflicting(grads_task)
        self.zero_grad()
        self._set_grad_vector(pc_grad, params)
        self.optimizer.step()
        return loss

    def _flatten_grads(self, grads, params):
        flat_grads = []
        for grad, param in zip(grads, params):
            if grad is None:
                flat_grads.append(torch.zeros_like(param).reshape(-1))
            else:
                flat_grads.append(grad.reshape(-1))
        return torch.cat(flat_grads)

    def _project_conflicting(self, grads_task):
        projected = []
        for grad_task in grads_task:
            grad_task = grad_task.clone()
            for grad_k in grads_task:
                inner_product = torch.dot(grad_task, grad_k)
                if inner_product < 0:
                    grad_task = grad_task - inner_product * grad_k / (grad_k.norm() ** 2)
            projected.append(grad_task)
        return torch.stack(projected).sum(dim=0)

    def _set_grad_vector(self, grad_vector, params):
        pointer = 0
        for param in params:
            num_param = param.numel()
            param.grad = (
                grad_vector[pointer : pointer + num_param]
                .view_as(param)
                .detach()
                .clone()
            )
            pointer += num_param
