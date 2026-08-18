"""JAX-PI style gradient-norm loss balancing for the PyTorch backend."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

import deepxde as dde


class AdaptiveLossWeights:
    """Mutable loss weights accepted by PINNacle's PyTorch ``Model.compile``."""

    def __init__(self, weights):
        values = np.asarray(weights, dtype=np.float32)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("loss weights must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("loss weights must be positive and finite")
        self._weights = values.copy()

    def __call__(self):
        return self._weights

    def get_numpy(self):
        return self._weights.copy()

    def set(self, weights):
        values = np.asarray(weights, dtype=np.float32)
        if values.shape != self._weights.shape:
            raise ValueError(
                f"expected loss weights with shape {self._weights.shape}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("loss weights must be positive and finite")
        self._weights = values.copy()


def grad_norm_weights(losses, parameters, epsilon=1e-12):
    """Compute ``mean(||grad L_i||) / ||grad L_i||`` for scalar losses."""
    losses = list(losses)
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not losses:
        raise ValueError("at least one loss is required")
    if not parameters:
        raise ValueError("at least one trainable parameter is required")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")

    norms = []
    for index, loss in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index + 1 < len(losses),
            allow_unused=True,
        )
        squared_norm = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for gradient in gradients:
            if gradient is not None:
                squared_norm = squared_norm + torch.sum(gradient.detach().pow(2))
        norms.append(torch.sqrt(squared_norm).clamp_min(float(epsilon)))

    norms_tensor = torch.stack(norms)
    weights = torch.mean(norms_tensor) / norms_tensor
    if not bool(torch.all(torch.isfinite(weights))):
        raise FloatingPointError("Grad Norm produced non-finite loss weights")
    return weights, norms_tensor


class GradNormCallback(dde.callbacks.Callback):
    """Update DeepXDE loss weights with the algorithm used by JAX-PI."""

    def __init__(
        self,
        adapter: AdaptiveLossWeights,
        loss_names,
        momentum=0.9,
        update_every=1000,
        log_path=None,
    ):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        if update_every <= 0:
            raise ValueError("update_every must be positive")
        self.adapter = adapter
        self.loss_names = list(loss_names)
        self.momentum = float(momentum)
        self.update_every = int(update_every)
        self.log_path = log_path
        self.records = []

    def on_epoch_end(self):
        # JAX-PI performs its first update after optimizer step zero, then every
        # ``update_every`` steps. DeepXDE's public step counter starts at one.
        step = int(self.model.train_state.step)
        if (step - 1) % self.update_every != 0:
            return
        self.update(step)

    def update(self, step):
        inputs = self.model.train_state.X_train
        targets = self.model.train_state.y_train
        _, weighted_losses = self.model.outputs_losses_train(inputs, targets)
        old_weights = self.adapter.get_numpy()
        if len(weighted_losses) != len(old_weights):
            raise ValueError(
                f"Grad Norm received {len(weighted_losses)} losses for "
                f"{len(old_weights)} weights"
            )
        if len(self.loss_names) != len(old_weights):
            raise ValueError("loss_names must contain one name per loss weight")

        weight_tensor = torch.as_tensor(
            old_weights,
            dtype=weighted_losses.dtype,
            device=weighted_losses.device,
        )
        raw_losses = [
            weighted_losses[index] / weight_tensor[index]
            for index in range(len(old_weights))
        ]
        proposed, norms = grad_norm_weights(raw_losses, self.model.net.parameters())
        proposed_np = proposed.detach().cpu().numpy().astype(np.float32)
        updated = self.momentum * old_weights + (1.0 - self.momentum) * proposed_np
        self.adapter.set(updated)
        self.model.losshistory.set_loss_weights(updated.copy())

        record = {
            "step": int(step),
            "weights": {
                name: float(value) for name, value in zip(self.loss_names, updated)
            },
            "gradient_norms": {
                name: float(value)
                for name, value in zip(self.loss_names, norms.detach().cpu().numpy())
            },
        }
        self.records.append(record)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(record, sort_keys=True) + "\n")

