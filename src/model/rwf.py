"""Random Weight Factorization layers and networks for PyTorch PINNs."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from deepxde import config
from deepxde.nn.pytorch.nn import NN


class RWFLinear(torch.nn.Module):
    """Linear layer with ``W = diag(exp(s)) V`` parameterization.

    The effective initial weight is sampled with Xavier normal exactly once.
    ``V`` is then chosen to compensate the sampled scales, so factorization
    does not alter the baseline Glorot initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mu: float = 1.0,
        sigma: float = 0.1,
    ):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not math.isfinite(mu):
            raise ValueError("RWF mu must be finite")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("RWF sigma must be non-negative and finite")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        dtype = config.real(torch)
        initial_weight = torch.empty(self.out_features, self.in_features, dtype=dtype)
        torch.nn.init.xavier_normal_(initial_weight)
        initial_scale = torch.empty(self.out_features, dtype=dtype)
        torch.nn.init.normal_(initial_scale, mean=float(mu), std=float(sigma))

        self.s = torch.nn.Parameter(initial_scale)
        self.V = torch.nn.Parameter(initial_weight / torch.exp(initial_scale).unsqueeze(1))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Materialize the effective dense weight without a diagonal matrix."""
        return torch.exp(self.s).unsqueeze(1) * self.V

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional.linear(inputs, self.weight, self.bias)


class RWFMLP(NN):
    """Vanilla tanh MLP with RWF applied to every dense weight matrix."""

    def __init__(
        self,
        layer_sizes,
        mu: float = 1.0,
        sigma: float = 0.1,
    ):
        super().__init__()
        if len(layer_sizes) < 2 or any(int(size) <= 0 for size in layer_sizes):
            raise ValueError("layer_sizes must contain at least two positive sizes")
        self.linears = torch.nn.ModuleList(
            RWFLinear(in_size, out_size, mu=mu, sigma=sigma)
            for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_inputs = inputs
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for layer in self.linears[:-1]:
            x = torch.tanh(layer(x))
        outputs = self.linears[-1](x)
        if self._output_transform is not None:
            outputs = self._output_transform(original_inputs, outputs)
        return outputs
