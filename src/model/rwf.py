"""Random Weight Factorization layers and networks for PyTorch PINNs."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as functional

from deepxde import config
from deepxde.nn.pytorch.nn import NN

from .sfli import (
    SFLIConfig,
    SFLIGaussianFirstLayer,
    apply_cosine_sfli,
    apply_tanh_sfli,
)


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
    """Vanilla RWF MLP with an optional structured first feature layer."""

    def __init__(
        self,
        layer_sizes,
        mu: float = 1.0,
        sigma: float = 0.1,
        sfli: Optional[SFLIConfig] = None,
    ):
        super().__init__()
        if len(layer_sizes) < 2 or any(int(size) <= 0 for size in layer_sizes):
            raise ValueError("layer_sizes must contain at least two positive sizes")
        if sfli is not None and not isinstance(sfli, SFLIConfig):
            raise TypeError("sfli must be an SFLIConfig instance or None")
        if sfli is not None and len(layer_sizes) < 3:
            raise ValueError("SFLI requires at least one hidden layer")
        all_linears = [
            RWFLinear(in_size, out_size, mu=mu, sigma=sigma)
            for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.sfli_type = "baseline" if sfli is None else sfli.type
        self.sfli_variant_class = {
            "baseline": "rwf_tanh_baseline",
            "tanh": "rwf_first_layer_tanh_initialization",
            "cosine": "rwf_first_layer_with_cos_activation",
            "gaussian": "radial_first_feature_layer",
        }[self.sfli_type]
        self.first_activation = {
            "baseline": "tanh",
            "tanh": "tanh",
            "cosine": "cos",
            "gaussian": "radial_gaussian",
        }[self.sfli_type]
        self.subsequent_activation = "tanh"
        if self.sfli_type == "gaussian":
            first_linear = all_linears[0]
            self.first_feature_layer = SFLIGaussianFirstLayer(
                int(layer_sizes[0]),
                int(layer_sizes[1]),
                sfli,
                device=first_linear.V.device,
                dtype=first_linear.V.dtype,
            )
            self.linears = torch.nn.ModuleList(all_linears[1:])
        else:
            self.register_module("first_feature_layer", None)
            self.linears = torch.nn.ModuleList(all_linears)
        if self.sfli_type == "tanh":
            apply_tanh_sfli(self.linears[0], sfli)
        elif self.sfli_type == "cosine":
            apply_cosine_sfli(self.linears[0], sfli)

    def first_layer_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return outputs of the baseline or SFLI first feature layer."""
        if self.sfli_type != "gaussian" and len(self.linears) < 2:
            raise ValueError("First-layer features require at least one hidden layer")
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        if self.sfli_type == "gaussian":
            return self.first_feature_layer(x)
        first_preactivation = self.linears[0](x)
        if self.sfli_type == "cosine":
            return torch.cos(first_preactivation)
        return torch.tanh(first_preactivation)

    def total_trainable_parameters(self) -> int:
        """Return the trainable parameter count for experiment logging."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def first_layer_trainable_parameters(self) -> int:
        """Return the parameter count of the first feature-producing layer."""
        layer = (
            self.first_feature_layer
            if self.sfli_type == "gaussian"
            else self.linears[0]
        )
        return sum(
            parameter.numel()
            for parameter in layer.parameters()
            if parameter.requires_grad
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_inputs = inputs
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        if self.sfli_type == "baseline" and len(self.linears) == 1:
            outputs = self.linears[0](x)
        elif self.sfli_type == "gaussian":
            x = self.first_feature_layer(x)
            for layer in self.linears[:-1]:
                x = torch.tanh(layer(x))
            outputs = self.linears[-1](x)
        else:
            first_preactivation = self.linears[0](x)
            x = (
                torch.cos(first_preactivation)
                if self.sfli_type == "cosine"
                else torch.tanh(first_preactivation)
            )
            for layer in self.linears[1:-1]:
                x = torch.tanh(layer(x))
            outputs = self.linears[-1](x)
        if self._output_transform is not None:
            outputs = self._output_transform(original_inputs, outputs)
        return outputs
