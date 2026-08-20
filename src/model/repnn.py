"""RepNN networks for physics-informed neural networks.

``RepNN`` implements the first-layer parameterization from
"RepNN: Tackling spectral bias in deep neural networks via parameter
reparameterization". ``RepNNRWF`` is an experimental combination of that
parameterization with the project's existing Random Weight Factorization.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as functional

from deepxde import config
from deepxde.nn.pytorch.nn import NN

from .fnn import FNN
from .rwf import RWFLinear, RWFMLP


def _validate_layer_sizes(layer_sizes: Sequence[int]) -> list[int]:
    sizes = [int(size) for size in layer_sizes]
    if len(sizes) < 3 or any(size <= 0 for size in sizes):
        raise ValueError(
            "layer_sizes must contain positive input, hidden, and output sizes"
        )
    return sizes


def _bounds_tensor(values: Any, size: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=config.real(torch)).flatten()
    if tensor.numel() != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


class RepNNFirstLayer(torch.nn.Module):
    """Dense first layer with a trainable shift for every weight.

    The layer computes ``sum_j weight_ij * (x_j + b_tilde_ij)`` and therefore
    deliberately has no independent bias parameter.
    """

    def __init__(self, in_features: int, out_features: int, nu_s: float = 10.0):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not math.isfinite(nu_s) or nu_s <= 0:
            raise ValueError("nu_s must be positive and finite")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        dtype = config.real(torch)
        self.weight = torch.nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=dtype)
        )
        self.b_tilde = torch.nn.Parameter(torch.empty_like(self.weight))
        torch.nn.init.normal_(self.weight, mean=0.0, std=float(nu_s))
        torch.nn.init.uniform_(self.b_tilde, -1.0, 1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        effective_bias = (self.weight * self.b_tilde).sum(dim=-1)
        return functional.linear(inputs, self.weight, effective_bias)


class RepNNRWFFirstLayer(torch.nn.Module):
    """RepNN first layer whose effective weight uses the existing RWF layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nu_s: float = 10.0,
        mu: float = 1.0,
        sigma: float = 0.1,
    ):
        super().__init__()
        if not math.isfinite(nu_s) or nu_s <= 0:
            raise ValueError("nu_s must be positive and finite")

        # RWFLinear owns the project's canonical exp(s) * V parameterization.
        # Rescale V once so its effective initial weight follows RepNN's N(0, nu_s^2).
        self.rwf = RWFLinear(
            in_features, out_features, bias=False, mu=mu, sigma=sigma
        )
        with torch.no_grad():
            initial_weight = torch.empty_like(self.rwf.V)
            torch.nn.init.normal_(initial_weight, mean=0.0, std=float(nu_s))
            self.rwf.V.copy_(initial_weight / torch.exp(self.rwf.s).unsqueeze(1))
        self.b_tilde = torch.nn.Parameter(torch.empty_like(self.rwf.V))
        torch.nn.init.uniform_(self.b_tilde, -1.0, 1.0)

    @property
    def weight(self) -> torch.Tensor:
        """Return the differentiable effective RWF weight."""
        return self.rwf.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        effective_weight = self.rwf.weight
        effective_bias = (effective_weight * self.b_tilde).sum(dim=-1)
        return functional.linear(inputs, effective_weight, effective_bias)


class _NormalizedRepNN(NN):
    """Shared coordinate normalization and DeepXDE transform behavior."""

    def _set_bounds(self, lb: Any, ub: Any, input_size: int) -> None:
        lower = _bounds_tensor(lb, input_size, "lb")
        upper = _bounds_tensor(ub, input_size, "ub")
        if not torch.all(upper > lower):
            raise ValueError("every ub coordinate must be greater than lb")
        self.register_buffer("lb", lower)
        self.register_buffer("ub", upper)

    def _normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._normalize(inputs)
        if self._input_transform is not None:
            x = self._input_transform(x)
        return x

    def _finish(self, original_inputs: torch.Tensor, outputs: torch.Tensor):
        if self._output_transform is not None:
            outputs = self._output_transform(original_inputs, outputs)
        return outputs


class RepNN(_NormalizedRepNN):
    """RepNN with one reparameterized first layer and ordinary later layers."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        lb: Any,
        ub: Any,
        nu_s: float = 10.0,
    ):
        super().__init__()
        sizes = _validate_layer_sizes(layer_sizes)
        self._set_bounds(lb, ub, sizes[0])
        self.first_layer = RepNNFirstLayer(sizes[0], sizes[1], nu_s=nu_s)
        self.hidden_layers = torch.nn.ModuleList(
            torch.nn.Linear(in_size, out_size, dtype=config.real(torch))
            for in_size, out_size in zip(sizes[1:-2], sizes[2:-1])
        )
        self.output_layer = torch.nn.Linear(
            sizes[-2], sizes[-1], dtype=config.real(torch)
        )

        for layer in self.hidden_layers:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.normal_(
            self.output_layer.weight, mean=0.0, std=sizes[-2] ** -0.5
        )
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_inputs = inputs
        x = torch.tanh(self.first_layer(self._prepare_inputs(inputs)))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return self._finish(original_inputs, self.output_layer(x))


class RepNNRWF(_NormalizedRepNN):
    """Experimental RepNN first-layer shifts combined with project RWF layers."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        lb: Any,
        ub: Any,
        nu_s: float = 10.0,
        mu: float = 1.0,
        sigma: float = 0.1,
    ):
        super().__init__()
        sizes = _validate_layer_sizes(layer_sizes)
        self._set_bounds(lb, ub, sizes[0])
        self.first_layer = RepNNRWFFirstLayer(
            sizes[0], sizes[1], nu_s=nu_s, mu=mu, sigma=sigma
        )
        self.hidden_layers = torch.nn.ModuleList(
            RWFLinear(in_size, out_size, mu=mu, sigma=sigma)
            for in_size, out_size in zip(sizes[1:-2], sizes[2:-1])
        )
        # RWFMLP applies RWF to its output layer as well; preserve that behavior.
        self.output_layer = RWFLinear(sizes[-2], sizes[-1], mu=mu, sigma=sigma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_inputs = inputs
        x = torch.tanh(self.first_layer(self._prepare_inputs(inputs)))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return self._finish(original_inputs, self.output_layer(x))


def build_network(
    config_dict: Mapping[str, Any],
    input_dim: int,
    output_dim: int,
    *,
    lb: Optional[Any] = None,
    ub: Optional[Any] = None,
) -> NN:
    """Build one of ``mlp``, ``rwf_mlp``, ``repnn``, or ``repnn_rwf``.

    ``config_dict`` may be the model mapping itself or contain it under ``model``.
    RepNN variants default to four hidden layers of width 300 and tanh activation.
    """
    model_config = config_dict.get("model", config_dict)
    if not isinstance(model_config, Mapping):
        raise TypeError("model config must be a mapping")
    model_type = str(model_config.get("type", "mlp")).lower().replace("-", "_")
    hidden_layers = int(model_config.get("hidden_layers", 4))
    hidden_width = int(model_config.get("hidden_width", 300))
    if hidden_layers <= 0 or hidden_width <= 0:
        raise ValueError("hidden_layers and hidden_width must be positive")
    layer_sizes = [int(input_dim)] + [hidden_width] * hidden_layers + [int(output_dim)]

    if model_type == "mlp":
        return FNN(
            layer_sizes,
            model_config.get("activation", "tanh"),
            model_config.get("kernel_initializer", "Glorot normal"),
        )
    if model_type == "rwf_mlp":
        return RWFMLP(
            layer_sizes,
            mu=float(model_config.get("rwf_mu", 1.0)),
            sigma=float(model_config.get("rwf_sigma", 0.1)),
        )
    if model_type not in {"repnn", "repnn_rwf"}:
        raise ValueError(f"unsupported model.type: {model_type!r}")

    lower = model_config.get("lb", lb)
    upper = model_config.get("ub", ub)
    if lower is None or upper is None:
        raise ValueError("RepNN models require lb and ub coordinate bounds")
    common = {
        "layer_sizes": layer_sizes,
        "lb": lower,
        "ub": upper,
        "nu_s": float(model_config.get("nu_s", 10.0)),
    }
    if model_type == "repnn":
        return RepNN(**common)
    return RepNNRWF(
        **common,
        mu=float(model_config.get("rwf_mu", 1.0)),
        sigma=float(model_config.get("rwf_sigma", 0.1)),
    )
