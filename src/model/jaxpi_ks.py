"""PyTorch network components used by the JAX-PI KS reproduction."""

from __future__ import annotations

import math

import torch

from deepxde import config
from deepxde.nn.pytorch.fnn import FNN
from deepxde.nn.pytorch.nn import NN

from .rwf import RWFLinear, RWFMLP


def _linear(
    in_features: int,
    out_features: int,
    *,
    use_rwf: bool = False,
    rwf_mu: float = 1.0,
    rwf_sigma: float = 0.1,
) -> torch.nn.Module:
    if use_rwf:
        return RWFLinear(
            in_features,
            out_features,
            mu=rwf_mu,
            sigma=rwf_sigma,
        )
    layer = torch.nn.Linear(
        in_features,
        out_features,
        dtype=config.real(torch),
    )
    torch.nn.init.xavier_normal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer


class JaxpiKSFeatures(torch.nn.Module):
    """Periodic KS coordinates with the optional trainable JAX-PI Fourier map.

    PINNacle represents a point as ``(x, t)``. JAX-PI feeds its network
    ``(t / window_length, cos(x), sin(x))`` and optionally projects those
    features through a trainable Gaussian Fourier matrix.  The periodic map can
    be disabled, in which case the coordinates are ``(x, t / window_length)``.
    """

    def __init__(
        self,
        time_scale: float,
        use_fourier: bool = True,
        periodic_encoding: bool = True,
        fourier_dim: int = 256,
        fourier_scale: float = 1.0,
    ):
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be positive and finite")
        if use_fourier and (fourier_dim <= 0 or fourier_dim % 2):
            raise ValueError("fourier_dim must be a positive even integer")
        if not math.isfinite(fourier_scale) or fourier_scale <= 0:
            raise ValueError("fourier_scale must be positive and finite")

        self.time_scale = float(time_scale)
        self.use_fourier = bool(use_fourier)
        self.periodic_encoding = bool(periodic_encoding)
        self.fourier_dim = int(fourier_dim)
        self.fourier_scale = float(fourier_scale)
        self.base_dim = 3 if self.periodic_encoding else 2
        if self.use_fourier:
            kernel = torch.empty(
                self.base_dim,
                self.fourier_dim // 2,
                dtype=config.real(torch),
            )
            torch.nn.init.normal_(kernel, mean=0.0, std=self.fourier_scale)
            self.kernel = torch.nn.Parameter(kernel)
        else:
            self.register_parameter("kernel", None)

    @property
    def out_dim(self) -> int:
        return self.fourier_dim if self.use_fourier else self.base_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("JaxpiKSFeatures expects a [N, 2] tensor ordered as (x, t)")
        x = inputs[:, 0:1]
        t = inputs[:, 1:2] / self.time_scale
        features = (
            torch.cat((t, torch.cos(x), torch.sin(x)), dim=1)
            if self.periodic_encoding
            else torch.cat((x, t), dim=1)
        )
        if not self.use_fourier:
            return features
        projected = features @ self.kernel
        return torch.cat((torch.cos(projected), torch.sin(projected)), dim=1)


class JaxpiKSNetwork(NN):
    """Plain or modified MLP matching the architecture used by JAX-PI."""

    def __init__(
        self,
        time_scale: float,
        num_layers: int = 4,
        hidden_dim: int = 256,
        modified_mlp: bool = True,
        fourier_features: bool = True,
        periodic_encoding: bool = True,
        fourier_dim: int = 256,
        fourier_scale: float = 1.0,
        use_rwf: bool = False,
        rwf_mu: float = 1.0,
        rwf_sigma: float = 0.1,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.modified_mlp = bool(modified_mlp)
        self.use_rwf = bool(use_rwf)
        self.features = JaxpiKSFeatures(
            time_scale=time_scale,
            use_fourier=fourier_features,
            periodic_encoding=periodic_encoding,
            fourier_dim=fourier_dim,
            fourier_scale=fourier_scale,
        )
        input_dim = self.features.out_dim
        linear_options = {
            "use_rwf": self.use_rwf,
            "rwf_mu": rwf_mu,
            "rwf_sigma": rwf_sigma,
        }
        self.hidden_layers = torch.nn.ModuleList()
        if self.modified_mlp:
            self.encoder_u = _linear(input_dim, hidden_dim, **linear_options)
            self.encoder_v = _linear(input_dim, hidden_dim, **linear_options)
            self.hidden_layers.append(_linear(input_dim, hidden_dim, **linear_options))
            for _ in range(1, num_layers):
                self.hidden_layers.append(_linear(hidden_dim, hidden_dim, **linear_options))
        else:
            self.register_module("encoder_u", None)
            self.register_module("encoder_v", None)
            self.hidden_layers.append(_linear(input_dim, hidden_dim, **linear_options))
            for _ in range(1, num_layers):
                self.hidden_layers.append(_linear(hidden_dim, hidden_dim, **linear_options))
        self.output_layer = _linear(hidden_dim, 1, **linear_options)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_inputs = inputs
        x = self.features(inputs)
        if self._input_transform is not None:
            x = self._input_transform(x)

        if self.modified_mlp:
            u = torch.tanh(self.encoder_u(x))
            v = torch.tanh(self.encoder_v(x))
            delta = u - v
            for layer in self.hidden_layers:
                gate = torch.tanh(layer(x))
                x = v + gate * delta
        else:
            for layer in self.hidden_layers:
                x = torch.tanh(layer(x))

        outputs = self.output_layer(x)
        if self._output_transform is not None:
            outputs = self._output_transform(original_inputs, outputs)
        return outputs


class PinnacleKSFNN(NN):
    """Standard DeepXDE FNN with the selectable KS input representation."""

    def __init__(
        self,
        time_scale: float,
        num_layers: int = 4,
        hidden_dim: int = 256,
        fourier_features: bool = False,
        periodic_encoding: bool = False,
        fourier_dim: int = 256,
        fourier_scale: float = 1.0,
        use_rwf: bool = False,
        rwf_mu: float = 1.0,
        rwf_sigma: float = 0.1,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.features = JaxpiKSFeatures(
            time_scale=time_scale,
            use_fourier=fourier_features,
            periodic_encoding=periodic_encoding,
            fourier_dim=fourier_dim,
            fourier_scale=fourier_scale,
        )
        layer_sizes = [self.features.out_dim, *([hidden_dim] * num_layers), 1]
        self.fnn = (
            RWFMLP(layer_sizes, mu=rwf_mu, sigma=rwf_sigma)
            if use_rwf
            else FNN(layer_sizes, "tanh", "Glorot normal")
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_inputs = inputs
        x = self.features(inputs)
        if self._input_transform is not None:
            x = self._input_transform(x)
        outputs = self.fnn(x)
        if self._output_transform is not None:
            outputs = self._output_transform(original_inputs, outputs)
        return outputs
