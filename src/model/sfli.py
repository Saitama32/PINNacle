"""Structured first-layer initialization utilities for vanilla RWF MLPs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from deepxde import config


DiagnosticValue = Union[int, float, torch.Tensor]


@dataclass(frozen=True)
class SFLIConfig:
    """Configuration shared by the supported SFLI first-layer variants.

    ``bounds`` must describe the coordinates seen by the network after any
    input normalization.
    """

    bounds: Sequence[Tuple[float, float]]
    gamma: Optional[float] = None
    C: float = 1.0
    seed: int = 0
    shift_min: float = 0.0
    shift_max: Optional[float] = None
    type: str = "tanh"

    def __post_init__(self) -> None:
        normalized_bounds = tuple((float(lo), float(hi)) for lo, hi in self.bounds)
        if not normalized_bounds:
            raise ValueError("SFLI bounds must contain at least one dimension")
        for lo, hi in normalized_bounds:
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError("SFLI bounds must be finite")
            if hi <= lo:
                raise ValueError("Each SFLI upper bound must exceed its lower bound")
        domain_volume = math.prod(hi - lo for lo, hi in normalized_bounds)
        if not math.isfinite(domain_volume) or domain_volume <= 0:
            raise ValueError("SFLI domain volume must be positive and finite")

        if self.gamma is not None and (
            not math.isfinite(float(self.gamma)) or float(self.gamma) <= 0
        ):
            raise ValueError("SFLI gamma must be positive and finite")
        if not math.isfinite(float(self.C)) or float(self.C) <= 0:
            raise ValueError("SFLI C must be positive and finite")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("SFLI seed must be a non-negative integer")
        if not math.isfinite(float(self.shift_min)) or float(self.shift_min) < 0:
            raise ValueError("SFLI shift_min must be non-negative and finite")
        if self.shift_max is not None:
            if not math.isfinite(float(self.shift_max)):
                raise ValueError("SFLI shift_max must be finite")
            if float(self.shift_max) <= float(self.shift_min):
                raise ValueError("SFLI shift_max must exceed shift_min")
        sfli_type = str(self.type).lower()
        if sfli_type not in {"tanh", "cosine", "gaussian"}:
            raise ValueError("SFLI type must be one of: tanh, cosine, gaussian")

        object.__setattr__(self, "bounds", normalized_bounds)
        if self.gamma is not None:
            object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "C", float(self.C))
        object.__setattr__(self, "shift_min", float(self.shift_min))
        if self.shift_max is not None:
            object.__setattr__(self, "shift_max", float(self.shift_max))
        object.__setattr__(self, "type", sfli_type)

    @property
    def domain_volume(self) -> float:
        return math.prod(hi - lo for lo, hi in self.bounds)

    @property
    def characteristic_length(self) -> float:
        return self.domain_volume ** (1.0 / len(self.bounds))


@dataclass(frozen=True)
class TanhSFLIInitialization:
    """Generated SFLI values, exposed for verification and diagnostics."""

    alpha: torch.Tensor
    shifts: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    gamma: float
    gamma_mode: str
    shift_interval: Tuple[float, float]
    domain_volume: float


# Keep the original tanh-only public name as a compatibility alias.
TanhSFLIConfig = SFLIConfig


@dataclass(frozen=True)
class CosineSFLIInitialization:
    """Generated initialization values for the cosine RWF first layer."""

    alpha: torch.Tensor
    shifts: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    gamma: float
    gamma_mode: str
    shift_interval: Tuple[float, float]
    domain_volume: float


@dataclass(frozen=True)
class GaussianSFLIInitialization:
    """Generated centers and metadata for a Gaussian first-feature layer."""

    centers: torch.Tensor
    gamma: float
    gamma_mode: str
    domain_volume: float


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _validate_generation_inputs(
    width: int,
    input_dim: int,
    sfli: SFLIConfig,
    expected_type: str,
) -> None:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("SFLI width must be a positive integer")
    if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
        raise ValueError("SFLI input_dim must be a positive integer")
    if not isinstance(sfli, SFLIConfig):
        raise TypeError("sfli must be an SFLIConfig instance")
    if sfli.type != expected_type:
        raise ValueError(f"Expected SFLI type '{expected_type}', got '{sfli.type}'")
    if len(sfli.bounds) != input_dim:
        raise ValueError(
            "SFLI bounds dimensionality must match the first layer input dimension"
        )


def _resolve_device_dtype(
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
) -> Tuple[torch.device, torch.dtype]:
    resolved_device = torch.empty(0).device if device is None else torch.device(device)
    resolved_dtype = config.real(torch) if dtype is None else dtype
    if not resolved_dtype.is_floating_point:
        raise ValueError("SFLI dtype must be floating point")
    return resolved_device, resolved_dtype


def _resolve_gamma(width: int, input_dim: int, sfli: SFLIConfig) -> Tuple[float, str]:
    if sfli.gamma is not None:
        return sfli.gamma, "explicit"
    gamma = sfli.C * (
        width ** (1.0 / input_dim) - 1.0
    ) / sfli.characteristic_length
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError(
            "The SFLI gamma formula must produce a positive finite value; "
            "use width > 1 or provide an explicit gamma"
        )
    return gamma, "formula"


def _sample_shifts(
    width: int,
    sfli: SFLIConfig,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    *,
    use_configured_min_for_default: bool,
) -> Tuple[torch.Tensor, Tuple[float, float]]:
    if sfli.shift_max is None:
        shift_interval = (
            sfli.shift_min if use_configured_min_for_default else 0.0,
            sfli.characteristic_length,
        )
    else:
        shift_interval = (sfli.shift_min, sfli.shift_max)
    if shift_interval[1] <= shift_interval[0]:
        raise ValueError("SFLI shift interval must have positive length")

    # The SFLI paper specifies c_i >= 0 sampled uniformly from an interval
    # associated with the input domain, but does not provide a unique
    # multidimensional formula.  |Omega|^(1/d) is the configurable default.
    shifts = torch.empty(width, device=device, dtype=dtype)
    shifts.uniform_(shift_interval[0], shift_interval[1], generator=generator)
    return shifts, shift_interval


def generate_tanh_sfli(
    width: int,
    input_dim: int,
    sfli: TanhSFLIConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> TanhSFLIInitialization:
    """Generate the target first-layer weight and bias for tanh SFLI."""

    _validate_generation_inputs(width, input_dim, sfli, "tanh")
    resolved_device, resolved_dtype = _resolve_device_dtype(device, dtype)
    gamma, gamma_mode = _resolve_gamma(width, input_dim, sfli)

    generator = _generator_for(resolved_device, sfli.seed)
    alpha = torch.randn(
        width,
        input_dim,
        generator=generator,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    alpha = alpha / alpha.norm(dim=1, keepdim=True).clamp_min(1e-12)

    shifts, shift_interval = _sample_shifts(
        width,
        sfli,
        generator,
        resolved_device,
        resolved_dtype,
        use_configured_min_for_default=False,
    )

    weight = alpha * gamma
    bias = shifts * gamma
    return TanhSFLIInitialization(
        alpha=alpha,
        shifts=shifts,
        weight=weight,
        bias=bias,
        gamma=gamma,
        gamma_mode=gamma_mode,
        shift_interval=shift_interval,
        domain_volume=sfli.domain_volume,
    )


def generate_cosine_sfli(
    width: int,
    input_dim: int,
    sfli: SFLIConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> CosineSFLIInitialization:
    """Generate the unnormalized Gaussian directions used by Cosine-SFLI."""

    _validate_generation_inputs(width, input_dim, sfli, "cosine")
    resolved_device, resolved_dtype = _resolve_device_dtype(device, dtype)
    gamma, gamma_mode = _resolve_gamma(width, input_dim, sfli)
    generator = _generator_for(resolved_device, sfli.seed)
    alpha = torch.randn(
        width,
        input_dim,
        generator=generator,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    # Paper baseline: alpha ~ N(0, I_d).  Do not normalize row norms.
    shifts, shift_interval = _sample_shifts(
        width,
        sfli,
        generator,
        resolved_device,
        resolved_dtype,
        use_configured_min_for_default=True,
    )
    return CosineSFLIInitialization(
        alpha=alpha,
        shifts=shifts,
        weight=gamma * alpha,
        bias=gamma * shifts,
        gamma=gamma,
        gamma_mode=gamma_mode,
        shift_interval=shift_interval,
        domain_volume=sfli.domain_volume,
    )


def generate_gaussian_sfli(
    width: int,
    input_dim: int,
    sfli: SFLIConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> GaussianSFLIInitialization:
    """Sample trainable Gaussian centers uniformly and independently in a box."""

    _validate_generation_inputs(width, input_dim, sfli, "gaussian")
    resolved_device, resolved_dtype = _resolve_device_dtype(device, dtype)
    gamma, gamma_mode = _resolve_gamma(width, input_dim, sfli)
    generator = _generator_for(resolved_device, sfli.seed)
    centers = torch.empty(
        width,
        input_dim,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    for dimension, (lo, hi) in enumerate(sfli.bounds):
        centers[:, dimension].uniform_(lo, hi, generator=generator)
    return GaussianSFLIInitialization(
        centers=centers,
        gamma=gamma,
        gamma_mode=gamma_mode,
        domain_volume=sfli.domain_volume,
    )


def apply_tanh_sfli(layer, sfli: TanhSFLIConfig) -> TanhSFLIInitialization:
    """Apply tanh SFLI to an initialized RWF layer without resampling ``s``."""

    if not isinstance(sfli, TanhSFLIConfig):
        raise TypeError("sfli must be a TanhSFLIConfig instance")
    required_attributes = ("in_features", "out_features", "s", "V", "bias")
    if any(not hasattr(layer, attribute) for attribute in required_attributes):
        raise TypeError("SFLI target must be an RWF-compatible linear layer")
    if layer.bias is None:
        raise ValueError("Tanh SFLI requires a first-layer bias")

    initialization = generate_tanh_sfli(
        layer.out_features,
        layer.in_features,
        sfli,
        device=layer.V.device,
        dtype=layer.V.dtype,
    )
    with torch.no_grad():
        scale = torch.exp(layer.s).unsqueeze(1)
        layer.V.copy_(initialization.weight / scale)
        layer.bias.copy_(initialization.bias)
    return initialization


def apply_cosine_sfli(layer, sfli: SFLIConfig) -> CosineSFLIInitialization:
    """Apply Cosine-SFLI to an initialized RWF layer without resampling ``s``."""

    if not isinstance(sfli, SFLIConfig):
        raise TypeError("sfli must be an SFLIConfig instance")
    required_attributes = ("in_features", "out_features", "s", "V", "bias")
    if any(not hasattr(layer, attribute) for attribute in required_attributes):
        raise TypeError("SFLI target must be an RWF-compatible linear layer")
    if layer.bias is None:
        raise ValueError("Cosine-SFLI requires a first-layer bias")

    initialization = generate_cosine_sfli(
        layer.out_features,
        layer.in_features,
        sfli,
        device=layer.V.device,
        dtype=layer.V.dtype,
    )
    with torch.no_grad():
        layer.V.copy_(initialization.weight / torch.exp(layer.s).unsqueeze(1))
        layer.bias.copy_(initialization.bias)
    return initialization


class SFLIGaussianFirstLayer(torch.nn.Module):
    """Trainable radial Gaussian first-feature layer."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        sfli: SFLIConfig,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        initialization = generate_gaussian_sfli(
            width,
            input_dim,
            sfli,
            device=device,
            dtype=dtype,
        )
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.centers = torch.nn.Parameter(initialization.centers)
        self.gamma = float(initialization.gamma)
        self.gamma_mode = initialization.gamma_mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"Gaussian SFLI expects a [N, {self.input_dim}] input tensor"
            )
        diff = inputs[:, None, :] - self.centers[None, :, :]
        squared_distance = (diff * diff).sum(dim=-1)
        return torch.exp(-(self.gamma**2) * squared_distance)


def initial_feature_diagnostics(
    features: torch.Tensor,
    epsilon: float = 1.0e-6,
    feature_type: Optional[str] = None,
) -> Dict[str, DiagnosticValue]:
    """Compute the initial Gram spectrum and epsilon-rank of a feature matrix."""

    if features.ndim != 2:
        raise ValueError("Initial feature matrix must have shape [N, width]")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Initial feature matrix dimensions must be non-zero")
    if not features.dtype.is_floating_point:
        raise ValueError("Initial feature matrix must be floating point")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("Diagnostic epsilon must be positive and finite")
    if feature_type is not None:
        feature_type = str(feature_type).lower()
        if feature_type not in {"tanh", "cosine", "gaussian"}:
            raise ValueError("Diagnostic feature_type is not supported")

    detached = features.detach()
    gram = detached.transpose(0, 1) @ detached / detached.shape[0]
    eigenvalues = torch.linalg.eigvalsh(gram)
    singular_values = torch.linalg.svdvals(detached)
    retained = eigenvalues > float(epsilon)
    rank = int(retained.sum().item())
    width = detached.shape[1]
    lambda_max = float(eigenvalues[-1].item())
    if rank:
        lambda_min_positive = float(eigenvalues[retained][0].item())
        condition_number = lambda_max / lambda_min_positive
    else:
        lambda_min_positive = 0.0
        condition_number = math.inf

    diagnostics = {
        "initial_feature_rank": rank,
        "initial_rank_fraction": rank / width,
        "initial_lambda_max": lambda_max,
        "initial_lambda_min_positive": lambda_min_positive,
        "initial_condition_number": condition_number,
        "initial_feature_singular_values": singular_values,
        "initial_gram_eigenvalues": eigenvalues,
        "feature_mean": float(detached.mean().item()),
        "feature_std": float(detached.std(unbiased=False).item()),
    }
    if detached.shape[1] > 1:
        centered = detached - detached.mean(dim=0, keepdim=True)
        norms = torch.linalg.vector_norm(centered, dim=0).clamp_min(1e-12)
        correlations = (centered.transpose(0, 1) @ centered) / (
            norms[:, None] * norms[None, :]
        )
        off_diagonal = ~torch.eye(
            detached.shape[1], device=detached.device, dtype=torch.bool
        )
        average_correlation = float(correlations[off_diagonal].mean().item())
    else:
        average_correlation = 0.0
    diagnostics["average_pairwise_feature_correlation"] = average_correlation

    if feature_type == "gaussian":
        diagnostics["gaussian_feature_mean"] = diagnostics["feature_mean"]
        diagnostics["gaussian_feature_std"] = diagnostics["feature_std"]
        diagnostics["gaussian_dead_fraction"] = float(
            (detached < 1.0e-8).to(detached.dtype).mean().item()
        )
        diagnostics["gaussian_peak_fraction"] = float(
            (detached > 0.99).to(detached.dtype).mean().item()
        )
    return diagnostics
