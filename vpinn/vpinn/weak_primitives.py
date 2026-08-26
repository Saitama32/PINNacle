"""Stateless quadrature and Legendre-bubble primitives for weak-form losses."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import torch

from .GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights
from .lengendre import legendre_bubble, legendre_bubble_derivative


@dataclass(frozen=True)
class SpatialQuadrature:
    """Tensor-product quadrature and test data on a uniform cell partition."""

    points: torch.Tensor
    weights: torch.Tensor
    test_values: torch.Tensor
    test_gradients: torch.Tensor
    cell_volumes: torch.Tensor
    spatial_dim: int

    @property
    def num_cells(self) -> int:
        return int(self.points.shape[0])

    @property
    def points_per_cell(self) -> int:
        return int(self.points.shape[1])

    @property
    def num_test_functions(self) -> int:
        return int(self.test_values.shape[0])


def gauss_lobatto_legendre(order: int, *, device, dtype):
    """Return Gauss-Lobatto-Legendre nodes and weights without global caches."""

    if int(order) < 2:
        raise ValueError("quadrature_order must be at least 2.")
    nodes, weights = GaussLobattoJacobiWeights(int(order), 0, 0)
    return (
        torch.as_tensor(nodes, device=device, dtype=dtype),
        torch.as_tensor(weights, device=device, dtype=dtype),
    )


def _normalize_cells(spatial_cells, spatial_dim: int) -> tuple[int, ...]:
    if isinstance(spatial_cells, int):
        cells = (int(spatial_cells),) * spatial_dim
    else:
        cells = tuple(int(value) for value in spatial_cells)
    if len(cells) != spatial_dim:
        raise ValueError(
            f"Expected {spatial_dim} spatial cell counts, received {len(cells)}."
        )
    if any(value <= 0 for value in cells):
        raise ValueError("All spatial cell counts must be positive.")
    return cells


def build_spatial_quadrature(
    bounds: Sequence[Sequence[float]],
    spatial_cells,
    quadrature_order: int,
    test_function_count: int,
    *,
    device,
    dtype,
) -> SpatialQuadrature:
    """Build physical quadrature and an H_0^1 Legendre basis for 1D or 2D space."""

    normalized_bounds = tuple((float(pair[0]), float(pair[1])) for pair in bounds)
    spatial_dim = len(normalized_bounds)
    if spatial_dim not in (1, 2):
        raise ValueError("Weak spatial quadrature currently supports 1D or 2D space.")
    if any(not lower < upper for lower, upper in normalized_bounds):
        raise ValueError("Every spatial bound must satisfy lower < upper.")
    if int(test_function_count) <= 0:
        raise ValueError("test_function_count must be positive.")

    cells = _normalize_cells(spatial_cells, spatial_dim)
    nodes_1d, weights_1d = gauss_lobatto_legendre(
        quadrature_order, device=device, dtype=dtype
    )

    node_mesh = torch.meshgrid(*([nodes_1d] * spatial_dim), indexing="ij")
    weight_mesh = torch.meshgrid(*([weights_1d] * spatial_dim), indexing="ij")
    reference_points = torch.stack(
        [component.reshape(-1) for component in node_mesh], dim=1
    )
    reference_weights = torch.ones_like(weight_mesh[0])
    for component in weight_mesh:
        reference_weights = reference_weights * component
    reference_weights = reference_weights.reshape(-1)

    cell_widths = torch.tensor(
        [(upper - lower) / count for (lower, upper), count in zip(normalized_bounds, cells)],
        device=device,
        dtype=dtype,
    )
    half_widths = 0.5 * cell_widths
    center_axes = [
        torch.linspace(
            lower + 0.5 * float(width),
            upper - 0.5 * float(width),
            count,
            device=device,
            dtype=dtype,
        )
        for (lower, upper), count, width in zip(normalized_bounds, cells, cell_widths)
    ]
    center_mesh = torch.meshgrid(*center_axes, indexing="ij")
    centers = torch.stack([component.reshape(-1) for component in center_mesh], dim=1)
    points = centers[:, None, :] + reference_points[None, :, :] * half_widths

    jacobian = torch.prod(half_widths)
    weights = reference_weights[None, :].expand(points.shape[0], -1) * jacobian
    cell_volume = torch.prod(cell_widths)
    cell_volumes = torch.ones(
        (points.shape[0],), device=device, dtype=dtype
    ) * cell_volume

    modes = range(1, int(test_function_count) + 1)
    mode_indices = list(itertools.product(modes, repeat=spatial_dim))
    test_values = []
    test_gradients = []
    for mode_index in mode_indices:
        values_by_dim = [
            legendre_bubble(mode, reference_points[:, dim])
            for dim, mode in enumerate(mode_index)
        ]
        derivatives_by_dim = [
            legendre_bubble_derivative(mode, reference_points[:, dim])
            for dim, mode in enumerate(mode_index)
        ]
        value = torch.ones_like(reference_weights)
        for component in values_by_dim:
            value = value * component
        gradient_components = []
        for derivative_dim in range(spatial_dim):
            component = derivatives_by_dim[derivative_dim]
            for dim, other_value in enumerate(values_by_dim):
                if dim != derivative_dim:
                    component = component * other_value
            component = component * (2.0 / cell_widths[derivative_dim])
            gradient_components.append(component)
        test_values.append(value)
        test_gradients.append(torch.stack(gradient_components, dim=-1))

    return SpatialQuadrature(
        points=points,
        weights=weights,
        test_values=torch.stack(test_values, dim=0),
        test_gradients=torch.stack(test_gradients, dim=0),
        cell_volumes=cell_volumes,
        spatial_dim=spatial_dim,
    )
