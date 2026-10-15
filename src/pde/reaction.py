from functools import lru_cache

import deepxde as dde
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import baseclass


def gaussian_initial_condition(x):
    """Initial condition used by the canonical PINN failure-mode benchmark."""
    x_coord = x[:, 0:1]
    return np.exp(-((x_coord - np.pi) ** 2) / (2 * (np.pi / 4) ** 2))


def _logistic_step(u, rho, dt):
    """Exact flow map for du/dt = rho*u*(1-u)."""
    return u / (u + (1.0 - u) * np.exp(-rho * dt))


@lru_cache(maxsize=8)
def _reaction_diffusion_reference(nu, rho, x_points=256, t_points=101, steps_per_output=100):
    """Generate a periodic reference solution with second-order Strang splitting."""
    x_grid = np.linspace(0.0, 2 * np.pi, x_points, endpoint=False)
    t_grid = np.linspace(0.0, 1.0, t_points)
    dt = (t_grid[1] - t_grid[0]) / steps_per_output

    u = gaussian_initial_condition(x_grid[:, None])[:, 0]
    solution = np.empty((t_points, x_points), dtype=np.float64)
    solution[0] = u

    wave_numbers = np.fft.fftfreq(x_points, d=1.0 / x_points)
    diffusion_multiplier = np.exp(-nu * wave_numbers**2 * dt)

    for time_index in range(1, t_points):
        for _ in range(steps_per_output):
            u = _logistic_step(u, rho, 0.5 * dt)
            u = np.fft.ifft(np.fft.fft(u) * diffusion_multiplier).real
            u = _logistic_step(u, rho, 0.5 * dt)
        solution[time_index] = u

    # Duplicate the periodic endpoint so interpolation is continuous at x=2*pi.
    periodic_x = np.append(x_grid, 2 * np.pi)
    periodic_solution = np.concatenate((solution, solution[:, :1]), axis=1)
    return RegularGridInterpolator(
        (t_grid, periodic_x),
        periodic_solution,
        method="linear",
        bounds_error=True,
    )


class _ReactionBase(baseclass.BaseTimePDE):

    def __init__(self, rho, geom, time):
        super().__init__()
        if rho <= 0:
            raise ValueError("rho must be positive")
        if geom[0] >= geom[1] or time[0] >= time[1]:
            raise ValueError("geometry and time intervals must be increasing")

        self.rho = rho
        self.output_dim = 1
        self.geom = dde.geometry.Interval(*geom)
        timedomain = dde.geometry.TimeDomain(*time)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = list(geom) + list(time)

    def _add_initial_and_periodic_conditions(self, include_derivative=False):
        x_min, x_max = self.bbox[:2]

        def boundary_x(x, on_boundary):
            return on_boundary and (np.isclose(x[0], x_min) or np.isclose(x[0], x_max))

        conditions = [{
            'component': 0,
            'function': gaussian_initial_condition,
            'bc': lambda _, on_initial: on_initial,
            'type': 'ic',
        }, {
            'component': 0,
            'type': 'periodic',
            'component_x': 0,
            'bc': boundary_x,
        }]
        if include_derivative:
            conditions.append({
                'component': 0,
                'type': 'periodic',
                'component_x': 0,
                'derivative_order': 1,
                'bc': boundary_x,
            })
        self.add_bcs(conditions)
        self.training_points()


class Reaction1D(_ReactionBase):
    """Logistic reaction equation u_t - rho*u*(1-u) = 0."""

    def __init__(self, rho=10.0, geom=(0.0, 2 * np.pi), time=(0.0, 1.0)):
        super().__init__(rho, geom, time)

        def reaction_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            return u_t - rho * u * (1.0 - u)

        self.pde = reaction_pde
        self.set_pdeloss()

        def ref_sol(x):
            initial = gaussian_initial_condition(x)
            elapsed = x[:, 1:2] - time[0]
            return initial / (initial + (1.0 - initial) * np.exp(-rho * elapsed))

        self.ref_sol = ref_sol
        self._add_initial_and_periodic_conditions()


class ReactionDiffusion1D(_ReactionBase):
    """Fisher reaction-diffusion equation on a periodic one-dimensional domain."""

    def __init__(self, nu=5.0, rho=5.0, geom=(0.0, 2 * np.pi), time=(0.0, 1.0)):
        if nu <= 0:
            raise ValueError("nu must be positive")
        if tuple(geom) != (0.0, 2 * np.pi) or tuple(time) != (0.0, 1.0):
            raise ValueError("the built-in reference solution requires geom=(0, 2*pi) and time=(0, 1)")
        super().__init__(rho, geom, time)
        self.nu = nu

        def reaction_diffusion_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_t - nu * u_xx - rho * u * (1.0 - u)

        self.pde = reaction_diffusion_pde
        self.set_pdeloss()

        interpolator = _reaction_diffusion_reference(float(nu), float(rho))

        def ref_sol(x):
            query = np.column_stack((x[:, 1], np.mod(x[:, 0], 2 * np.pi)))
            return interpolator(query)[:, None]

        self.ref_sol = ref_sol
        self._add_initial_and_periodic_conditions(include_derivative=True)
