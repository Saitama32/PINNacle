import numpy as np
import torch
from scipy import interpolate

import deepxde as dde
from . import baseclass


def _wave_network_derivatives(net, inputs, spatial_dim):
    """Return first spatial derivatives and the second time derivative."""

    values = net(inputs)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("Wave weak-form adapters require a scalar network output.")
    first = torch.autograd.grad(
        values,
        inputs,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
    )[0]
    time_first = first[:, spatial_dim : spatial_dim + 1]
    if time_first.requires_grad:
        time_second = torch.autograd.grad(
            time_first,
            inputs,
            grad_outputs=torch.ones_like(time_first),
            create_graph=True,
        )[0][:, spatial_dim : spatial_dim + 1]
    else:
        # Exact constant/linear trial functions legitimately have a zero
        # second derivative and may produce a first derivative without a
        # grad_fn.  Keep a harmless network dependency for backward().
        time_second = torch.zeros_like(time_first) + 0.0 * values
    return first[:, :spatial_dim], time_second


class Wave1DWeakFormAdapter:
    """Spatial weak form of ``u_tt - C^2 u_xx = 0``."""

    def __init__(self, pde):
        self.pde = pde
        self.spatial_bounds = ((pde.bbox[0], pde.bbox[1]),)
        self.time_bounds = (pde.bbox[2], pde.bbox[3])

    def weak_residuals(self, net, quadrature, times):
        count_t = int(times.numel())
        count_c, count_q, _ = quadrature.points.shape
        spatial = quadrature.points[None, :, :, :].expand(count_t, -1, -1, -1)
        time_grid = times[:, None, None, None].expand(-1, count_c, count_q, 1)
        inputs = torch.cat((spatial, time_grid), dim=-1).reshape(-1, 2)
        inputs = inputs.requires_grad_(True)
        spatial_gradient, time_second = _wave_network_derivatives(net, inputs, 1)
        u_x = spatial_gradient[:, 0].reshape(count_t, count_c, count_q)
        u_tt = time_second[:, 0].reshape(count_t, count_c, count_q)

        mass = torch.einsum(
            "tcq,mq,cq->tcm",
            u_tt,
            quadrature.test_values,
            quadrature.weights,
        )
        stiffness = torch.einsum(
            "tcq,mq,cq->tcm",
            u_x,
            quadrature.test_gradients[:, :, 0],
            quadrature.weights,
        )
        return mass + float(self.pde.C) ** 2 * stiffness


class Wave2DHeterogeneousWeakFormAdapter:
    """Spatial weak form of ``Delta u - u_tt / a(x, y) = 0``."""

    def __init__(self, pde):
        self.pde = pde
        self.spatial_bounds = (
            (pde.bbox[0], pde.bbox[1]),
            (pde.bbox[2], pde.bbox[3]),
        )
        self.time_bounds = (pde.bbox[4], pde.bbox[5])
        self._coefficient_cache = {}

    def _coefficient(self, quadrature):
        points = quadrature.points
        key = (points.data_ptr(), str(points.device), str(points.dtype))
        coefficient = self._coefficient_cache.get(key)
        if coefficient is None:
            coefficient = self.pde.wave_coefficient(points.reshape(-1, 2)).reshape(
                points.shape[0], points.shape[1]
            )
            if not torch.isfinite(coefficient).all():
                raise RuntimeError("Wave coefficient is non-finite on weak quadrature points.")
            self._coefficient_cache[key] = coefficient
        return coefficient

    def weak_residuals(self, net, quadrature, times):
        count_t = int(times.numel())
        count_c, count_q, _ = quadrature.points.shape
        spatial = quadrature.points[None, :, :, :].expand(count_t, -1, -1, -1)
        time_grid = times[:, None, None, None].expand(-1, count_c, count_q, 1)
        inputs = torch.cat((spatial, time_grid), dim=-1).reshape(-1, 3)
        inputs = inputs.requires_grad_(True)
        spatial_gradient, time_second = _wave_network_derivatives(net, inputs, 2)
        spatial_gradient = spatial_gradient.reshape(count_t, count_c, count_q, 2)
        u_tt = time_second[:, 0].reshape(count_t, count_c, count_q)
        coefficient = self._coefficient(quadrature)

        stiffness = torch.einsum(
            "tcqd,mqd,cq->tcm",
            spatial_gradient,
            quadrature.test_gradients,
            quadrature.weights,
        )
        mass = torch.einsum(
            "tcq,mq,cq->tcm",
            u_tt / coefficient[None, :, :],
            quadrature.test_values,
            quadrature.weights,
        )
        return stiffness + mass


class Wave1D(baseclass.BasePDE):

    def __init__(self, C=2, bbox=[0, 1, 0, 1], scale=1, a=4):
        super().__init__()
        # output dim
        self.output_dim = 1
        self.C = float(C)
        # geom
        self.bbox = [0, scale, 0, scale]
        self.geom = dde.geometry.Rectangle(xmin=[self.bbox[0], self.bbox[2]], xmax=[self.bbox[1], self.bbox[3]])

        # define PDE
        def wave_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_tt = dde.grad.hessian(u, x, i=1, j=1)

            return u_tt - self.C**2 * u_xx

        self.pde = wave_pde
        self.set_pdeloss(num=1)

        def ref_sol(x):
            x = x / scale
            return (np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(a * np.pi * x[:, 0:1]) * np.cos(2 * a * np.pi * x[:, 1:2]))

        self.ref_sol = ref_sol

        def boundary_x0(x, on_boundary):
            return on_boundary and (np.isclose(x[0], self.bbox[0]) or np.isclose(x[0], self.bbox[1]))

        def boundary_t0(x, on_boundary):
            return on_boundary and np.isclose(x[1], self.bbox[2])

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'neumann'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': boundary_t0,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': boundary_x0,
            'type': 'dirichlet'
        }])

        # training config
        self.training_points()

    def weak_form_adapter(self):
        return Wave1DWeakFormAdapter(self)


class Wave2D_Heterogeneous(baseclass.BasePDE):

    def __init__(self, datapath="ref/wave_darcy.dat", bbox=[-1, 1, -1, 1, 0, 5], mu=(-0.5, 0), sigma=0.3):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        # NOTE: no circ are deleted, since the pde is currently not regraded as TimePDE and 3D-CSGDifference is difficult)
        self.bbox = bbox
        self.geom = dde.geometry.Hypercube(xmin=(self.bbox[0], self.bbox[2], self.bbox[4]), xmax=(self.bbox[1], self.bbox[3], self.bbox[5]))

        # PDE
        # self.darcy_2d_coef = generate_darcy_2d_coef(N_res=256, alpha=4, bbox=bbox[0:4])
        self.darcy_2d_coef = np.loadtxt("ref/darcy_2d_coef_256.dat")
        self._wave_coefficient_value_cache = {}

        def wave_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
            u_tt = dde.grad.hessian(u, x, i=2, j=2)

            return u_xx - u_tt / self.wave_coefficient(x)

        self.pde = wave_pde
        self.set_pdeloss(num=1)

        self.load_ref_data(datapath, t_transpose=True)

        # BCs
        def boundary_t0(x, on_initial):
            return np.isclose(x[2], bbox[4])

        def boundary_rec(x, on_boundary):
            return on_boundary and (np.isclose(x[0], bbox[0]) or np.isclose(x[0], bbox[1]) or np.isclose(x[1], bbox[2]) or np.isclose(x[1], bbox[3]))

        def initial_condition(x):
            return np.exp(-((x[:, 0:1] - mu[0])**2 + (x[:, 1:2] - mu[1])**2) / (2 * sigma**2))

        self.add_bcs([{
            'component': 0,
            'function': initial_condition,
            'bc': boundary_t0,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'neumann'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_rec,
            'type': 'neumann',
        }])

        # training config
        self.training_points(mul=4)

    def wave_coefficient(self, x):
        """Evaluate the fixed heterogeneous coefficient on a torch point set."""

        spatial = x.detach()[:, 0:2]
        cache = getattr(self, "_wave_coefficient_value_cache", None)
        if cache is None:
            cache = self._wave_coefficient_value_cache = {}
        cache_key = (
            tuple(spatial.shape),
            str(spatial.device),
            str(spatial.dtype),
            float(torch.sum(spatial).cpu()),
            float(torch.sum(spatial.square()).cpu()),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        query = (x.detach().cpu().numpy()[:, 0:2] + 1.0) / 2.0
        values = interpolate.griddata(
            self.darcy_2d_coef[:, 0:2],
            self.darcy_2d_coef[:, 2],
            query,
            method="linear",
        )
        missing = ~np.isfinite(values)
        if np.any(missing):
            values[missing] = interpolate.griddata(
                self.darcy_2d_coef[:, 0:2],
                self.darcy_2d_coef[:, 2],
                query[missing],
                method="nearest",
            )
        result = torch.as_tensor(values, device=x.device, dtype=x.dtype).unsqueeze(-1)
        cache[cache_key] = result
        return result

    def weak_form_adapter(self):
        return Wave2DHeterogeneousWeakFormAdapter(self)


class Wave2D_LongTime(baseclass.BaseTimePDE):

    def __init__(self, bbox=[0, 1, 0, 1, 0, 100], a=np.sqrt(2), m1=1, m2=3, n1=1, n2=2, p1=1, p2=1):
        super().__init__()

        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # pde
        INITIAL_COEF_1 = 1
        INITIAL_COEF_2 = 1

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_tt = dde.grad.hessian(u, x, i=2, j=2)

            return [u_tt - (u_xx + a * a * u_yy)]

        self.pde = pde
        self.set_pdeloss(num=1)

        # BCs
        def ref_sol(x):
            return (
                INITIAL_COEF_1 * np.sin(m1 * np.pi * x[:, 0:1]) * np.sinh(n1 * np.pi * x[:, 1:2]) * np.cos(p1 * np.pi * x[:, 2:3])
                + INITIAL_COEF_2 * np.sinh(m2 * np.pi * x[:, 0:1]) * np.sin(n2 * np.pi * x[:, 1:2]) * np.cos(p2 * np.pi * x[:, 2:3])
            )

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

        # training config
        self.training_points(mul=4)
