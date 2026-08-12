import deepxde as dde
import numpy as np

from . import baseclass


class Convection1D(baseclass.BaseTimePDE):

    def __init__(self, beta=50, geom=(0, 2 * np.pi), time=(0, 1)):
        super().__init__()
        # output dim
        self.output_dim = 1
        # domain
        self.geom = dde.geometry.Interval(*geom)
        timedomain = dde.geometry.TimeDomain(*time)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = list(geom) + list(time)

        # PDE
        def convection_pde(x, u):
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            return u_t + beta * u_x

        self.pde = convection_pde
        self.set_pdeloss()

        def ref_sol(x):
            return np.sin(x[:, 0:1] - beta * x[:, 1:2])

        self.ref_sol = ref_sol

        # BCs
        def boundary_x(x, on_boundary):
            return on_boundary and (np.isclose(x[0], geom[0]) or np.isclose(x[0], geom[1]))

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'type': 'periodic',
            'component_x': 0,
            'bc': boundary_x,
        }])

        # train settings
        self.training_points()
