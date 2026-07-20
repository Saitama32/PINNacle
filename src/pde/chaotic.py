import numpy as np
import deepxde as dde

from . import baseclass


def build_ks_terms(u, u_t, u_x, u_xx, u_xxxx, alpha, beta, gamma):
    raw_adv = u * u_x
    term_t = u_t
    term_adv = alpha * raw_adv
    term_diff = beta * u_xx
    term_hyper = gamma * u_xxxx
    residual = term_t + term_adv + term_diff + term_hyper
    return {
        "u": u,
        "u_t": u_t,
        "u_x": u_x,
        "u_xx": u_xx,
        "u_xxxx": u_xxxx,
        "raw_u_u_x": raw_adv,
        "term_t": term_t,
        "term_adv": term_adv,
        "term_diff": term_diff,
        "term_hyper": term_hyper,
        "residual": residual,
    }


class GrayScottEquation(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/grayscott.dat", bbox=[-1, 1, -1, 1, 0, 200], b=0.04, d=0.1, epsilon=(1e-5, 5e-6)):
        super().__init__()
        # output dim
        self.output_dim = 2

        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle((self.bbox[0], self.bbox[2]), (self.bbox[1], self.bbox[3]))
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # PDE
        def pde(x, y):
            u, v = y[:, 0:1], y[:, 1:2]

            u_t = dde.grad.jacobian(u, x, i=0, j=2)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)

            v_t = dde.grad.jacobian(v, x, i=0, j=2)
            v_xx = dde.grad.hessian(v, x, i=0, j=0)
            v_yy = dde.grad.hessian(v, x, i=1, j=1)
            return [u_t - (epsilon[0] * (u_xx + u_yy) + b * (1 - u) - u * (v**2)), v_t - (epsilon[1] * (v_xx + v_yy) - d * v + u * (v**2))]

        self.pde = pde
        self.set_pdeloss(num=2)

        self.load_ref_data(datapath, t_transpose=False)

        # BC
        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[2], bbox[4])

        def ic_func(x, component):
            if component == 0:
                return 1 - np.exp(-80 * ((x[:, 0] + 0.05)**2 + (x[:, 1] + 0.02)**2))
            else:
                return np.exp(-80 * ((x[:, 0] - 0.05)**2 + (x[:, 1] - 0.02)**2))

        self.add_bcs([{
            'component': 0,
            'function': (lambda x: ic_func(x, component=0)),
            'bc': boundary_ic,
            'type': 'ic'
        }, {
            'component': 1,
            'function': (lambda x: ic_func(x, component=1)),
            'bc': boundary_ic,
            'type': 'ic'
        }])

        self.training_points(mul=4)


class KuramotoSivashinskyEquation(baseclass.BaseTimePDE):

    def __init__(
        self,
        datapath="ref/Kuramoto_Sivashinsky.dat",
        bbox=[0, 2 * np.pi, 0, 1],
        alpha=100 / 16,
        beta=100 / (16 * 16),
        gamma=100 / (16**4),
    ):
        super().__init__()
        # output dim
        self.output_dim = 1
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Interval(bbox[0], bbox[1])
        timedomain = dde.geometry.TimeDomain(bbox[2], bbox[3])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # PDE
        def pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            spatial = self.spatial_terms(x, u)
            return u_t + spatial["g"]

        self.pde = pde
        self.set_pdeloss(num=1)

        self.load_ref_data(datapath, t_transpose=False)

        # BCs
        self.add_bcs([{
            'component': 0,
            'function': (lambda x: np.cos(x[:, 0:1]) * (1 + np.sin(x[:, 0:1]))),
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }])

        # training point
        self.training_points()

    def build_terms(self, u, u_t, u_x, u_xx, u_xxxx):
        return build_ks_terms(
            u=u,
            u_t=u_t,
            u_x=u_x,
            u_xx=u_xx,
            u_xxxx=u_xxxx,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

    def spatial_terms(self, x, u):
        u_x = dde.grad.jacobian(u, x, i=0, j=0)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        u_xxxx = dde.grad.hessian(u_xx, x, i=0, j=0)
        terms = self.build_terms(
            u=u,
            u_t=0.0,
            u_x=u_x,
            u_xx=u_xx,
            u_xxxx=u_xxxx,
        )
        return {
            "u_x": u_x,
            "u_xx": u_xx,
            "u_xxxx": u_xxxx,
            "raw_u_u_x": terms["raw_u_u_x"],
            "term_adv": terms["term_adv"],
            "term_diff": terms["term_diff"],
            "term_hyper": terms["term_hyper"],
            "g": terms["term_adv"] + terms["term_diff"] + terms["term_hyper"],
        }

    def ks_spatial_operator(self, x, u):
        return self.spatial_terms(x, u)["g"]
