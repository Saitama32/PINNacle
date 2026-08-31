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


def build_factorized_ks_terms(
    u, q, u_t, u_x, u_xx, q_xx, alpha, beta, gamma
):
    """Build the two residuals of the second-order factorized KS system."""

    raw_adv = u * u_x
    compatibility = q - beta * u - gamma * u_xx
    dynamics = u_t + alpha * raw_adv + q_xx
    return {
        "u": u,
        "q": q,
        "u_t": u_t,
        "u_x": u_x,
        "u_xx": u_xx,
        "q_xx": q_xx,
        "raw_u_u_x": raw_adv,
        "term_adv": alpha * raw_adv,
        "compatibility": compatibility,
        "dynamics": dynamics,
    }


def build_factorized_ks_reference(data, beta, gamma):
    """Append spectral ``q = beta*u + gamma*u_xx`` to rectangular KS data."""

    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Factorized KS reference data require x, t, and u columns")
    x = np.unique(data[:, 0])
    t = np.unique(data[:, 1])
    if len(data) != len(x) * len(t):
        raise ValueError("Factorized KS reference data must form a rectangular grid")
    x_indices = np.searchsorted(x, data[:, 0])
    t_indices = np.searchsorted(t, data[:, 1])
    u_grid = np.empty((len(t), len(x)), dtype=np.float64)
    occupancy = np.zeros_like(u_grid, dtype=np.uint8)
    u_grid[t_indices, x_indices] = data[:, 2]
    np.add.at(occupancy, (t_indices, x_indices), 1)
    if not np.all(occupancy == 1):
        raise ValueError("Factorized KS reference grid has duplicates or missing points")

    duplicate_endpoint = (
        len(x) > 2
        and np.allclose(u_grid[:, 0], u_grid[:, -1], rtol=1e-5, atol=1e-7)
    )
    spectral_x = x[:-1] if duplicate_endpoint else x
    spectral_u = u_grid[:, :-1] if duplicate_endpoint else u_grid
    if len(spectral_x) < 2:
        raise ValueError("Factorized KS reference requires at least two spatial points")
    spacing = np.diff(spectral_x)
    spacing_rtol = 5e-5 if data.dtype == np.float32 else 1e-8
    spacing_atol = 1e-7 if data.dtype == np.float32 else 1e-12
    if not np.allclose(
        spacing, spacing[0], rtol=spacing_rtol, atol=spacing_atol
    ):
        raise ValueError("Factorized KS spectral reference requires a uniform x grid")

    modes = 2.0 * np.pi * np.fft.fftfreq(
        len(spectral_x), d=float(spacing[0])
    )
    coefficients = np.fft.fft(spectral_u, axis=1)
    u_xx = np.fft.ifft(-(modes**2) * coefficients, axis=1).real
    q_grid = beta * spectral_u + gamma * u_xx
    if duplicate_endpoint:
        q_grid = np.c_[q_grid, q_grid[:, 0]]
    q = q_grid[t_indices, x_indices].astype(data.dtype, copy=False)
    return np.column_stack((data[:, :3], q))


def build_conservative_ks_terms(
    u, flux, u_t, u_x, u_xxx, flux_x, alpha, beta, gamma
):
    """Build conservation and flux-compatibility residuals for KS."""

    expected_flux = 0.5 * alpha * u**2 + beta * u_x + gamma * u_xxx
    conservation = u_t + flux_x
    flux_compatibility = flux - expected_flux
    return {
        "u": u,
        "flux": flux,
        "u_t": u_t,
        "u_x": u_x,
        "u_xxx": u_xxx,
        "flux_x": flux_x,
        "expected_flux": expected_flux,
        "conservation": conservation,
        "flux_compatibility": flux_compatibility,
    }


def build_conservative_ks_reference(data, alpha, beta, gamma):
    """Append spectral KS flux to rectangular periodic ``x, t, u`` data."""

    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Conservative KS reference data require x, t, and u columns")
    x = np.unique(data[:, 0])
    t = np.unique(data[:, 1])
    if len(data) != len(x) * len(t):
        raise ValueError("Conservative KS reference data must form a rectangular grid")
    x_indices = np.searchsorted(x, data[:, 0])
    t_indices = np.searchsorted(t, data[:, 1])
    u_grid = np.empty((len(t), len(x)), dtype=np.float64)
    occupancy = np.zeros_like(u_grid, dtype=np.uint8)
    u_grid[t_indices, x_indices] = data[:, 2]
    np.add.at(occupancy, (t_indices, x_indices), 1)
    if not np.all(occupancy == 1):
        raise ValueError("Conservative KS reference grid has duplicates or missing points")

    duplicate_endpoint = (
        len(x) > 2
        and np.allclose(u_grid[:, 0], u_grid[:, -1], rtol=1e-5, atol=1e-7)
    )
    spectral_x = x[:-1] if duplicate_endpoint else x
    spectral_u = u_grid[:, :-1] if duplicate_endpoint else u_grid
    if len(spectral_x) < 2:
        raise ValueError("Conservative KS reference requires at least two spatial points")
    spacing = np.diff(spectral_x)
    spacing_rtol = 5e-5 if data.dtype == np.float32 else 1e-8
    spacing_atol = 1e-7 if data.dtype == np.float32 else 1e-12
    if not np.allclose(
        spacing, spacing[0], rtol=spacing_rtol, atol=spacing_atol
    ):
        raise ValueError("Conservative KS spectral reference requires a uniform x grid")

    modes = 2.0 * np.pi * np.fft.fftfreq(
        len(spectral_x), d=float(spacing[0])
    )
    coefficients = np.fft.fft(spectral_u, axis=1)
    u_x = np.fft.ifft((1j * modes) * coefficients, axis=1).real
    u_xxx = np.fft.ifft((1j * modes) ** 3 * coefficients, axis=1).real
    flux_grid = 0.5 * alpha * spectral_u**2 + beta * u_x + gamma * u_xxx
    if duplicate_endpoint:
        flux_grid = np.c_[flux_grid, flux_grid[:, 0]]
    flux = flux_grid[t_indices, x_indices].astype(data.dtype, copy=False)
    return np.column_stack((data[:, :3], flux))


def build_first_order_ks_terms(
    u, p, q, r, u_t, u_x, p_x, q_x, r_x, alpha, beta, gamma
):
    """Build the four residuals of the fully first-order KS system."""

    return {
        "u": u,
        "p": p,
        "q": q,
        "r": r,
        "u_t": u_t,
        "u_x": u_x,
        "p_x": p_x,
        "q_x": q_x,
        "r_x": r_x,
        "p_compatibility": p - u_x,
        "q_compatibility": q - p_x,
        "r_compatibility": r - q_x,
        "dynamics": u_t + alpha * u * p + beta * q + gamma * r_x,
    }


def build_first_order_ks_reference(data):
    """Append spectral ``p=u_x``, ``q=u_xx``, and ``r=u_xxx`` to KS data."""

    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("First-order KS reference data require x, t, and u columns")
    x = np.unique(data[:, 0])
    t = np.unique(data[:, 1])
    if len(data) != len(x) * len(t):
        raise ValueError("First-order KS reference data must form a rectangular grid")
    x_indices = np.searchsorted(x, data[:, 0])
    t_indices = np.searchsorted(t, data[:, 1])
    u_grid = np.empty((len(t), len(x)), dtype=np.float64)
    occupancy = np.zeros_like(u_grid, dtype=np.uint8)
    u_grid[t_indices, x_indices] = data[:, 2]
    np.add.at(occupancy, (t_indices, x_indices), 1)
    if not np.all(occupancy == 1):
        raise ValueError("First-order KS reference grid has duplicates or missing points")

    duplicate_endpoint = (
        len(x) > 2
        and np.allclose(u_grid[:, 0], u_grid[:, -1], rtol=1e-5, atol=1e-7)
    )
    spectral_x = x[:-1] if duplicate_endpoint else x
    spectral_u = u_grid[:, :-1] if duplicate_endpoint else u_grid
    if len(spectral_x) < 2:
        raise ValueError("First-order KS reference requires at least two spatial points")
    spacing = np.diff(spectral_x)
    spacing_rtol = 5e-5 if data.dtype == np.float32 else 1e-8
    spacing_atol = 1e-7 if data.dtype == np.float32 else 1e-12
    if not np.allclose(
        spacing, spacing[0], rtol=spacing_rtol, atol=spacing_atol
    ):
        raise ValueError("First-order KS spectral reference requires a uniform x grid")

    modes = 2.0 * np.pi * np.fft.fftfreq(
        len(spectral_x), d=float(spacing[0])
    )
    coefficients = np.fft.fft(spectral_u, axis=1)
    derivatives = [
        np.fft.ifft((1j * modes) ** order * coefficients, axis=1).real
        for order in (1, 2, 3)
    ]
    if duplicate_endpoint:
        derivatives = [np.c_[field, field[:, 0]] for field in derivatives]
    auxiliary = [
        field[t_indices, x_indices].astype(data.dtype, copy=False)
        for field in derivatives
    ]
    return np.column_stack((data[:, :3], *auxiliary))


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


class FactorizedKuramotoSivashinskyEquation(baseclass.BaseTimePDE):
    """Second-order two-field system equivalent to the classical KS equation.

    The repository uses ``alpha`` for advection, ``beta`` for second-order
    diffusion, and ``gamma`` for fourth-order hyperdiffusion. Introducing

        q = beta * u + gamma * u_xx

    gives the equivalent pair

        q - beta*u - gamma*u_xx = 0,
        u_t + alpha*u*u_x + q_xx = 0.

    Thus the network has two outputs ``(u, q)`` and automatic differentiation
    never goes beyond second spatial derivatives.
    """

    def __init__(
        self,
        datapath="ref/Kuramoto_Sivashinsky.dat",
        bbox=[0, 2 * np.pi, 0, 1],
        alpha=100 / 16,
        beta=100 / (16 * 16),
        gamma=100 / (16**4),
    ):
        super().__init__()
        coefficients = np.asarray([alpha, beta, gamma], dtype=np.float64)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("KS coefficients must be finite")
        if beta == 0.0 or gamma == 0.0:
            raise ValueError("Factorized KS beta and gamma must be non-zero")

        self.output_config = [{"name": "u"}, {"name": "q"}]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.bbox = list(bbox)
        self.geom = dde.geometry.Interval(self.bbox[0], self.bbox[1])
        timedomain = dde.geometry.TimeDomain(self.bbox[2], self.bbox[3])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        def pde(x, fields):
            u = fields[:, 0:1]
            q = fields[:, 1:2]
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            q_xx = dde.grad.hessian(q, x, i=0, j=0)
            terms = self.build_terms(u, q, u_t, u_x, u_xx, q_xx)
            return [terms["compatibility"], terms["dynamics"]]

        self.pde = pde
        self.set_pdeloss(names=["ks_compatibility", "ks_dynamics"])
        self.load_ref_data(datapath, t_transpose=False)
        self.ref_data = build_factorized_ks_reference(
            self.ref_data, beta=self.beta, gamma=self.gamma
        )
        self.add_bcs(
            [
                {
                    "component": 0,
                    "function": (
                        lambda x: np.cos(x[:, 0:1])
                        * (1.0 + np.sin(x[:, 0:1]))
                    ),
                    "bc": (lambda _, on_initial: on_initial),
                    "type": "ic",
                }
            ]
        )
        self.training_points()

    def build_terms(self, u, q, u_t, u_x, u_xx, q_xx):
        return build_factorized_ks_terms(
            u=u,
            q=q,
            u_t=u_t,
            u_x=u_x,
            u_xx=u_xx,
            q_xx=q_xx,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )


class ConservativeKuramotoSivashinskyEquation(baseclass.BaseTimePDE):
    """Two-field conservative/flux system equivalent to classical KS.

    With the coefficient convention used in this repository,

        J = alpha*u**2/2 + beta*u_x + gamma*u_xxx,

    and the equivalent system is

        u_t + J_x = 0,
        J - (alpha*u**2/2 + beta*u_x + gamma*u_xxx) = 0.

    The network predicts ``(u, J)`` and differentiation reaches third order
    instead of the fourth order required by the classical strong form.
    """

    def __init__(
        self,
        datapath="ref/Kuramoto_Sivashinsky.dat",
        bbox=[0, 2 * np.pi, 0, 1],
        alpha=100 / 16,
        beta=100 / (16 * 16),
        gamma=100 / (16**4),
    ):
        super().__init__()
        coefficients = np.asarray([alpha, beta, gamma], dtype=np.float64)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("KS coefficients must be finite")

        self.output_config = [{"name": "u"}, {"name": "flux"}]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.bbox = list(bbox)
        self.geom = dde.geometry.Interval(self.bbox[0], self.bbox[1])
        timedomain = dde.geometry.TimeDomain(self.bbox[2], self.bbox[3])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        def pde(x, fields):
            u = fields[:, 0:1]
            flux = fields[:, 1:2]
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_xxx = dde.grad.jacobian(u_xx, x, i=0, j=0)
            flux_x = dde.grad.jacobian(flux, x, i=0, j=0)
            terms = self.build_terms(u, flux, u_t, u_x, u_xxx, flux_x)
            return [terms["conservation"], terms["flux_compatibility"]]

        self.pde = pde
        self.set_pdeloss(names=["ks_conservation", "ks_flux_compatibility"])
        self.load_ref_data(datapath, t_transpose=False)
        self.ref_data = build_conservative_ks_reference(
            self.ref_data,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        self.add_bcs(
            [
                {
                    "component": 0,
                    "function": (
                        lambda x: np.cos(x[:, 0:1])
                        * (1.0 + np.sin(x[:, 0:1]))
                    ),
                    "bc": (lambda _, on_initial: on_initial),
                    "type": "ic",
                }
            ]
        )
        self.training_points()

    def build_terms(self, u, flux, u_t, u_x, u_xxx, flux_x):
        return build_conservative_ks_terms(
            u=u,
            flux=flux,
            u_t=u_t,
            u_x=u_x,
            u_xxx=u_xxx,
            flux_x=flux_x,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )


class FirstOrderKuramotoSivashinskyEquation(baseclass.BaseTimePDE):
    """Four-field first-order system equivalent to the classical KS equation.

    The learned auxiliary fields are ``p=u_x``, ``q=p_x``, and ``r=q_x``.
    With this repository's coefficient convention, the system is

        p - u_x = 0,
        q - p_x = 0,
        r - q_x = 0,
        u_t + alpha*u*p + beta*q + gamma*r_x = 0.

    Every derivative evaluated by automatic differentiation is first order.
    """

    def __init__(
        self,
        datapath="ref/Kuramoto_Sivashinsky.dat",
        bbox=[0, 2 * np.pi, 0, 1],
        alpha=100 / 16,
        beta=100 / (16 * 16),
        gamma=100 / (16**4),
    ):
        super().__init__()
        coefficients = np.asarray([alpha, beta, gamma], dtype=np.float64)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("KS coefficients must be finite")

        self.output_config = [
            {"name": "u"},
            {"name": "p"},
            {"name": "q"},
            {"name": "r"},
        ]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.bbox = list(bbox)
        self.geom = dde.geometry.Interval(self.bbox[0], self.bbox[1])
        timedomain = dde.geometry.TimeDomain(self.bbox[2], self.bbox[3])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        def pde(x, fields):
            u = fields[:, 0:1]
            p = fields[:, 1:2]
            q = fields[:, 2:3]
            r = fields[:, 3:4]
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            p_x = dde.grad.jacobian(p, x, i=0, j=0)
            q_x = dde.grad.jacobian(q, x, i=0, j=0)
            r_x = dde.grad.jacobian(r, x, i=0, j=0)
            terms = self.build_terms(u, p, q, r, u_t, u_x, p_x, q_x, r_x)
            return [
                terms["p_compatibility"],
                terms["q_compatibility"],
                terms["r_compatibility"],
                terms["dynamics"],
            ]

        self.pde = pde
        self.set_pdeloss(
            names=[
                "ks_p_compatibility",
                "ks_q_compatibility",
                "ks_r_compatibility",
                "ks_dynamics",
            ]
        )
        self.load_ref_data(datapath, t_transpose=False)
        self.ref_data = build_first_order_ks_reference(self.ref_data)
        self.add_bcs(
            [
                {
                    "component": 0,
                    "function": (
                        lambda x: np.cos(x[:, 0:1])
                        * (1.0 + np.sin(x[:, 0:1]))
                    ),
                    "bc": (lambda _, on_initial: on_initial),
                    "type": "ic",
                },
                *[
                    {
                        "component": component,
                        "component_x": 0,
                        "bc": (lambda _, on_boundary: on_boundary),
                        "type": "periodic",
                        "name": f"periodic_{name}",
                    }
                    for component, name in enumerate(("u", "p", "q", "r"))
                ],
            ]
        )
        self.training_points()

    def build_terms(self, u, p, q, r, u_t, u_x, p_x, q_x, r_x):
        return build_first_order_ks_terms(
            u=u,
            p=p,
            q=q,
            r=r,
            u_t=u_t,
            u_x=u_x,
            p_x=p_x,
            q_x=q_x,
            r_x=r_x,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )


class CanonicalKuramotoSivashinskyEquation(KuramotoSivashinskyEquation):
    """Dimensionless strong-form equivalent of the physical KS equation.

    The physical equation is

        u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx = 0.

    With ``x = L * xi``, ``t = T * tau`` and ``u = U * v``, the scales
    below make every coefficient in the equation for ``v`` equal to one.
    The network, geometry, boundary condition, and ``ref_data`` therefore use
    canonical coordinates and values.  The conversion methods provide an
    explicit bridge back to physical units for future metrics and callbacks.
    """

    def __init__(
        self,
        datapath="ref/Kuramoto_Sivashinsky.dat",
        bbox=[0, 2 * np.pi, 0, 1],
        alpha=100 / 16,
        beta=100 / (16 * 16),
        gamma=100 / (16**4),
    ):
        coefficients = np.asarray([alpha, beta, gamma], dtype=np.float64)
        if not np.all(np.isfinite(coefficients)) or np.any(coefficients <= 0.0):
            raise ValueError("KS physical coefficients must be finite and positive")

        self.physical_alpha = float(alpha)
        self.physical_beta = float(beta)
        self.physical_gamma = float(gamma)
        self.length_scale = float(np.sqrt(self.physical_gamma / self.physical_beta))
        self.time_scale = float(
            self.physical_gamma / (self.physical_beta * self.physical_beta)
        )
        self.solution_scale = float(
            self.physical_beta ** 1.5
            / (self.physical_alpha * np.sqrt(self.physical_gamma))
        )
        scales = np.asarray(
            [self.length_scale, self.time_scale, self.solution_scale],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("KS canonical scales must be finite and positive")

        self.residual_scale = self.solution_scale / self.time_scale
        self.residual_mse_scale = self.residual_scale**2
        self.physical_bbox = self._validated_bbox(bbox)
        canonical_bbox = [
            self.physical_bbox[0] / self.length_scale,
            self.physical_bbox[1] / self.length_scale,
            self.physical_bbox[2] / self.time_scale,
            self.physical_bbox[3] / self.time_scale,
        ]

        super().__init__(
            datapath=datapath,
            bbox=canonical_bbox,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
        )

        self.ref_data = self.to_canonical_data(self.ref_data)

        # The parent IC is expressed in its input coordinate directly. Replace
        # it with v(xi, tau_0) = u_0(L * xi) / U.
        self.bcs = None
        self.loss_config = [
            config for config in self.loss_config if config["type"] == "pde"
        ]
        self.add_bcs(
            [
                {
                    "component": 0,
                    "function": self.canonical_initial_condition,
                    "bc": (lambda _, on_initial: on_initial),
                    "type": "ic",
                }
            ]
        )

    @staticmethod
    def _validated_bbox(bbox):
        values = np.asarray(bbox, dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("KS bbox must contain four finite values")
        if values[0] >= values[1] or values[2] >= values[3]:
            raise ValueError("KS bbox lower bounds must be smaller than upper bounds")
        return values.tolist()

    @staticmethod
    def _copy_array(values):
        clone = getattr(values, "clone", None)
        if callable(clone):
            return clone()
        array = np.asarray(values)
        dtype = np.result_type(array.dtype, np.float32)
        return np.array(array, dtype=dtype, copy=True)

    @staticmethod
    def _scale_value(values, scale):
        try:
            return values * scale
        except TypeError:
            return np.asarray(values) * scale

    def _scale_columns(self, values, scales, minimum_columns, label):
        result = self._copy_array(values)
        if result.ndim == 0 or result.shape[-1] < minimum_columns:
            raise ValueError(
                f"{label} must have at least {minimum_columns} columns"
            )
        for column, scale in enumerate(scales):
            result[..., column] = result[..., column] * scale
        return result

    def canonical_initial_condition(self, points):
        physical_x = points[:, 0:1] * self.length_scale
        return (
            np.cos(physical_x) * (1.0 + np.sin(physical_x))
            / self.solution_scale
        )

    def to_canonical_points(self, points):
        return self._scale_columns(
            points,
            (1.0 / self.length_scale, 1.0 / self.time_scale),
            minimum_columns=2,
            label="KS points",
        )

    def to_physical_points(self, points):
        return self._scale_columns(
            points,
            (self.length_scale, self.time_scale),
            minimum_columns=2,
            label="KS points",
        )

    def to_canonical_outputs(self, outputs):
        return self._scale_value(outputs, 1.0 / self.solution_scale)

    def to_physical_outputs(self, outputs):
        return self._scale_value(outputs, self.solution_scale)

    def to_canonical_data(self, data):
        return self._scale_columns(
            data,
            (
                1.0 / self.length_scale,
                1.0 / self.time_scale,
                1.0 / self.solution_scale,
            ),
            minimum_columns=3,
            label="KS data",
        )

    def to_physical_data(self, data):
        return self._scale_columns(
            data,
            (self.length_scale, self.time_scale, self.solution_scale),
            minimum_columns=3,
            label="KS data",
        )

    def to_canonical_derivatives(self, *, u_t, u_x, u_xx, u_xxxx):
        return {
            "v_tau": self._scale_value(u_t, self.time_scale / self.solution_scale),
            "v_xi": self._scale_value(u_x, self.length_scale / self.solution_scale),
            "v_xixi": self._scale_value(
                u_xx, self.length_scale**2 / self.solution_scale
            ),
            "v_xixixixi": self._scale_value(
                u_xxxx, self.length_scale**4 / self.solution_scale
            ),
        }

    def to_physical_derivatives(self, *, v_tau, v_xi, v_xixi, v_xixixixi):
        return {
            "u_t": self._scale_value(v_tau, self.solution_scale / self.time_scale),
            "u_x": self._scale_value(v_xi, self.solution_scale / self.length_scale),
            "u_xx": self._scale_value(
                v_xixi, self.solution_scale / self.length_scale**2
            ),
            "u_xxxx": self._scale_value(
                v_xixixixi, self.solution_scale / self.length_scale**4
            ),
        }

    def to_physical_residual(self, canonical_residual):
        return self._scale_value(canonical_residual, self.residual_scale)

    def to_canonical_residual(self, physical_residual):
        return self._scale_value(physical_residual, 1.0 / self.residual_scale)

    def to_physical_residual_mse(self, canonical_residual_mse):
        return self._scale_value(canonical_residual_mse, self.residual_mse_scale)

    def to_canonical_residual_mse(self, physical_residual_mse):
        return self._scale_value(
            physical_residual_mse, 1.0 / self.residual_mse_scale
        )
