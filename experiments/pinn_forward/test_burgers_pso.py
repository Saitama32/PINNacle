"""Quick PSO test on Burgers 1D PDE - minimized for speed."""

import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

# Reduced points for speed
data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=500, num_boundary=40, num_initial=40
)
# Smaller net for speed
net = dde.nn.FNN([2] + [16] * 2 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# PSO with small pop and few iterations
dde.optimizers.set_PSO_options(pop_size=10, n_iter=50, lr=0)
model.compile("PSO")
losshistory, train_state = model.train(iterations=50, display_every=10)

print("PSO Burgers test done. Final train loss:", np.sum(train_state.loss_train))
