import math

import torch


class PeriodicFourierFeatures(torch.nn.Module):
    """Periodic Fourier feature map for space + time inputs."""

    def __init__(
        self,
        x_period,
        num_modes_x,
        y_period=None,
        include_t=True,
        include_raw_x=False,
        include_bias=True,
    ):
        super().__init__()
        self.x_period = float(x_period)
        self.num_modes_x = int(num_modes_x)
        self.y_period = None if y_period is None else float(y_period)
        self.include_t = bool(include_t)
        self.include_raw_x = bool(include_raw_x)
        self.include_bias = bool(include_bias)
        self.register_buffer("modes", torch.arange(1, self.num_modes_x + 1).float())

    @property
    def out_dim(self):
        dim = 2 * self.num_modes_x
        if self.y_period is not None:
            dim += 2 * self.num_modes_x
        if self.include_t:
            dim += 1
        if self.include_raw_x:
            dim += 2 if self.y_period is not None else 1
        if self.include_bias:
            dim += 1
        return dim

    def forward(self, inputs):
        x = inputs[:, 0:1]
        y = inputs[:, 1:2] if self.y_period is not None else None
        t = inputs[:, -1:]
        feats = []

        if self.include_t:
            feats.append(t)
        if self.include_bias:
            feats.append(torch.ones_like(t))
        if self.include_raw_x:
            feats.append(x if y is None else torch.cat([x, y], dim=1))

        modes = self.modes.to(device=inputs.device, dtype=inputs.dtype)
        phase_x = 2.0 * math.pi * x * modes.reshape(1, -1) / self.x_period
        feats.append(torch.cos(phase_x))
        feats.append(torch.sin(phase_x))
        if y is not None:
            phase_y = 2.0 * math.pi * y * modes.reshape(1, -1) / self.y_period
            feats.append(torch.cos(phase_y))
            feats.append(torch.sin(phase_y))
        return torch.cat(feats, dim=1)
