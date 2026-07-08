import torch

from deepxde import config
from deepxde.nn import NN

from .fnn import activation_dict, initializer_dict


class ResidualBlock(torch.nn.Module):
    def __init__(self, width, activation, kernel_initializer):
        super().__init__()
        initializer = initializer_dict[kernel_initializer]
        initializer_zero = initializer_dict["zeros"]

        self.activation = activation_dict[activation]
        self.linear1 = torch.nn.Linear(width, width, dtype=config.real(torch))
        self.linear2 = torch.nn.Linear(width, width, dtype=config.real(torch))

        initializer(self.linear1.weight)
        initializer_zero(self.linear1.bias)
        initializer(self.linear2.weight)
        initializer_zero(self.linear2.bias)

    def forward(self, inputs):
        x = self.activation(self.linear1(inputs))
        x = self.linear2(x)
        x = x + inputs
        return self.activation(x)


class ResNet(NN):
    """Residual fully-connected network for PyTorch DeepXDE backend."""

    def __init__(
        self,
        input_size,
        output_size,
        width,
        num_blocks,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        initializer = initializer_dict[kernel_initializer]
        initializer_zero = initializer_dict["zeros"]

        self.stem = torch.nn.Linear(input_size, width, dtype=config.real(torch))
        initializer(self.stem.weight)
        initializer_zero(self.stem.bias)

        self.activation = activation_dict[activation]
        self.blocks = torch.nn.ModuleList(
            [
                ResidualBlock(width, activation, kernel_initializer)
                for _ in range(num_blocks)
            ]
        )

        self.head = torch.nn.Linear(width, output_size, dtype=config.real(torch))
        initializer(self.head.weight)
        initializer_zero(self.head.bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        x = self.activation(self.stem(x))
        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
