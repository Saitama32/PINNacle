import re

import torch

from .nn import NN
from .. import activations
from .. import initializers
from ... import config


class FNN(NN):
    """Fully-connected neural network."""

    supports_feature_reuse = True

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(
        self,
        inputs,
        starting_id=0,
        return_interval=False,
        original_inputs=None,
    ):
        x = inputs
        if starting_id == 0:
            original_inputs = inputs
        elif original_inputs is None:
            raise ValueError("original_inputs must be provided when starting_id > 0")

        if starting_id == 0 and self._input_transform is not None:
            x = self._input_transform(x)
        intervals = [x] if return_interval else None

        for j, linear in enumerate(self.linears[:-1]):
            if j < starting_id:
                continue
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
            if return_interval:
                intervals.append(x)
        x = self.linears[-1](x) if starting_id <= len(self.linears) - 1 else x
        if self._output_transform is not None:
            x = self._output_transform(original_inputs, x)
        if return_interval:
            return x, intervals
        return x

    def parameter_to_layer_index(self, param_name):
        match = re.search(r"linears\.(\d+)\.(weight|weight_orig|bias)$", param_name)
        if match is None:
            return None
        return int(match.group(1))


class PFNN(NN):
    """Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers, each of which is exclusively
            used by one output. Note that `len(layer_sizes[i])` should equal the number
            of outputs. Every number specifies the number of neurons in that layer.
    """

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        if len(layer_sizes) <= 1:
            raise ValueError("must specify input and output sizes")
        if not isinstance(layer_sizes[0], int):
            raise ValueError("input size must be integer")
        if not isinstance(layer_sizes[-1], int):
            raise ValueError("output size must be integer")

        n_output = layer_sizes[-1]

        def make_linear(n_input, n_output):
            linear = torch.nn.Linear(n_input, n_output, dtype=config.real(torch))
            initializer(linear.weight)
            initializer_zero(linear.bias)
            return linear

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError(
                        "number of sub-layers should equal number of network outputs"
                    )
                if isinstance(prev_layer_size, (list, tuple)):
                    # e.g. [8, 8, 8] -> [16, 16, 16]
                    self.layers.append(
                        torch.nn.ModuleList(
                            [
                                make_linear(prev_layer_size[j], curr_layer_size[j])
                                for j in range(n_output)
                            ]
                        )
                    )
                else:  # e.g. 64 -> [8, 8, 8]
                    self.layers.append(
                        torch.nn.ModuleList(
                            [
                                make_linear(prev_layer_size, curr_layer_size[j])
                                for j in range(n_output)
                            ]
                        )
                    )
            else:  # e.g. 64 -> 64
                if not isinstance(prev_layer_size, int):
                    raise ValueError(
                        "cannot rejoin parallel subnetworks after splitting"
                    )
                self.layers.append(make_linear(prev_layer_size, curr_layer_size))

        # output layers
        if isinstance(layer_sizes[-2], (list, tuple)):  # e.g. [3, 3, 3] -> 3
            self.layers.append(
                torch.nn.ModuleList(
                    [make_linear(layer_sizes[-2][j], 1) for j in range(n_output)]
                )
            )
        else:
            self.layers.append(make_linear(layer_sizes[-2], n_output))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for layer in self.layers[:-1]:
            if isinstance(layer, torch.nn.ModuleList):
                if isinstance(x, list):
                    x = [self.activation(f(x_)) for f, x_ in zip(layer, x)]
                else:
                    x = [self.activation(f(x)) for f in layer]
            else:
                x = self.activation(layer(x))

        # output layers
        if isinstance(x, list):
            x = torch.cat([f(x_) for f, x_ in zip(self.layers[-1], x)], dim=1)
        else:
            x = self.layers[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
