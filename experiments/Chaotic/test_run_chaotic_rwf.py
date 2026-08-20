import os
import sys
from types import SimpleNamespace

os.environ["DDEBACKEND"] = "pytorch"

import pytest
import torch

from deepxde.optimizers.pytorch.optimizers import get as get_optimizer
from experiments.Chaotic import run_chaotic
from src.dynamic_freezing.optimizer_adapter import MaskedOptimizerAdapter
from src.dynamic_freezing.weight_groups import WeightGroupCollection
from src.model import RWFLinear, RWFMLP


def test_build_network_creates_rwf_mlp():
    pde = SimpleNamespace(input_dim=2, output_dim=1)

    network = run_chaotic.build_network(
        pde,
        hidden_layers="8*2",
        net_type="rwf-mlp",
        rwf_mu=0.5,
        rwf_sigma=0.0,
    )

    assert isinstance(network, RWFMLP)
    assert len(network.linears) == 3
    assert all(isinstance(layer, RWFLinear) for layer in network.linears)
    assert all(torch.equal(layer.s, torch.full_like(layer.s, 0.5)) for layer in network.linears)
    assert network(torch.zeros(4, 2)).shape == (4, 1)


def test_rwf_dynamic_freezing_groups_only_scale_parameters():
    network = RWFMLP([2, 4, 4, 1])

    groups = WeightGroupCollection(network, group_size=2)

    assert groups.grouping_mode == "rwf_scales"
    assert {group.parameter_name for group in groups.groups} == {
        "linears.0.s",
        "linears.1.s",
        "linears.2.s",
    }
    assert all(group.parameter.ndim == 1 for group in groups.groups)
    assert all(not name.endswith(".V") for name, _, _ in groups.weight_parameters)


@pytest.mark.parametrize("optimizer_name", ["muon", "soap"])
def test_rwf_freezing_keeps_scales_fixed_while_v_matrices_train(optimizer_name):
    torch.manual_seed(7)
    network = RWFMLP([2, 8, 8, 1]).float()
    groups = WeightGroupCollection(network, group_size=4)
    frozen_group = groups.groups[0]
    groups.set_frozen({frozen_group.group_id})
    optimizer, _ = get_optimizer(
        network.parameters(),
        optimizer_name,
        learning_rate=5e-4,
        model=network,
    )
    adapter = MaskedOptimizerAdapter(network, optimizer, groups).install()
    x = torch.randn(16, 2)
    target = torch.randn(16, 1)
    scales_before = {
        name: parameter.detach().clone()
        for name, parameter in network.named_parameters()
        if name.endswith(".s")
    }
    matrices_before = {
        name: parameter.detach().clone()
        for name, parameter in network.named_parameters()
        if name.endswith(".V")
    }

    optimizer.zero_grad()
    torch.mean((network(x) - target) ** 2).backward()
    optimizer.step()
    adapter.uninstall()

    frozen_after = frozen_group.parameter.detach().reshape(-1)[
        frozen_group.flat_start : frozen_group.flat_end
    ]
    frozen_before = scales_before[frozen_group.parameter_name].reshape(-1)[
        frozen_group.flat_start : frozen_group.flat_end
    ]
    assert torch.equal(frozen_after, frozen_before)
    assert all(
        not torch.equal(before, dict(network.named_parameters())[name].detach())
        for name, before in matrices_before.items()
    )
    assert any(
        not torch.equal(before, dict(network.named_parameters())[name].detach())
        for name, before in scales_before.items()
        if name != frozen_group.parameter_name
    )
    assert all(torch.isfinite(parameter).all() for parameter in network.parameters())
    state = optimizer.state[frozen_group.parameter]
    for key in ("exp_avg", "exp_avg_sq"):
        values = state[key].reshape(-1)[frozen_group.flat_start : frozen_group.flat_end]
        assert torch.count_nonzero(values) == 0


def test_parse_args_exposes_rwf_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_chaotic.py", "--net", "rwf-mlp", "--rwf-mu", "0.75", "--rwf-sigma", "0.2"],
    )

    args = run_chaotic.parse_args()

    assert args.net == "rwf-mlp"
    assert args.rwf_mu == pytest.approx(0.75)
    assert args.rwf_sigma == pytest.approx(0.2)
    run_chaotic.validate_args(args)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--rwf-mu", "nan", "--rwf-mu must be finite"),
        ("--rwf-sigma", "-0.1", "--rwf-sigma must be non-negative and finite"),
    ],
)
def test_validate_args_rejects_invalid_rwf_parameters(
    monkeypatch, option, value, message
):
    monkeypatch.setattr(sys, "argv", ["run_chaotic.py", option, value])

    with pytest.raises(ValueError, match=message):
        run_chaotic.validate_args(run_chaotic.parse_args())
