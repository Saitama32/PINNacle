from unittest.mock import patch

import pytest
import torch

from RL.rl_algorithms import DQNAgent
from RL.rl_utils.hybrid_policy import HybridHeadPolicy


OPTIMIZERS = {
    "Adam": {"lr": [1e-3, 1e-4], "epochs": [100, 1000]},
    "LBFGS": {"lr": [1.0, 0.5], "epochs": [100, 500]},
}


def _state():
    return {
        "loss_total": torch.randn(26, 26),
        "loss_oper": torch.randn(26, 26),
        "loss_bnd": torch.randn(26, 26),
    }


@pytest.mark.parametrize(
    "representation", ["legacy_flat", "separate", "shared_hidden"]
)
def test_split_greedy_selection_matches_original_agent(representation):
    torch.manual_seed(7)
    agent = DQNAgent(
        optimizer_dict=OPTIMIZERS,
        device="cpu",
        parameter_representation=representation,
        network_normalization="groupnorm",
        group_norm_groups=8,
        exp=None,
    )
    agent.model_optim.eval()
    agent.model_params.eval()
    state = _state()

    expected_action, expected_raw = agent.select_greedy_action(state)
    with patch("RL.rl_algorithms.random.random", return_value=1.0):
        actual_action, actual_raw, is_model = agent.select_action(
            state, force_greedy=True
        )

    assert is_model
    assert actual_raw == expected_raw
    assert actual_action == expected_action


class _FakeAgent:
    def __init__(self, optimizer_choice, parameter_choice):
        self.optimizer_dict = OPTIMIZERS
        self.i2opt = {0: "Adam", 1: "LBFGS"}
        self.optimizer_choice = optimizer_choice
        self.parameter_choice = parameter_choice
        self.parameter_requests = []
        self.steps_done = 0

    def select_optimizer_greedy(self, state):
        del state
        index = 0 if self.optimizer_choice == "Adam" else 1
        return index, self.optimizer_choice, torch.tensor([0.0, 1.0])

    def select_parameters_greedy(self, state, optim_name):
        del state
        self.parameter_requests.append(optim_name)
        return self.parameter_choice

    def get_random_action(self):
        return 0, 0, {"lr": 0}

    def post_proc_model(self, optim_class, epochs_class, param_class):
        name = self.i2opt[optim_class]
        return {
            "type": name,
            "epochs": OPTIMIZERS[name]["epochs"][epochs_class],
            "params": {
                key: OPTIMIZERS[name][key][value]
                for key, value in param_class.items()
            },
        }


def test_hybrid_uses_optimizer_from_first_and_parameters_from_second():
    optimizer_agent = _FakeAgent("LBFGS", (0, {"lr": 0}))
    parameter_agent = _FakeAgent("Adam", (1, {"lr": 1}))
    policy = HybridHeadPolicy(
        optimizer_agent,
        parameter_agent,
        optimizer_source="good",
        parameter_source="bad",
    )

    action, raw, is_model, diagnostics = policy.select_action(
        state={}, epsilon=0.0
    )

    assert is_model
    assert raw == (1, 1, {"lr": 1})
    assert action == {"type": "LBFGS", "epochs": 500, "params": {"lr": 0.5}}
    assert parameter_agent.parameter_requests == ["LBFGS"]
    assert diagnostics["optimizer_source"] == "good"
    assert diagnostics["parameter_source"] == "bad"
    assert optimizer_agent.steps_done == 1


def test_hybrid_preserves_epsilon_random_action():
    agent = _FakeAgent("LBFGS", (1, {"lr": 1}))
    policy = HybridHeadPolicy(agent, agent, "good", "good")

    action, raw, is_model, diagnostics = policy.select_action(
        state={}, epsilon=1.0
    )

    assert not is_model
    assert raw == (0, 0, {"lr": 0})
    assert action["type"] == "Adam"
    assert diagnostics["optimizer_source"] == "epsilon-random"
