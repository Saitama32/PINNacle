import random
import unittest

import torch
import torch.nn as nn

from RL.rl_algorithms import DQNAgent
from RL.rl_utils.per_buffer import Transition, UniformReplayBuffer


OPTIMIZERS = {
    "Adam": {"lr": [1e-3, 1e-4], "epochs": [10, 20]},
    "LBFGS": {"lr": [1.0, 0.5], "epochs": [10, 20]},
}


def make_state(value):
    tensor = torch.full((8, 8), float(value), dtype=torch.float32)
    return {
        "loss_total": tensor.clone(),
        "loss_oper": tensor.clone(),
        "loss_bnd": tensor.clone(),
        "delta": torch.zeros_like(tensor),
    }


def fill_buffer(agent):
    for index in range(6):
        action_index = index % len(OPTIMIZERS)
        action = (action_index, {"lr": index % 2, "epochs": (index + 1) % 2})
        done = 1 if index in (2, 5) else 0
        agent.replay_buffer.push(
            make_state(index / 10),
            make_state((index + 1) / 10),
            action,
            float(index + 1) / 10,
            done,
            1.0 if done else 0.0,
            action_index,
        )


class FixedOptimizerQ(nn.Module):
    def __init__(self, q_values):
        super().__init__()
        self.register_buffer("q_values", torch.tensor(q_values, dtype=torch.float32))

    def forward(self, state):
        q_values = self.q_values.expand(state.shape[0], -1)
        return state.flatten(start_dim=1), q_values


class FailingTarget(nn.Module):
    def forward(self, state):
        raise AssertionError("target model must not be evaluated when masking is disabled")


class AgentAblationTests(unittest.TestCase):
    def make_agent(self, **flags):
        return DQNAgent(
            optimizer_dict=OPTIMIZERS,
            memory_size=16,
            batch_size=2,
            gamma=0.9,
            warmup_updates=0,
            success_frac=0.0,
            device="cpu",
            exp=None,
            **flags,
        )

    def test_uniform_replay_has_unit_weights_and_no_priorities(self):
        buffer = UniformReplayBuffer(capacity=8)
        for index in range(4):
            buffer.push(
                make_state(index),
                make_state(index + 1),
                (0, {"lr": 0, "epochs": 0}),
                0.0,
                1 if index == 2 else 0,
                1.0 if index == 2 else 0.0,
                0,
            )

        sequences, indexes, weights = buffer.sample_sequences(2, 4, device="cpu")
        self.assertEqual(buffer.prior, [])
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
        self.assertEqual(len(sequences), 2)
        self.assertTrue(all(sequence for sequence in sequences))
        self.assertTrue(all(sum(tr.done != 0 for tr in sequence) <= 1 for sequence in sequences))

        buffer.update_priorities(indexes, torch.full((2,), 100.0))
        self.assertEqual(buffer.prior, [])

    def test_one_step_target_uses_online_action_and_target_value(self):
        agent = self.make_agent(use_soft_watkins=False)
        agent.model_optim = FixedOptimizerQ([1.0, 3.0])
        agent.target_model_optim = FixedOptimizerQ([10.0, 20.0])

        target = agent._one_step_optimizer_targets(
            torch.zeros(2, 4, 8, 8),
            torch.tensor([2.0, 5.0]),
            torch.tensor([0.0, 1.0]),
        )
        self.assertTrue(torch.allclose(target, torch.tensor([20.0, 5.0])))

    def test_disabled_trust_region_keeps_every_sample(self):
        agent = self.make_agent(use_trust_region_masking=False)
        agent.target_model_optim = FailingTarget()
        dropped, kept, _, _ = agent._trust_region_keep_mask(
            torch.tensor([100.0, -100.0]),
            torch.tensor([0.0, 0.0]),
            torch.zeros(2, 4, 8, 8),
            torch.tensor([0, 1]),
        )
        self.assertFalse(dropped.any())
        self.assertTrue(torch.equal(kept, torch.ones_like(kept)))

    def test_all_presets_complete_a_synthetic_update(self):
        presets = (
            {},
            {"use_prioritized_replay": False},
            {"use_soft_watkins": False},
            {"use_trust_region_masking": False},
        )
        for index, flags in enumerate(presets):
            with self.subTest(flags=flags):
                torch.manual_seed(100 + index)
                random.seed(100 + index)
                agent = self.make_agent(**flags)
                fill_buffer(agent)
                optimizer_losses, parameter_losses = agent.optim_(iters=1)
                self.assertEqual(len(optimizer_losses), 1)
                self.assertEqual(len(parameter_losses), 1)
                self.assertTrue(torch.isfinite(torch.tensor(optimizer_losses)).all())
                self.assertTrue(torch.isfinite(torch.tensor(parameter_losses)).all())

    def test_default_mode_matches_explicit_full_mode(self):
        torch.manual_seed(777)
        random.seed(777)
        default_agent = self.make_agent()
        fill_buffer(default_agent)

        torch.manual_seed(777)
        random.seed(777)
        explicit_agent = self.make_agent(
            use_prioritized_replay=True,
            use_soft_watkins=True,
            use_trust_region_masking=True,
        )
        fill_buffer(explicit_agent)

        torch.manual_seed(888)
        random.seed(888)
        default_losses = default_agent.optim_(iters=1)
        torch.manual_seed(888)
        random.seed(888)
        explicit_losses = explicit_agent.optim_(iters=1)

        self.assertEqual(default_losses, explicit_losses)


if __name__ == "__main__":
    unittest.main()
