"""Regression tests for derivative-cache cleanup in basin hopping."""

from types import SimpleNamespace

import pytest
import torch

from experiments.Chaotic.run_basin_hopping_integral import (
    OneShotBasinHopper,
    build_parser,
    validate_args,
)


class _FakeNet:
    def __init__(self):
        self.zero_grad_calls = []

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls.append(set_to_none)


class _FakeModel:
    def __init__(self):
        self.net = _FakeNet()


class _EvaluationLoss:
    def __init__(self, fail=False):
        self.fail = fail
        self.last_diagnostics = {}

    def _set_cached_endpoints(self, x, t, step):
        pass

    def _endpoint_ptrs(self, x, t):
        return x.data_ptr(), t.data_ptr()

    def compute_raw_loss(self, step, endpoints):
        if self.fail:
            raise RuntimeError("synthetic evaluation failure")
        value = torch.tensor(float(step + 1))
        self.last_diagnostics = {
            "integral_loss_raw": value,
            "global_integral_loss": value,
            "local_integral_loss": value * 0,
            "initial_condition_loss": value * 0,
            "periodic_loss": value * 0,
        }


def _hopper():
    hopper = object.__new__(OneShotBasinHopper)
    hopper.model = _FakeModel()
    return hopper


def test_fixed_evaluation_clears_derivative_cache_per_batch(monkeypatch):
    clear_calls = []
    monkeypatch.setattr(
        "experiments.Chaotic.run_basin_hopping_integral.dde.grad.clear",
        lambda: clear_calls.append(True),
    )
    tensor = torch.zeros(1, 1)
    batches = [(tensor, tensor, {}), (tensor, tensor, {})]
    hopper = _hopper()

    hopper._evaluate(_EvaluationLoss(), batches)

    assert len(clear_calls) == len(batches)
    assert hopper.model.net.zero_grad_calls == [True]


def test_fixed_evaluation_clears_derivative_cache_after_failure(monkeypatch):
    clear_calls = []
    monkeypatch.setattr(
        "experiments.Chaotic.run_basin_hopping_integral.dde.grad.clear",
        lambda: clear_calls.append(True),
    )
    tensor = torch.zeros(1, 1)

    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        _hopper()._evaluate(_EvaluationLoss(fail=True), [(tensor, tensor, {})])

    assert clear_calls == [True]


def test_local_relaxation_clears_cache_and_releases_gradients(monkeypatch):
    class FakeOptimizer:
        def __init__(self):
            self.zero_grad_calls = []

        def zero_grad(self, set_to_none=False):
            self.zero_grad_calls.append(set_to_none)

        def step(self, closure):
            return closure()

    class RelaxationLoss:
        @staticmethod
        def compute_weighted_loss(step):
            return torch.tensor(float(step), requires_grad=True)

    clear_calls = []
    monkeypatch.setattr(
        "experiments.Chaotic.run_basin_hopping_integral.dde.grad.clear",
        lambda: clear_calls.append(True),
    )
    hopper = _hopper()
    hopper.args = SimpleNamespace(basin_hopping_local_steps=1, basin_hopping_step=10)
    hopper._evaluate = lambda loss, batches: {
        "total": 1.0, "global": 1.0, "local": 0.0, "ic": 0.0, "periodic": 0.0,
    }
    optimizer = FakeOptimizer()
    trajectory = []

    hopper._relax(RelaxationLoss(), optimizer, 0, [], trajectory)

    assert clear_calls == [True]
    assert optimizer.zero_grad_calls == [True, True]
    assert trajectory[0]["local_step"] == 1


def test_deoptimization_phase_is_enabled_for_first_1000_steps_by_default():
    args = build_parser().parse_args([])

    assert args.basin_hopping is True
    assert args.basin_hopping_step == 1000
    assert args.integral_warmup_steps == 0
    assert args.parameter_lower == -1.0
    assert args.parameter_upper == 1.0
    assert args.integral_periodic_enabled is True


@pytest.mark.parametrize(
    "lower, upper",
    [(1.0, 1.0), (2.0, -2.0), (float("nan"), 1.0)],
)
def test_deoptimization_parameter_bounds_are_validated(lower, upper):
    args = build_parser().parse_args([])
    args.parameter_lower = lower
    args.parameter_upper = upper

    with pytest.raises(ValueError, match="parameter bounds"):
        validate_args(args)
