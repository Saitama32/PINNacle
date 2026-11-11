from collections import OrderedDict

import torch

from RL.rl_utils.DQN_classes import DQN_optim, DQN_params
from experiments.agent_ablation.heads_ablation.reaction.checkpoint_model import (
    build_checkpoint_heads,
    infer_checkpoint_spec,
)
from experiments.agent_ablation.heads_ablation.reaction.comet_io import (
    classify_asset,
    pair_checkpoints,
)
from experiments.agent_ablation.heads_ablation.reaction.dataset import (
    build_records,
    coverage_rows,
    decode_action,
)
from experiments.agent_ablation.heads_ablation.reaction.diagnostics import (
    analyze_checkpoint,
    optimizer_calibration_rows,
)
from experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis import (
    _chronological_records,
)
from experiments.agent_ablation.heads_ablation.reaction.hybrid_comparison import (
    compare_hybrid_policies,
)


OPTIMIZERS = OrderedDict(
    {
        "Adam": {"lr": [0.1, 0.01], "epochs": [10, 20]},
        "LBFGS": {"lr": [1.0, 0.5], "epochs": [5, 10]},
    }
)


def _state(value):
    base = torch.full((6, 6), float(value))
    return {
        "loss_total": base,
        "loss_oper": base * 0.5,
        "loss_bnd": base * 0.25,
        "delta": torch.zeros_like(base),
    }


def test_asset_pairing_uses_exact_steps_only():
    raw_assets = [
        {"assetId": "o1", "fileName": "model_optim_step_10.pt"},
        {"assetId": "p1", "fileName": "model_params_step_10.pt"},
        {"assetId": "o2", "fileName": "rl_model_snapshots/model_optim_step_20.pt"},
        {"assetId": "p2", "fileName": "model_params_step_21.pt"},
        {"assetId": "t1", "fileName": "entry_step_7.pt"},
    ]
    assets = [classify_asset(asset) for asset in raw_assets]
    pairs = pair_checkpoints(asset for asset in assets if asset is not None)
    assert [pair.step for pair in pairs] == [10]


def test_shared_hidden_checkpoint_and_all_head_outputs():
    optim = DQN_optim(len(OPTIMIZERS), normalization="groupnorm")
    params = DQN_params(OPTIMIZERS)
    spec = infer_checkpoint_spec(
        optim.state_dict(),
        params.state_dict(),
        {"network_normalization": "groupnorm"},
    )
    assert spec.parameter_representation == "shared_hidden"
    assert spec.optimizer_count == 2

    model = build_checkpoint_heads(
        OPTIMIZERS,
        optim.state_dict(),
        params.state_dict(),
        {"network_normalization": "groupnorm"},
    )
    optimizer_q, parameter_q = model.forward_all(
        torch.stack([
            torch.stack(tuple(_state(1.0).values())),
            torch.stack(tuple(_state(2.0).values())),
        ])
    )
    assert optimizer_q.shape == (2, 2)
    assert set(parameter_q) == {"Adam", "LBFGS"}
    assert parameter_q["Adam"]["lr"].shape == (2, 2)


def test_records_mc_returns_coverage_and_checkpoint_diagnostics():
    entries = [
        {
            "state": _state(2.0),
            "action": (0, 1, {"lr": 0}),
            "reward_model": 1.0,
            "done": 0,
        },
        {
            "state": _state(1.0),
            "action": (1, 0, {"lr": 1}),
            "reward_model": 2.0,
            "done": 1,
        },
    ]
    records = build_records([("online", "key", entries)], gamma=0.5)
    assert records[0].mc_return == 2.0
    assert records[1].mc_return == 2.0
    assert decode_action(records[0].action, OPTIMIZERS)["parameter_indices"] == {
        "epochs": 1,
        "lr": 0,
    }
    coverage = coverage_rows(records, OPTIMIZERS)
    assert sum(row["transition_count"] for row in coverage) == 2
    assert {row["source_key"] for row in coverage} == {"key"}

    optim = DQN_optim(len(OPTIMIZERS), normalization="groupnorm")
    params = DQN_params(OPTIMIZERS)
    model = build_checkpoint_heads(
        OPTIMIZERS,
        optim.state_dict(),
        params.state_dict(),
        {"network_normalization": "groupnorm"},
    )
    optimizer_row, parameter_rows, factual_rows = analyze_checkpoint(
        model,
        records,
        OPTIMIZERS,
        checkpoint_step=10,
        experiment_label="test",
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert optimizer_row["state_count"] == 2
    assert optimizer_row["evaluation_scope"] == "fixed_all_online"
    assert len(parameter_rows) == 4
    assert len(factual_rows) == 4
    assert {row["parameter"] for row in factual_rows} == {"epochs", "lr"}
    optimizer_calibration = optimizer_calibration_rows(factual_rows)
    assert len(optimizer_calibration) == 1
    assert optimizer_calibration[0]["count"] == 2

    hybrid_rows, agreement_rows, hybrid_summary = compare_hybrid_policies(
        {"G": model, "B": model},
        records,
        OPTIMIZERS,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert len(hybrid_rows) == 8
    assert {row["hybrid"] for row in hybrid_rows} == {"GG", "GB", "BG", "BB"}
    assert all(row["agreement_fraction"] == 1.0 for row in agreement_rows)
    assert len(hybrid_summary) == 4


def test_chronological_records_cut_future_rewards_before_mc_return():
    entries = [
        {
            "state": _state(3.0),
            "action": (0, 0, {"lr": 0}),
            "reward_model": 1.0,
            "done": 0,
            "_source": {"asset_step": 1},
        },
        {
            "state": _state(2.0),
            "action": (0, 0, {"lr": 0}),
            "reward_model": 2.0,
            "done": 0,
            "_source": {"asset_step": 2},
        },
        {
            "state": _state(1.0),
            "action": (0, 0, {"lr": 0}),
            "reward_model": 100.0,
            "done": 1,
            "_source": {"asset_step": 3},
        },
    ]
    records = _chronological_records(
        [("online", "agent", entries), ("online", "other", entries)],
        experiment_key_value="agent",
        step=2,
        gamma=0.5,
    )
    assert len(records) == 2
    assert records[0].mc_return == 2.0
    assert records[1].mc_return == 2.0
