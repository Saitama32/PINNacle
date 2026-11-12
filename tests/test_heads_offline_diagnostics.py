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
from experiments.agent_ablation.heads_ablation.reaction.counterfactual_sweep import (
    select_stratified_records,
    summarize_groups,
    summarize_outcomes,
)
from experiments.agent_ablation.heads_ablation.reaction.run_counterfactual_sweep import (
    SweepProgress,
)
from experiments.agent_ablation.heads_ablation.reaction.return_transfer import (
    analyze_return_transfer,
    experiment_family,
    family_transfer_pair_rows,
    family_transfer_summary_rows,
    source_relation,
    transfer_gap_rows,
)
from experiments.agent_ablation.heads_ablation.reaction.config import (
    AGENT_SOAP_EXPERIMENTS,
    ALL_POLICY_TRANSFER_EXPERIMENTS,
    CREDIT_SPECIALIZATION_EXPERIMENTS,
    UNNORMALIZED_TARGET_EXPERIMENTS,
)


OPTIMIZERS = OrderedDict(
    {
        "Adam": {"lr": [0.1, 0.01], "epochs": [10, 20]},
        "LBFGS": {"lr": [1.0, 0.5], "epochs": [5, 10]},
    }
)


def test_credit_specialization_cohort_uses_verified_keys():
    expected_unnormalized = {
        "93af321143f5432dbdf953788c3e7b9a",
        "6fe1c5ad480443098bde98e40df3bcf8",
        "c39a13908b6244e2b8e1123be6042aca",
        "cdae3dc015c04359a96507588895b369",
        "c6af2a222fb24744b6103a23c6a5994b",
        "96067f30c5054231bcc05970489bffe4",
        "444bf8c164e14313bd3ad77b819ff273",
        "6f5cc25711524ed5abf3acaef0bf3e37",
    }
    assert {
        config["key"] for config in UNNORMALIZED_TARGET_EXPERIMENTS.values()
    } == expected_unnormalized
    assert len(CREDIT_SPECIALIZATION_EXPERIMENTS) == 14


def test_all_policy_transfer_includes_every_verified_agent_soap_run():
    expected_agent_soap = {
        "766133376b114b46ad3cd5d0168176f6",
        "1298a4d80f7b4fa4b06b0ba7a069bf55",
        "a940859302ee451993364556a6d0c3e0",
        "a42715bce3c14d41970149bd46490843",
        "df40139ee7164e0c808be909e951968d",
        "4d432fe2e7824cec90a6cb0360ff892c",
        "86ff8b0675c7465eab51afc9fe551a4e",
        "4385849a327b42bc835b6ac835b19bbb",
        "7ec5c49e0fb24792bb2dfd5915a0c8a3",
        "606c56ee9e0549e9b49fad506a1a00bf",
        "aed567fc4f19490299046aa118ac4611",
        "ffe0c354e8804f75a6049bb4ae96e513",
    }
    assert {
        config["key"] for config in AGENT_SOAP_EXPERIMENTS.values()
    } == expected_agent_soap
    assert len(ALL_POLICY_TRANSFER_EXPERIMENTS) == 38
    assert len(
        {config["key"] for config in ALL_POLICY_TRANSFER_EXPERIMENTS.values()}
    ) == len(ALL_POLICY_TRANSFER_EXPERIMENTS)


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
            "current_loss": 4.0,
            "done": 0,
        },
        {
            "state": _state(1.0),
            "action": (1, 0, {"lr": 1}),
            "reward_model": 2.0,
            "current_loss": 1.0,
            "done": 1,
        },
    ]
    records = build_records([("online", "key", entries)], gamma=0.5)
    assert records[0].mc_return == 2.0
    assert records[1].mc_return == 2.0
    assert abs(records[0].best_future_progress - torch.log(torch.tensor(4.0)).item()) < 1e-6
    assert records[0].best_future_horizon == 1
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


def test_best_future_progress_uses_best_later_loss_not_rewards():
    entries = [
        {
            "state": _state(3.0 - index),
            "action": (0, 0, {"lr": 0}),
            "reward_model": reward,
            "current_loss": (10.0, 2.0, 5.0)[index],
            "done": int(index == 2),
        }
        for index, reward in enumerate((1.0, 4.0, -20.0))
    ]
    records = build_records([("online", "agent", entries)], gamma=0.5)
    assert records[0].mc_return == -2.0
    assert abs(records[0].best_future_progress - torch.log(torch.tensor(5.0)).item()) < 1e-6
    assert records[0].best_future_horizon == 1
    assert records[1].best_future_progress == 0.0
    assert records[1].best_future_horizon == 0


def test_return_transfer_separates_sources_targets_and_heads():
    entries = []
    for index, reward in enumerate((1.0, 2.0, 3.0)):
        entries.append(
            {
                "state": _state(index + 1),
                "action": (index % 2, index % 2, {"lr": index % 2}),
                "reward_model": reward,
                "current_loss": 10.0 - index,
                "done": 1,
            }
        )
    blocks = [
        ("online", "agent", entries),
        ("online", "other-agent", entries),
        ("farm", "farm-agent", entries),
    ]
    records = build_records(blocks, gamma=0.9)
    assert source_relation(records[0], "agent") == "own"
    assert source_relation(records[3], "agent") == "other"
    assert source_relation(records[6], "agent") == "farm"

    optim = DQN_optim(len(OPTIMIZERS), normalization="groupnorm")
    params = DQN_params(OPTIMIZERS)
    model = build_checkpoint_heads(
        OPTIMIZERS,
        optim.state_dict(),
        params.state_dict(),
        {"network_normalization": "groupnorm"},
    )
    rows = analyze_return_transfer(
        model,
        records,
        OPTIMIZERS,
        checkpoint_step=10,
        checkpoint_label="A",
        checkpoint_key="agent",
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert {row["source_relation"] for row in rows} == {"own", "farm", "other"}
    assert {row["return_target"] for row in rows} == {
        "one_step",
        "mc",
        "best_future",
    }
    assert {row["head"] for row in rows} == {"optimizer", "parameter"}
    gaps = transfer_gap_rows(rows)
    assert {row["external_relation"] for row in gaps} == {"farm", "other"}


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


def test_counterfactual_selection_uses_previous_transition_and_summary_ranks():
    entries = []
    for step in range(4):
        entries.append(
            {
                "state": _state(4 - step),
                "action": (0, 0, {"lr": 0}),
                "reward_model": float(step),
                "done": int(step == 3),
                "_source": {
                    "asset_step": step + 10,
                    "asset_file": f"entry_step_{step + 10}.pt",
                    "local_index": 0,
                },
            }
        )
    records = build_records([("farm", "farm-key", entries)], gamma=0.9)
    selected = select_stratified_records(
        records, per_source_phase=1, suffix_horizon=1
    )
    assert selected
    for _, current, previous, suffix in selected:
        assert current.chain_step == previous.chain_step + 1
        assert current.asset_step == previous.asset_step + 1
        assert len(suffix) == 1
        assert suffix[0].chain_step == current.chain_step + 1

    rows = []
    for epochs_index in range(3):
        for lr_index in range(3):
            horizon_loss = float(1 + epochs_index * 3 + lr_index)
            immediate_loss = 10.0 - horizon_loss
            rows.append(
                {
                    "selection_id": 0,
                    "repeat": 0,
                    "source_key": "farm-key",
                    "phase": "middle",
                    "optimizer": "Adam",
                    "g_epochs_index": 0,
                    "g_lr_index": 0,
                    "b_epochs_index": 1,
                    "b_lr_index": 1,
                    "epochs_index": epochs_index,
                    "lr_index": lr_index,
                    "final_loss": immediate_loss,
                    "objective_loss": horizon_loss,
                    "score": -immediate_loss,
                    "immediate_score": -immediate_loss,
                    "horizon_score": -horizon_loss,
                }
            )
    summaries, aggregate = summarize_outcomes(rows)
    assert summaries[0]["g_exact_oracle"] == 1
    assert summaries[0]["g_rank"] == 1
    assert summaries[0]["b_rank"] == 5
    assert summaries[0]["g_good_action"] == 1
    assert summaries[0]["b_good_action"] == 0
    assert abs(summaries[0]["immediate_horizon_correlation"] + 1.0) < 1e-12
    assert aggregate["g_oracle_accuracy"] == 1.0
    grouped = summarize_groups(summaries)
    overall = next(
        row for row in grouped if row["optimizer"] == "all" and row["phase"] == "all"
    )
    assert overall["states"] == 1
    assert overall["g_mean_rank"] == 1.0
    assert overall["b_mean_rank"] == 5.0

    partial_summaries, partial_aggregate = summarize_outcomes(rows[:2])
    assert partial_summaries == []
    assert partial_aggregate == {"evaluated_states": 0, "incomplete_states": 1}


def test_counterfactual_progress_writes_scale_and_eta(tmp_path):
    progress = SweepProgress(tmp_path, 9, 0, [], "Adam")
    progress.advance(
        10.0,
        {"selection_id": 3, "g_optimizer": "Adam"},
        {"epochs_index": 0, "lr_index": 1},
    )
    import json

    snapshot = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert snapshot["completed_actions"] == 1
    assert snapshot["total_actions"] == 9
    assert abs(snapshot["percent"] - 100.0 / 9.0) < 1e-12
    assert snapshot["eta_seconds"] == 80.0


def test_family_transfer_distinguishes_own_seed_and_cross_configuration():
    assert experiment_family("GLOBAL_1STEP_U_4") == "GLOBAL_1STEP_U"
    assert experiment_family("G") == "G"
    rows = []
    for source_key, relation in (
        ("a1", "own"),
        ("a2", "other"),
        ("b1", "other"),
        ("farm1", "farm"),
    ):
        for parameter, count, rho in (("lr", 2, 0.2), ("epochs", 6, 0.6)):
            rows.append(
                {
                    "checkpoint_label": "CFG_A_1",
                    "checkpoint_key": "a1",
                    "checkpoint_step": 10,
                    "head": "parameter",
                    "optimizer": "Adam",
                    "parameter": parameter,
                    "source_relation": relation,
                    "source_scope": "experiment",
                    "source_group": source_key,
                    "return_target": "mc",
                    "count": count,
                    "pearson_q_target": rho,
                    "spearman_q_target": rho,
                    "top_q_quartile_standardized_lift": rho,
                    "observed_argmax_fraction": rho,
                    "mean_observed_rank": 2.0,
                    "mean_max_gap": 1.0,
                }
            )
    labels = {"a1": "CFG_A_1", "a2": "CFG_A_2", "b1": "CFG_B_1"}
    pairs = family_transfer_pair_rows(rows, labels)
    assert len(pairs) == 4
    by_source = {row["source_key"]: row for row in pairs}
    assert by_source["a1"]["transfer_scope"] == "exact_own"
    assert by_source["a2"]["transfer_scope"] == "same_family_other"
    assert by_source["b1"]["transfer_scope"] == "cross_family"
    assert by_source["farm1"]["transfer_scope"] == "farm"
    assert abs(by_source["a1"]["spearman_q_target"] - 0.5) < 1e-12
    summaries = family_transfer_summary_rows(pairs)
    assert len(summaries) == 4
    assert all(row["checkpoint_order"] == 1 for row in summaries)
