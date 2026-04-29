from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch

from RL.rl_utils.load_buffer.load_exps_from_comet import (
    WORKSPACE,
    _extract_loss_scalar_from_state,
    _repair_equal_states_from_previous_next_states,
    add_delta_to_all_entries,
    api,
    get_duration_hours,
    get_end_time,
    get_param_value,
    load_single_experiment_transitions,
    shift_done_rewards,
)


OPTIMIZER_NAMES = ["Adam", "LBFGS", "PSO"]
OUTPUT_COLUMNS = [
    "last_optimizer",
    "current_reward",
    "reward_prev_penalty",
    "chain",
]


def _scalar_to_int(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def decode_optimizer_from_action(action, optimizer_names=None):
    optimizer_names = optimizer_names or OPTIMIZER_NAMES

    if isinstance(action, dict):
        opt_type = action.get("type")
        return str(opt_type) if opt_type else "UNKNOWN"

    if isinstance(action, (tuple, list)) and len(action) > 0:
        first = action[0]
        if isinstance(first, str):
            return first

        opt_idx = _scalar_to_int(first)
        if opt_idx is not None and 0 <= opt_idx < len(optimizer_names):
            return optimizer_names[opt_idx]

    return "UNKNOWN"


def _safe_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_reward_prev_penalty(tr, prev_loss, next_loss):
    reward_loss = _safe_float(tr.get("reward_loss"))
    if reward_loss is not None:
        return reward_loss

    reward_model = _safe_float(tr.get("reward_model"))
    if tr.get("done") in (1, -1) and reward_model is not None:
        return reward_model

    loss_prev = _safe_float(tr.get("loss_prev"))
    loss_current = _safe_float(tr.get("loss_current"))
    if loss_prev is not None and loss_current is not None:
        return float(loss_prev - loss_current)

    if prev_loss is not None and next_loss is not None:
        return float(prev_loss - next_loss)

    return None


def add_loss_reward_to_non_terminal_sequence(seq, loss_key="loss_total"):
    if not seq:
        return

    _repair_equal_states_from_previous_next_states(seq, loss_key=loss_key)

    for tr in seq:
        if tr.get("done") in (1, -1):
            continue

        prev_loss = _extract_loss_scalar_from_state(tr["state"], loss_key=loss_key)
        next_loss = _extract_loss_scalar_from_state(tr["next_state"], loss_key=loss_key)
        loss_reward = float(prev_loss - next_loss)

        if "reward_model_original" not in tr and "reward_model" in tr:
            tr["reward_model_original"] = float(tr["reward_model"])
        if "reward_model_raw_original" not in tr and "reward_model_raw" in tr:
            tr["reward_model_raw_original"] = float(tr["reward_model_raw"])

        tr["reward_loss"] = loss_reward
        tr["reward_model"] = loss_reward
        if "reward_model_raw" in tr:
            tr["reward_model_raw"] = loss_reward

        tr["loss_prev"] = float(prev_loss)
        tr["loss_current"] = float(next_loss)


def add_loss_reward_to_non_terminal_transitions(entries, loss_key="loss_total"):
    sequences = []
    curr_seq = []

    for tr in entries:
        curr_seq.append(tr)
        if tr.get("done") in (1, -1):
            sequences.append(curr_seq)
            curr_seq = []

    if curr_seq:
        sequences.append(curr_seq)

    updated = 0
    for seq in sequences:
        add_loss_reward_to_non_terminal_sequence(seq, loss_key=loss_key)
        updated += sum(1 for tr in seq if tr.get("done") not in (1, -1))

    print(
        f"\nRecomputed loss-based reward_model for {updated} "
        f"non-terminal transitions using '{loss_key}'."
    )
    return entries


def build_loss_reward_tolerance_rows(entries, loss_key="loss_total"):
    rows = []
    chain = []

    for tr in entries:
        last_optimizer = decode_optimizer_from_action(tr.get("action"))
        chain.append(last_optimizer)

        prev_loss = _extract_loss_scalar_from_state(tr.get("state"), loss_key=loss_key)
        next_loss = _extract_loss_scalar_from_state(tr.get("next_state"), loss_key=loss_key)
        rows.append(
            {
                "last_optimizer": last_optimizer,
                "current_reward": next_loss,
                "reward_prev_penalty": _get_reward_prev_penalty(tr, prev_loss, next_loss),
                "chain": ", ".join(chain),
            }
        )

        if tr.get("done") in (1, -1):
            chain = []

    return rows


def _resolve_num_workers(num_workers, total_experiments):
    if total_experiments <= 0:
        return 1
    if num_workers is None:
        return min(8, total_experiments)
    return max(1, min(int(num_workers), total_experiments))


def _load_project_transitions(
    proj_name,
    max_exps_last=10,
    duration_grater_hours=1,
    save_dir=None,
    tolerance=0.0,
    prev_tol=0.0,
    use_tol=True,
    num_workers=None,
):
    print(f"\nLoading Comet transitions for project: {proj_name}")
    experiments = list(api.get_experiments(workspace=WORKSPACE, project_name=proj_name))
    experiments_sorted = sorted(experiments, key=get_end_time, reverse=True)
    experiments_sorted_duration = [
        exp
        for exp in experiments_sorted
        if get_duration_hours(exp) >= duration_grater_hours
    ]

    experiments_sorted_duration = experiments_sorted_duration[:max_exps_last]
    if prev_tol > 0.0 and use_tol:
        experiments_selected = [
            exp
            for exp in experiments_sorted_duration
            if float(get_param_value(exp, "tolerance", 0.0)) >= prev_tol
        ]
    elif prev_tol == 0 and use_tol:
        experiments_selected = [
            exp
            for exp in experiments_sorted_duration
            if float(get_param_value(exp, "tolerance", 0.0)) >= tolerance
        ]
    else:
        experiments_selected = experiments_sorted_duration

    print(f"Selected {len(experiments_selected)} experiments.")
    all_transitions = []
    worker_count = _resolve_num_workers(num_workers, len(experiments_selected))
    indexed_experiments = list(enumerate(experiments_selected, 1))

    if worker_count <= 1:
        experiment_results = [
            load_single_experiment_transitions(exp, index, save_dir=save_dir)
            for index, exp in indexed_experiments
        ]
    else:
        experiment_results = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    load_single_experiment_transitions,
                    exp,
                    index,
                    save_dir,
                ): index
                for index, exp in indexed_experiments
            }
            for future in as_completed(future_to_index):
                try:
                    experiment_results.append(future.result())
                except Exception as exc:
                    print(f"Failed to load experiment #{future_to_index[future]}: {exc}")

    experiment_results.sort(key=lambda result: result.index)
    for result in experiment_results:
        if result.error:
            print(f"[{result.index}] {result.exp_name}: {result.error}")
            continue
        all_transitions.extend(result.transitions)
        print(
            f"[{result.index}] {result.exp_name}: "
            f"{len(result.transitions)} transitions "
            f"({len(all_transitions)} total)"
        )

    return all_transitions


def collect_loss_reward_tolerance_rows(
    proj_name,
    max_exps_last=10,
    duration_grater_hours=1,
    save_dir=None,
    tolerance=0.0,
    prev_tol=0.0,
    use_tol=True,
    num_workers=None,
    loss_key="loss_total",
):
    transitions = _load_project_transitions(
        proj_name=proj_name,
        max_exps_last=max_exps_last,
        duration_grater_hours=duration_grater_hours,
        save_dir=save_dir,
        tolerance=tolerance,
        prev_tol=prev_tol,
        use_tol=use_tol,
        num_workers=num_workers,
    )

    if not transitions:
        return []

    transitions = shift_done_rewards(transitions, done=-1, shift_value=-5)
    entries = add_delta_to_all_entries(transitions)
    entries = add_loss_reward_to_non_terminal_transitions(entries, loss_key=loss_key)
    return build_loss_reward_tolerance_rows(entries, loss_key=loss_key)


def collect_loss_reward_tolerance_dataframe(**kwargs):
    rows = collect_loss_reward_tolerance_rows(**kwargs)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
