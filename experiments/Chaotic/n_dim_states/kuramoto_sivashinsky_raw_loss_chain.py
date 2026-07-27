import os

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(PROJECT_ROOT)

import deepxde as dde
import numpy as np
import torch
from comet_ml import API, start
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from landscape_visualization._aux.PINN_loss_data import PINNLossData
from RL.rl_utils.load_buffer.load_exps_from_comet import (
    add_delta_to_sequence,
    collect_comet_transition_entries,
    load_single_experiment_transitions,
)
from RL.rl_utils.load_buffer.rebuild_states_from_solver_models import (
    clone_state_dict,
    restore_solver_models,
)
from src.pde.chaotic import KuramotoSivashinskyEquation
from src.utils.args import parse_hidden_layers


WORKSPACE = "saitama32"
DEFAULT_SOURCE_PROJECT_NAME = "rlpinn-kuramoto-sivashinsky-tolerance"
TARGET_PROJECT_NAME = "rlpinn_ks_rebuild_buffer_raw_loss_no_log"
LOCAL_OUTPUT_DIR = os.path.join("transitions_rebuilt", "kuramoto_sivashinsky_raw_loss")
LOSS_KEYS = ("loss_total", "loss_oper", "loss_bnd")
LOG_LOSS = False


dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)


def build_get_model_kuramoto_sivashinsky(hidden_layers: str):
    def get_model():
        pde = KuramotoSivashinskyEquation()

        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, c in enumerate(pde.loss_config):
            loss_type = c.get("type", "")
            if loss_type in ("boundary", "initial", "ic"):
                loss_weights[i] = 100.0
            elif loss_type == "pde":
                loss_weights[i] = 1.0
            else:
                loss_weights[i] = 1.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model


def signed_log1p_abs(value):
    return torch.sign(value) * torch.log1p(torch.abs(value))


def maybe_transform_loss(value, log_loss):
    return signed_log1p_abs(value) if log_loss else value


def make_zero_state_like(state):
    return {
        key: torch.zeros_like(state[key])
        for key in LOSS_KEYS
    }


def build_loss_compute(get_model, device):
    dde_model, loss_weights = get_model()
    dde_model.net = dde_model.net.float()
    if device.startswith("cuda") and torch.cuda.is_available():
        dde_model.net.to(device)
    dde_model.compile(
        torch.optim.Adam(dde_model.net.parameters(), lr=0.001),
        loss_weights=loss_weights,
    )
    return dde_model, PINNLossData(dde_model, cache_points=True, use_train=True)


def compute_transformed_loss_state(solver_models, dde_model, loss_compute, *, log_loss):
    state = {key: [] for key in LOSS_KEYS}

    for solver_model in solver_models:
        dde_model.net.load_state_dict(solver_model.state_dict(), strict=True)
        loss_dict = loss_compute.evaluate(save_graph=False)

        for key in LOSS_KEYS:
            loss_value = loss_dict[key].detach().cpu().float()
            state[key].append(maybe_transform_loss(loss_value, log_loss))

    return {
        key: torch.stack(values).float()
        for key, values in state.items()
    }


def split_transition_sequences(transitions):
    sequences = []
    current_sequence = []

    for transition in transitions:
        current_sequence.append(transition)
        if int(transition.get("done", 0)) in (1, -1):
            sequences.append(current_sequence)
            current_sequence = []

    if current_sequence:
        sequences.append(current_sequence)

    return sequences


def rebuild_raw_loss_states(
    transitions,
    *,
    get_model,
    device,
    log_loss,
    on_rebuilt_entry=None,
):
    dde_model, loss_compute = build_loss_compute(get_model, device)

    rebuilt_entries = []
    skipped = 0
    loss_time_total = 0.0
    loss_time_count = 0

    def flush_rebuilt_sequence(rebuilt_sequence):
        add_delta_to_sequence(rebuilt_sequence)
        for rebuilt_entry in rebuilt_sequence:
            rebuilt_entries.append(rebuilt_entry)
            if on_rebuilt_entry is not None:
                on_rebuilt_entry(rebuilt_entry, len(rebuilt_entries))

    for seq_i, sequence in enumerate(split_transition_sequences(transitions), 1):
        previous_next_state = None
        rebuilt_sequence = []

        for transition_i, transition in enumerate(sequence):
            try:
                solver_models = restore_solver_models(transition.get("solver_models"))
                loss_started_at = time.perf_counter()
                next_state = compute_transformed_loss_state(
                    solver_models,
                    dde_model,
                    loss_compute,
                    log_loss=log_loss,
                )
                loss_time_total += time.perf_counter() - loss_started_at
                loss_time_count += 1
            except Exception as exc:
                skipped += 1
                print(
                    "Skipping transition during raw-loss rebuild "
                    f"(sequence={seq_i}, index={transition_i}): {exc}"
                )
                flush_rebuilt_sequence(rebuilt_sequence)
                rebuilt_sequence = []
                previous_next_state = None
                continue

            if previous_next_state is None:
                state = make_zero_state_like(next_state)
            else:
                state = clone_state_dict(previous_next_state)

            rebuilt_entry = dict(transition)
            rebuilt_entry["state"] = state
            rebuilt_entry["next_state"] = next_state
            rebuilt_sequence.append(rebuilt_entry)

            previous_next_state = clone_state_dict(next_state)

        flush_rebuilt_sequence(rebuilt_sequence)

    avg_loss_time = loss_time_total / loss_time_count if loss_time_count else 0.0
    print(
        "Rebuilt raw-loss transition states from solver_models: "
        f"{len(rebuilt_entries)} kept, {skipped} skipped. "
        f"loss eval avg: {avg_loss_time:.2f}s over {loss_time_count} runs "
        f"(total {loss_time_total:.2f}s)."
    )
    return rebuilt_entries


def rebuild_single_comet_experiment_raw_loss_transitions(
    *,
    source_project_name,
    source_experiment_key,
    target_experiment,
    output_dir,
    get_model,
    device,
    log_loss,
    workspace=WORKSPACE,
):
    if not source_project_name:
        raise ValueError("source_project_name is required.")
    if not source_experiment_key:
        raise ValueError("source_experiment_key is required.")
    if target_experiment is None:
        raise ValueError("target_experiment is required.")

    os.makedirs(output_dir, exist_ok=True)
    api = API(api_key=os.getenv("COMET_API_KEY"))

    print(
        "Loading source Comet experiment: "
        f"workspace={workspace}, project={source_project_name}, "
        f"experiment={source_experiment_key}"
    )
    source_exp = api.get_experiment(
        workspace=workspace,
        project_name=source_project_name,
        experiment=source_experiment_key,
    )
    if source_exp is None:
        raise ValueError(
            "Source Comet experiment was not found. "
            f"workspace={workspace}, project={source_project_name}, "
            f"experiment={source_experiment_key}"
        )

    load_result = load_single_experiment_transitions(source_exp, index=1)
    if load_result.error:
        raise RuntimeError(load_result.error)
    if not load_result.transitions:
        print("No transitions loaded from source experiment.")
        return []

    print(
        f"Loaded {len(load_result.transitions)} transitions from "
        f"{load_result.exp_name} ({load_result.exp_id})."
    )

    def log_rebuilt_entry(entry, step):
        file_path = os.path.join(output_dir, f"transitions_{step}.pt")
        torch.save(entry, file_path)
        target_experiment.log_asset(
            file_path,
            file_name=f"entry_step_{step}.pt",
            step=step,
            overwrite=True,
        )
        print(f"Logged raw-loss transition to Comet: entry_step_{step}.pt")

    rebuilt_entries = rebuild_raw_loss_states(
        load_result.transitions,
        get_model=get_model,
        device=device,
        log_loss=log_loss,
        on_rebuilt_entry=log_rebuilt_entry,
    )

    print(
        "Raw-loss rebuild buffer upload complete: "
        f"{len(rebuilt_entries)} entries logged to Comet, "
        f"local output_dir={output_dir}"
    )
    return rebuilt_entries


def rebuild_project_comet_raw_loss_transitions(
    *,
    source_project_name,
    target_experiment,
    output_dir,
    get_model,
    device,
    log_loss,
    max_exps_last,
    duration_grater_hours,
    tolerance,
    prev_tol,
    use_tol,
    new_tol,
    num_workers,
    experiment_keys=None,
):
    if not source_project_name:
        raise ValueError("source_project_name is required.")
    if target_experiment is None:
        raise ValueError("target_experiment is required.")

    os.makedirs(output_dir, exist_ok=True)
    if experiment_keys is not None:
        print(
            "Loading source Comet project transitions by explicit experiment keys: "
            f"project={source_project_name}, keys={len(experiment_keys)}"
        )
    else:
        print(
            "Loading source Comet project transitions by filters: "
            f"project={source_project_name}, max_exps_last={max_exps_last}, "
            f"duration_grater_hours={duration_grater_hours}, tolerance={tolerance}, "
            f"prev_tol={prev_tol}, use_tol={use_tol}, new_tol={new_tol}"
        )
    transitions = collect_comet_transition_entries(
        max_exps_last=max_exps_last,
        duration_grater_hours=duration_grater_hours,
        save_dir=None,
        tolerance=tolerance,
        prev_tol=prev_tol,
        use_tol=use_tol,
        new_tol=new_tol,
        proj_name=source_project_name,
        mark_states=None,
        num_workers=num_workers,
        experiment_keys=experiment_keys,
    )
    if not transitions:
        print("No transitions loaded from source project.")
        return []

    print(f"Loaded {len(transitions)} transitions from source project.")

    def log_rebuilt_entry(entry, step):
        file_path = os.path.join(output_dir, f"transitions_{step}.pt")
        torch.save(entry, file_path)
        target_experiment.log_asset(
            file_path,
            file_name=f"entry_step_{step}.pt",
            step=step,
            overwrite=True,
        )
        print(f"Logged raw-loss transition to Comet: entry_step_{step}.pt")

    rebuilt_entries = rebuild_raw_loss_states(
        transitions,
        get_model=get_model,
        device=device,
        log_loss=log_loss,
        on_rebuilt_entry=log_rebuilt_entry,
    )

    print(
        "Raw-loss project rebuild buffer upload complete: "
        f"{len(rebuilt_entries)} entries logged to Comet, "
        f"local output_dir={output_dir}"
    )
    return rebuilt_entries


def normalize_experiment_keys(experiment_keys):
    if experiment_keys is None:
        return None
    keys = tuple(key.strip() for key in experiment_keys if key and key.strip())
    if not keys:
        raise ValueError("--source-experiment-keys was provided but no keys were passed.")
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="kuramoto_sivashinsky_raw_loss_rebuild")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--source-project-name", type=str, default=DEFAULT_SOURCE_PROJECT_NAME)
    parser.add_argument("--source-experiment-key", type=str, default=None)
    parser.add_argument("--source-experiment-keys", nargs="*", default=None)
    parser.add_argument("--target-project-name", type=str, default=TARGET_PROJECT_NAME)
    parser.add_argument("--out", type=str, default=LOCAL_OUTPUT_DIR)
    parser.add_argument("--max-exps-last", type=int, default=10)
    parser.add_argument("--duration-grater-hours", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--prev-tol", type=float, default=0.0)
    parser.add_argument("--new-tol", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-use-tol", action="store_true")

    args = parser.parse_args()

    api_key = os.getenv("COMET_API_KEY")
    experiment = start(
        api_key=api_key,
        project_name=args.target_project_name,
        workspace=WORKSPACE,
    )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    source_experiment_keys = None if args.source_experiment_key else normalize_experiment_keys(args.source_experiment_keys)
    if args.source_experiment_key:
        source_mode = "single_experiment"
        source_run_name = args.source_experiment_key
    elif source_experiment_keys is not None:
        source_mode = "experiment_key_list"
        source_run_name = f"explicit_keys_{len(source_experiment_keys)}"
    else:
        source_mode = "filtered_project"
        source_run_name = args.source_project_name
    output_dir = os.path.join(args.out, source_run_name)

    experiment.log_parameters({
        "param": "raw_loss_v1",
        "description": "rebuild_kuramoto_sivashinsky_buffer_raw_loss_without_autoencoder",
        "source_project_name": args.source_project_name,
        "source_experiment_key": args.source_experiment_key,
        "source_mode": source_mode,
        "source_experiment_key_count": len(source_experiment_keys or []),
        "max_exps_last": args.max_exps_last,
        "duration_grater_hours": args.duration_grater_hours,
        "tolerance": args.tolerance,
        "prev_tol": args.prev_tol,
        "use_tol": not args.no_use_tol,
        "new_tol": args.new_tol,
        "num_workers": args.num_workers,
        "log_loss": LOG_LOSS,
        "state_keys": "/".join([*LOSS_KEYS, "delta"]),
        "loss_transform": "sign(x) * log1p(abs(x))" if LOG_LOSS else "identity",
        "delta_source": "add_delta_to_sequence over logged loss_total states" if LOG_LOSS else "add_delta_to_sequence over raw loss_total states",
        "cache_train_points": True,
        "device": device,
        "local_output_dir": output_dir,
    })

    get_model = build_get_model_kuramoto_sivashinsky(args.hidden_layers)

    if args.source_experiment_key:
        rebuild_single_comet_experiment_raw_loss_transitions(
            source_project_name=args.source_project_name,
            source_experiment_key=args.source_experiment_key,
            target_experiment=experiment,
            output_dir=output_dir,
            get_model=get_model,
            device=device,
            log_loss=LOG_LOSS,
        )
    else:
        rebuild_project_comet_raw_loss_transitions(
            source_project_name=args.source_project_name,
            target_experiment=experiment,
            output_dir=output_dir,
            get_model=get_model,
            device=device,
            log_loss=LOG_LOSS,
            max_exps_last=args.max_exps_last,
            duration_grater_hours=args.duration_grater_hours,
            tolerance=args.tolerance,
            prev_tol=args.prev_tol,
            use_tol=not args.no_use_tol,
            new_tol=args.new_tol,
            num_workers=args.num_workers,
            experiment_keys=source_experiment_keys,
        )


if __name__ == "__main__":
    main()
