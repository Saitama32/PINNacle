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
from src.pde.heat import Heat2D_Multiscale
from src.utils.args import parse_hidden_layers


WORKSPACE = "saitama32"
DEFAULT_SOURCE_PROJECT_NAME = "rlpinn-heat2d-multiscale-tolerance-with-models"
TARGET_PROJECT_NAME = "rlpinn_heat2d_multiscale_rebuild_buffer_raw_loss"
LOCAL_OUTPUT_DIR = os.path.join("transitions_rebuilt", "heat2d_multiscale_raw_loss")
LOSS_KEYS = ("loss_total", "loss_oper", "loss_bnd")
DEFAULT_SOURCE_EXPERIMENT_KEYS = (
    "daaacfec31b948da937d2a59c58e690e",
    "ef910a5d5a52437a89ac65f597a1e80d",
    "716952811ba14bfe82858f284803f2d1",
    "a5f025273e974d598d22bc8f180f2e5d",
    "d76ef7bc98fe477da2eb6da77d4c295a",
    "c35c23091bf44ce3bcc830fbf8676b6b",
    "77c4cc4eb29647e69571c08fe45693db",
    "0713a1068ca04b398c7cc95125e95b16",
    "fd70101c66344f089ce1350f98791e18",
    "a85dd943edc04beeafea69b6cfc1e425",
    "51ede5dafa354b688439c0f46d56cd10",
    "37209b01c3f241bc9022323242d8e096",
    "6c395024763a4ab3ae2779e1d8efad76",
    "0cb6b60d4af14c2ca1a0260b62ebdfa2",
    "bfb397f2ac4043f9846eda6116a4c73c",
    "f261d270af61461f99ba27370d07fbf5",
    "e9ffe2b0afa54bb39711b5389d98b76b",
    "8f16582064f94b2b88fe31c02317e957",
    "abf6bf2959f746aebe95bd42b27b04fc",
    "8d149b9adc6b4747ada196f2b251905b",
    "61e3fb0cf6094c6cbf1cccc3519d79c2",
    "eb40bb2a134f44d8ada09b707d08e957",
    "89e579090b25478fbea30aa334a70ad0",
    "32063b9747b5497e9be2c58677d9b81b",
    "f4bf8ff90134439e9137ebf43a689dc5",
    "0c2ca8b6f7c94c88aa83f35c5e9f531b",
    "47467044f1ed4f048d054a1c80406e62",
    "58a2a82fc18945faa8e40e3a66cfb153",
    "5c87b095fda343318cc89e8732f226f4",
    "9b5703254409435cb7966c9dba9ae267",
    "de58aaa3fc344a17ac4bf532181d402c",
    "bff2f5cf6d414523bf0535e6af881bac",
    "dcf22662f57a456498caa93718a4161d",
    "b0ab6c45c1424134be97a25352427b90",
    "17eb0bdd4e9b4d27acb58798f11a29c9",
    "c3c993bee8ed48dc9371c849b6a507e0",
    "66fb9407b95f4574832853f28c570425",
    "eb75d06d4236478ea33f68f73fec05b9",
    "3a1943db792047be9a96c69261398389",
    "8352624d124d40b6b8075def8cb078ce",
    "b50813462ac04ac3b197a4588d513892",
    "1c93fed1aeda45e7a1197b3cc69717b7",
    "fafe8b1b120c4fae8482b1ef4f0c6fad",
    "615597cf17c9433c8c4bb95a2ab70846",
    "231c4f2e6ca54372bc23158e0defade5",
    "e3072079296a4fd2b14070a5a76d0fdc",
    "b9bef4c4130b44bc8f73d4bc679d16c9",
    "2f81465f60034f799af0f932262c3275",
    "af880862baee4353b01798cddbb82e9e",
    "e36cf040f9be43c7a2a96be8133e2a36",
    "6433273f30f047aa8473fd05f397692c",
    "b79cc090e38546cab8ad49a61db9b387",
    "29104361cd474814b11f3f689408ac0f",
    "92abc752f99e41dcafe2ac39f74b7122",
    "8dc6414975fa4e1ab885f02105e2ebc1",
    "7b271fbfb7dc48efb296c65bc87283a3",
    "3623e5c5e5c549e2b2b4209cb2e93592",
    "ca918a90aada422bb2a0d4ab82df702c",
    "f73ac339f04344348b19e472f77c2b15",
    "4bbf8db23b164005bb19eb30063f23ec",
    "de1edfe7110549b3979e535bf4c6a326",
    "cad86879e8f6418187e8ac0a05e09d08",
    "645bde1dc2a14b19ab50da15ed8be06e",
    "26ec5857d2704ed7aeff408f4d02a4c7",
    "bf1ef1fc2eff4acc829d15c38c57cd5f",
    "460766e4455843da8b99da373a018251",
    "e8f113d6899541d9913eaf15a64fe5c5",
    "5e6a1fc431364ede8e0c34be1d07fd0c",
    "84c92b8d44074391bb101696d34210ef",
    "3a2bf300100e49b3adbf5f7a79e0a35b",
    "9cbc7b81448947ff9930073393937bc1",
    "2f64eee884034d4aab1f8dac949f1316",
    "1f6c543d48cc47b6a71a2cb02efceae2",
    "205ffc2dbbad482885f786b10c35ae5d",
    "34091af5401c4ea6908e7efa632d8e20",
    "2535731c44eb41db9f2da5b63b7280d8",
    "41fce27ab3fa42cdb97749249de38df6",
    "16cd5f86687c493e8886650e010d0fa2",
    "b7bbc36c3b20436095b1eae14c8077c4",
    "75b0b0c7e0bb4e8fb5a6c809760f2d85",
)


dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)


def build_get_model_heat2d_multiscale(hidden_layers: str, **pde_kwargs):
    def get_model():
        pde = Heat2D_Multiscale(**pde_kwargs)

        layers = [pde.input_dim] + parse_hidden_layers(argparse.Namespace(hidden_layers=hidden_layers)) + [pde.output_dim]
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, c in enumerate(pde.loss_config):
            loss_type = c.get("type", "")
            if loss_type in ("boundary", "initial"):
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


def compute_transformed_loss_state(solver_models, dde_model, loss_compute):
    state = {key: [] for key in LOSS_KEYS}

    for solver_model in solver_models:
        dde_model.net.load_state_dict(solver_model.state_dict(), strict=True)
        loss_dict = loss_compute.evaluate(save_graph=False)

        for key in LOSS_KEYS:
            loss_value = loss_dict[key].detach().cpu().float()
            state[key].append(signed_log1p_abs(loss_value))

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
        on_rebuilt_entry=log_rebuilt_entry,
    )

    print(
        "Raw-loss project rebuild buffer upload complete: "
        f"{len(rebuilt_entries)} entries logged to Comet, "
        f"local output_dir={output_dir}"
    )
    return rebuilt_entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="heat2d_multiscale_raw_loss_rebuild")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--source-project-name", type=str, default=DEFAULT_SOURCE_PROJECT_NAME)
    parser.add_argument("--source-experiment-key", type=str, default=None)
    parser.add_argument("--target-project-name", type=str, default=TARGET_PROJECT_NAME)
    parser.add_argument("--out", type=str, default=LOCAL_OUTPUT_DIR)
    parser.add_argument("--max-exps-last", type=int, default=10)
    parser.add_argument("--duration-grater-hours", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--prev-tol", type=float, default=0.0)
    parser.add_argument("--new-tol", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-use-tol", action="store_true")
    parser.add_argument("--pde-coef-x", type=float, default=1 / (500 * np.pi) ** 2, help="PDE coefficient for x diffusion")
    parser.add_argument("--pde-coef-y", type=float, default=1 / (np.pi ** 2), help="PDE coefficient for y diffusion")
    parser.add_argument("--init-coef-x", type=float, default=20 * np.pi, help="Initial condition frequency in x")
    parser.add_argument("--init-coef-y", type=float, default=np.pi, help="Initial condition frequency in y")

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
    source_experiment_keys = None if args.source_experiment_key else DEFAULT_SOURCE_EXPERIMENT_KEYS
    source_run_name = args.source_experiment_key or args.source_project_name
    output_dir = os.path.join(args.out, source_run_name)

    experiment.log_parameters({
        "param": "raw_loss_v1",
        "description": "rebuild_heat2d_multiscale_buffer_raw_loss_without_autoencoder",
        "source_project_name": args.source_project_name,
        "source_experiment_key": args.source_experiment_key,
        "source_mode": "single_experiment" if args.source_experiment_key else "experiment_key_list",
        "source_experiment_key_count": len(source_experiment_keys or []),
        "max_exps_last": args.max_exps_last,
        "duration_grater_hours": args.duration_grater_hours,
        "tolerance": args.tolerance,
        "prev_tol": args.prev_tol,
        "use_tol": not args.no_use_tol,
        "new_tol": args.new_tol,
        "num_workers": args.num_workers,
        "state_keys": "/".join([*LOSS_KEYS, "delta"]),
        "loss_transform": "sign(x) * log1p(abs(x))",
        "delta_source": "add_delta_to_sequence over transformed loss_total states",
        "cache_train_points": True,
        "device": device,
        "local_output_dir": output_dir,
    })

    pde_kwargs = dict(
        pde_coef=(args.pde_coef_x, args.pde_coef_y),
        init_coef=(args.init_coef_x, args.init_coef_y),
    )
    get_model = build_get_model_heat2d_multiscale(args.hidden_layers, **pde_kwargs)

    if args.source_experiment_key:
        rebuild_single_comet_experiment_raw_loss_transitions(
            source_project_name=args.source_project_name,
            source_experiment_key=args.source_experiment_key,
            target_experiment=experiment,
            output_dir=output_dir,
            get_model=get_model,
            device=device,
        )
    else:
        rebuild_project_comet_raw_loss_transitions(
            source_project_name=args.source_project_name,
            target_experiment=experiment,
            output_dir=output_dir,
            get_model=get_model,
            device=device,
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
