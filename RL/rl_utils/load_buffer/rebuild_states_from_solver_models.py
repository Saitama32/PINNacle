import copy
import time

import torch
import deepxde as dde

from landscape_visualization._aux.early_stopping_plot import EarlyStopping
from landscape_visualization._aux.plot_loss_surface import PlotLossSurface
from landscape_visualization._aux.visualization_model import VisualizationModel


def compute_delta_map(loss_t, loss_t1, eps=1e-6):
    raw_delta = loss_t1 - loss_t
    delta = torch.sign(raw_delta) * torch.log1p(torch.abs(raw_delta))
    delta = delta / (delta.abs().max() + eps)
    return delta.clamp(-1, 1)


def infer_fnn_layers(state_dict):
    layer_ids = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("linears.") and key.endswith(".weight")
        }
    )
    if not layer_ids:
        raise ValueError("No linears.N.weight tensors found in state_dict.")

    layers = []
    previous_out = None
    for layer_id in layer_ids:
        weight_key = f"linears.{layer_id}.weight"
        bias_key = f"linears.{layer_id}.bias"
        if bias_key not in state_dict:
            raise ValueError(f"Missing bias tensor: {bias_key}")

        weight = state_dict[weight_key]
        bias = state_dict[bias_key]
        if weight.ndim != 2:
            raise ValueError(f"{weight_key} must be 2D, got {tuple(weight.shape)}")
        if bias.ndim != 1:
            raise ValueError(f"{bias_key} must be 1D, got {tuple(bias.shape)}")

        out_features, in_features = weight.shape
        if bias.shape[0] != out_features:
            raise ValueError(
                f"{bias_key} shape {tuple(bias.shape)} does not match "
                f"{weight_key} out_features={out_features}"
            )
        if previous_out is not None and in_features != previous_out:
            raise ValueError(
                f"Layer {layer_id} input size {in_features} does not match "
                f"previous layer output size {previous_out}"
            )

        if not layers:
            layers.append(in_features)
        layers.append(out_features)
        previous_out = out_features

    return layers


def restore_solver_models(serialized_solver_models):
    if not serialized_solver_models:
        raise ValueError("Missing solver_models.")

    solver_models = []
    for model_state in serialized_solver_models:
        if model_state is None:
            continue
        class_name = model_state.get("class_name")
        state_dict = model_state.get("state_dict")
        if class_name != "FNN":
            raise ValueError(f"Unsupported solver model class: {class_name!r}")
        if not isinstance(state_dict, dict):
            raise ValueError("solver model state_dict must be a dict.")

        state_dict = {
            key: value.detach().to("cpu") if torch.is_tensor(value) else value
            for key, value in state_dict.items()
        }
        net = dde.nn.FNN(infer_fnn_layers(state_dict), "tanh", "Glorot normal").float()
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        solver_models.append(net)

    if not solver_models:
        raise ValueError("No valid solver models were restored.")
    return solver_models


def clone_state_dict(state):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state.items()
    }


def make_zero_state_like(state):
    zero_state = {}
    for key in ("loss_total", "loss_oper", "loss_bnd"):
        if key in state:
            zero_state[key] = torch.zeros_like(state[key])
    if "loss_total" not in zero_state:
        raise ValueError("Cannot build zero state without loss_total.")
    zero_state["delta"] = torch.zeros_like(zero_state["loss_total"])
    return zero_state


def rebuild_next_state_from_solver_models(
    solver_models,
    *,
    AE_model_params,
    AE_train_params,
    loss_surface_params,
    counter,
):
    visualization_model = VisualizationModel(**copy.deepcopy(AE_model_params))

    finetune_AE_model = AE_train_params["finetune_AE_model"]
    AE_params = AE_train_params[
        "other_RL_epoch_AE_params" if finetune_AE_model else "first_RL_epoch_AE_params"
    ]
    cb_es = EarlyStopping(patience=AE_params["patience_scheduler"])

    AEmodel = visualization_model.train(
        AE_train_params["learning_rate"],
        AE_params["cosine_scheduler_patience"],
        AE_params["epochs"],
        AE_train_params["every_epoch"],
        AE_train_params["batch_size"],
        AE_train_params["resume"],
        callbacks=[cb_es],
        solver_models=solver_models,
        finetune_AE_model=finetune_AE_model,
    )

    current_loss_surface_params = copy.deepcopy(loss_surface_params)
    current_loss_surface_params["solver_models"] = solver_models
    current_loss_surface_params["AE_model"] = AEmodel

    plot_loss_surface = PlotLossSurface(**current_loss_surface_params)
    plot_loss_surface.counter = counter
    return plot_loss_surface.save_equation_loss_surface(
        log_key=AE_train_params.get("log_key", False)
    )


def rebuild_transitions_states_from_solver_models(
    transitions,
    *,
    AE_model_params,
    AE_train_params,
    loss_surface_params,
    on_rebuilt_entry=None,
):
    if AE_model_params is None or AE_train_params is None or loss_surface_params is None:
        raise ValueError(
            "AE_model_params, AE_train_params, and loss_surface_params are required "
            "when rebuilding states from solver_models."
        )

    rebuilt_entries = []
    current_sequence = []
    sequences = []
    for transition in transitions:
        current_sequence.append(transition)
        if int(transition.get("done", 0)) in (1, -1):
            sequences.append(current_sequence)
            current_sequence = []
    if current_sequence:
        sequences.append(current_sequence)

    skipped = 0
    rebuild_time_total = 0.0
    rebuild_time_count = 0
    counter = 1
    for seq_i, sequence in enumerate(sequences, 1):
        previous_next_state = None
        for transition_i, transition in enumerate(sequence):
            try:
                solver_models = restore_solver_models(transition.get("solver_models"))
                rebuild_started_at = time.perf_counter()
                next_state = rebuild_next_state_from_solver_models(
                    solver_models,
                    AE_model_params=AE_model_params,
                    AE_train_params=AE_train_params,
                    loss_surface_params=loss_surface_params,
                    counter=counter,
                )
                rebuild_time_total += time.perf_counter() - rebuild_started_at
                rebuild_time_count += 1
            except Exception as exc:
                skipped += 1
                print(
                    "Skipping transition during state rebuild "
                    f"(sequence={seq_i}, index={transition_i}): {exc}"
                )
                previous_next_state = None
                continue

            if previous_next_state is None:
                state = make_zero_state_like(next_state)
            else:
                state = clone_state_dict(previous_next_state)

            next_state["delta"] = compute_delta_map(
                state["loss_total"],
                next_state["loss_total"],
            )
            if "delta" not in state:
                state["delta"] = torch.zeros_like(state["loss_total"])

            rebuilt_entry = dict(transition)
            rebuilt_entry["state"] = state
            rebuilt_entry["next_state"] = next_state
            rebuilt_entries.append(rebuilt_entry)
            if on_rebuilt_entry is not None:
                on_rebuilt_entry(rebuilt_entry, len(rebuilt_entries))

            previous_next_state = clone_state_dict(next_state)
            counter += 1

    avg_rebuild_time = (
        rebuild_time_total / rebuild_time_count
        if rebuild_time_count
        else 0.0
    )
    print(
        "Rebuilt transition states from solver_models: "
        f"{len(rebuilt_entries)} kept, {skipped} skipped. "
        f"next_state rebuild avg: {avg_rebuild_time:.2f}s "
        f"over {rebuild_time_count} runs "
        f"(total {rebuild_time_total:.2f}s)."
    )
    return rebuilt_entries
