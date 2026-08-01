import copy

import numpy as np
import torch

from landscape_visualization._aux.PINN_loss_data import PINNLossData


def _safe_log1p_signed(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _loss_component_indexes(pde):
    indexes = {
        "oper": [],
        "ic": [],
        "bnd": [],
    }

    for i, config in enumerate(getattr(pde, "loss_config", [])):
        loss_type = str(config.get("type", "")).lower()
        name = str(config.get("name", "")).lower()
        if loss_type == "pde":
            indexes["oper"].append(i)
        elif loss_type in ("ic", "initial") or name.startswith("ic") or "initial" in name:
            indexes["ic"].append(i)
        else:
            indexes["bnd"].append(i)

    return indexes


def _sum_loss_vec(loss_vec, indexes):
    if not indexes:
        return 0.0
    return float(np.sum(loss_vec[indexes]))


def _component_values(loss_vec, pde):
    component_indexes = _loss_component_indexes(pde)
    if component_indexes["oper"] or component_indexes["ic"] or component_indexes["bnd"]:
        loss_oper = _sum_loss_vec(loss_vec, component_indexes["oper"])
        loss_ic = _sum_loss_vec(loss_vec, component_indexes["ic"])
        loss_bnd = _sum_loss_vec(loss_vec, component_indexes["bnd"])
        return loss_oper, loss_ic + loss_bnd

    num_pde = int(getattr(pde, "num_pde", 0))
    return float(np.sum(loss_vec[:num_pde])), float(np.sum(loss_vec[num_pde:]))


def build_raw_loss_state_from_solver_models(
    solver_models,
    *,
    dde_pde_model,
    state_len=None,
    log_key=False,
):
    if not solver_models:
        raise ValueError("Missing solver models for raw-loss state.")
    if dde_pde_model is None:
        raise ValueError("loss_surface_params['dde_pde_model'] is required for raw-loss state.")

    dde_model, loss_weights = dde_pde_model()
    dde_model.compile(
        torch.optim.Adam(dde_model.net.parameters(), lr=0.001),
        loss_weights=loss_weights,
    )
    loss_compute = PINNLossData(dde_model, cache_points=True, use_train=True)

    values = {
        "loss_total": [],
        "loss_oper": [],
        "loss_bnd": [],
    }

    selected_models = solver_models
    if state_len is not None and len(selected_models) > int(state_len):
        selected_models = selected_models[-int(state_len):]

    for solver_model in selected_models:
        dde_model.net.load_state_dict(copy.deepcopy(solver_model.state_dict()), strict=True)
        loss_dict = loss_compute.evaluate(save_graph=False)
        loss_vec = loss_dict["loss_vec"].detach().cpu().numpy()

        loss_oper, loss_bnd = _component_values(loss_vec, dde_model.pde)
        values["loss_total"].append(float(np.sum(loss_vec)))
        values["loss_oper"].append(loss_oper)
        values["loss_bnd"].append(loss_bnd)

    state = {
        key: torch.tensor(value, dtype=torch.float32)
        for key, value in values.items()
    }

    if state_len is not None and len(selected_models) < int(state_len):
        pad = int(state_len) - len(selected_models)
        for key, value in state.items():
            if value.numel() == 0:
                state[key] = torch.zeros(int(state_len), dtype=torch.float32)
            else:
                state[key] = torch.cat([value[:1].repeat(pad), value], dim=0)

    if log_key:
        for key in ("loss_total", "loss_oper", "loss_bnd"):
            state[key] = _safe_log1p_signed(state[key])

    return state
