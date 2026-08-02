import copy
from typing import Any, Iterable

import numpy as np
import torch

from landscape_visualization._aux.PINN_loss_data import PINNLossData


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _loss_component_indexes(pde) -> dict[str, list[int]]:
    indexes = {"oper": [], "ic": [], "bnd": []}
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


def _sum_loss_vec(loss_vec: np.ndarray, indexes: list[int]) -> float:
    if not indexes:
        return 0.0
    return float(np.sum(loss_vec[indexes]))


def _as_state_dict(model_or_payload: Any) -> dict:
    if isinstance(model_or_payload, dict):
        if "state_dict" in model_or_payload:
            return model_or_payload["state_dict"]
        return model_or_payload
    if hasattr(model_or_payload, "state_dict"):
        return model_or_payload.state_dict()
    raise TypeError(f"Unsupported solver model payload: {type(model_or_payload)!r}")


def _left_pad(values: list[float], state_len: int | None) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot build raw-loss state from an empty value sequence.")

    if state_len is not None:
        state_len = int(state_len)
        if state_len <= 0:
            raise ValueError(f"state_len must be positive, got {state_len}.")
        values = values[-state_len:]
        if len(values) < state_len:
            values = [values[0]] * (state_len - len(values)) + values

    return torch.tensor(values, dtype=torch.float32)


def build_raw_loss_state_from_solver_models(
    solver_models: Iterable[Any],
    *,
    dde_pde_model,
    state_len: int | None = None,
    log_key: bool = False,
) -> dict[str, torch.Tensor]:
    if solver_models is None:
        raise ValueError("solver_models is required for raw_loss state.")
    if dde_pde_model is None:
        raise ValueError("dde_pde_model is required for raw_loss state.")

    solver_models = list(solver_models)
    if not solver_models:
        raise ValueError("solver_models is empty; cannot build raw_loss state.")

    selected_models = solver_models[-int(state_len):] if state_len is not None else solver_models

    dde_model, loss_weights = dde_pde_model()
    dde_model.compile(
        torch.optim.Adam(dde_model.net.parameters(), lr=0.001),
        loss_weights=loss_weights,
    )
    loss_compute = PINNLossData(dde_model, cache_points=True, use_train=True)
    component_indexes = _loss_component_indexes(dde_model.pde)

    values = {
        "loss_total": [],
        "loss_oper": [],
        "loss_bnd": [],
    }

    for model_payload in selected_models:
        dde_model.net.load_state_dict(copy.deepcopy(_as_state_dict(model_payload)), strict=True)
        loss_dict = loss_compute.evaluate(save_graph=False)
        loss_vec = loss_dict["loss_vec"].detach().cpu().numpy()

        values["loss_total"].append(float(np.sum(loss_vec)))
        values["loss_oper"].append(_sum_loss_vec(loss_vec, component_indexes["oper"]))
        values["loss_bnd"].append(
            _sum_loss_vec(loss_vec, component_indexes["ic"])
            + _sum_loss_vec(loss_vec, component_indexes["bnd"])
        )

    state = {key: _left_pad(key_values, state_len) for key, key_values in values.items()}
    if log_key:
        state = {key: _signed_log1p(value) for key, value in state.items()}

    return state
