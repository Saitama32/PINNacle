"""Checkpoint-compatible, inference-only view of both DQN heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from RL.rl_utils.DQN_classes import ConvEncoder, DQN_optim, DQN_params, DuelingHead


@dataclass(frozen=True)
class CheckpointSpec:
    parameter_representation: str
    network_normalization: str
    group_norm_groups: int
    optimizer_count: int


def _last_dimension(state: dict, suffixes: tuple[str, ...]) -> int | None:
    for key, value in state.items():
        if any(key.endswith(suffix) for suffix in suffixes) and torch.is_tensor(value):
            if value.ndim >= 1:
                return int(value.shape[0])
    return None


def infer_checkpoint_spec(
    optim_state: dict,
    params_state: dict,
    experiment_parameters: dict[str, Any] | None = None,
) -> CheckpointSpec:
    experiment_parameters = experiment_parameters or {}
    params_has_encoder = any(key.startswith("encoder.") for key in params_state)
    representation = str(
        experiment_parameters.get("parameter_representation") or ""
    ).lower()
    if representation not in {"legacy_flat", "separate", "shared_hidden"}:
        representation = "separate" if params_has_encoder else "shared_hidden"

    has_running_stats = any(
        key.startswith("encoder.")
        and (key.endswith("running_mean") or key.endswith("running_var"))
        for key in optim_state
    )
    normalization = str(
        experiment_parameters.get("network_normalization") or ""
    ).lower().replace("_", "")
    if normalization not in {"batchnorm", "groupnorm"}:
        normalization = "batchnorm" if has_running_stats else "groupnorm"

    try:
        groups = int(experiment_parameters.get("group_norm_groups", 8))
    except (TypeError, ValueError):
        groups = 8

    optimizer_count = _last_dimension(
        optim_state,
        (
            "head.adv.2.weight",
            "fc_optim_class.weight",
        ),
    )
    if optimizer_count is None:
        raise ValueError("Could not infer optimizer output size from checkpoint")
    return CheckpointSpec(
        parameter_representation=representation,
        network_normalization=normalization,
        group_norm_groups=groups,
        optimizer_count=optimizer_count,
    )


class EncodedParameterModel(nn.Module):
    """Historical parameter model with an optional private encoder."""

    def __init__(
        self,
        optimizer_dict,
        normalization: str,
        group_norm_groups: int,
    ):
        super().__init__()
        self.encoder = ConvEncoder(
            normalization=normalization,
            group_norm_groups=group_norm_groups,
        )
        self.fc_param_by_opt = nn.ModuleDict()
        for optimizer, parameter_dict in optimizer_dict.items():
            self.fc_param_by_opt[optimizer] = nn.ModuleDict(
                {
                    parameter: DuelingHead(128, len(values))
                    for parameter, values in parameter_dict.items()
                }
            )

    def forward_from_hidden(self, hidden, optimizer: str):
        return {
            name: head(hidden)
            for name, head in self.fc_param_by_opt[optimizer].items()
        }


def _remap_legacy_encoder_keys(state: dict) -> dict:
    remapped = {}
    for key, value in state.items():
        if key.startswith("encoder.backbone."):
            key = key.replace("encoder.backbone.", "encoder.backbone_2d.", 1)
        remapped[key] = value
    return remapped


def _allowed_missing(key: str) -> bool:
    return (
        key.startswith("encoder.backbone_1d.")
        or key.startswith("encoder.backbone_3d.")
    )


def _load_compatible(module: nn.Module, state: dict, label: str) -> None:
    result = module.load_state_dict(_remap_legacy_encoder_keys(state), strict=False)
    missing = [key for key in result.missing_keys if not _allowed_missing(key)]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Incompatible {label} checkpoint: missing={missing}, "
            f"unexpected={result.unexpected_keys}"
        )


class CheckpointHeads(nn.Module):
    """Unified inference API returning Q for every optimizer and parameter."""

    def __init__(self, optimizer_dict, spec: CheckpointSpec):
        super().__init__()
        if len(optimizer_dict) != spec.optimizer_count:
            raise ValueError(
                "Action-space/checkpoint mismatch: config has "
                f"{len(optimizer_dict)} optimizers, checkpoint has "
                f"{spec.optimizer_count}"
            )
        self.optimizer_dict = optimizer_dict
        self.optimizer_names = tuple(optimizer_dict)
        self.spec = spec
        self.model_optim = DQN_optim(
            len(optimizer_dict),
            normalization=spec.network_normalization,
            group_norm_groups=spec.group_norm_groups,
        )
        if spec.parameter_representation == "shared_hidden":
            self.model_params = DQN_params(optimizer_dict)
        else:
            self.model_params = EncodedParameterModel(
                optimizer_dict,
                normalization=spec.network_normalization,
                group_norm_groups=spec.group_norm_groups,
            )

    def load_checkpoint(self, optim_state: dict, params_state: dict) -> None:
        _load_compatible(self.model_optim, optim_state, "optimizer-head")
        _load_compatible(self.model_params, params_state, "parameter-head")
        self.eval()

    @torch.inference_mode()
    def forward_all(self, states: torch.Tensor):
        flat, hidden = self.model_optim.encoder(states)
        optimizer_q = self.model_optim.head(hidden)
        all_parameter_q = {}
        for optimizer in self.optimizer_names:
            if self.spec.parameter_representation == "shared_hidden":
                values = self.model_params(hidden, [optimizer] * len(states))
                all_parameter_q[optimizer] = {
                    name: torch.stack([item[name] for item in values])
                    for name in values[0]
                }
                continue

            if self.spec.parameter_representation == "separate":
                _, parameter_hidden = self.model_params.encoder(states)
            else:
                parameter_hidden = flat
            all_parameter_q[optimizer] = self.model_params.forward_from_hidden(
                parameter_hidden,
                optimizer,
            )
        return optimizer_q, all_parameter_q


def build_checkpoint_heads(
    optimizer_dict,
    optim_state: dict,
    params_state: dict,
    experiment_parameters: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> CheckpointHeads:
    spec = infer_checkpoint_spec(
        optim_state,
        params_state,
        experiment_parameters=experiment_parameters,
    )
    model = CheckpointHeads(optimizer_dict, spec).to(device)
    model.load_checkpoint(optim_state, params_state)
    return model

