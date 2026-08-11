from dataclasses import dataclass

import torch


@dataclass
class WeightGroup:
    group_id: int
    layer_name: str
    parameter_name: str
    parameter: torch.nn.Parameter
    flat_start: int
    flat_end: int
    weight_offset: int
    is_frozen: bool = False

    @property
    def num_weights(self):
        return self.flat_end - self.flat_start

    def values(self):
        return self.parameter.detach().reshape(-1)[self.flat_start : self.flat_end]


class WeightGroupCollection:
    """Stable groups that never cross parameter-matrix boundaries."""

    def __init__(self, module, group_size):
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.module = module
        self.group_size = int(group_size)
        self.groups = []
        self.weight_parameters = []
        self._groups_by_parameter = {}
        offset = 0
        for parameter_name, parameter in module.named_parameters():
            # In the supported PINNs, matrices are weights and vectors are biases.
            if parameter.ndim < 2 or not parameter.requires_grad:
                continue
            self.weight_parameters.append((parameter_name, parameter, offset))
            parameter_groups = []
            for start in range(0, parameter.numel(), self.group_size):
                group = WeightGroup(
                    group_id=len(self.groups),
                    layer_name=parameter_name.rsplit(".", 1)[0],
                    parameter_name=parameter_name,
                    parameter=parameter,
                    flat_start=start,
                    flat_end=min(start + self.group_size, parameter.numel()),
                    weight_offset=offset + start,
                )
                self.groups.append(group)
                parameter_groups.append(group)
            self._groups_by_parameter[parameter] = parameter_groups
            offset += parameter.numel()
        self.num_weights = offset
        if not self.groups:
            raise ValueError("No trainable weight matrices were found")

    def __len__(self):
        return len(self.groups)

    def groups_for(self, parameter):
        return self._groups_by_parameter.get(parameter, ())

    @property
    def frozen_groups(self):
        return [group for group in self.groups if group.is_frozen]

    @property
    def trainable_groups(self):
        return [group for group in self.groups if not group.is_frozen]

    def set_frozen(self, group_ids):
        frozen = {int(group_id) for group_id in group_ids}
        unknown = frozen.difference(range(len(self.groups)))
        if unknown:
            raise ValueError(f"Unknown weight group ids: {sorted(unknown)}")
        for group in self.groups:
            group.is_frozen = group.group_id in frozen

    def metadata(self):
        return [
            {
                "group_id": group.group_id,
                "layer_name": group.layer_name,
                "parameter_name": group.parameter_name,
                "flat_start": group.flat_start,
                "flat_end": group.flat_end,
                "num_weights": group.num_weights,
                "is_frozen": group.is_frozen,
            }
            for group in self.groups
        ]

    def apply_group_delta(self, group, delta):
        if delta.numel() != group.num_weights:
            raise ValueError("Group delta has an incompatible size")
        with torch.no_grad():
            flat = group.parameter.reshape(-1)
            flat[group.flat_start : group.flat_end].add_(delta.reshape(-1))

    def frozen_mask_for(self, parameter):
        groups = self.groups_for(parameter)
        if not groups or not any(group.is_frozen for group in groups):
            return None
        mask = torch.zeros(parameter.numel(), dtype=torch.bool, device=parameter.device)
        for group in groups:
            if group.is_frozen:
                mask[group.flat_start : group.flat_end] = True
        return mask.reshape(parameter.shape)
