import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128):
        super().__init__()
        self.backbone_1d = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.backbone_2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.backbone_3d = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.gap_1d = nn.AdaptiveAvgPool1d(1)
        self.gap_2d = nn.AdaptiveAvgPool2d(1)
        self.gap_3d = nn.AdaptiveAvgPool3d(1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = self.backbone_1d(x)
            flat = self.gap_1d(x).view(x.size(0), -1)
        elif x.dim() == 4:
            x = self.backbone_2d(x)
            flat = self.gap_2d(x).view(x.size(0), -1)
        elif x.dim() == 5:
            x = self.backbone_3d(x)
            flat = self.gap_3d(x).view(x.size(0), -1)
        elif x.dim() > 5:
            x = x.flatten(start_dim=2)
            x = self.backbone_1d(x)
            flat = self.gap_1d(x).view(x.size(0), -1)
        else:
            raise ValueError(
                f"Expected state tensor with shape (B, C, *spatial_dims), got {tuple(x.shape)}."
            )

        h = self.mlp(flat)
        return flat, h


def _remap_legacy_encoder_state_dict(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.backbone."):
            key = key.replace("encoder.backbone.", "encoder.backbone_2d.", 1)
        remapped[key] = value
    return remapped


def _load_with_legacy_2d_fallback(module, state_dict, strict=True, assign=False):
    remapped = _remap_legacy_encoder_state_dict(state_dict)
    def load(module_, state_dict_, strict_):
        try:
            return nn.Module.load_state_dict(
                module_,
                state_dict_,
                strict=strict_,
                assign=assign,
            )
        except TypeError:
            return nn.Module.load_state_dict(module_, state_dict_, strict=strict_)

    try:
        return load(module, remapped, strict)
    except RuntimeError as exc:
        legacy_2d = any(key.startswith("encoder.backbone.") for key in state_dict)
        if not legacy_2d:
            raise
        if strict:
            try:
                return load(module, remapped, False)
            except RuntimeError:
                raise exc
        raise


class DuelingHead(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.adv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )
        self.val = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.fc_out_adv = self.adv[-1]

    def forward(self, x):
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def __getitem__(self, index):
        return self.adv[index]


class DQN_optim(nn.Module):
    def __init__(self, optim_n, in_channels=4, hidden_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.head = DuelingHead(hidden_dim, optim_n)
        self.fc_optim_class = self.head.fc_out_adv

    def forward(self, x):
        flat, h = self.encoder(x)
        q = self.head(h)
        return flat, q

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return _load_with_legacy_2d_fallback(self, state_dict, strict=strict, assign=assign)


class DQN_params(nn.Module):
    def __init__(self, optimizer_dict, in_channels=4, hidden_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.optimizer_dict = optimizer_dict
        self.fc_param_by_opt = nn.ModuleDict()

        for opt_name, param_dict in optimizer_dict.items():
            self.fc_param_by_opt[opt_name] = nn.ModuleDict()
            for param_name, values in param_dict.items():
                self.fc_param_by_opt[opt_name][param_name] = DuelingHead(hidden_dim, len(values))

    def forward(self, x, optim_name_ar):
        if x.dim() >= 3:
            flat, h = self.encoder(x)
        else:
            flat, h = x, x

        out = []
        for i, opt_name in enumerate(optim_name_ar):
            heads = self.fc_param_by_opt[opt_name]
            q_dict = {
                pname: heads[pname](h[i:i + 1]).squeeze(0)
                for pname in heads
            }
            out.append(q_dict)
        return out

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return _load_with_legacy_2d_fallback(self, state_dict, strict=strict, assign=assign)
