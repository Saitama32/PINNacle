import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = 128

        self.backbone_1d = self._make_backbone(nn.Conv1d, nn.BatchNorm1d, in_channels)
        self.backbone_2d = self._make_backbone(nn.Conv2d, nn.BatchNorm2d, in_channels)
        self.backbone_3d = self._make_backbone(nn.Conv3d, nn.BatchNorm3d, in_channels)

        self.gap_1d = nn.AdaptiveAvgPool1d(1)
        self.gap_2d = nn.AdaptiveAvgPool2d(1)
        self.gap_3d = nn.AdaptiveAvgPool3d(1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    @staticmethod
    def _make_backbone(conv_cls, norm_cls, in_channels):
        return nn.Sequential(
            conv_cls(in_channels, 32, kernel_size=3, stride=1, padding=1),
            norm_cls(32),
            nn.ReLU(inplace=True),
            conv_cls(32, 64, kernel_size=3, stride=2, padding=1),
            norm_cls(64),
            nn.ReLU(inplace=True),
            conv_cls(64, 128, kernel_size=3, stride=2, padding=1),
            norm_cls(128),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() < 3:
            raise ValueError(
                f"Expected state tensor with shape (B, C, *spatial_dims), got {tuple(x.shape)}"
            )

        spatial_ndim = x.dim() - 2
        if spatial_ndim == 1:
            x = self.backbone_1d(x)
            flat = self.gap_1d(x).view(x.size(0), -1)
        elif spatial_ndim == 2:
            x = self.backbone_2d(x)
            flat = self.gap_2d(x).view(x.size(0), -1)
        elif spatial_ndim == 3:
            x = self.backbone_3d(x)
            flat = self.gap_3d(x).view(x.size(0), -1)
        else:
            x = x.flatten(start_dim=2)
            x = self.backbone_1d(x)
            flat = self.gap_1d(x).view(x.size(0), -1)

        h = self.mlp(flat)
        return flat, h


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

    def __getitem__(self, index):
        return self.adv[index]

    def forward(self, x):
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


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
                param_name: heads[param_name](h[i:i + 1]).squeeze(0)
                for param_name in heads
            }
            out.append(q_dict)
        return out
