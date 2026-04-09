import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
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

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        flat = self.gap(x).view(x.size(0), -1)
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
        if x.dim() == 4:
            flat, h = self.encoder(x)
        else:
            flat, h = x, x   # ожидается (B, hidden_dim)

        out = []
        for i, opt_name in enumerate(optim_name_ar):
            heads = self.fc_param_by_opt[opt_name]
            q_dict = {
                pname: heads[pname](h[i:i+1]).squeeze(0)
                for pname in heads
            }
            out.append(q_dict)
        return out