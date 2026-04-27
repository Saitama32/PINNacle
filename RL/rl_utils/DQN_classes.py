import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ---- общий экстрактор (оставь свой, если отличается) ----
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_1d = nn.Sequential(
            nn.Conv1d(4, 32, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.backbone_2d = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.backbone_3d = nn.Sequential(
            nn.Conv3d(4, 32, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.gap_1d = nn.AdaptiveAvgPool1d(1)
        self.gap_2d = nn.AdaptiveAvgPool2d(1)
        self.gap_3d = nn.AdaptiveAvgPool3d(1)
        self.mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        if x.dim() < 3:
            raise ValueError(f"Expected state tensor with shape (B, C, *spatial_dims), got {tuple(x.shape)}")

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
        h = self.mlp(flat)              # (B,64)
        return flat, h

# ---- дуэлинговая голова (скалярные Q) ----
class DuelingHead(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.adv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.val = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # удобно для твоей старой визуализации «весов класса»
        self.fc_out_adv = self.adv[-1]

    def forward(self, h):                # (B,hidden_dim) -> (B,A)
        adv = self.adv(h)
        val = self.val(h)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

# ---- выбор оптимизатора (действие верхнего уровня) ----
class DQN_optim(nn.Module):
    def __init__(self, optim_n):
        super().__init__()
        self.encoder = ConvEncoder()
        self.head    = DuelingHead(64, optim_n)
        self.fc_optim_class = self.head.fc_out_adv  # совместимость с твоими графиками

    def forward(self, x):
        flat, h = self.encoder(x)
        q = self.head(h)                 # (B, optim_n)
        return flat, q

# ---- выбор параметров по оптимизатору ----
class DQN_params(nn.Module):
    """
    optimizer_dict = {
        'adam': {'epochs': [...], 'lr': [...], 'betas': [...]},
        'lbfgs': {'epochs': [...], 'history_size': [...], ...},
        ...
    }
    """
    def __init__(self, optimizer_dict):
        super().__init__()
        self.encoder = ConvEncoder()
        self.optimizer_dict = optimizer_dict
        self.fc_param_by_opt = nn.ModuleDict()
        for opt_name, param_dict in optimizer_dict.items():
            self.fc_param_by_opt[opt_name] = nn.ModuleDict()
            for param_name, values in param_dict.items():
                self.fc_param_by_opt[opt_name][param_name] = DuelingHead(64, len(values))

    def forward(self, x, optim_name_ar):
        """
        x: (B,4,*spatial_dims) if raw state maps/volumes are passed.
        Либо подай уже (B,64) — тогда распознаю и не буду повторно кодировать.
        """
        if x.dim() >= 3:
            flat, h = self.encoder(x)
        else:
            flat, h = x, x

        out = []
        for i, opt_name in enumerate(optim_name_ar):
            heads = self.fc_param_by_opt[opt_name]
            q_dict = {pname: heads[pname](h[i:i+1]).squeeze(0) for pname in heads}  # (n_choices,)
            out.append(q_dict)
        return out
