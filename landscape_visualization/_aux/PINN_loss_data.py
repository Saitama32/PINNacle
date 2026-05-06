# PINN_loss_data.py
import numpy as np
import torch
from typing import Dict, Optional

class PINNLossData:
    """
    DeepXDE-версия.
    Держит ссылку на dde.Model (уже compile() сделан),
    и умеет возвращать loss_dict в твоём формате.
    """

    def __init__(self, dde_model, *, cache_points: bool = True, use_train: bool = True):
        self.model = dde_model
        self.cache_points = cache_points
        self.use_train = use_train

        self._cached = False
        self._X = None
        self._y = None
        self._aux = None

    def _ensure_points(self):
        if self._cached and self.cache_points:
            return

        if self.use_train:
            X, y, aux = self.model.data.train_next_batch(None)
        else:
            X, y = self.model.data.test()
            aux = None

        self._X, self._y, self._aux = X, y, aux
        self._cached = True

    def evaluate(self, save_graph: bool = False) -> Dict[str, torch.Tensor]:
        """
        Возвращает dict с ключами как раньше:
        loss_total, loss_oper, loss_bnd, loss_normalized (+ при желании operator/bval_diff можно добавить).
        """

        self._ensure_points()

        # DeepXDE отдаёт (y_pred, losses_vector) в numpy
        _, loss_vec = self.model._outputs_losses(
            True if self.use_train else False,
            self._X,
            self._y,
            self._aux,
        )
        loss_vec = np.asarray(loss_vec, dtype=np.float64)  # shape (num_loss,)

        num_pde = getattr(self.model.pde, "num_pde", None)
        if num_pde is None:
            raise AttributeError("dde_model.pde.num_pde not found. Expected PINNacle BasePDE-like object.")

        loss_oper = float(np.sum(loss_vec[:num_pde]))
        loss_bnd = float(np.sum(loss_vec[num_pde:]))
        loss_total = float(np.sum(loss_vec))

        lw = getattr(self.model.losshistory, "loss_weights", None)
        if lw is not None:
            lw = np.asarray(lw, dtype=np.float64)
            loss_vec_norm = loss_vec / (lw + 1e-30)
            loss_normalized = float(np.sum(loss_vec_norm))
        else:
            loss_normalized = loss_total

        # save_graph у тебя раньше использовался чтобы detach и чистить cuda
        return {
            "loss_total": torch.tensor(loss_total, dtype=torch.float32),
            "loss_oper": torch.tensor(loss_oper, dtype=torch.float32),
            "loss_bnd": torch.tensor(loss_bnd, dtype=torch.float32),
            "loss_normalized": torch.tensor(loss_normalized, dtype=torch.float32),
            "loss_vec": torch.tensor(loss_vec, dtype=torch.float32),
        }
    

def extract_layers_from_dde_fnn(net: torch.nn.Module):
    linears = None
    if hasattr(net, "linears"):
        linears = getattr(net, "linears")
    elif hasattr(net, "layers"):
        linears = getattr(net, "layers")

    if linears is None:
        # fallback: ищем все Linear в порядке обхода
        linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]

    if not linears:
        raise ValueError("Can't find Linear layers inside dde FNN net.")

    sizes = [linears[0].in_features] + [l.out_features for l in linears]
    return sizes


def get_PINN(layer_sizes, device):
    layers = []
    for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
        layer = torch.nn.Linear(i, j)
        layers.append(layer)
        layers.append(torch.nn.Tanh())
    layers = layers[:-1]
    return torch.nn.Sequential(*layers).to(device)
