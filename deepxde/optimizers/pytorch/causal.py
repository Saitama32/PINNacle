from contextlib import contextmanager

import numpy as np
import torch


class CausalOptimizer(torch.optim.Optimizer):
    """Causality-aware wrapper for PyTorch optimizers.

    Modes:
        bc_mode="all": IC/BC are always active, PDE points are causal.
        bc_mode="causal": IC is always active, BC and PDE points are causal.
    """

    def __init__(
        self,
        base_optimizer,
        base_optimizer_name,
        n_time_bins=20,
        start_bins=1,
        time_index=-1,
        unlock_every=1000,
        unlock_tol=None,
        min_steps_per_bin=200,
        bc_mode="causal",
        min_points_per_bc=1,
        causal_strategy="prefix",
        steps_per_window=200,
        state_alpha=0.8,
        x_state=None,
        window_ic_weight=100.0,
        verbose=False,
    ):
        self.base_optimizer = base_optimizer
        self.base_optimizer_name = base_optimizer_name
        self.name = "Causal"

        self.n_time_bins = int(n_time_bins)
        self.active_bins = int(start_bins)
        self.time_index = int(time_index)
        self.unlock_every = unlock_every
        self.unlock_tol = unlock_tol
        self.min_steps_per_bin = int(min_steps_per_bin)
        self.bc_mode = bc_mode
        self.min_points_per_bc = int(min_points_per_bc)
        self.causal_strategy = causal_strategy
        self.steps_per_window = int(steps_per_window)
        self.state_alpha = float(state_alpha)
        self.x_state = x_state
        self.window_ic_weight = float(window_ic_weight)
        self.verbose = bool(verbose)

        self.global_step = 0
        self.steps_since_unlock = 0
        self.current_window = 0
        self.current_cycle = 0
        self.steps_in_window = 0
        self.request_state_update = False
        self.window_states = None
        self.model = None
        self.last_loss = None
        self.losses = None
        self._thresholds = None
        self._t_min = None
        self._t_max = None

        if self.n_time_bins < 1:
            raise ValueError("n_time_bins must be >= 1")
        if self.active_bins < 1:
            raise ValueError("start_bins must be >= 1")
        if self.bc_mode not in ["all", "causal"]:
            raise ValueError('bc_mode must be either "all" or "causal"')
        if self.min_points_per_bc < 0:
            raise ValueError("min_points_per_bc must be >= 0")
        if self.causal_strategy not in ["prefix", "cyclic_windows"]:
            raise ValueError('causal_strategy must be "prefix" or "cyclic_windows"')
        if self.steps_per_window < 1:
            raise ValueError("steps_per_window must be >= 1")
        if not 0.0 <= self.state_alpha <= 1.0:
            raise ValueError("state_alpha must be in [0, 1]")
        self.active_bins = min(self.active_bins, self.n_time_bins)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @property
    def state(self):
        return self.base_optimizer.state

    def __getattr__(self, name):
        return getattr(self.base_optimizer, name)

    def zero_grad(self, *args, **kwargs):
        return self.base_optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "base_optimizer_name": self.base_optimizer_name,
            "n_time_bins": self.n_time_bins,
            "active_bins": self.active_bins,
            "time_index": self.time_index,
            "unlock_every": self.unlock_every,
            "unlock_tol": self.unlock_tol,
            "min_steps_per_bin": self.min_steps_per_bin,
            "bc_mode": self.bc_mode,
            "min_points_per_bc": self.min_points_per_bc,
            "causal_strategy": self.causal_strategy,
            "steps_per_window": self.steps_per_window,
            "state_alpha": self.state_alpha,
            "x_state": self.x_state,
            "window_ic_weight": self.window_ic_weight,
            "current_window": self.current_window,
            "current_cycle": self.current_cycle,
            "steps_in_window": self.steps_in_window,
            "window_states": self.window_states,
            "global_step": self.global_step,
            "steps_since_unlock": self.steps_since_unlock,
            "last_loss": self.last_loss,
            "thresholds": self._thresholds,
            "t_min": self._t_min,
            "t_max": self._t_max,
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.active_bins = state_dict.get("active_bins", self.active_bins)
        self.global_step = state_dict.get("global_step", self.global_step)
        self.steps_since_unlock = state_dict.get(
            "steps_since_unlock", self.steps_since_unlock
        )
        self.last_loss = state_dict.get("last_loss", self.last_loss)
        self.bc_mode = state_dict.get("bc_mode", self.bc_mode)
        self.min_points_per_bc = state_dict.get(
            "min_points_per_bc", self.min_points_per_bc
        )
        self.causal_strategy = state_dict.get("causal_strategy", self.causal_strategy)
        self.steps_per_window = state_dict.get(
            "steps_per_window", self.steps_per_window
        )
        self.state_alpha = state_dict.get("state_alpha", self.state_alpha)
        self.x_state = state_dict.get("x_state", self.x_state)
        self.window_ic_weight = state_dict.get(
            "window_ic_weight", self.window_ic_weight
        )
        self.current_window = state_dict.get("current_window", self.current_window)
        self.current_cycle = state_dict.get("current_cycle", self.current_cycle)
        self.steps_in_window = state_dict.get("steps_in_window", self.steps_in_window)
        self.window_states = state_dict.get("window_states", self.window_states)
        self._thresholds = state_dict.get("thresholds", self._thresholds)
        self._t_min = state_dict.get("t_min", self._t_min)
        self._t_max = state_dict.get("t_max", self._t_max)

    def attach_model(self, model):
        self.model = model

    def _to_numpy_time(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()[:, self.time_index]
        return x[:, self.time_index]

    def _take(self, x, idx):
        if x is None:
            return None
        if isinstance(x, tuple):
            return x
        if torch.is_tensor(x):
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=x.device)
            return x[idx_t]
        return x[idx]

    def _concat(self, blocks):
        blocks = [block for block in blocks if len(block) > 0]
        if not blocks:
            return None
        if torch.is_tensor(blocks[0]):
            return torch.cat(blocks, dim=0)
        return np.concatenate(blocks, axis=0)

    def _is_ic(self, bc):
        return bc.__class__.__name__ == "IC"

    def _init_time_bins(self, inputs, data):
        if hasattr(data, "num_bcs") and data.num_bcs is not None:
            n_bc = int(np.sum(data.num_bcs))
            x_time_source = inputs[n_bc:]
            if len(x_time_source) == 0:
                x_time_source = inputs
        else:
            x_time_source = inputs

        t = self._to_numpy_time(x_time_source)
        self._t_min = float(np.min(t))
        self._t_max = float(np.max(t))
        self._thresholds = np.linspace(
            self._t_min,
            self._t_max,
            self.n_time_bins + 1,
            dtype=t.dtype,
        )[1:]

        if self.verbose:
            print(
                "[CausalOptimizer] "
                f"base={self.base_optimizer_name}, "
                f"bc_mode={self.bc_mode}, "
                f"strategy={self.causal_strategy}, "
                f"time=[{self._t_min:.6g}, {self._t_max:.6g}], "
                f"bins={self.n_time_bins}, "
                f"active={self.active_bins}"
            )

    def current_t_threshold(self):
        if self._thresholds is None:
            return None
        return self._thresholds[self.active_bins - 1]

    def current_window_bounds(self):
        if self._thresholds is None:
            return None, None
        if self.causal_strategy == "prefix":
            return self._t_min, self.current_t_threshold()

        left = self._t_min
        if self.current_window > 0:
            left = self._thresholds[self.current_window - 1]
        right = self._thresholds[self.current_window]
        return left, right

    def _active_time_indices(self, block):
        t = self._to_numpy_time(block)
        if self.causal_strategy == "prefix":
            return np.where(t <= self.current_t_threshold())[0]

        t_left, t_right = self.current_window_bounds()
        return np.where((t >= t_left) & (t <= t_right))[0]

    def _ensure_min_bc_points(self, block, idx):
        if len(idx) > 0 or self.min_points_per_bc <= 0 or len(block) == 0:
            return idx
        t = self._to_numpy_time(block)
        keep = min(self.min_points_per_bc, len(block))
        return np.argsort(t)[:keep]

    def _build_active_data(self, inputs, targets, data):
        if isinstance(inputs, tuple):
            return inputs, targets, None

        if not hasattr(data, "num_bcs") or data.num_bcs is None:
            return inputs, targets, None

        if self._thresholds is None:
            self._init_time_bins(inputs, data)

        old_num_bcs = list(data.num_bcs)
        bcs = getattr(data, "bcs", [None] * len(old_num_bcs))

        active_blocks = []
        active_indices_global = []
        active_num_bcs = []
        start = 0

        for bc, n in zip(bcs, old_num_bcs):
            n = int(n)
            end = start + n
            block = inputs[start:end]

            if self._is_ic(bc):
                if self.causal_strategy == "cyclic_windows" and self.current_window > 0:
                    local_idx = np.array([], dtype=int)
                else:
                    local_idx = np.arange(n)
            elif self.bc_mode == "all" and self.causal_strategy == "prefix":
                local_idx = np.arange(n)
            else:
                local_idx = self._active_time_indices(block)
                local_idx = self._ensure_min_bc_points(block, local_idx)

            active_blocks.append(self._take(block, local_idx))
            active_num_bcs.append(len(local_idx))
            active_indices_global.append(start + local_idx)
            start = end

        x_pde = inputs[start:]
        if len(x_pde) > 0:
            pde_local_idx = self._active_time_indices(x_pde)
            active_blocks.append(self._take(x_pde, pde_local_idx))
            active_indices_global.append(start + pde_local_idx)

        active_inputs = self._concat(active_blocks)
        if active_inputs is None:
            return inputs, targets, None

        active_indices_global = np.concatenate(active_indices_global)
        if targets is None:
            active_targets = None
        else:
            try:
                active_targets = self._take(targets, active_indices_global)
            except Exception:
                active_targets = targets

        return active_inputs, active_targets, active_num_bcs

    @contextmanager
    def causal_context(self, inputs, targets, data):
        active_inputs, active_targets, active_num_bcs = self._build_active_data(
            inputs, targets, data
        )
        old_num_bcs = None
        old_train_x = None

        if active_num_bcs is not None:
            old_num_bcs = data.num_bcs
            old_train_x = getattr(data, "train_x", None)
            data.num_bcs = active_num_bcs
            data.train_x = active_inputs

        try:
            yield active_inputs, active_targets
        finally:
            if old_num_bcs is not None:
                data.num_bcs = old_num_bcs
                data.train_x = old_train_x

    def _extract_loss_value(self, loss):
        if loss is None:
            return None

        try:
            return float(loss.detach().cpu())
        except Exception:
            pass

        try:
            return float(loss)
        except Exception:
            return None

    def _maybe_unlock(self):
        if self.causal_strategy != "prefix":
            return False

        if self.active_bins >= self.n_time_bins:
            return False

        self.steps_since_unlock += 1
        unlock_by_steps = (
            self.unlock_every is not None
            and self.steps_since_unlock >= self.unlock_every
        )
        unlock_by_tol = (
            self.unlock_tol is not None
            and self.last_loss is not None
            and self.last_loss <= self.unlock_tol
            and self.steps_since_unlock >= self.min_steps_per_bin
        )

        if unlock_by_steps or unlock_by_tol:
            self.active_bins += 1
            self.steps_since_unlock = 0

            if self.verbose:
                threshold = self.current_t_threshold()
                print(
                    "[CausalOptimizer] "
                    f"unlocked bin {self.active_bins}/{self.n_time_bins}, "
                    f"t <= {threshold:.6g}"
                )
            return True

        return False

    def _init_window_states_if_needed(self):
        if self.window_states is None:
            self.window_states = [None for _ in range(self.n_time_bins + 1)]

    def _make_state_points(self, t_value):
        if self.x_state is None:
            raise RuntimeError("x_state must be provided for cyclic_windows strategy.")

        if torch.is_tensor(self.x_state):
            x_np = self.x_state.detach().cpu().numpy()
        else:
            x_np = np.asarray(self.x_state)

        t_col = np.full((x_np.shape[0], 1), float(t_value), dtype=x_np.dtype)
        return np.concatenate([x_np, t_col], axis=1)

    def _predict_state_at(self, t_value):
        if self.model is None:
            raise RuntimeError("CausalOptimizer needs attach_model(model).")

        with torch.no_grad():
            return self.model.predict(self._make_state_points(t_value))

    def _update_next_window_state(self):
        self._init_window_states_if_needed()
        _, t_right = self.current_window_bounds()
        next_state = self._predict_state_at(t_right)
        idx = self.current_window + 1

        if self.window_states[idx] is None:
            self.window_states[idx] = next_state
        else:
            self.window_states[idx] = (
                self.state_alpha * self.window_states[idx]
                + (1.0 - self.state_alpha) * next_state
            )

        if self.verbose:
            print(
                "[CausalOptimizer] "
                f"updated state[{idx}] with alpha={self.state_alpha}"
            )

    def _advance_window(self):
        self.steps_in_window = 0
        self.current_window += 1

        if self.current_window >= self.n_time_bins:
            self.current_window = 0
            self.current_cycle += 1

        if self.verbose:
            t_left, t_right = self.current_window_bounds()
            print(
                "[CausalOptimizer] "
                f"cycle={self.current_cycle}, "
                f"window={self.current_window + 1}/{self.n_time_bins}, "
                f"t in [{t_left:.6g}, {t_right:.6g}]"
            )

    def after_train_step(self):
        if self.causal_strategy != "cyclic_windows":
            return
        if not self.request_state_update:
            return

        self._update_next_window_state()
        self._advance_window()
        self.request_state_update = False

    def window_ic_loss(self):
        if self.causal_strategy != "cyclic_windows":
            return None
        if self.current_window == 0:
            return None
        if self.model is None:
            raise RuntimeError("CausalOptimizer needs attach_model(model).")

        self._init_window_states_if_needed()
        target = self.window_states[self.current_window]
        if target is None:
            return None

        t_left, _ = self.current_window_bounds()
        x = self._make_state_points(t_left)
        param = next(self.model.net.parameters())
        x_torch = torch.as_tensor(x, dtype=param.dtype, device=param.device)
        y_torch = torch.as_tensor(target, dtype=param.dtype, device=param.device)
        pred = self.model.net(x_torch)
        return self.window_ic_weight * torch.mean((pred - y_torch) ** 2)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("CausalOptimizer requires a closure.")

        loss = self.base_optimizer.step(closure)
        loss_value = self._extract_loss_value(loss)
        if loss_value is not None:
            self.last_loss = loss_value

        self.global_step += 1
        if self.causal_strategy == "cyclic_windows":
            self.steps_in_window += 1
            if self.steps_in_window >= self.steps_per_window:
                self.request_state_update = True
        else:
            self._maybe_unlock()
        return loss
