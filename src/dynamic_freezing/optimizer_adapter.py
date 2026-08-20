import copy
from contextlib import contextmanager

import torch


_RESTORABLE_STATE_KEYS = {
    "Adam": {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"},
    "AdamW": {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"},
    "PCGrad": {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"},
    "SOAP": {"exp_avg", "exp_avg_sq"},
    "MuonWithAuxAdam": {"momentum_buffer", "exp_avg", "exp_avg_sq"},
}


def _clone_model_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def preview_optimizer_step(module, optimizer, closure=None):
    """Return a real optimizer proposal and restore model and optimizer exactly."""
    model_before = _clone_model_state(module)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    parameters_before = {parameter: parameter.detach().clone() for parameter in module.parameters()}
    try:
        optimizer.step(closure)
        return {
            parameter: parameter.detach().clone() - parameters_before[parameter]
            for parameter in module.parameters()
        }
    finally:
        module.load_state_dict(model_before)
        optimizer.load_state_dict(optimizer_before)


class MaskedOptimizerAdapter:
    """Apply a sub-tensor mask without changing the model or optimizer topology."""

    SUPPORTED = {"Adam", "AdamW", "PCGrad", "SOAP", "MuonWithAuxAdam"}

    def __init__(self, module, optimizer, groups, controller=None):
        optimizer_name = optimizer.__class__.__name__
        if optimizer_name not in self.SUPPORTED:
            raise ValueError(
                f"Dynamic freezing does not support {optimizer_name}; "
                "supported optimizers are Adam, PCGrad, SOAP and MuonWithAuxAdam."
            )
        self.module = module
        self.optimizer = optimizer
        self.groups = groups
        self.controller = controller
        self.optimizer_name = optimizer_name
        self._original_step = optimizer.step
        self._installed = False
        self._inside_step = False

    def install(self):
        if not self._installed:
            self.optimizer.step = self.step
            self._installed = True
        return self

    def uninstall(self):
        if self._installed:
            self.optimizer.step = self._original_step
            self._installed = False

    @contextmanager
    def uninstalled(self):
        was_installed = self._installed
        if was_installed:
            self.uninstall()
        try:
            yield
        finally:
            if was_installed:
                self.install()

    def _parameter_snapshot(self):
        return {parameter: parameter.detach().clone() for parameter in self.module.parameters()}

    def _state_snapshot(self):
        keys = _RESTORABLE_STATE_KEYS.get(self.optimizer_name, set())
        snapshots = {}
        for parameter in self.module.parameters():
            state = self.optimizer.state.get(parameter, {})
            snapshots[parameter] = {
                key: value.detach().clone()
                for key, value in state.items()
                if key in keys and torch.is_tensor(value) and value.shape == parameter.shape
            }
        return snapshots

    def _restore_frozen(self, parameter_before, state_before):
        with torch.no_grad():
            for parameter, before in parameter_before.items():
                mask = self.groups.frozen_mask_for(parameter)
                if mask is None:
                    continue
                parameter[mask] = before[mask]
                # SOAP matrix moments use a rotating basis; one-dimensional
                # fallback parameters use ordinary Adam moments.
                if self.optimizer_name == "SOAP" and parameter.ndim == 2:
                    continue
                state = self.optimizer.state.get(parameter, {})
                old_state = state_before.get(parameter, {})
                for key in _RESTORABLE_STATE_KEYS.get(self.optimizer_name, set()):
                    value = state.get(key)
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        old_value = old_state.get(key)
                        if old_value is None:
                            old_value = torch.zeros_like(value)
                        value[mask] = old_value[mask]

    def step(self, closure=None, *args, **kwargs):
        if self._inside_step:
            return self._original_step(closure, *args, **kwargs)
        self._inside_step = True
        try:
            parameter_before = self._parameter_snapshot()
            state_before = self._state_snapshot()
            event_due = self.controller is not None and self.controller.should_trigger_before_step()
            if not event_due:
                result = self._original_step(closure, *args, **kwargs)
                self._restore_frozen(parameter_before, state_before)
                return result

            optimizer_before = copy.deepcopy(self.optimizer.state_dict())
            result = self._original_step(closure, *args, **kwargs)
            parameter_after = self._parameter_snapshot()
            optimizer_after = copy.deepcopy(self.optimizer.state_dict())
            proposal = {
                parameter: parameter_after[parameter] - parameter_before[parameter]
                for parameter in parameter_before
            }

            # Event losses are evaluated at theta before the proposed step.
            with torch.no_grad():
                for parameter, value in parameter_before.items():
                    parameter.copy_(value)
            self.optimizer.load_state_dict(optimizer_before)
            self.controller.run_event(proposal)

            # Commit the already computed optimizer transition, then mask it.
            with torch.no_grad():
                for parameter, value in parameter_after.items():
                    parameter.copy_(value)
            self.optimizer.load_state_dict(optimizer_after)
            self._restore_frozen(parameter_before, state_before)
            return result
        finally:
            self._inside_step = False
