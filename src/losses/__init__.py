"""Physics-loss implementations that can be attached to PINNacle models."""

from .weak_form import WeakFormConfig, WeakFormLoss, attach_weak_form_loss

__all__ = ["WeakFormConfig", "WeakFormLoss", "attach_weak_form_loss"]
