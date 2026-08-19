__all__ = ["get", "is_external_optimizer"]

import torch

# from .nncg import NNCG
from .causal import CausalOptimizer
from .muon import MuonWithAuxAdam
from .pcgrad import PCGrad
from .pso import PSO
from .soap import SOAP
from .ssbroyden import SSBroyden
from .zo_cge import ZOCGE
from ..config import (
    CAUSAL_options,
    LBFGS_options,
    NNCG_options,
    MUON_options,
    PCGRAD_options,
    PSO_options,
    SSBROYDEN_options,
    SOAP_options,
    ZOCGE_options,
)


# NOTE: edited
def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B", "NNCG", "PSO"]


def _iter_linear_modules(module):
    if isinstance(module, torch.nn.Linear) or (
        isinstance(module, torch.nn.Module)
        and isinstance(getattr(module, "V", None), torch.nn.Parameter)
        and isinstance(getattr(module, "s", None), torch.nn.Parameter)
        and module.V.ndim == 2
    ):
        yield module
        return
    if isinstance(module, torch.nn.Module):
        for child in module.children():
            yield from _iter_linear_modules(child)


def _hidden_linear_weight_ids(model):
    if model is None:
        return set()

    if hasattr(model, "linears"):
        layers = list(model.linears)
        hidden_layers = layers[1:-1]
    elif hasattr(model, "layers"):
        layers = list(model.layers)
        hidden_layers = layers[1:-1]
    else:
        layers = list(_iter_linear_modules(model))
        hidden_layers = layers[1:-1]

    muon_param_ids = set()
    for layer in hidden_layers:
        for linear in _iter_linear_modules(layer):
            matrix = linear.V if hasattr(linear, "V") else linear.weight
            if matrix.requires_grad:
                muon_param_ids.add(id(matrix))
    return muon_param_ids


def _make_muon_optimizer(params, learning_rate, weight_decay=0, model=None):
    if learning_rate is None:
        raise ValueError("No learning rate for muon.")

    params = list(params)
    muon_param_ids = _hidden_linear_weight_ids(model)
    muon_params = [p for p in params if id(p) in muon_param_ids and p.ndim == 2]
    muon_param_ids = {id(p) for p in muon_params}
    aux_params = [p for p in params if id(p) not in muon_param_ids]

    if not muon_params:
        print(
            "Warning: muon found no hidden Linear.weight parameters; "
            "all parameters will use auxiliary Adam."
        )

    param_groups = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": learning_rate,
                "momentum": MUON_options["momentum"],
                "nesterov": MUON_options["nesterov"],
                "ns_steps": MUON_options["ns_steps"],
                "weight_decay": MUON_options["muon_weight_decay"] or weight_decay,
            }
        )
    if aux_params:
        param_groups.append(
            {
                "params": aux_params,
                "use_muon": False,
                "lr": learning_rate if MUON_options["adam_lr"] is None else MUON_options["adam_lr"],
                "betas": MUON_options["adam_betas"],
                "eps": MUON_options["adam_eps"],
                "weight_decay": MUON_options["adam_weight_decay"] or weight_decay,
            }
        )
    return MuonWithAuxAdam(param_groups)


def _make_base_optimizer(params, optimizer, learning_rate=None, decay=None, weight_decay=0, model=None):
    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        if weight_decay > 0:
            raise ValueError("L-BFGS optimizer doesn't support weight_decay > 0")
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        return torch.optim.LBFGS(
            params,
            lr=LBFGS_options["lr"] if LBFGS_options["lr"] is not None else 1,
            max_iter=LBFGS_options["iter_per_step"],
            max_eval=LBFGS_options["fun_per_step"],
            tolerance_grad=LBFGS_options["gtol"],
            tolerance_change=LBFGS_options["ftol"],
            history_size=LBFGS_options["maxcor"],
            line_search_fn=("strong_wolfe" if LBFGS_options["maxls"] > 0 else None),
        )

    if optimizer == "PSO":
        if weight_decay > 0:
            raise ValueError("PSO optimizer doesn't support weight_decay > 0")
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        return PSO(
            params,
            pop_size=PSO_options["pop_size"],
            b=PSO_options["b"],
            c1=PSO_options["c1"],
            c2=PSO_options["c2"],
            lr=PSO_options["lr"],
            betas=PSO_options["betas"],
            c_decrease=PSO_options["c_decrease"],
            variance=PSO_options["variance"],
            epsilon=PSO_options["epsilon"],
            n_iter=PSO_options["n_iter"],
        )

    if optimizer == "ZOCGE":
        if learning_rate is None:
            raise ValueError("No learning rate for ZOCGE.")
        return ZOCGE(
            params,
            lr=learning_rate,
            mu=ZOCGE_options["mu"],
            weight_decay=weight_decay,
            sparsity=ZOCGE_options["sparsity"],
            prune_method=ZOCGE_options["prune_method"],
            remask_interval=ZOCGE_options["remask_interval"],
            feature_reuse=ZOCGE_options["feature_reuse"],
            grasp_sample_size=ZOCGE_options["grasp_sample_size"],
        )

    if optimizer == "adam":
        if learning_rate is None:
            raise ValueError("No learning rate for adam.")
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    if optimizer == "soap":
        if learning_rate is None:
            raise ValueError("No learning rate for soap.")
        return SOAP(
            params,
            lr=learning_rate,
            betas=(SOAP_options["beta1"], SOAP_options["beta2"]),
            shampoo_beta=SOAP_options["shampoo_beta"],
            eps=SOAP_options["epsilon"],
            weight_decay=weight_decay,
            precondition_frequency=SOAP_options["precondition_frequency"],
            max_precondition_dim=SOAP_options["max_precondition_dim"],
            bias_correction=SOAP_options["bias_correction"],
        )

    if optimizer in ["muon", "Muon"]:
        return _make_muon_optimizer(params, learning_rate, weight_decay=weight_decay, model=model)

    raise NotImplementedError(
        f"Causal base optimizer {optimizer} is not supported. "
        "Use one of: adam, soap, muon, L-BFGS, L-BFGS-B, PSO."
    )


def get(params, optimizer, learning_rate=None, decay=None, weight_decay=0, model=None):
    """Retrieves an Optimizer instance."""
    # Custom Optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optim = optimizer
    elif optimizer == "Causal":
        base_name = CAUSAL_options["base_optimizer"]
        if base_name == "Causal":
            raise ValueError("Causal optimizer cannot wrap itself.")
        if (
            base_name == "PSO"
            and CAUSAL_options["causal_strategy"] == "cyclic_windows"
        ):
            raise ValueError(
                "Causal cyclic_windows strategy does not support PSO yet. "
                "Use adam, soap, muon, L-BFGS, or L-BFGS-B."
            )

        base_optim = _make_base_optimizer(
            params,
            base_name,
            learning_rate=learning_rate,
            decay=decay,
            weight_decay=weight_decay,
            model=model,
        )
        optim = CausalOptimizer(
            base_optimizer=base_optim,
            base_optimizer_name=base_name,
            n_time_bins=CAUSAL_options["n_time_bins"],
            start_bins=CAUSAL_options["start_bins"],
            time_index=CAUSAL_options["time_index"],
            unlock_every=CAUSAL_options["unlock_every"],
            unlock_tol=CAUSAL_options["unlock_tol"],
            min_steps_per_bin=CAUSAL_options["min_steps_per_bin"],
            bc_mode=CAUSAL_options["bc_mode"],
            min_points_per_bc=CAUSAL_options["min_points_per_bc"],
            causal_strategy=CAUSAL_options["causal_strategy"],
            steps_per_window=CAUSAL_options["steps_per_window"],
            state_alpha=CAUSAL_options["state_alpha"],
            x_state=CAUSAL_options["x_state"],
            window_ic_weight=CAUSAL_options["window_ic_weight"],
            verbose=CAUSAL_options["verbose"],
        )
    elif optimizer in ["L-BFGS", "L-BFGS-B", "PSO"]:
        optim = _make_base_optimizer(
            params,
            optimizer,
            learning_rate=learning_rate,
            decay=decay,
            weight_decay=weight_decay,
            model=model,
        )
    elif optimizer == "NNCG":
        if weight_decay > 0:
            raise ValueError("NNCG optimizer doesn't support weight_decay > 0")
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        optim = NNCG(
            params,
            lr=NNCG_options["lr"],
            rank=NNCG_options["rank"],
            mu=NNCG_options["mu"],
            update_freq=NNCG_options["updatefreq"],
            chunk_size=NNCG_options["chunksz"],
            cg_tol=NNCG_options["cgtol"],
            cg_max_iters=NNCG_options["cgmaxiter"],
            line_search_fn=NNCG_options["lsfun"],
            verbose=NNCG_options["verbose"],
        )
    else:
        if learning_rate is None:
            raise ValueError("No learning rate for {}.".format(optimizer))
        if optimizer == "sgd":
            optim = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "adam":
            optim = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "soap":
            optim = SOAP(
                params,
                lr=learning_rate,
                betas=(SOAP_options["beta1"], SOAP_options["beta2"]),
                shampoo_beta=SOAP_options["shampoo_beta"],
                eps=SOAP_options["epsilon"],
                weight_decay=weight_decay,
                precondition_frequency=SOAP_options["precondition_frequency"],
                max_precondition_dim=SOAP_options["max_precondition_dim"],
                bias_correction=SOAP_options["bias_correction"],
            )
        elif optimizer in ["muon", "Muon"]:
            optim = _make_muon_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
                model=model,
            )
        elif optimizer == "ZOCGE":
            optim = ZOCGE(
                params,
                lr=learning_rate,
                mu=ZOCGE_options["mu"],
                weight_decay=weight_decay,
                sparsity=ZOCGE_options["sparsity"],
                prune_method=ZOCGE_options["prune_method"],
                remask_interval=ZOCGE_options["remask_interval"],
                feature_reuse=ZOCGE_options["feature_reuse"],
                grasp_sample_size=ZOCGE_options["grasp_sample_size"],
            )
        elif optimizer == "pcgrad":
            if PCGRAD_options["base_optimizer"] != "adam":
                raise NotImplementedError("PCGrad currently wraps only adam.")
            optim = PCGrad(
                torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
            )
        elif optimizer in ["SSBroyden", "ssbroyden"]:
            if weight_decay > 0:
                raise ValueError("SSBroyden optimizer doesn't support weight_decay > 0")
            if decay is not None:
                print("Warning: learning rate scheduler is ignored for {}".format(optimizer))
            optim = SSBroyden(
                params,
                lr=SSBROYDEN_options["lr"],
                tolerance_grad=SSBROYDEN_options["tolerance_grad"],
                debug=SSBROYDEN_options["debug"],
                debug_every=SSBROYDEN_options["debug_every"],
            )
        elif optimizer == "adamw":
            if weight_decay == 0:
                raise ValueError("AdamW optimizer requires non-zero weight decay")
            optim = torch.optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(
                f"{optimizer} to be implemented for backend pytorch."
            )
    lr_scheduler = _get_learningrate_scheduler(optim, decay)
    return optim, lr_scheduler


def _get_learningrate_scheduler(optim, decay):
    if decay is None:
        return None

    # NOTE: edited
    if (
        isinstance(decay, torch.optim.lr_scheduler._LRScheduler)
        or decay.__class__.__name__ == "ReduceLROnPlateau"
    ):
        return decay

    if decay[0] == "step":
        return torch.optim.lr_scheduler.StepLR(
            optim, step_size=decay[1], gamma=decay[2]
        )

    if decay[0] == "exponential":
        decay_steps = int(decay[1])
        decay_rate = float(decay[2])
        if decay_steps <= 0:
            raise ValueError("exponential decay_steps must be positive")
        if decay_rate <= 0:
            raise ValueError("exponential decay_rate must be positive")
        return torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lambda step: decay_rate ** (step / decay_steps),
        )

    # TODO: More learning rate scheduler
    raise NotImplementedError(
        f"{decay[0]} learning rate scheduler to be implemented for backend pytorch."
    )
