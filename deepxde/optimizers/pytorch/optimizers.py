__all__ = ["get", "is_external_optimizer"]

import torch

# from .nncg import NNCG
from .causal import CausalOptimizer
from .muon import MuonWithAuxAdam
from .muown import MuownWithAuxAdam
from .mop import MOPWithAuxAdam
from .polargrad import PolarGradWithAuxAdam
from .mousse import MousseWithAuxLion
from .psgd_pro import PSGDPro
from .klopt import KLOptWithAuxAdam
from .rekls_v3 import ReklsV3WithAuxAdam
from .kl_m_soap import KlMSoapWithAuxMAdam
from .madam import MAdam
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
    MUOWN_options,
    MOP_options,
    POLARGRAD_options,
    MOUSSE_options,
    PSGDPRO_options,
    KLOPT_options,
    REKLSV3_options,
    KLMSOAP_options,
    MADAM_options,
    PCGRAD_options,
    PSO_options,
    SSBROYDEN_options,
    SOAP_options,
    ZOCGE_options,
)


_KLOPT_NAMES = {"klopt", "kl-shampoo", "klshampoo", "kl-soap", "klsoap"}
_REKLSV3_NAMES = {"rekls", "reklsv3", "rekls-v3", "rekls_v3"}
_KLMSOAP_NAMES = {"kl-m-soap", "kl_m_soap", "klmsoap", "kl-msoap"}
_MADAM_NAMES = {"madam", "m-adam", "m_adam"}
_MUOWN_NAMES = {"muown", "mu-own", "mu_own"}
_PSGDPRO_NAMES = {"psgdpro", "psgd-pro", "psgd_pro", "pcggradpro"}
_POLARGRAD_NAMES = {"polargrad", "polar-grad", "polar_grad"}


def _resolve_torch_dtype(value):
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = value.lower().replace("torch.", "")
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    raise ValueError(f"Unsupported PyTorch dtype: {value!r}")


def _make_klopt_optimizer(params, learning_rate, weight_decay=0, mode=None):
    if learning_rate is None:
        raise ValueError("No learning rate for KLOpt.")
    flat_params = []
    for item in params:
        if isinstance(item, dict):
            flat_params.extend(item["params"])
        else:
            flat_params.append(item)
    params = [p for p in flat_params if p.requires_grad]
    kl_params = [
        p for p in params if 2 <= sum(size != 1 for size in p.shape) <= 3
    ]
    kl_ids = {id(p) for p in kl_params}
    aux_params = [p for p in params if id(p) not in kl_ids]
    groups = []
    if kl_params:
        groups.append({"params": kl_params, "use_klopt": True})
    if aux_params:
        groups.append({"params": aux_params, "use_klopt": False})
    if not groups:
        raise ValueError("KLOpt has no trainable parameters.")

    normalized_mode = None if mode is None else mode.lower()
    using_klsoap = KLOPT_options["using_klsoap"]
    if normalized_mode in {"kl-soap", "klsoap"}:
        using_klsoap = True
    elif normalized_mode in {"kl-shampoo", "klshampoo"}:
        using_klsoap = False
    return KLOptWithAuxAdam(
        groups,
        lr=learning_rate,
        betas=(KLOPT_options["beta1"], KLOPT_options["beta2"]),
        shampoo_beta=KLOPT_options["shampoo_beta"],
        eps=KLOPT_options["epsilon"],
        weight_decay=weight_decay,
        precondition_frequency=KLOPT_options["precondition_frequency"],
        using_klsoap=using_klsoap,
        normalize_grads=KLOPT_options["normalize_grads"],
        init_factor=KLOPT_options["init_factor"],
        using_damping=KLOPT_options["using_damping"],
        using_clamping=KLOPT_options["using_clamping"],
        max_clamp_value=KLOPT_options["max_clamp_value"],
        cast_dtype=_resolve_torch_dtype(KLOPT_options["cast_dtype"]),
    )


def _make_reklsv3_optimizer(params, learning_rate, weight_decay=0):
    if learning_rate is None:
        raise ValueError("No learning rate for REKLS V3.")
    flat_params = []
    for item in params:
        if isinstance(item, dict):
            flat_params.extend(item["params"])
        else:
            flat_params.append(item)
    params = [parameter for parameter in flat_params if parameter.requires_grad]
    rekls_params = [parameter for parameter in params if parameter.ndim == 2]
    rekls_ids = {id(parameter) for parameter in rekls_params}
    auxiliary_params = [
        parameter for parameter in params if id(parameter) not in rekls_ids
    ]

    groups = []
    if rekls_params:
        groups.append(
            {
                "params": rekls_params,
                "use_rekls": True,
                "lr": learning_rate,
                "weight_decay": (
                    weight_decay
                    if weight_decay != 0
                    else REKLSV3_options["rekls_weight_decay"]
                ),
            }
        )
    if auxiliary_params:
        groups.append(
            {
                "params": auxiliary_params,
                "use_rekls": False,
                "lr": (
                    learning_rate
                    if REKLSV3_options["auxiliary_lr"] is None
                    else REKLSV3_options["auxiliary_lr"]
                ),
                "weight_decay": REKLSV3_options["auxiliary_weight_decay"],
            }
        )
    if not groups:
        raise ValueError("REKLS V3 has no trainable parameters.")
    return ReklsV3WithAuxAdam(
        groups,
        lr=learning_rate,
        betas=REKLSV3_options["betas"],
        shampoo_beta=REKLSV3_options["shampoo_beta"],
        eps=REKLSV3_options["epsilon"],
        base_optimizer=REKLSV3_options["base_optimizer"],
        scale_log2=REKLSV3_options["scale_log2"],
        auxiliary_betas=REKLSV3_options["auxiliary_betas"],
        auxiliary_eps=REKLSV3_options["auxiliary_epsilon"],
        auxiliary_scale_log2=REKLSV3_options["auxiliary_scale_log2"],
    )


def _make_kl_m_soap_optimizer(params, learning_rate, weight_decay=0):
    if learning_rate is None:
        raise ValueError("No learning rate for KL-M-SOAP.")
    flat_params = []
    for item in params:
        if isinstance(item, dict):
            flat_params.extend(item["params"])
        else:
            flat_params.append(item)
    params = [parameter for parameter in flat_params if parameter.requires_grad]
    matrix_params = [parameter for parameter in params if parameter.ndim == 2]
    matrix_ids = {id(parameter) for parameter in matrix_params}
    auxiliary_params = [
        parameter for parameter in params if id(parameter) not in matrix_ids
    ]
    groups = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "use_kl_m_soap": True,
                "lr": learning_rate,
                "weight_decay": (
                    weight_decay
                    if weight_decay != 0
                    else KLMSOAP_options["kl_m_soap_weight_decay"]
                ),
            }
        )
    if auxiliary_params:
        groups.append(
            {
                "params": auxiliary_params,
                "use_kl_m_soap": False,
                "lr": (
                    learning_rate
                    if KLMSOAP_options["auxiliary_lr"] is None
                    else KLMSOAP_options["auxiliary_lr"]
                ),
                "weight_decay": KLMSOAP_options["auxiliary_weight_decay"],
            }
        )
    if not groups:
        raise ValueError("KL-M-SOAP has no trainable parameters.")
    return KlMSoapWithAuxMAdam(
        groups,
        lr=learning_rate,
        betas=KLMSOAP_options["betas"],
        shampoo_beta=KLMSOAP_options["shampoo_beta"],
        eps=KLMSOAP_options["epsilon"],
        scale_log2=KLMSOAP_options["scale_log2"],
        auxiliary_betas=KLMSOAP_options["auxiliary_betas"],
        auxiliary_scale_log2=KLMSOAP_options["auxiliary_scale_log2"],
    )


def _make_madam_optimizer(params, learning_rate, weight_decay=0):
    if learning_rate is None:
        raise ValueError("No learning rate for MAdam.")
    return MAdam(
        params,
        lr=learning_rate,
        betas=MADAM_options["betas"],
        weight_decay=weight_decay,
        scale_log2=MADAM_options["scale_log2"],
        correct_bias=MADAM_options["correct_bias"],
    )


def _make_mousse_optimizer(params, learning_rate, weight_decay=0, model=None):
    if learning_rate is None:
        raise ValueError("No learning rate for Mousse.")
    params = [p for p in params if p.requires_grad]
    matrix_ids = _hidden_linear_weight_ids(model)
    matrix_params = [p for p in params if id(p) in matrix_ids and p.ndim == 2]
    selected_ids = {id(p) for p in matrix_params}
    auxiliary = [p for p in params if id(p) not in selected_ids]
    groups = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "algorithm": "mousse",
                "weight_decay": (
                    weight_decay
                    if weight_decay != 0
                    else MOUSSE_options["mousse_weight_decay"]
                ),
            }
        )
    if auxiliary:
        groups.append(
            {
                "params": auxiliary,
                "algorithm": "lion",
                "weight_decay": MOUSSE_options["lion_weight_decay"],
            }
        )
    if not groups:
        raise ValueError("Mousse has no trainable parameters.")
    return MousseWithAuxLion(
        groups,
        lr=learning_rate,
        mu=MOUSSE_options["momentum"],
        betas=MOUSSE_options["lion_betas"],
        epsilon=MOUSSE_options["epsilon"],
        nesterov=MOUSSE_options["nesterov"],
        adjust_lr=MOUSSE_options["adjust_lr"],
        shampoo_epsilon=MOUSSE_options["shampoo_epsilon"],
        shampoo_beta=MOUSSE_options["shampoo_beta"],
        shampoo_update_freq=MOUSSE_options["shampoo_update_frequency"],
        shampoo_alpha=MOUSSE_options["shampoo_alpha"],
        lr_correction=MOUSSE_options["lr_correction"],
        apply_norm=MOUSSE_options["apply_norm"],
        use_l_or_r=MOUSSE_options["use_l_or_r"],
    )


def _make_psgdpro_optimizer(params, learning_rate, weight_decay=0):
    if learning_rate is None:
        raise ValueError("No learning rate for PSGDPro.")
    params = [p for p in params if p.requires_grad]
    psgd_params = [p for p in params if p.ndim >= 1]
    psgd_ids = {id(p) for p in psgd_params}
    auxiliary = [p for p in params if id(p) not in psgd_ids]
    groups = []
    if psgd_params:
        groups.append(
            {
                "params": psgd_params,
                "use_psgd": True,
                "weight_decay": (
                    weight_decay
                    if weight_decay != 0
                    else PSGDPRO_options["psgd_weight_decay"]
                ),
            }
        )
    if auxiliary:
        groups.append(
            {
                "params": auxiliary,
                "use_psgd": False,
                "weight_decay": PSGDPRO_options["auxiliary_weight_decay"],
            }
        )
    if not groups:
        raise ValueError("PSGDPro has no trainable parameters.")
    return PSGDPro(
        groups,
        lr=learning_rate,
        momentum=PSGDPRO_options["momentum"],
        beta_lip=PSGDPRO_options["beta_lip"],
        precond_lr=PSGDPRO_options["preconditioner_lr"],
        precond_init_scale=PSGDPRO_options["preconditioner_init_scale"],
        damping_noise_scale=PSGDPRO_options["damping_noise_scale"],
        min_precond_lr=PSGDPRO_options["min_preconditioner_lr"],
        warmup_steps=PSGDPRO_options["warmup_steps"],
        max_update_rms=PSGDPRO_options["max_update_rms"],
        weight_decay_method=PSGDPRO_options["weight_decay_method"],
        auxiliary_betas=PSGDPRO_options["auxiliary_betas"],
        auxiliary_eps=PSGDPRO_options["auxiliary_epsilon"],
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
        hidden_layers = (
            layers[:-1]
            if getattr(model, "sfli_type", None) == "gaussian"
            else layers[1:-1]
        )
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


def _make_auxiliary_group(
    params, use_flag, options, learning_rate, weight_decay
):
    """Build an Adam/SOAP fallback group for a matrix optimizer."""
    optimizer_name = options["auxiliary_optimizer"]
    group = {
        "params": params,
        use_flag: False,
        "auxiliary_optimizer": optimizer_name,
        "lr": (
            learning_rate
            if options["auxiliary_lr"] is None
            else options["auxiliary_lr"]
        ),
        "weight_decay": options["auxiliary_weight_decay"] or weight_decay,
    }
    if optimizer_name == "soap":
        group.update(
            betas=(SOAP_options["beta1"], SOAP_options["beta2"]),
            shampoo_beta=SOAP_options["shampoo_beta"],
            eps=SOAP_options["epsilon"],
            precondition_frequency=SOAP_options["precondition_frequency"],
            max_precondition_dim=SOAP_options["max_precondition_dim"],
            bias_correction=SOAP_options["bias_correction"],
        )
    else:
        group.update(
            betas=options["adam_betas"],
            eps=options["adam_eps"],
        )
    return group


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
            f"all parameters will use auxiliary {MUON_options['auxiliary_optimizer']}."
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
            _make_auxiliary_group(
                aux_params, "use_muon", MUON_options, learning_rate, weight_decay
            )
        )
    return MuonWithAuxAdam(param_groups)


def _make_muown_optimizer(params, learning_rate, weight_decay=0, model=None):
    if learning_rate is None:
        raise ValueError("No learning rate for MuOwn.")
    params = list(params)
    selected_ids = _hidden_linear_weight_ids(model)
    matrix_params = [
        parameter
        for parameter in params
        if id(parameter) in selected_ids and parameter.ndim == 2
    ]
    selected_ids = {id(parameter) for parameter in matrix_params}
    auxiliary_params = [
        parameter for parameter in params if id(parameter) not in selected_ids
    ]
    if not matrix_params:
        print(
            "Warning: MuOwn found no hidden Linear.weight parameters; "
            f"all parameters will use auxiliary {MUOWN_options['auxiliary_optimizer']}."
        )

    groups = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "use_muown": True,
                "lr": learning_rate,
                "momentum": MUOWN_options["momentum"],
                "betas": MUOWN_options["betas"],
                "adam_eps": MUOWN_options["adam_eps"],
                "fp32_matmul_precision": MUOWN_options["fp32_matmul_precision"],
                "coefficient_type": MUOWN_options["coefficient_type"],
                "ns_steps": MUOWN_options["ns_steps"],
                "scale_mode": MUOWN_options["scale_mode"],
                "extra_scale_factor": MUOWN_options["extra_scale_factor"],
                "weight_decay": MUOWN_options["muown_weight_decay"] or weight_decay,
            }
        )
    if auxiliary_params:
        fallback_options = {
            "auxiliary_optimizer": MUOWN_options["auxiliary_optimizer"],
            "auxiliary_lr": MUOWN_options["auxiliary_lr"],
            "auxiliary_weight_decay": MUOWN_options["auxiliary_weight_decay"],
            "adam_betas": MUOWN_options["auxiliary_betas"],
            "adam_eps": MUOWN_options["auxiliary_eps"],
        }
        groups.append(
            _make_auxiliary_group(
                auxiliary_params,
                "use_muown",
                fallback_options,
                learning_rate,
                weight_decay,
            )
        )
    return MuownWithAuxAdam(groups)


def _make_mop_optimizer(params, learning_rate, weight_decay=0, model=None):
    if learning_rate is None:
        raise ValueError("No learning rate for MOP.")

    params = list(params)
    mop_ids = _hidden_linear_weight_ids(model)
    mop_params = [p for p in params if id(p) in mop_ids and p.ndim == 2]
    mop_ids = {id(p) for p in mop_params}
    aux_params = [p for p in params if id(p) not in mop_ids]

    if not mop_params:
        print(
            "Warning: MOP found no hidden Linear.weight parameters; "
            f"all parameters will use auxiliary {MOP_options['auxiliary_optimizer']}."
        )

    param_groups = []
    if mop_params:
        param_groups.append(
            {
                "params": mop_params,
                "use_mop": True,
                "lr": learning_rate,
                "momentum": MOP_options["momentum"],
                "nesterov": MOP_options["nesterov"],
                "scale_mode": MOP_options["scale_mode"],
                "extra_scale_factor": MOP_options["extra_scale_factor"],
                "weight_decay": MOP_options["mop_weight_decay"] or weight_decay,
            }
        )
    if aux_params:
        param_groups.append(
            _make_auxiliary_group(
                aux_params, "use_mop", MOP_options, learning_rate, weight_decay
            )
        )
    return MOPWithAuxAdam(param_groups)


def _make_polargrad_optimizer(params, learning_rate, weight_decay=0, model=None):
    if learning_rate is None:
        raise ValueError("No learning rate for PolarGrad.")

    params = [parameter for parameter in params if parameter.requires_grad]
    polargrad_ids = _hidden_linear_weight_ids(model)
    polargrad_params = [
        parameter
        for parameter in params
        if id(parameter) in polargrad_ids and parameter.ndim == 2
    ]
    selected_ids = {id(parameter) for parameter in polargrad_params}
    auxiliary_params = [
        parameter for parameter in params if id(parameter) not in selected_ids
    ]

    if not polargrad_params:
        print(
            "Warning: PolarGrad found no hidden Linear.weight parameters; "
            "all parameters will use auxiliary "
            f"{POLARGRAD_options['auxiliary_optimizer']}."
        )

    groups = []
    if polargrad_params:
        groups.append(
            {
                "params": polargrad_params,
                "use_polargrad": True,
                "lr": learning_rate,
                "momentum": POLARGRAD_options["momentum"],
                "polar_first": POLARGRAD_options["polar_first"],
                "method": POLARGRAD_options["method"],
                "inner_steps": POLARGRAD_options["inner_steps"],
                "a": POLARGRAD_options["a"],
                "b": POLARGRAD_options["b"],
                "c": POLARGRAD_options["c"],
                "weight_decay": (
                    POLARGRAD_options["polargrad_weight_decay"] or weight_decay
                ),
            }
        )
    if auxiliary_params:
        groups.append(
            _make_auxiliary_group(
                auxiliary_params,
                "use_polargrad",
                POLARGRAD_options,
                learning_rate,
                weight_decay,
            )
        )
    if not groups:
        raise ValueError("PolarGrad has no trainable parameters.")
    return PolarGradWithAuxAdam(groups)


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

    if isinstance(optimizer, str) and optimizer.lower() in _KLOPT_NAMES:
        return _make_klopt_optimizer(
            params,
            learning_rate,
            weight_decay=weight_decay,
            mode=optimizer,
        )

    if isinstance(optimizer, str) and optimizer.lower() in _REKLSV3_NAMES:
        return _make_reklsv3_optimizer(
            params, learning_rate, weight_decay=weight_decay
        )

    if isinstance(optimizer, str) and optimizer.lower() in _KLMSOAP_NAMES:
        return _make_kl_m_soap_optimizer(
            params, learning_rate, weight_decay=weight_decay
        )

    if isinstance(optimizer, str) and optimizer.lower() in _MADAM_NAMES:
        return _make_madam_optimizer(params, learning_rate, weight_decay=weight_decay)

    if optimizer in ["muon", "Muon"]:
        return _make_muon_optimizer(params, learning_rate, weight_decay=weight_decay, model=model)

    if isinstance(optimizer, str) and optimizer.lower() in _MUOWN_NAMES:
        return _make_muown_optimizer(
            params, learning_rate, weight_decay=weight_decay, model=model
        )

    if optimizer in ["mop", "MOP"]:
        return _make_mop_optimizer(params, learning_rate, weight_decay=weight_decay, model=model)

    if isinstance(optimizer, str) and optimizer.lower() in _POLARGRAD_NAMES:
        return _make_polargrad_optimizer(
            params, learning_rate, weight_decay=weight_decay, model=model
        )

    if optimizer in ["mousse", "Mousse"]:
        return _make_mousse_optimizer(
            params, learning_rate, weight_decay=weight_decay, model=model
        )

    if isinstance(optimizer, str) and optimizer.lower() in _PSGDPRO_NAMES:
        return _make_psgdpro_optimizer(
            params, learning_rate, weight_decay=weight_decay
        )

    raise NotImplementedError(
        f"Causal base optimizer {optimizer} is not supported. "
        "Use one of: adam, madam, soap, klopt, kl-shampoo, kl-soap, kl-m-soap, "
        "rekls-v3, muon, muown, MOP, "
        "mousse, psgdpro, polargrad, "
        "L-BFGS, L-BFGS-B, PSO."
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
        elif isinstance(optimizer, str) and optimizer.lower() in _KLOPT_NAMES:
            optim = _make_klopt_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
                mode=optimizer,
            )
        elif isinstance(optimizer, str) and optimizer.lower() in _REKLSV3_NAMES:
            optim = _make_reklsv3_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
            )
        elif isinstance(optimizer, str) and optimizer.lower() in _KLMSOAP_NAMES:
            optim = _make_kl_m_soap_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
            )
        elif isinstance(optimizer, str) and optimizer.lower() in _MADAM_NAMES:
            optim = _make_madam_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer in ["muon", "Muon"]:
            optim = _make_muon_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
                model=model,
            )
        elif isinstance(optimizer, str) and optimizer.lower() in _MUOWN_NAMES:
            optim = _make_muown_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
                model=model,
            )
        elif optimizer in ["mop", "MOP"]:
            optim = _make_mop_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
                model=model,
            )
        elif isinstance(optimizer, str) and optimizer.lower() in _POLARGRAD_NAMES:
            optim = _make_polargrad_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
                model=model,
            )
        elif optimizer in ["mousse", "Mousse"]:
            optim = _make_mousse_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
                model=model,
            )
        elif isinstance(optimizer, str) and optimizer.lower() in _PSGDPRO_NAMES:
            optim = _make_psgdpro_optimizer(
                params,
                learning_rate,
                weight_decay=weight_decay,
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
