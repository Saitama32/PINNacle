__all__ = [
    "set_LBFGS_options",
    "set_NNCG_options",
    "set_PSO_options",
    "set_ZOCGE_options",
    "set_SOAP_options",
    "KLOPT_options",
    "set_KLOPT_options",
    "MUON_options",
    "set_MUON_options",
    "MOP_options",
    "set_MOP_options",
    "MOUSSE_options",
    "set_MOUSSE_options",
    "PSGDPRO_options",
    "set_PSGDPRO_options",
    "set_PCGRAD_options",
    "set_SSBROYDEN_options",
    "set_CAUSAL_options",
    "set_hvd_opt_options",
]

from ..backend import backend_name
# from ..config import hvd

LBFGS_options = {}
NNCG_options = {}
PSO_options = {}
ZOCGE_options = {}
SOAP_options = {}
KLOPT_options = {}
MUON_options = {}
MOP_options = {}
MOUSSE_options = {}
PSGDPRO_options = {}
PCGRAD_options = {}
SSBROYDEN_options = {}
CAUSAL_options = {}
hvd = None
if hvd is not None:
    hvd_opt_options = {}

set_ZOCGE_defaults = {
    "mu": 1e-3,
    "sparsity": 0.0,
    "prune_method": "random",
    "remask_interval": 0,
    "feature_reuse": False,
    "grasp_sample_size": 0,
}
ZOCGE_options.update(set_ZOCGE_defaults)


def set_LBFGS_options(
    lr=None,
    maxcor=100,
    ftol=0,
    gtol=1e-8,
    maxiter=15000,
    maxfun=None,
    maxls=50,
):
    """Sets the hyperparameters of L-BFGS.

    The L-BFGS optimizer used in each backend:

    - TensorFlow 1.x: `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_
    - TensorFlow 2.x: `tfp.optimizer.lbfgs_minimize <https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize>`_
    - PyTorch: `torch.optim.LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_
    - Paddle: `paddle.incubate.optimizers.LBFGS <https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/incubate/optimizer/LBFGS_en.html>`_

    I find empirically that torch.optim.LBFGS and scipy.optimize.minimize are better than
    tfp.optimizer.lbfgs_minimize in terms of the final loss value.

    Args:
        maxcor (int): `maxcor` (scipy), `num_correction_pairs` (tfp), `history_size` (torch), `history_size` (paddle).
            The maximum number of variable metric corrections used to define the limited
            memory matrix. (The limited memory BFGS method does not store the full
            hessian but uses this many terms in an approximation to it.)
        ftol (float): `ftol` (scipy), `f_relative_tolerance` (tfp), `tolerance_change` (torch), `tolerance_change` (paddle).
            The iteration stops when `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol`.
        gtol (float): `gtol` (scipy), `tolerance` (tfp), `tolerance_grad` (torch), `tolerance_grad` (paddle).
            The iteration will stop when `max{|proj g_i | i = 1, ..., n} <= gtol` where
            `pg_i` is the i-th component of the projected gradient.
        maxiter (int): `maxiter` (scipy), `max_iterations` (tfp), `max_iter` (torch), `max_iter` (paddle).
            Maximum number of iterations.
        maxfun (int): `maxfun` (scipy), `max_eval` (torch), `max_eval` (paddle).
            Maximum number of function evaluations. If ``None``, `maxiter` * 1.25.
        maxls (int): `maxls` (scipy), `max_line_search_iterations` (tfp).
            Maximum number of line search steps (per iteration).

    Warning:
        If L-BFGS stops earlier than expected, set the default float type to 'float64':

        .. code-block:: python

            dde.config.set_default_float("float64")
    """
    global LBFGS_options
    LBFGS_options["lr"] = lr
    LBFGS_options["maxcor"] = maxcor
    LBFGS_options["ftol"] = ftol
    LBFGS_options["gtol"] = gtol
    LBFGS_options["maxiter"] = maxiter
    LBFGS_options["maxfun"] = maxfun if maxfun is not None else int(maxiter * 1.25)
    LBFGS_options["maxls"] = maxls


def set_NNCG_options(
    lr=1,
    rank=50,
    mu=1e-1,
    updatefreq=20,
    chunksz=1,
    cgtol=1e-16,
    cgmaxiter=1000,
    lsfun="armijo",
    verbose=False,
):
    """Sets the hyperparameters of NysNewtonCG (NNCG).

    The NNCG optimizer only supports PyTorch.

    Args:
        lr (float):
            Learning rate (before line search).
        rank (int):
            Rank of preconditioner matrix used in preconditioned conjugate gradient.
        mu (float):
            Hessian damping parameter.
        updatefreq (int):
            How often the preconditioner matrix in preconditioned
            conjugate gradient is updated. This parameter is not directly used in NNCG,
            instead it is used in _train_pytorch_nncg in deepxde/model.py.
        chunksz (int):
            Number of Hessian-vector products to compute in parallel when constructing
            preconditioner. If `chunk_size` is 1, the Hessian-vector products are
            computed serially.
        cgtol (float):
            Convergence tolerance for the conjugate gradient method. The iteration stops
            when `||r||_2 <= cgtol`, where `r` is the residual. Note that this condition
            is based on the absolute tolerance, not the relative tolerance.
        cgmaxiter (int):
            Maximum number of iterations for the conjugate gradient method.
        lsfun (str):
            The line search function used to find the step size. The default value is
            "armijo". The other option is None.
        verbose (bool):
            If `True`, prints the eigenvalues of the Nyström approximation
            of the Hessian.
    """
    NNCG_options["lr"] = lr
    NNCG_options["rank"] = rank
    NNCG_options["mu"] = mu
    NNCG_options["updatefreq"] = updatefreq
    NNCG_options["chunksz"] = chunksz
    NNCG_options["cgtol"] = cgtol
    NNCG_options["cgmaxiter"] = cgmaxiter
    NNCG_options["lsfun"] = lsfun
    NNCG_options["verbose"] = verbose


def set_PSO_options(
    pop_size=30,
    b=0.9,
    c1=8e-2,
    c2=5e-1,
    lr=1e-3,
    betas=(0.99, 0.999),
    c_decrease=False,
    variance=1,
    epsilon=1e-8,
    n_iter=2000,
):
    """Sets the hyperparameters of Particle Swarm Optimization (PSO).

    The PSO optimizer only supports PyTorch.

    Args:
        pop_size (int): Population size of the PSO swarm.
        b (float): Inertia of the particles.
        c1 (float): The p-best coefficient.
        c2 (float): The g-best coefficient.
        lr (float): Learning rate for gradient descent component (0 disables it).
        betas (tuple): Same coefficients as in Adam algorithm.
        c_decrease (bool): Whether to decrease c1 and increase c2 over iterations.
        variance (float): Variance parameter for swarm initialization.
        epsilon (float): Small constant for gradient descent (like in Adam).
        n_iter (int): Number of iterations for c_decrease schedule.
    """
    PSO_options["pop_size"] = pop_size
    PSO_options["b"] = b
    PSO_options["c1"] = c1
    PSO_options["c2"] = c2
    PSO_options["lr"] = lr
    PSO_options["betas"] = betas
    PSO_options["c_decrease"] = c_decrease
    PSO_options["variance"] = variance
    PSO_options["epsilon"] = epsilon
    PSO_options["n_iter"] = n_iter


def set_ZOCGE_options(
    mu=1e-3,
    sparsity=0.0,
    prune_method="random",
    remask_interval=0,
    feature_reuse=False,
    grasp_sample_size=0,
):
    """Sets the hyperparameters of zeroth-order full CGE optimizer."""
    ZOCGE_options["mu"] = mu
    ZOCGE_options["sparsity"] = sparsity
    ZOCGE_options["prune_method"] = prune_method
    ZOCGE_options["remask_interval"] = remask_interval
    ZOCGE_options["feature_reuse"] = feature_reuse
    ZOCGE_options["grasp_sample_size"] = grasp_sample_size


def set_SOAP_options(
    beta1=0.99,
    beta2=0.999,
    shampoo_beta=None,
    epsilon=1e-8,
    precondition_frequency=10,
    max_precondition_dim=4096,
    bias_correction=True,
):
    """Sets the hyperparameters of SOAP optimizer.

    The SOAP optimizer only supports PyTorch. In this implementation, matrix
    preconditioning is applied to 2D tensors, while 1D tensors and scalars use
    an AdamW-style fallback.

    Args:
        beta1 (float): Gradient momentum coefficient.
        beta2 (float): Second-moment coefficient for AdamW fallback.
        shampoo_beta (float): Matrix statistics coefficient. If ``None``, uses beta2.
        epsilon (float): Numerical stability constant.
        precondition_frequency (int): Matrix inverse-root update frequency.
        max_precondition_dim (int): Maximum matrix side for preconditioning.
        bias_correction (bool): Whether to apply Adam-style bias correction.
    """
    SOAP_options["beta1"] = beta1
    SOAP_options["beta2"] = beta2
    SOAP_options["shampoo_beta"] = beta2 if shampoo_beta is None else shampoo_beta
    SOAP_options["epsilon"] = epsilon
    SOAP_options["precondition_frequency"] = precondition_frequency
    SOAP_options["max_precondition_dim"] = max_precondition_dim
    SOAP_options["bias_correction"] = bias_correction


def set_KLOPT_options(
    beta1=0.9,
    beta2=0.98,
    shampoo_beta=None,
    epsilon=1e-8,
    precondition_frequency=10,
    using_klsoap=False,
    normalize_grads=False,
    init_factor=0.1,
    using_damping=False,
    using_clamping=True,
    max_clamp_value=4000,
    cast_dtype="bfloat16",
):
    """Sets the hyperparameters of KL-Shampoo/KL-SOAP.

    The optimizer only supports PyTorch. Supported matrix/tensor parameters use
    the prototype KLOpt algorithm; biases, scalars, and singleton matrices use
    an AdamW fallback so it can be selected through the regular DeepXDE API.

    Args:
        beta1 (float): First-moment coefficient.
        beta2 (float): KL curvature and KL-SOAP second-moment coefficient.
        shampoo_beta (float): Curvature coefficient. If ``None``, uses beta2.
        epsilon (float): Numerical stability constant and optional damping.
        precondition_frequency (int): Eigenbasis QR update frequency.
        using_klsoap (bool): Select KL-SOAP instead of KL-Shampoo for ``klopt``.
        normalize_grads (bool): Normalize each KL-preconditioned update.
        init_factor (float): Initial eigenvalue estimate.
        using_damping (bool): Add damping while learning curvature factors.
        using_clamping (bool): Clamp inverse square-root eigenvalue estimates.
        max_clamp_value (int): Upper bound used by eigenvalue clamping.
        cast_dtype: PyTorch dtype or one of ``float32``, ``float64``,
            ``float16``, and ``bfloat16``.
    """
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("KLOpt beta values must be in [0, 1)")
    if shampoo_beta is not None and not 0 <= shampoo_beta < 1:
        raise ValueError("shampoo_beta must be in [0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if precondition_frequency < 1:
        raise ValueError("precondition_frequency must be >= 1")
    if init_factor <= 0:
        raise ValueError("init_factor must be positive")
    if max_clamp_value < 1:
        raise ValueError("max_clamp_value must be >= 1")
    KLOPT_options["beta1"] = beta1
    KLOPT_options["beta2"] = beta2
    KLOPT_options["shampoo_beta"] = -1 if shampoo_beta is None else shampoo_beta
    KLOPT_options["epsilon"] = epsilon
    KLOPT_options["precondition_frequency"] = precondition_frequency
    KLOPT_options["using_klsoap"] = using_klsoap
    KLOPT_options["normalize_grads"] = normalize_grads
    KLOPT_options["init_factor"] = init_factor
    KLOPT_options["using_damping"] = using_damping
    KLOPT_options["using_clamping"] = using_clamping
    KLOPT_options["max_clamp_value"] = max_clamp_value
    KLOPT_options["cast_dtype"] = cast_dtype


def set_MUON_options(
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    adam_lr=3e-4,
    adam_betas=(0.9, 0.95),
    adam_eps=1e-10,
    muon_weight_decay=0.0,
    adam_weight_decay=0.0,
):
    """Sets the hyperparameters of MuonWithAuxAdam.

    The PyTorch Muon integration applies Muon only to hidden ``Linear.weight``
    matrices. Input/output layer weights, biases, external trainable variables,
    and any non-linear-layer parameters are handled by the auxiliary Adam
    branch.

    Args:
        momentum (float): Muon momentum coefficient.
        nesterov (bool): Whether to use Nesterov momentum for Muon updates.
        ns_steps (int): Number of Newton-Schulz iterations.
        adam_lr (float): Auxiliary Adam learning rate. If ``None``, uses the
            ``lr`` passed to ``Model.compile``.
        adam_betas (tuple): Adam beta coefficients for auxiliary parameters.
        adam_eps (float): Adam numerical stability constant.
        muon_weight_decay (float): Decoupled weight decay for Muon parameters.
        adam_weight_decay (float): Decoupled weight decay for auxiliary
            parameters.
    """
    MUON_options["momentum"] = momentum
    MUON_options["nesterov"] = nesterov
    MUON_options["ns_steps"] = ns_steps
    MUON_options["adam_lr"] = adam_lr
    MUON_options["adam_betas"] = adam_betas
    MUON_options["adam_eps"] = adam_eps
    MUON_options["muon_weight_decay"] = muon_weight_decay
    MUON_options["adam_weight_decay"] = adam_weight_decay


def set_MOP_options(
    momentum=0.95,
    nesterov=False,
    scale_mode="nuclear_norm",
    extra_scale_factor=1.0,
    adam_lr=3e-4,
    adam_betas=(0.9, 0.95),
    adam_eps=1e-10,
    mop_weight_decay=0.01,
    adam_weight_decay=0.0,
):
    """Sets the hyperparameters of MOPWithAuxAdam.

    Hidden ``Linear.weight`` matrices use NVIDIA MOP's exact SVD polar
    decomposition. Other parameters use the auxiliary Adam branch.
    """
    if scale_mode not in {"nuclear_norm", "shape_scaling", "spectral", "unit_rms_norm"}:
        raise ValueError(f"Invalid MOP scale mode: {scale_mode}")
    MOP_options["momentum"] = momentum
    MOP_options["nesterov"] = nesterov
    MOP_options["scale_mode"] = scale_mode
    MOP_options["extra_scale_factor"] = extra_scale_factor
    MOP_options["adam_lr"] = adam_lr
    MOP_options["adam_betas"] = adam_betas
    MOP_options["adam_eps"] = adam_eps
    MOP_options["mop_weight_decay"] = mop_weight_decay
    MOP_options["adam_weight_decay"] = adam_weight_decay


def set_MOUSSE_options(
    momentum=0.95,
    lion_betas=(0.9, 0.95),
    epsilon=1e-8,
    nesterov=False,
    adjust_lr="spectral_norm",
    shampoo_epsilon=1e-10,
    shampoo_beta=0.95,
    shampoo_update_frequency=10,
    shampoo_alpha=0.125,
    lr_correction=True,
    apply_norm=True,
    use_l_or_r=0,
    mousse_weight_decay=0.01,
    lion_weight_decay=0.0,
):
    """Sets hyperparameters for Mousse with an auxiliary Lion optimizer."""
    if not 0 <= momentum < 1:
        raise ValueError("Mousse momentum must be in [0, 1)")
    if len(lion_betas) != 2 or any(not 0 <= beta < 1 for beta in lion_betas):
        raise ValueError("Mousse Lion betas must be in [0, 1)")
    if epsilon <= 0:
        raise ValueError("Mousse epsilon must be positive")
    if adjust_lr not in ("spectral_norm", "rms_norm", "None", None):
        raise ValueError("adjust_lr must be spectral_norm, rms_norm, or None")
    if shampoo_epsilon <= 0 or not 0 <= shampoo_beta < 1:
        raise ValueError("Invalid Mousse Shampoo epsilon or beta")
    if shampoo_update_frequency < 1 or shampoo_alpha < 0:
        raise ValueError("Invalid Mousse Shampoo frequency or alpha")
    if use_l_or_r not in (0, 1, 2):
        raise ValueError("use_l_or_r must be 0, 1, or 2")
    if mousse_weight_decay < 0 or lion_weight_decay < 0:
        raise ValueError("Mousse weight decay values must be non-negative")
    MOUSSE_options.update(
        momentum=momentum,
        lion_betas=lion_betas,
        epsilon=epsilon,
        nesterov=nesterov,
        adjust_lr=adjust_lr,
        shampoo_epsilon=shampoo_epsilon,
        shampoo_beta=shampoo_beta,
        shampoo_update_frequency=shampoo_update_frequency,
        shampoo_alpha=shampoo_alpha,
        lr_correction=lr_correction,
        apply_norm=apply_norm,
        use_l_or_r=use_l_or_r,
        mousse_weight_decay=mousse_weight_decay,
        lion_weight_decay=lion_weight_decay,
    )


def set_PSGDPRO_options(
    momentum=0.9,
    beta_lip=0.9,
    preconditioner_lr=0.1,
    preconditioner_init_scale=1.0,
    damping_noise_scale=0.1,
    min_preconditioner_lr=0.01,
    warmup_steps=10000,
    max_update_rms=0.0,
    weight_decay_method="decoupled",
    psgd_weight_decay=0.01,
    auxiliary_betas=(0.9, 0.999),
    auxiliary_epsilon=1e-8,
    auxiliary_weight_decay=0.0,
):
    """Sets hyperparameters for NVIDIA PSGDPro."""
    if not 0 <= momentum < 1 or not 0 <= beta_lip < 1:
        raise ValueError("PSGDPro momentum and beta_lip must be in [0, 1)")
    if preconditioner_lr <= 0 or min_preconditioner_lr < 0:
        raise ValueError("PSGDPro preconditioner learning rates are invalid")
    if preconditioner_init_scale <= 0 or warmup_steps < 1:
        raise ValueError("PSGDPro initialization scale and warmup must be positive")
    if damping_noise_scale < 0 or max_update_rms < 0:
        raise ValueError("PSGDPro damping and maximum RMS must be non-negative")
    if weight_decay_method not in ("decoupled", "independent", "l2", "palm"):
        raise ValueError("Invalid PSGDPro weight decay method")
    if len(auxiliary_betas) != 2 or any(
        not 0 <= beta < 1 for beta in auxiliary_betas
    ):
        raise ValueError("PSGDPro auxiliary betas must be in [0, 1)")
    if auxiliary_epsilon <= 0:
        raise ValueError("PSGDPro auxiliary epsilon must be positive")
    if psgd_weight_decay < 0 or auxiliary_weight_decay < 0:
        raise ValueError("PSGDPro weight decay values must be non-negative")
    PSGDPRO_options.update(
        momentum=momentum,
        beta_lip=beta_lip,
        preconditioner_lr=preconditioner_lr,
        preconditioner_init_scale=preconditioner_init_scale,
        damping_noise_scale=damping_noise_scale,
        min_preconditioner_lr=min_preconditioner_lr,
        warmup_steps=warmup_steps,
        max_update_rms=max_update_rms,
        weight_decay_method=weight_decay_method,
        psgd_weight_decay=psgd_weight_decay,
        auxiliary_betas=auxiliary_betas,
        auxiliary_epsilon=auxiliary_epsilon,
        auxiliary_weight_decay=auxiliary_weight_decay,
    )


def set_PCGRAD_options(base_optimizer="adam"):
    """Sets the hyperparameters of PCGrad optimizer wrapper.

    Args:
        base_optimizer (str): Wrapped optimizer. Currently only "adam" is used,
            matching the public PCGrad example that wraps Adam.
    """
    PCGRAD_options["base_optimizer"] = base_optimizer


def set_SSBROYDEN_options(
    lr=1.0,
    tolerance_grad=1e-10,
    debug=False,
    debug_every=100,
):
    """Sets the hyperparameters of SSBroyden optimizer.

    The SSBroyden optimizer only supports PyTorch.

    Args:
        lr (float): Initial line-search step.
        tolerance_grad (float): First-order optimality tolerance.
        debug (bool): Whether to print SSBroyden step diagnostics.
        debug_every (int): Print diagnostics every this many optimizer steps.
    """
    SSBROYDEN_options["lr"] = lr
    SSBROYDEN_options["tolerance_grad"] = tolerance_grad
    SSBROYDEN_options["debug"] = debug
    SSBROYDEN_options["debug_every"] = debug_every


def set_CAUSAL_options(
    base_optimizer="adam",
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
    """Sets the hyperparameters of causal optimizer wrapper.

    Args:
        base_optimizer: One of "adam", "soap", "muon", "L-BFGS", "L-BFGS-B", "PSO".
        n_time_bins: Number of temporal bins.
        start_bins: Number of initially active temporal bins.
        time_index: Column index of time in train_x. Usually -1.
        unlock_every: Open one more bin every N optimizer steps.
        unlock_tol: If not None, open next bin when active loss <= unlock_tol.
        min_steps_per_bin: Minimum steps before tolerance-based unlock.
        bc_mode: "all" keeps IC/BC full and PDE causal; "causal" keeps
            IC full and makes BC/PDE causal.
        min_points_per_bc: If bc_mode="causal" and a BC block becomes empty,
            keep this many earliest points from that BC block.
        causal_strategy: "prefix" for growing causal prefix or "cyclic_windows"
            for rolling windows.
        steps_per_window: Number of optimizer steps per cyclic window.
        state_alpha: Smoothing factor for stored cyclic window states.
        x_state: Spatial coordinates used for cyclic pseudo-IC states.
        window_ic_weight: Weight of pseudo-IC loss for cyclic windows.
        verbose: Print unlock logs.
    """
    CAUSAL_options["base_optimizer"] = base_optimizer
    CAUSAL_options["n_time_bins"] = n_time_bins
    CAUSAL_options["start_bins"] = start_bins
    CAUSAL_options["time_index"] = time_index
    CAUSAL_options["unlock_every"] = unlock_every
    CAUSAL_options["unlock_tol"] = unlock_tol
    CAUSAL_options["min_steps_per_bin"] = min_steps_per_bin
    CAUSAL_options["bc_mode"] = bc_mode
    CAUSAL_options["min_points_per_bc"] = min_points_per_bc
    CAUSAL_options["causal_strategy"] = causal_strategy
    CAUSAL_options["steps_per_window"] = steps_per_window
    CAUSAL_options["state_alpha"] = state_alpha
    CAUSAL_options["x_state"] = x_state
    CAUSAL_options["window_ic_weight"] = window_ic_weight
    CAUSAL_options["verbose"] = verbose


def set_hvd_opt_options(
    compression=None,
    op=None,
    backward_passes_per_step=1,
    average_aggregated_gradients=False,
):
    """Sets the parameters of hvd.DistributedOptimizer.

    The default parameters are the same as for `hvd.DistributedOptimizer <https://horovod.readthedocs.io/en/stable/api.html>`_.

    Args:
        compression: Compression algorithm used to reduce the amount of data
            sent and received by each worker node.  Defaults to not using compression.
        op: The reduction operation to use when combining gradients across different ranks. Defaults to Average.
        backward_passes_per_step (int): Number of backward passes to perform before calling
            hvd.allreduce. This allows accumulating updates over multiple mini-batches before reducing and applying them.
        average_aggregated_gradients (bool): Whether to average the aggregated gradients that have been accumulated over
            multiple mini-batches. If true divides gradient updates by backward_passes_per_step. Only applicable for
            backward_passes_per_step > 1.
    """
    if compression is None:
        compression = hvd.compression.Compression.none
    if op is None:
        op = hvd.Average
    hvd_opt_options["compression"] = compression
    hvd_opt_options["op"] = op
    hvd_opt_options["backward_passes_per_step"] = backward_passes_per_step
    hvd_opt_options["average_aggregated_gradients"] = average_aggregated_gradients


set_LBFGS_options()
set_NNCG_options()
set_PSO_options()
set_SOAP_options()
set_KLOPT_options()
set_MUON_options()
set_MOP_options()
set_MOUSSE_options()
set_PSGDPRO_options()
set_PCGRAD_options()
set_SSBROYDEN_options()
set_CAUSAL_options()
if hvd is not None:
    set_hvd_opt_options()


# Backend-dependent options
if backend_name in ["pytorch", "paddle"]:
    # number of iterations per optimization call
    LBFGS_options["iter_per_step"] = min(1000, LBFGS_options["maxiter"])
    LBFGS_options["fun_per_step"] = (
        LBFGS_options["maxfun"]
        * LBFGS_options["iter_per_step"]
        // LBFGS_options["maxiter"]
    )
