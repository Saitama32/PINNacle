__all__ = [
    "set_LBFGS_options",
    "set_NNCG_options",
    "set_PSO_options",
    "set_SOAP_options",
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
SOAP_options = {}
PCGRAD_options = {}
SSBROYDEN_options = {}
CAUSAL_options = {}
hvd = None
if hvd is not None:
    hvd_opt_options = {}


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


def set_PCGRAD_options(base_optimizer="adam"):
    """Sets the hyperparameters of PCGrad optimizer wrapper.

    Args:
        base_optimizer (str): Wrapped optimizer. Currently only "adam" is used,
            matching the public PCGrad example that wraps Adam.
    """
    PCGRAD_options["base_optimizer"] = base_optimizer


def set_SSBROYDEN_options(lr=1.0, tolerance_grad=1e-10):
    """Sets the hyperparameters of SSBroyden optimizer.

    The SSBroyden optimizer only supports PyTorch.

    Args:
        lr (float): Initial line-search step.
        tolerance_grad (float): First-order optimality tolerance.
    """
    SSBROYDEN_options["lr"] = lr
    SSBROYDEN_options["tolerance_grad"] = tolerance_grad


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
        base_optimizer: One of "adam", "soap", "L-BFGS", "L-BFGS-B", "PSO".
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
