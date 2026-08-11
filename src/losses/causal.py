import torch


def temporal_chunk_losses(residual, t, num_chunks):
    """Return time-ordered mean squared residuals for equal-sized chunks."""
    num_chunks = int(num_chunks)
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")

    t = t.reshape(-1)
    r2 = residual.reshape(-1).pow(2)
    if t.numel() != r2.numel():
        raise ValueError(
            "residual and t must contain the same number of scalar values, "
            f"got {r2.numel()} and {t.numel()}"
        )

    idx = torch.argsort(t)
    r2 = r2[idx]

    n = r2.numel()
    if n < num_chunks:
        raise ValueError(f"num_chunks={num_chunks} > number of residual points={n}")

    chunk_size = n // num_chunks
    used = chunk_size * num_chunks
    return r2[:used].reshape(num_chunks, chunk_size).mean(dim=1)


def causal_loss_with_fixed_weights(residual, t, num_chunks, fixed_weights):
    """Evaluate causal residual loss without recomputing its curriculum weights."""
    chunk_losses = temporal_chunk_losses(residual, t, num_chunks)
    weights = torch.as_tensor(
        fixed_weights,
        dtype=chunk_losses.dtype,
        device=chunk_losses.device,
    ).reshape(-1)
    if weights.numel() != chunk_losses.numel():
        raise ValueError(
            "fixed_weights must have one value per temporal chunk, "
            f"got {weights.numel()} weights for {chunk_losses.numel()} chunks"
        )
    return torch.mean(weights.detach() * chunk_losses)


def causal_residual_loss(
    residual,
    t,
    num_chunks,
    tol,
    include_ic_in_weights=False,
    ic_loss=None,
    ic_weight_in_causal=0.0,
    return_details=False,
):
    """Reweight time-ordered PDE residual chunks for causal training."""
    chunk_losses = temporal_chunk_losses(residual, t, num_chunks)

    matrix = torch.tril(
        torch.ones(
            int(num_chunks),
            int(num_chunks),
            device=chunk_losses.device,
            dtype=chunk_losses.dtype,
        ),
        diagonal=-1,
    )
    cumulative_prev = matrix @ chunk_losses

    if include_ic_in_weights:
        if ic_loss is None:
            raise ValueError("ic_loss must be provided when include_ic_in_weights=True")
        cumulative_prev = cumulative_prev + float(ic_weight_in_causal) * ic_loss.detach()

    weights = torch.exp(-float(tol) * cumulative_prev.detach())
    loss = torch.mean(weights * chunk_losses)

    diagnostics = {
        "causal_weight_min": weights.min().detach(),
        "causal_weight_mean": weights.mean().detach(),
        "causal_weight_max": weights.max().detach(),
        "causal_chunk_loss_min": chunk_losses.min().detach(),
        "causal_chunk_loss_mean": chunk_losses.mean().detach(),
        "causal_chunk_loss_max": chunk_losses.max().detach(),
    }
    if return_details:
        details = {
            "weights": weights.detach(),
            "chunk_losses": chunk_losses.detach(),
        }
        return loss, diagnostics, details
    return loss, diagnostics
