import torch


def temporal_bin_indices(t, num_chunks, t_min, t_max):
    """Map times to fixed, equal-width bins over ``[t_min, t_max]``."""
    num_chunks = int(num_chunks)
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")

    t = t.reshape(-1)
    if t.numel() == 0:
        raise ValueError("t must contain at least one value")
    t_min = float(t_min)
    t_max = float(t_max)
    if not t_min < t_max:
        raise ValueError(f"t_min must be smaller than t_max, got {t_min} and {t_max}")

    bounds = torch.linspace(
        t_min,
        t_max,
        num_chunks + 1,
        dtype=t.dtype,
        device=t.device,
    )
    tolerance = torch.finfo(t.dtype).eps * max(1.0, abs(t_min), abs(t_max)) * 8
    if bool(torch.any(t < t_min - tolerance)) or bool(torch.any(t > t_max + tolerance)):
        raise ValueError(
            f"time values must lie within the fixed PDE interval [{t_min}, {t_max}]"
        )
    indices = torch.bucketize(t, bounds[1:-1], right=True)
    return indices, bounds


def temporal_chunk_losses(
    residual,
    t,
    num_chunks,
    t_min,
    t_max,
    return_details=False,
):
    """Return mean squared residuals in fixed, equal-width temporal bins."""
    num_chunks = int(num_chunks)

    t = t.reshape(-1)
    r2 = residual.reshape(-1).pow(2)
    if t.numel() != r2.numel():
        raise ValueError(
            "residual and t must contain the same number of scalar values, "
            f"got {r2.numel()} and {t.numel()}"
        )

    bin_indices, bounds = temporal_bin_indices(t, num_chunks, t_min, t_max)
    counts = torch.bincount(bin_indices, minlength=num_chunks)
    empty = torch.nonzero(counts == 0, as_tuple=False).reshape(-1)
    if empty.numel() > 0:
        empty_ids = ", ".join(str(int(value)) for value in empty.detach().cpu())
        raise ValueError(
            "causal temporal bins contain no PDE residual points: "
            f"{empty_ids}; sampling must cover every equal-width time bin"
        )

    sums = torch.zeros(num_chunks, dtype=r2.dtype, device=r2.device)
    sums.scatter_add_(0, bin_indices, r2)
    chunk_losses = sums / counts.to(dtype=r2.dtype)
    if return_details:
        return chunk_losses, {
            "t_min": bounds[:-1].detach(),
            "t_max": bounds[1:].detach(),
            "counts": counts.detach(),
        }
    return chunk_losses


def causal_loss_with_fixed_weights(
    residual, t, num_chunks, fixed_weights, t_min, t_max
):
    """Evaluate causal residual loss without recomputing its curriculum weights."""
    chunk_losses = temporal_chunk_losses(
        residual, t, num_chunks, t_min=t_min, t_max=t_max
    )
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
    t_min,
    t_max,
    include_ic_in_weights=False,
    ic_loss=None,
    ic_weight_in_causal=0.0,
    return_details=False,
):
    """Reweight time-ordered PDE residual chunks for causal training."""
    chunk_losses, chunk_details = temporal_chunk_losses(
        residual,
        t,
        num_chunks,
        t_min=t_min,
        t_max=t_max,
        return_details=True,
    )

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
    ic_offset = torch.zeros((), device=chunk_losses.device, dtype=chunk_losses.dtype)

    if include_ic_in_weights:
        if ic_loss is None:
            raise ValueError("ic_loss must be provided when include_ic_in_weights=True")
        ic_offset = float(ic_weight_in_causal) * ic_loss.detach()
        cumulative_prev = cumulative_prev + ic_offset

    weights = torch.exp(-float(tol) * cumulative_prev.detach())
    post_chunk_weights = torch.exp(
        -float(tol) * (ic_offset + torch.cumsum(chunk_losses.detach(), dim=0))
    )
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
            "post_chunk_weights": post_chunk_weights.detach(),
            "t_min": chunk_details["t_min"],
            "t_max": chunk_details["t_max"],
            "counts": chunk_details["counts"],
        }
        return loss, diagnostics, details
    return loss, diagnostics
