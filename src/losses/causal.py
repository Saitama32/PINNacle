import torch


def causal_residual_loss(
    residual,
    t,
    num_chunks,
    tol,
    include_ic_in_weights=False,
    ic_loss=None,
    ic_weight_in_causal=0.0,
):
    """Causal PDE residual loss.

    The PDE residual is computed by the existing equation code. This function
    only reweights residual-squared chunks ordered by time.
    """
    t = t.reshape(-1)
    r2 = residual.reshape(-1).pow(2)

    idx = torch.argsort(t)
    r2 = r2[idx]

    n = r2.numel()
    if n < num_chunks:
        raise ValueError(f"num_chunks={num_chunks} > number of residual points={n}")

    chunk_size = n // num_chunks
    used = chunk_size * num_chunks
    r2 = r2[:used]
    chunk_losses = r2.reshape(num_chunks, chunk_size).mean(dim=1)

    matrix = torch.tril(
        torch.ones(num_chunks, num_chunks, device=r2.device, dtype=r2.dtype),
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
    return loss, diagnostics

