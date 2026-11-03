import torch


def mc_return(seq, gamma, *, device):
    ret = torch.zeros((), dtype=torch.float32, device=device)
    discount = torch.tensor(1.0, dtype=torch.float32, device=device)
    gamma_t = torch.tensor(float(gamma), dtype=torch.float32, device=device)

    for tr in seq:
        ret = ret + discount * torch.tensor(float(tr.reward), dtype=torch.float32, device=device)
        discount = discount * gamma_t

    return ret


def collect_chain_mc_metrics(agent, seqs, q_sa, gamma, weights=None):
    if not seqs:
        return {
            "chain_mc_abs_error_mean": 0.0,
            "chain_mc_mse": 0.0,
            "chain_mc_return_abs_mean": 0.0,
            "chain_mc_error_norm": 0.0,
            "chain_mc_q_corr": 0.0,
        }

    with torch.no_grad():
        mc_returns = torch.stack([
            mc_return(seq, gamma, device=agent.device)
            for seq in seqs
        ])
        q_values = q_sa.detach()
        mc_errors = q_values - mc_returns
        if weights is None:
            weights = torch.ones_like(mc_returns)
        else:
            weights = weights.detach().to(
                device=agent.device, dtype=mc_returns.dtype
            )
        weight_sum = weights.sum().clamp_min(1e-8)

        def weighted_mean(values):
            return (values * weights).sum() / weight_sum

        mc_return_abs_mean = weighted_mean(mc_returns.abs()).clamp_min(1e-8)
        q_centered = q_values - weighted_mean(q_values)
        mc_centered = mc_returns - weighted_mean(mc_returns)
        q_std = weighted_mean(q_centered.square()).sqrt()
        mc_std = weighted_mean(mc_centered.square()).sqrt()

        if q_std.item() < 1e-8 or mc_std.item() < 1e-8:
            q_corr = 0.0
        else:
            covariance = weighted_mean(q_centered * mc_centered)
            q_corr = float(covariance.div(q_std * mc_std).item())

        return {
            "chain_mc_abs_error_mean": float(weighted_mean(mc_errors.abs()).item()),
            "chain_mc_mse": float(weighted_mean(mc_errors.square()).item()),
            "chain_mc_return_abs_mean": float(mc_return_abs_mean.item()),
            "chain_mc_error_norm": float(
                weighted_mean(mc_errors.abs()).div(mc_return_abs_mean).item()
            ),
            "chain_mc_q_corr": q_corr,
        }
