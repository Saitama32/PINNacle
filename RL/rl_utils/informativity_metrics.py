import torch


def mc_return(seq, gamma, *, device):
    ret = torch.zeros((), dtype=torch.float32, device=device)
    discount = torch.tensor(1.0, dtype=torch.float32, device=device)
    gamma_t = torch.tensor(float(gamma), dtype=torch.float32, device=device)

    for tr in seq:
        ret = ret + discount * torch.tensor(float(tr.reward), dtype=torch.float32, device=device)
        discount = discount * gamma_t

    return ret


def collect_chain_mc_metrics(agent, seqs, q_sa, gamma):
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

        mc_return_abs_mean = mc_returns.abs().mean().clamp_min(1e-8)
        q_centered = q_values - q_values.mean()
        mc_centered = mc_returns - mc_returns.mean()
        q_std = q_values.std(unbiased=False)
        mc_std = mc_returns.std(unbiased=False)

        if q_std.item() < 1e-8 or mc_std.item() < 1e-8:
            q_corr = 0.0
        else:
            q_corr = float((q_centered * mc_centered).mean().div(q_std * mc_std).item())

        return {
            "chain_mc_abs_error_mean": float(mc_errors.abs().mean().item()),
            "chain_mc_mse": float((mc_errors ** 2).mean().item()),
            "chain_mc_return_abs_mean": float(mc_return_abs_mean.item()),
            "chain_mc_error_norm": float(mc_errors.abs().mean().div(mc_return_abs_mean).item()),
            "chain_mc_q_corr": q_corr,
        }
