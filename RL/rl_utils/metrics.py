import math
from collections import defaultdict

import torch


def policy_entropy_from_counts(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for v in counts.values():
        if v <= 0:
            continue
        p = v / total
        entropy -= p * math.log(p + 1e-12)

    return float(entropy)


def collect_policy_metrics_from_q(agent, q_opt, prefix):
    """
    q_opt: Tensor[B, n_actions]

    Evaluates the current greedy policy of agent.model_optim
    on a batch of replay states.

    This function does not affect training.
    """
    with torch.no_grad():
        greedy_actions = q_opt.argmax(dim=1).detach().cpu().tolist()

    counts = defaultdict(int)

    for action_idx in greedy_actions:
        optim_name = agent.i2opt[int(action_idx)]
        counts[optim_name] += 1

    total = len(greedy_actions)
    metrics = {}

    for _, optim_name in agent.i2opt.items():
        count = counts.get(optim_name, 0)
        frac = count / total if total > 0 else 0.0

        metrics[f"{prefix}/frac/{optim_name}"] = frac

    metrics[f"{prefix}/entropy"] = policy_entropy_from_counts(counts)

    return metrics


def collect_policy_metrics_by_seq_position(agent, seqs):
    """
    Evaluates what the current greedy policy would choose
    for replay states from sampled sequences.

    Buckets:
        - early
        - middle
        - late

    This is policy evaluation on replay states, not logging actions
    that were actually taken in the environment.
    """
    buckets = {
        "early": [],
        "middle": [],
        "late": [],
    }

    for seq in seqs:
        T = len(seq)
        if T == 0:
            continue

        for t, tr in enumerate(seq):
            pos = t / max(T - 1, 1)

            if pos < 0.33:
                bucket = "early"
            elif pos < 0.66:
                bucket = "middle"
            else:
                bucket = "late"

            buckets[bucket].append(tr.state)

    metrics = {}

    for bucket_name, states_list in buckets.items():
        if len(states_list) == 0:
            continue

        states_tensor = torch.stack([
            agent._stack_state(s) for s in states_list
        ]).to(agent.device)

        with torch.no_grad():
            _, q_opt = agent.model_optim(states_tensor)

        metrics.update(
            collect_policy_metrics_from_q(
                agent,
                q_opt,
                prefix=f"policy_eval/{bucket_name}"
            )
        )

    return metrics