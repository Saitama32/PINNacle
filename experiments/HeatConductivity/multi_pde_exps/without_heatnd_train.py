import os
import sys

os.environ["DDEBACKEND"] = "pytorch"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from comet_ml import start

COMET_API_KEY = "aP71fQTYPNqfsYWvudPPmoBl5"
COMET_WORKSPACE = "saitama32"
COMET_TARGET_PROJECT = "rlpinn-heat-pde-agent-opt-without-heatnd"

experiment = start(
    api_key=COMET_API_KEY,
    project_name=COMET_TARGET_PROJECT,
    workspace=COMET_WORKSPACE,
)

import argparse
import statistics

COMET_EXPERIMENT_METADATA = {
    "param": "v_1",
    "reward_function": "v_2",
    "description": "no_heatnd_agent_optimization",
}
COMET_SOURCE_PROJECTS = {
    "rlpinn-heat-2d-vc-farm-transitions": {
        "n_exps": 200,
        "tolerance": 0.0585015359142309,
        "prev_tol": 0.0,
        "use_tol": True,
        "new_tol": False,
        "use_log_state": False,
    },
    "rlpinn-heat2d-multiscale-tolerance-corrected": {
        "n_exps": 200,
        "tolerance": 0.006643642,
        "prev_tol": 0.0,
        "use_tol": False,
        "new_tol": True,
        "use_log_state": False,
    },
    "rlpinn-heat-2d-cg-farm-trans": {
        "n_exps": 200,
        "tolerance": 0.0455133201723103,
        "prev_tol": 0.0,
        "use_tol": True,
        "new_tol": False,
        "use_log_state": False,
    },
    "rlpinn-heat2d-longtime-tolerance": {
        "n_exps": 200,
        "tolerance": 1.06494992027684,
        "prev_tol": 0.0,
        "use_tol": False,
        "new_tol": True,
        "use_log_state": False,
    },
    "rlpinn-heatinv-tolerance": {
        "n_exps": 200,
        "tolerance": 0.0588327879086136,
        "prev_tol": 0.0,
        "use_tol": False,
        "new_tol": True,
        "use_log_state": False,
    },
}
ALREADY_LOGGED_PROJECTS = {
    "rlpinn-burgers2d-tolerance",
    "rlpinn-grayscott-farm-transitions",
    "rlpinn-heat-2d-cg-farm-trans",
    "rlpinn-heat2d-longtime-tolerance",
    "rlpinn-heat2d-multiscale-tolerance-corrected",
    "rlpinn-heatinv-tolerance",
    "rlpinn-heatnd-tolerance",
    "rlpinn-ns2d-backstep-tolerance",
    "rlpinn-ns2d-liddriven-farm-transitions",
    "rlpinn-ns2d-longtime-tolerance",
    "rlpinn-poisson3d-complexgeometry-fram-trans",
    "rlpinn-poissoninv-tolerance",
    "rlpinn-poissonnd-farm-transitions",
    "rlpinn-wave2d-heterogeneous-tolerance",
}
COMET_LOAD_NUM_WORKERS = 8
SOURCE_BUFFER_CAPACITY = 500_000
SOURCE_BUFFER_KEEP_LAST = 5_000
MERGED_BUFFER_CAPACITY = 500_000

import torch

from RL.rl_algorithms import DQNAgent, PrioritizedReplayBuffer
from RL.rl_utils.load_buffer.load_exps_from_comet import collect_all_comet_transitions


OPTIMIZER_DICT = {
    "Adam": {
        "lr": [1e-2, 1e-3, 1e-4],
        "epochs": [100, 1000, 2500],
    },
    "LBFGS": {
        "lr": [1, 5e-1, 1e-1],
        "epochs": [100, 500, 1000],
    },
    "PSO": {
        "lr": [0.0, 1e-3, 1e-4],
        "epochs": [100, 200, 300],
    },
}


def load_project_buffers():
    buffers = {}

    for project_name, cfg in COMET_SOURCE_PROJECTS.items():
        print(f"Loading transitions from Comet project: {project_name}")
        buffers[project_name] = collect_all_comet_transitions(
            replay_buffer=PrioritizedReplayBuffer(capacity=SOURCE_BUFFER_CAPACITY),
            max_exps_last=cfg["n_exps"],
            proj_name=project_name,
            mark_states=True,
            tolerance=cfg["tolerance"],
            prev_tol=cfg["prev_tol"],
            use_tol=cfg["use_tol"],
            new_tol=cfg["new_tol"],
            use_log_state=cfg["use_log_state"],
            num_workers=COMET_LOAD_NUM_WORKERS,
        )
        print(f"Loaded {len(buffers[project_name])} transitions from {project_name}")

    return buffers


def shrink_buffers(buffers, keep_last):
    if keep_last is None or keep_last <= 0:
        return buffers

    trimmed = {}
    for project_name, old_buffer in buffers.items():
        new_buffer = PrioritizedReplayBuffer(capacity=keep_last)
        start_idx = max(0, len(old_buffer.memory) - keep_last)

        for transition, priority in zip(
            old_buffer.memory[start_idx:],
            old_buffer.prior[start_idx:],
        ):
            new_buffer.push(*transition, priority=priority)

        trimmed[project_name] = new_buffer
        print(
            f"Trimmed {project_name}: {len(old_buffer)} -> {len(new_buffer)} transitions"
        )

    return trimmed


def log_state_dict(state, log_losses=True, clip_delta=True):
    out = dict(state)

    if log_losses:
        for key in ("loss_total", "loss_oper", "loss_bnd"):
            if key in out and torch.is_tensor(out[key]):
                value = torch.nan_to_num(
                    out[key].float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                out[key] = torch.log1p(torch.clamp(value, min=0.0))

    if clip_delta and "delta" in out and torch.is_tensor(out["delta"]):
        delta = torch.nan_to_num(
            out["delta"].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        out["delta"] = delta.clamp(-1.0, 1.0)

    return out


def merge_buffers_no_priority(buffers, capacity=None, clip_delta=True):
    if capacity is None:
        capacity = sum(len(buffer) for buffer in buffers.values())

    merged = PrioritizedReplayBuffer(capacity=capacity)
    total = 0

    for project_name, buffer in buffers.items():
        log_losses = project_name not in ALREADY_LOGGED_PROJECTS
        print(f"Merging {project_name}: log_losses={log_losses}")

        for transition in buffer.memory:
            state = log_state_dict(
                transition.state,
                log_losses=log_losses,
                clip_delta=clip_delta,
            )
            next_state = log_state_dict(
                transition.next_state,
                log_losses=log_losses,
                clip_delta=clip_delta,
            )
            merged.push(
                state,
                next_state,
                transition.action,
                transition.reward,
                transition.done,
                transition.model_reward,
                transition.opt_model_i,
            )
            total += 1

    print(
        f"Merged transitions: {total}. Buffer size={len(merged)}, capacity={merged.capacity}"
    )
    return merged


def build_dqn_args(experiment, device, batch_size, warmup_updates, memory_size):
    dqn_args = {
        "n_observation": None,
        "n_action": len(OPTIMIZER_DICT),
        "optimizer_dict": OPTIMIZER_DICT,
        "memory_size": memory_size,
        "gamma": 0.95,
        "lr": 1e-3,
        "device": device,
        "batch_size": batch_size,
        "n_transitions_reinit": 2000,
        "exp": experiment,
        "warmup_updates": warmup_updates,
        "recalc_batch_size": batch_size,
    }
    if experiment is not None:
        experiment.log_parameters(dqn_args)
    return dqn_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-updates", type=int, default=350)
    parser.add_argument("--memory-size", type=int, default=MERGED_BUFFER_CAPACITY)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_buffers = load_project_buffers()
    trimmed_buffers = shrink_buffers(source_buffers, keep_last=SOURCE_BUFFER_KEEP_LAST)
    big_buffer = merge_buffers_no_priority(
        trimmed_buffers,
        capacity=MERGED_BUFFER_CAPACITY,
    )

    rl_agent = DQNAgent(
        **build_dqn_args(
            experiment=experiment,
            device=device,
            batch_size=args.batch_size,
            warmup_updates=args.warmup_updates,
            memory_size=args.memory_size,
        )
    )
    rl_agent.success_frac = 0.25
    rl_agent.replay_buffer = big_buffer

    print(f"Replay buffer size after Comet load: {len(rl_agent.replay_buffer)}")
    if len(rl_agent.replay_buffer) < rl_agent.batch_size:
        print(
            "Not enough transitions for optimization: "
            f"{len(rl_agent.replay_buffer)} < batch_size({rl_agent.batch_size})"
        )
        return

    for step in range(1, args.steps + 1):
        loss_optim, loss_param = rl_agent.optim_(iters=args.iters)
        rl_agent.steps_done += 1

        if loss_optim and loss_param:
            print(
                f"[{step}/{args.steps}] "
                f"optim_loss_mean={statistics.mean(loss_optim):.6f}, "
                f"param_loss_mean={statistics.mean(loss_param):.6f}"
            )
        else:
            print(f"[{step}/{args.steps}] no optimization updates were made.")
            break


if __name__ == "__main__":
    main()
