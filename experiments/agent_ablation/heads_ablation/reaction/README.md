# Offline diagnostics for reaction-diffusion heads

This directory analyzes saved RL checkpoints and transition assets without
changing or training the production agent.

The default labels are the historical `G` and `B` agents. They use the current
five-optimizer action space (`Adam`, `LBFGS`, `PSO`, `SOAP`, `Muon`). The loader
also verifies that the configured optimizer count matches the checkpoint output
layer, so an incompatible action-space profile fails instead of producing a
mislabelled report.

## 1. Inspect assets without downloading them

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis inventory
```

The inventory reports paired checkpoint steps, transition count, and known
asset sizes. Optimizer and parameter snapshots are paired only at exactly the
same step.

## 2. Analyze online transitions from G and B

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis analyze --device cuda
```

## 3. Add the fixed farm buffer

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis analyze --device cuda --include-farm
```

Large transition assets are downloaded only once. The cache stores state,
action, reward, terminal and loss information but removes `solver_models`.
Downloads use five workers by default, matching the current training-buffer
loader. Override this with `--download-workers N`; reducing it can help if the
Comet connection starts throttling.

The default farm source is the eight-experiment project
`rlpinn-reaction-diffusion-nu0-01-rho20-0-tolerance-expand-set`, which supplied
the original replay data associated with the G/B setup.

For a quick smoke run, select a few evenly spaced checkpoints:

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis analyze --max-checkpoints 3 --batch-size 256
```

Outputs:

- `coverage.csv`: unique physical transition/chain coverage by joint action;
- `optimizer_head_history.csv`: optimizer Q spread, margin, entropy and routing;
- `parameter_head_history.csv`: the same per optimizer and parameter;
- `factual_q_history.csv`: observed-action Q/rank/gap and MC return;
- `factual_calibration_history.csv`: factual Q-to-MC correlations.
- `optimizer_factual_calibration_history.csv`: the same factual calibration
  once per transition for the optimizer head.

Coverage alone can be regenerated from the cached transitions without running
any checkpoint inference:

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis coverage --include-farm
```

Calibration tables can likewise be regenerated from `factual_q_history.csv`:

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis calibrate
```

The final G/B optimizer and parameter checkpoints can be crossed on every
cached state as GG/GB/BG/BB without retraining:

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis compare --include-farm --device cuda
```

This writes per-state hybrid actions, G/B head-choice agreement, and factual
exact-match summaries. Exact-match rewards are observational diagnostics, not
counterfactual estimates for actions that were not logged.

History files contain two complementary `evaluation_scope` values:

- `fixed_all_online` evaluates every checkpoint on the same combined G+B
  archive, which makes checkpoints directly comparable;
- `chronological` uses the fixed farm archive plus only the checkpoint's own
  online transitions collected by that step. MC returns are rebuilt after the
  cutoff, so future rewards do not leak into an early snapshot.

The factual calibration files do not identify the best unobserved action. That
requires sufficient overlap between similar logged states or a separate
counterfactual PINN action sweep from saved solver models.
