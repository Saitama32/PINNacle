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

## 4. Return-transfer matrix without new PINN rollouts

To separate distribution specialization from temporal credit assignment, run
every saved checkpoint on three archived domains: its own online transitions,
the fixed farm buffer, and online transitions produced by other agents.  The
factual Q of the recorded action is compared with one-step reward, terminal MC
return, and the best future loss progress remaining in the physical trajectory:

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis return-matrix --cohort gb --device cuda
```

The larger target-mode cohort contains G/B plus four independent runs of each
legacy, global-one-step, joint-Soft-Watkins and per-parameter-Soft-Watkins
configuration:

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis return-matrix --cohort target_ablation --device cuda --checkpoint-stride 2
```

Use `--labels LABEL ...` or `--max-checkpoints N` for a smaller diagnostic
pass. `return_transfer_matrix.csv` contains optimizer- and parameter-head
calibration, ranking and top-Q-quartile lift for every checkpoint/source/target
cell. `return_transfer_gaps.csv` reports `own - farm` and `own - other` gaps.
A large own-only advantage indicates distribution specialization; a large
change between one-step and MC/best-future targets indicates credit-assignment
mismatch. Best-future follows the diagnostic definition
`log((L_t + eps) / (min_{k>=t} L_k + eps))`: it measures how close the current
transition is to the best later loss in the same saved physical chain.

The focused cohort named in the original diagnostic proposal contains G/B,
four unnormalized global-one-step agents, four unnormalized per-parameter
Soft-Watkins agents, and four normalized legacy agents:

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_offline_analysis return-matrix --cohort credit_specialization --device cuda --max-checkpoints 5
```

## 5. Counterfactual parameter sweep

The sweep restores the PINN weights immediately before a selected action and
evaluates all nine `(epochs, lr)` choices for the optimizer routed by the final
G optimizer head. By default every counterfactual checkpoint is then continued
with the next three recorded actions from the same physical chain. All nine
branches receive the same controlled historical suffix. Selection is
stratified by source experiment and by early/middle/late chain position, and
only states with a complete requested suffix are eligible:

```powershell
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_counterfactual_sweep select --device cuda --suffix-horizon 3
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_counterfactual_sweep sweep --device cuda --optimizer Adam --suffix-horizon 3
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_counterfactual_sweep sweep --device cuda --optimizer LBFGS --suffix-horizon 3
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_counterfactual_sweep summarize --suffix-horizon 3
```

Use repeated `--selection-id ID` arguments for a small targeted run. Results
are appended after every reached epoch threshold, so an interrupted run resumes
without discarding completed actions. `--repeats N` reconstructs the training
points with additional deterministic seeds.

The controlled objective is the best weighted loss reached by the first action
or any suffix action, including the starting checkpoint as step zero. Reports
contain both the immediate score and the horizon score, their correlation,
best-step position, final-vs-best stability gap, G/B horizon regret, and the
fraction of choices within log-regret `0.05` of the empirical best branch.

The original DeepXDE collocation points were not serialized. Therefore every
counterfactual choice for one state uses the same deterministically
reconstructed points, which supports fair within-state ranking but does not
reproduce the historical scalar loss exactly. Adam and LBFGS can be scheduled
separately because LBFGS is substantially more expensive.

For a multi-GPU Linux worker, run all shards and merge them automatically:

```bash
GPU_IDS=0,1,2,3 \
PYTHON_BIN=/path/to/venv/bin/python \
bash experiments/agent_ablation/heads_ablation/reaction/run_counterfactual_multi_gpu.sh
```

Each GPU has an independent cache, result CSV, and `Adam.log`/`LBFGS.log`.
Rerunning the command resumes completed grid points. To launch the two phases
manually, pass the same `--num-shards N` and a different `--shard-index` from
`0` through `N-1` to each process, then run `merge`.

Every completed counterfactual branch updates a 30-character progress bar and
an ETA estimated from the latest 30 branch durations. Each shard writes a
`progress.json`; the multi-GPU launcher prints their combined progress every
30 seconds. It can also be queried manually with:

```bash
python -u -m experiments.agent_ablation.heads_ablation.reaction.run_counterfactual_sweep status --suffix-horizon 3
```

Set `OPTIMIZERS=Adam` or `OPTIMIZERS=LBFGS` to run only one phase. This is
useful when the cluster has a wall-time limit because LBFGS is much slower.
The merged `state_summary.csv` contains per-state oracle ranks/regrets;
`group_summary.csv` aggregates them overall and by optimizer/trajectory phase,
including G-vs-B win fractions and the epochs-by-lr interaction magnitude.
