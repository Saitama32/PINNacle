#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${ITERATIONS:-40000}"
OPTIMIZER="${OPTIMIZER:-soap}"
OUT="${OUT:-runs_ks_ablation}"
PYTHON="${PYTHON:-python}"

RUNNER="experiments/Chaotic/run_chaotic.py"
COMMON=(
  "$RUNNER"
  --equation ks
  --iterations "$ITERATIONS"
  --optimizer "$OPTIMIZER"
  --out "$OUT"
)

run_experiment() {
  local name="$1"
  shift

  echo "Running ${name}"
  "$PYTHON" "${COMMON[@]}" --name "$name" "$@"
}

run_experiment E0_baseline

run_experiment E1_causal \
  --use-causal-loss \
  --causal-num-chunks 16 \
  --causal-tol 0.1

run_experiment E2_fourier \
  --use-fourier-features \
  --fourier-num-modes-x 16

run_experiment E3_resampling \
  --resample-collocation \
  --resample-every 1

run_experiment E4_causal_fourier \
  --use-causal-loss \
  --causal-num-chunks 16 \
  --causal-tol 0.1 \
  --use-fourier-features \
  --fourier-num-modes-x 16

run_experiment E5_causal_resampling \
  --use-causal-loss \
  --causal-num-chunks 16 \
  --causal-tol 0.1 \
  --resample-collocation \
  --resample-every 1

run_experiment E6_fourier_resampling \
  --use-fourier-features \
  --fourier-num-modes-x 16 \
  --resample-collocation \
  --resample-every 1

run_experiment E7_causal_fourier_resampling \
  --use-causal-loss \
  --causal-num-chunks 16 \
  --causal-tol 0.1 \
  --use-fourier-features \
  --fourier-num-modes-x 16 \
  --resample-collocation \
  --resample-every 1

run_experiment E8_all_windows \
  --bc-loss-weight 10000 \
  --use-causal-loss \
  --causal-num-chunks 16 \
  --causal-tol 0.1 \
  --causal-include-ic true \
  --causal-ic-weight 10000 \
  --use-fourier-features \
  --fourier-num-modes-x 16 \
  --resample-collocation \
  --resample-every 1 \
  --use-windows \
  --window-model-mode new_model \
  --window-state-source reference \
  --window-causal-tol-schedule 1e-3,1e-2,1e-1,1,10,100 \
  --num-windows 10
