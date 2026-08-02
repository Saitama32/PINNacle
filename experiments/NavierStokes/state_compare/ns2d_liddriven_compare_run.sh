#!/bin/bash

SCRIPT="experiments/NavierStokes/state_compare/ns2d_liddriven_compare.py"
STATE_TYPE=""
EXP_KEY_1=""
EXP_KEY_2=""
MODEL_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --state-type) STATE_TYPE="$2"; shift 2 ;;
        --exp-key-1) EXP_KEY_1="$2"; shift 2 ;;
        --exp-key-2) EXP_KEY_2="$2"; shift 2 ;;
        --model-step) MODEL_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

if [[ -z "$STATE_TYPE" || -z "$EXP_KEY_1" || -z "$EXP_KEY_2" ]]; then
    echo "Usage: $0 --state-type raw|log_raw|1d|2d|3d --exp-key-1 KEY --exp-key-2 KEY [--model-step STEP]"
    exit 2
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
EXTRA_ARGS=()
if [[ -n "$MODEL_STEP" ]]; then
    EXTRA_ARGS+=(--model_step "$MODEL_STEP")
fi

echo "Detected GPUs: $NUM_GPUS"
echo "state_type: $STATE_TYPE"

if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --state-type "$STATE_TYPE" --exp_key "$EXP_KEY_1" "${EXTRA_ARGS[@]}" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --state-type "$STATE_TYPE" --exp_key "$EXP_KEY_2" "${EXTRA_ARGS[@]}" &
else
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --state-type "$STATE_TYPE" --exp_key "$EXP_KEY_1" "${EXTRA_ARGS[@]}" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --state-type "$STATE_TYPE" --exp_key "$EXP_KEY_2" "${EXTRA_ARGS[@]}" &
fi

wait
echo "All processes finished."
