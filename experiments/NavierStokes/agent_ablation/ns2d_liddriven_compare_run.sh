#!/bin/bash

SCRIPT="experiments/NavierStokes/agent_ablation/ns2d_liddriven_compare.py"
STATE_TYPE="2d"
ABLATION=""
EXP_KEY_1=""
EXP_KEY_2=""
MODEL_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ablation) ABLATION="$2"; shift 2 ;;
        --exp-key-1) EXP_KEY_1="$2"; shift 2 ;;
        --exp-key-2) EXP_KEY_2="$2"; shift 2 ;;
        --model-step) MODEL_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

if [[ -z "$ABLATION" || -z "$EXP_KEY_1" || -z "$EXP_KEY_2" ]]; then
    echo "Usage: $0 --ablation no_per|no_soft_watkins|no_trust_region --exp-key-1 KEY --exp-key-2 KEY [--model-step STEP]"
    exit 2
fi

case "$ABLATION" in
    no_per|no_soft_watkins|no_trust_region) ;;
    *) echo "Unknown ablation type: $ABLATION"; exit 2 ;;
esac

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
EXTRA_ARGS=()
if [[ -n "$MODEL_STEP" ]]; then
    EXTRA_ARGS+=(--model_step "$MODEL_STEP")
fi

echo "Detected GPUs: $NUM_GPUS"
echo "state_type: $STATE_TYPE"
echo "ablation: $ABLATION"

if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --ablation "$ABLATION" --exp_key "$EXP_KEY_1" "${EXTRA_ARGS[@]}" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --ablation "$ABLATION" --exp_key "$EXP_KEY_2" "${EXTRA_ARGS[@]}" &
else
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --ablation "$ABLATION" --exp_key "$EXP_KEY_1" "${EXTRA_ARGS[@]}" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --ablation "$ABLATION" --exp_key "$EXP_KEY_2" "${EXTRA_ARGS[@]}" &
fi

wait
echo "All processes finished."
