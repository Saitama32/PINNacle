#!/bin/bash

SCRIPT="experiments/Poisson/agent_ablation/poisson3d_complexgeometry_compare.py"
STATE_TYPE="2d"
ABLATION=""
EXP_KEY=""
SEEDS_1=""
SEEDS_2=""
MODEL_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ablation) ABLATION="$2"; shift 2 ;;
        --exp-key) EXP_KEY="$2"; shift 2 ;;
        --seeds-1) SEEDS_1="$2"; shift 2 ;;
        --seeds-2) SEEDS_2="$2"; shift 2 ;;
        --model-step) MODEL_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

if [[ -z "$ABLATION" || -z "$EXP_KEY" || -z "$SEEDS_1" || -z "$SEEDS_2" ]]; then
    echo "Usage: $0 --ablation no_per|no_soft_watkins|no_trust_region --exp-key KEY --seeds-1 S1,S2,S3,S4,S5 --seeds-2 S6,S7,S8,S9,S10 [--model-step STEP]"
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
echo "experiment key: $EXP_KEY"
echo "seed group 1: $SEEDS_1"
echo "seed group 2: $SEEDS_2"

if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
    RL_TRANSITIONS_DIR="transitions/${ABLATION}_${EXP_KEY:0:8}_group_1" CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --name "poisson3d_complexgeometry_${ABLATION}_${EXP_KEY:0:8}_group_1" --ablation "$ABLATION" --exp_key "$EXP_KEY" --seeds "$SEEDS_1" "${EXTRA_ARGS[@]}" &
    PID_1=$!
    RL_TRANSITIONS_DIR="transitions/${ABLATION}_${EXP_KEY:0:8}_group_2" CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --name "poisson3d_complexgeometry_${ABLATION}_${EXP_KEY:0:8}_group_2" --ablation "$ABLATION" --exp_key "$EXP_KEY" --seeds "$SEEDS_2" "${EXTRA_ARGS[@]}" &
    PID_2=$!
else
    RL_TRANSITIONS_DIR="transitions/${ABLATION}_${EXP_KEY:0:8}_group_1" CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --name "poisson3d_complexgeometry_${ABLATION}_${EXP_KEY:0:8}_group_1" --ablation "$ABLATION" --exp_key "$EXP_KEY" --seeds "$SEEDS_1" "${EXTRA_ARGS[@]}" &
    PID_1=$!
    RL_TRANSITIONS_DIR="transitions/${ABLATION}_${EXP_KEY:0:8}_group_2" CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --name "poisson3d_complexgeometry_${ABLATION}_${EXP_KEY:0:8}_group_2" --ablation "$ABLATION" --exp_key "$EXP_KEY" --seeds "$SEEDS_2" "${EXTRA_ARGS[@]}" &
    PID_2=$!
fi

STATUS=0
wait "$PID_1" || STATUS=$?
wait "$PID_2" || STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
    echo "At least one comparison process failed with exit code $STATUS."
    exit "$STATUS"
fi
echo "All processes finished."
