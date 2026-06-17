#!/bin/bash
# Auto-run RL chain and distribute jobs across available GPUs.

SCRIPT="experiments/HeatConductivity/heatnd_chain.py"

LOG_KEY_FOR_NEW_STATE="true"
EXP_KEY_1="88a05788e04e476c9d235767fc924def"
STEP_1=1660
EXP_KEY_2="88a05788e04e476c9d235767fc924def"
STEP_2=2000

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "log_key: $LOG_KEY_FOR_NEW_STATE"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" &
fi

wait
echo "All processes finished."
