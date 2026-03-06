#!/bin/bash
# Automatic launch of gray_scott_comparison_chain.py with GPU distribution

SCRIPT="experiments/comparison_multi_pde/gray_scott_comparison_chain.py"

LOG_KEY="true"
# Replace these with experiment keys of the trained Gray-Scott RL agent you want to compare.
EXP_KEY_1="PUT_GRAY_SCOTT_EXP_KEY_1"
EXP_KEY_2="PUT_GRAY_SCOTT_EXP_KEY_2"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "log_key: $LOG_KEY"
echo "exp_key_1: $EXP_KEY_1"
echo "exp_key_2: $EXP_KEY_2"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" &
fi

wait
echo "All processes finished."
