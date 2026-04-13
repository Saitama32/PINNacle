#!/bin/bash
# Automatic launch of poisson_2d_manyarea_comparison_chain.py with GPU distribution

SCRIPT="experiments/Poisson/poisson_2d_manyarea_comparison_chain.py"

LOG_KEY_1="true"
LOG_KEY_2="true"
EXP_KEY_1="1d1abc4651214459a6be11df3f9ea299"
EXP_KEY_2="3c1a750f7bbc4979aad314729cd02327"

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "log_key: $LOG_KEY"
echo "exp_key: $EXP_KEY"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY_1" --exp_key "$EXP_KEY_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY_2" --exp_key "$EXP_KEY_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY_1" --exp_key "$EXP_KEY_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$LOG_KEY_2" --exp_key "$EXP_KEY_2" &
fi

wait
echo "All processes finished."
