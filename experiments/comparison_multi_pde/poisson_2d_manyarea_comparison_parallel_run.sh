#!/bin/bash
# Automatic launch of poisson_2d_manyarea_comparison_chain.py with GPU distribution

SCRIPT="experiments/comparison_multi_pde/poisson_2d_manyarea_comparison_chain.py"

LOG_KEY_1="false"
LOG_KEY_2="true"
EXP_KEY_1="b893ceac0e9c40778f4a9b3ccce76769"
EXP_KEY_2="7f7a91cef55d4aeba0e509024977456b"

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
