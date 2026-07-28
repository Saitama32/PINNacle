#!/bin/bash
# Parallel launch for Convection1D beta=50 RL optimizer chain.

SCRIPT="experiments/convection/convection_chain.py"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA device found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting 2 processes on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" &
elif [ "$NUM_GPUS" -ge 2 ]; then
    echo "Starting 1 process on each of the first 2 GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" &
else
    echo "Unexpected GPU count: $NUM_GPUS"
    exit 1
fi

wait
echo "All processes finished."
