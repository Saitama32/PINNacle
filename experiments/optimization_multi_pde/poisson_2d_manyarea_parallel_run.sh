#!/bin/bash
# Automatic launch of poisson_2d_manyarea_chain.py with GPU distribution

SCRIPT="experiments/optimization_multi_pde/poisson_2d_manyarea_chain.py"
SCRIPT_2="$SCRIPT"

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

log_enable="true"
echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_enable" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_enable" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_enable" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$log_enable" &
fi

wait
echo "All processes finished."

