#!/bin/bash
# Rebuild Poisson3D ComplexGeometry buffers from two source experiments and distribute jobs across GPUs.

SCRIPT="experiments/Poisson/n_dim_states/poisson3d_complexgeometry_chain.py"
SOURCE_EXPERIMENT_KEY_GPU0="$1"
SOURCE_EXPERIMENT_KEY_GPU1="$2"

if [ -z "$SOURCE_EXPERIMENT_KEY_GPU0" ] || [ -z "$SOURCE_EXPERIMENT_KEY_GPU1" ]; then
    echo "Usage: $0 <source_experiment_key_gpu0> <source_experiment_key_gpu1>"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on one GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --source-experiment-key "$SOURCE_EXPERIMENT_KEY_GPU0" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --source-experiment-key "$SOURCE_EXPERIMENT_KEY_GPU1" &
elif [ "$NUM_GPUS" -eq 2 ]; then
    echo "Launching 1 process on each of two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --source-experiment-key "$SOURCE_EXPERIMENT_KEY_GPU0" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --source-experiment-key "$SOURCE_EXPERIMENT_KEY_GPU1" &
else
    echo "More than 2 GPUs detected; using first two."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --source-experiment-key "$SOURCE_EXPERIMENT_KEY_GPU0" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --source-experiment-key "$SOURCE_EXPERIMENT_KEY_GPU1" &
fi

wait
echo "All processes finished."
