#!/bin/bash
# Run two fixed-chain Poisson 2d-C processes and distribute them over two GPUs.

SCRIPT="experiments/optuna/poisson2d_classic_fixed_chain.py"
GPU_0_SEEDS=()
GPU_1_SEEDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-0-seeds)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                GPU_0_SEEDS+=("$1")
                shift
            done
            ;;
        --gpu-1-seeds)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                GPU_1_SEEDS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 2
            ;;
    esac
done

if [[ ${#GPU_0_SEEDS[@]} -eq 0 || ${#GPU_1_SEEDS[@]} -eq 0 ]]; then
    echo "Usage: $0 --gpu-0-seeds SEED... --gpu-1-seeds SEED..."
    exit 2
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "GPU 0 seeds: ${GPU_0_SEEDS[*]}"
echo "GPU 1 seeds: ${GPU_1_SEEDS[*]}"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --seeds "${GPU_0_SEEDS[@]}" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --seeds "${GPU_1_SEEDS[@]}" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --seeds "${GPU_0_SEEDS[@]}" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --seeds "${GPU_1_SEEDS[@]}" &
fi

wait
echo "All processes finished."
