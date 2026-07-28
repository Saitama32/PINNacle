#!/bin/bash
# Auto-run RL chain and distribute jobs across available GPUs.

SCRIPT="experiments/Burgers/n_dim_states/burgers_1d_chain_3d_state.py"
SEEDS=(1234 12345)

run_seed() {
    local gpu="$1"
    local seed="$2"
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" --seed "$seed" &
}

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on one GPU..."
    run_seed 0 "${SEEDS[0]}"
    run_seed 0 "${SEEDS[1]}"
elif [ "$NUM_GPUS" -eq 2 ]; then
    echo "Launching 1 process on each of two GPUs..."
    run_seed 0 "${SEEDS[0]}"
    run_seed 1 "${SEEDS[1]}"
else
    echo "More than 2 GPUs detected; using first two."
    run_seed 0 "${SEEDS[0]}"
    run_seed 1 "${SEEDS[1]}"
fi

wait
echo "All processes finished."
