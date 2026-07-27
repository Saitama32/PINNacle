#!/bin/bash
# Rebuild Kuramoto-Sivashinsky 1d-state buffers from two source experiment batches and distribute jobs across GPUs.

SCRIPT="experiments/Chaotic/n_dim_states/kuramoto_sivashinsky_chain_1d_state.py"
SOURCE_EXPERIMENT_KEYS_GPU0="$1"
SOURCE_EXPERIMENT_KEYS_GPU1="$2"
SOURCE_PROJECT_NAME="${3:-rlpinn-kuramoto-sivashinsky-tolerance}"
TARGET_PROJECT_NAME="${4:-rlpinn_ks_rebuild_buffer_1_dim}"
NUM_WORKERS="${5:-5}"

if [ -z "$SOURCE_EXPERIMENT_KEYS_GPU0" ] || [ -z "$SOURCE_EXPERIMENT_KEYS_GPU1" ]; then
    echo "Usage: $0 <comma_separated_source_experiment_keys_gpu0> <comma_separated_source_experiment_keys_gpu1>"
    exit 1
fi

IFS=',' read -ra GPU0_KEYS <<< "$SOURCE_EXPERIMENT_KEYS_GPU0"
IFS=',' read -ra GPU1_KEYS <<< "$SOURCE_EXPERIMENT_KEYS_GPU1"

cleanup() {
    jobs -pr | xargs -r kill
}
trap cleanup INT TERM EXIT

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "GPU0 batch size: ${#GPU0_KEYS[@]}"
echo "GPU1 batch size: ${#GPU1_KEYS[@]}"
echo "Source project: $SOURCE_PROJECT_NAME"
echo "Target project: $TARGET_PROJECT_NAME"
echo "Num workers: $NUM_WORKERS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on one GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --source-project-name "$SOURCE_PROJECT_NAME" --target-project-name "$TARGET_PROJECT_NAME" --out "transitions_rebuilt/kuramoto_sivashinsky_1_dim/gpu_0" --num-workers "$NUM_WORKERS" --source-experiment-keys "${GPU0_KEYS[@]}" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --source-project-name "$SOURCE_PROJECT_NAME" --target-project-name "$TARGET_PROJECT_NAME" --out "transitions_rebuilt/kuramoto_sivashinsky_1_dim/gpu_1" --num-workers "$NUM_WORKERS" --source-experiment-keys "${GPU1_KEYS[@]}" &
elif [ "$NUM_GPUS" -eq 2 ]; then
    echo "Launching 1 process on each of two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --source-project-name "$SOURCE_PROJECT_NAME" --target-project-name "$TARGET_PROJECT_NAME" --out "transitions_rebuilt/kuramoto_sivashinsky_1_dim/gpu_0" --num-workers "$NUM_WORKERS" --source-experiment-keys "${GPU0_KEYS[@]}" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --source-project-name "$SOURCE_PROJECT_NAME" --target-project-name "$TARGET_PROJECT_NAME" --out "transitions_rebuilt/kuramoto_sivashinsky_1_dim/gpu_1" --num-workers "$NUM_WORKERS" --source-experiment-keys "${GPU1_KEYS[@]}" &
else
    echo "More than 2 GPUs detected; using first two."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --source-project-name "$SOURCE_PROJECT_NAME" --target-project-name "$TARGET_PROJECT_NAME" --out "transitions_rebuilt/kuramoto_sivashinsky_1_dim/gpu_0" --num-workers "$NUM_WORKERS" --source-experiment-keys "${GPU0_KEYS[@]}" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --source-project-name "$SOURCE_PROJECT_NAME" --target-project-name "$TARGET_PROJECT_NAME" --out "transitions_rebuilt/kuramoto_sivashinsky_1_dim/gpu_1" --num-workers "$NUM_WORKERS" --source-experiment-keys "${GPU1_KEYS[@]}" &
fi

wait
trap - INT TERM EXIT
echo "All processes finished."
