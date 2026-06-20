#!/bin/bash
# Automatic launch of burgers_1d_comparison_chain.py with GPU distribution

SCRIPT="experiments/Burgers/burgers_1d_chain.py"

LOG_KEY="true"
EXP_KEY_1="a32d8440fed44ee28ba7c1d21d1bd12f"
STEP_1=2500
EXP_KEY_2="c559d9d8b9b145caaf973b3c76b1ce13"
STEP_2=2500

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
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" &
fi

wait
echo "All processes finished."
