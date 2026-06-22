#!/bin/bash
# Automatic launch of gray_scott_comparison_chain.py with GPU distribution

SCRIPT="experiments/GrayScott/grayscott_chain.py"
LOG_KEY="true"
# Replace these with experiment keys of the trained Gray-Scott RL agent you want to compare.
EXP_KEY_1="7cb70254b82349c48ee19af951694f3d"
STEP_1=2002
EXP_KEY_2="88a05788e04e476c9d235767fc924def"
STEP_2=2000


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
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" &
fi

wait
echo "All processes finished."
