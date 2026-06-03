#!/bin/bash
# Automatic launch of wave_1d_comparison_chain.py with GPU distribution

SCRIPT="experiments/Wave/wave_1d_chain.py"

LOG_KEY_FOR_NEW_STATE="True"
EXP_KEY_1="f9341ba04ce64316b3ba006f7652f0cb"
STEP_1=999
EXP_KEY_2="9696d869f1ac4a86b93aabfd31c54939"
STEP_2=999

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "log_key: $LOG_KEY_FOR_NEW_STATE"
echo "exp_key: $EXP_KEY"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" &
fi

wait
echo "All processes finished."
