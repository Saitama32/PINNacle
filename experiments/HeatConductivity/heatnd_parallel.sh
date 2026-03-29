#!/bin/bash
# Auto-run RL chain and distribute jobs across available GPUs.

SCRIPT="experiments/HeatConductivity/heatnd_chain.py"

LOG_KEY="true"
EXP_KEY_1="b85fdbc28a9145ad9bc38803860ca5ce"
EXP_KEY_2="3c1187d5ba8e4bc29e3e84d48d867b71"

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
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" &
fi

wait
echo "All processes finished."
echo "All processes finished."
