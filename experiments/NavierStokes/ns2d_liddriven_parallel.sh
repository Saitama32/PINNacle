#!/bin/bash
# Автоматический запуск poisson_2d_cg_chain.py с распределением по GPU

SCRIPT="experiments/NavierStokes/ns2d_liddriven_chain.py"

LOG_KEY="true"
EXP_KEY_1="c0b0841538514927a70117ddc6af36e6"
EXP_KEY_2="82c99480791d4bb3b0a38830e8766b5f"

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
