#!/bin/bash
# Автоматический запуск poisson_2d_cg_chain.py с распределением по GPU

SCRIPT="experiments/NavierStokes/ns2d_liddriven_chain.py"

LOG_KEY_FOR_NEW_STATE="True"
EXP_KEY_1="2ab53eb432744456a31965c6de39ac94"
EXP_KEY_2="f1537b3917c342109adaa81d9d5f517c"


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
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key_for_new_state "$LOG_KEY_FOR_NEW_STATE" --exp_key "$EXP_KEY_2" &
fi

wait
echo "All processes finished."
