#!/bin/bash
# Automatic launch of ReactionDiffusion1D comparison runs with GPU distribution.

SCRIPT="experiments/Reaction/reaction_diffusion_chain_expand_set.py"

LOG_KEY="true"
EXP_KEY_1=""
STEP_1=1000
EXP_KEY_2=""
STEP_2=1000
NU_1=0.01
NU_2=0.01
RHO_1=20.0
RHO_2=20.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-key-1)
            EXP_KEY_1="$2"
            shift 2
            ;;
        --exp-key-2)
            EXP_KEY_2="$2"
            shift 2
            ;;
        --exp-key)
            EXP_KEY_1="$2"
            EXP_KEY_2="$2"
            shift 2
            ;;
        --nu-1)
            NU_1="$2"
            shift 2
            ;;
        --nu-2)
            NU_2="$2"
            shift 2
            ;;
        --rho-1)
            RHO_1="$2"
            shift 2
            ;;
        --rho-2)
            RHO_2="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$EXP_KEY_1" || -z "$EXP_KEY_2" || "$EXP_KEY_1" == PASTE_* || "$EXP_KEY_2" == PASTE_* ]]; then
    echo "Replace both Comet experiment key placeholders before launching."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"
echo "log_key: $LOG_KEY"
echo "exp_key_1: $EXP_KEY_1"
echo "exp_key_2: $EXP_KEY_2"
echo "nu_1: $NU_1"
echo "nu_2: $NU_2"
echo "rho_1: $RHO_1"
echo "rho_2: $RHO_2"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on a single GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" --nu "$NU_1" --rho "$RHO_1" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" --nu "$NU_2" --rho "$RHO_2" &
else
    echo "Launching one process per GPU on first two GPUs..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_1" --model_step "$STEP_1" --nu "$NU_1" --rho "$RHO_1" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$LOG_KEY" --exp_key "$EXP_KEY_2" --model_step "$STEP_2" --nu "$NU_2" --rho "$RHO_2" &
fi

wait
echo "All processes finished."
