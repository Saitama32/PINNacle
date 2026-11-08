#!/bin/bash
# Evaluate two hybrid-head combinations in parallel.

set -u

SCRIPT="experiments/Reaction/reaction_diffusion_chain_expand_set.py"
GOOD_KEY="037a7084956e4c2bae7ec477bb48d2cf"
BAD_KEY="f1bec88bf5974a86b46dd13c503e1a36"
GOOD_STEP=1000
BAD_STEP=1000
LOG_KEY="true"
PAIR_1="GG"
PAIR_2="BB"
SINGLE_PAIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --good-key) GOOD_KEY="$2"; shift 2 ;;
        --bad-key) BAD_KEY="$2"; shift 2 ;;
        --good-step) GOOD_STEP="$2"; shift 2 ;;
        --bad-step) BAD_STEP="$2"; shift 2 ;;
        --pair-1) PAIR_1="${2^^}"; shift 2 ;;
        --pair-2) PAIR_2="${2^^}"; shift 2 ;;
        # Backward-compatible single-pair/all interface.
        --pair) SINGLE_PAIR="${2^^}"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

validate_pair() {
    case "$1" in
        GG|BB|GB|BG) ;;
        *)
            echo "Invalid pair '$1'. Expected one of: GG, BB, GB, BG."
            exit 1
            ;;
    esac
}

if [[ -n "$SINGLE_PAIR" && "$SINGLE_PAIR" != "ALL" ]]; then
    validate_pair "$SINGLE_PAIR"
else
    validate_pair "$PAIR_1"
    validate_pair "$PAIR_2"
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

run_combo() {
    local gpu="$1"
    local label="$2"
    local optim_key="$3"
    local param_key="$4"
    local optim_step="$5"
    local param_step="$6"
    echo "Launching $label on GPU $gpu: optim=$optim_key params=$param_key"
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
        --name "reaction_diffusion_heads_${label}" \
        --log_key "$LOG_KEY" \
        --optimizer_exp_key "$optim_key" \
        --parameter_exp_key "$param_key" \
        --optimizer_model_step "$optim_step" \
        --parameter_model_step "$param_step"
}

run_pair() {
    local gpu="$1"
    local pair="$2"
    case "$pair" in
    GG)
        run_combo "$gpu" GG "$GOOD_KEY" "$GOOD_KEY" "$GOOD_STEP" "$GOOD_STEP"
        ;;
    BB)
        run_combo "$gpu" BB "$BAD_KEY" "$BAD_KEY" "$BAD_STEP" "$BAD_STEP"
        ;;
    GB)
        run_combo "$gpu" GB "$GOOD_KEY" "$BAD_KEY" "$GOOD_STEP" "$BAD_STEP"
        ;;
    BG)
        run_combo "$gpu" BG "$BAD_KEY" "$GOOD_KEY" "$BAD_STEP" "$GOOD_STEP"
        ;;
    esac
}

run_pair_batch() {
    local first="$1"
    local second="$2"
    local second_gpu=0
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        second_gpu=1
    fi

    echo "Launching pair batch: $first on GPU 0, $second on GPU $second_gpu"
    run_pair 0 "$first" &
    pid_a=$!
    run_pair "$second_gpu" "$second" &
    pid_b=$!
    wait "$pid_a" "$pid_b"
}

if [[ "$SINGLE_PAIR" == "ALL" ]]; then
    run_pair_batch GG BB
    run_pair_batch GB BG
    echo "All four hybrid-head comparisons finished."
elif [[ -n "$SINGLE_PAIR" ]]; then
    run_pair 0 "$SINGLE_PAIR"
    echo "Hybrid-head comparison '$SINGLE_PAIR' finished."
else
    run_pair_batch "$PAIR_1" "$PAIR_2"
    echo "Hybrid-head comparisons '$PAIR_1' and '$PAIR_2' finished."
fi
