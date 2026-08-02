#!/bin/bash
# Auto-run one NS2D Liddriven ablation and distribute two jobs across available GPUs.

SCRIPT="experiments/agent_ablation/ns2d_liddriven_agent_ablation.py"
ABLATION="${1:-}"

case "$ABLATION" in
    no_per|no_soft_watkins|no_trust_region)
        ;;
    *)
        echo "Usage: $0 {no_per|no_soft_watkins|no_trust_region}"
        exit 2
        ;;
esac

run_job() {
    local gpu="$1"
    local replica="$2"
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
        --ablation "$ABLATION" \
        --name "ns2d_liddriven_${ABLATION}_replica_${replica}" &
}

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Launching 2 processes on one GPU..."
    run_job 0 1
    run_job 0 2
elif [ "$NUM_GPUS" -eq 2 ]; then
    echo "Launching 1 process on each of two GPUs..."
    run_job 0 1
    run_job 1 2
else
    echo "More than 2 GPUs detected; using first two."
    run_job 0 1
    run_job 1 2
fi

wait
echo "All processes finished."
