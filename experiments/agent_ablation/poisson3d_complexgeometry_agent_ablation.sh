#!/bin/bash
# Launch one Poisson3D complex-geometry agent-ablation preset on the first available GPU.

set -euo pipefail

SCRIPT="experiments/agent_ablation/poisson3d_complexgeometry_agent_ablation.py"
ABLATION="${1:-}"

case "$ABLATION" in
    no_per|no_soft_watkins|no_trust_region)
        ;;
    *)
        echo "Usage: $0 {no_per|no_soft_watkins|no_trust_region}"
        exit 2
        ;;
esac

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA devices found. Exiting."
    exit 1
fi

echo "Launching Poisson3D complex-geometry ablation: $ABLATION"
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --ablation "$ABLATION"

