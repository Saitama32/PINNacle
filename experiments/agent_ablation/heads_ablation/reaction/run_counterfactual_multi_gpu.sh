#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_counterfactual_multi_gpu.sh 0,1,2,3 3 Adam,LBFGS 1 python
# Environment variables remain supported as fallbacks:
#   PYTHON_BIN=python REPEATS=1 OPTIMIZERS=Adam,LBFGS
#   OUTPUT_DIR=/path/to/output CACHE_DIR=/path/to/cache

GPU_IDS="${1:-${GPU_IDS:-0}}"
SUFFIX_HORIZON="${2:-${SUFFIX_HORIZON:-3}}"
OPTIMIZERS="${3:-${OPTIMIZERS:-Adam,LBFGS}}"
REPEATS="${4:-${REPEATS:-1}}"
PYTHON_BIN="${5:-${PYTHON_BIN:-python}}"
OUTPUT_DIR="${6:-${OUTPUT_DIR:-experiments/agent_ablation/heads_ablation/reaction/outputs/counterfactual}}"
CACHE_DIR="${7:-${CACHE_DIR:-experiments/agent_ablation/heads_ablation/reaction/cache}}"
MODULE="experiments.agent_ablation.heads_ablation.reaction.run_counterfactual_sweep"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_SHARDS="${#GPUS[@]}"

echo "Effective counterfactual configuration:"
echo "  GPU_IDS=${GPU_IDS} (num_shards=${NUM_SHARDS})"
echo "  SUFFIX_HORIZON=${SUFFIX_HORIZON}"
echo "  OPTIMIZERS=${OPTIMIZERS}"
echo "  REPEATS=${REPEATS}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"

"${PYTHON_BIN}" -u -m "${MODULE}" select \
  --suffix-horizon "${SUFFIX_HORIZON}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}"

run_phase() {
  local optimizer="$1"
  local pids=()
  echo "Starting ${optimizer} phase on ${NUM_SHARDS} GPUs"
  for shard in "${!GPUS[@]}"; do
    local gpu="${GPUS[$shard]}"
    local shard_dir="${OUTPUT_DIR}/suffix_h${SUFFIX_HORIZON}/shards/shard_$(printf '%02d' "${shard}")"
    mkdir -p "${shard_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m "${MODULE}" sweep \
      --device cuda \
      --optimizer "${optimizer}" \
      --suffix-horizon "${SUFFIX_HORIZON}" \
      --repeats "${REPEATS}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --output-dir "${OUTPUT_DIR}" \
      --cache-dir "${CACHE_DIR}/shard_${shard}" \
      > "${shard_dir}/${optimizer}.log" 2>&1 &
    pids+=("$!")
  done
  while true; do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
      fi
    done
    if [[ "${alive}" -eq 0 ]]; then
      break
    fi
    "${PYTHON_BIN}" -u -m "${MODULE}" status \
      --suffix-horizon "${SUFFIX_HORIZON}" \
      --output-dir "${OUTPUT_DIR}"
    sleep 30
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "${optimizer} phase failed; inspect ${OUTPUT_DIR}/shards/shard_*/${optimizer}.log" >&2
    return 1
  fi
}

IFS=',' read -r -a OPTIMIZER_PHASES <<< "${OPTIMIZERS}"
for optimizer in "${OPTIMIZER_PHASES[@]}"; do
  run_phase "${optimizer}"
done

"${PYTHON_BIN}" -u -m "${MODULE}" merge \
  --suffix-horizon "${SUFFIX_HORIZON}" \
  --output-dir "${OUTPUT_DIR}"
