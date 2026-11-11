#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU_IDS=0,1,2,3 bash run_counterfactual_multi_gpu.sh
# Optional environment variables:
#   PYTHON_BIN=python REPEATS=1 OPTIMIZERS=Adam,LBFGS
#   OUTPUT_DIR=/path/to/output CACHE_DIR=/path/to/cache

PYTHON_BIN="${PYTHON_BIN:-python}"
REPEATS="${REPEATS:-1}"
SUFFIX_HORIZON="${SUFFIX_HORIZON:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/agent_ablation/heads_ablation/reaction/outputs/counterfactual}"
CACHE_DIR="${CACHE_DIR:-experiments/agent_ablation/heads_ablation/reaction/cache}"
GPU_IDS="${GPU_IDS:-0}"
OPTIMIZERS="${OPTIMIZERS:-Adam,LBFGS}"
MODULE="experiments.agent_ablation.heads_ablation.reaction.run_counterfactual_sweep"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_SHARDS="${#GPUS[@]}"

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
