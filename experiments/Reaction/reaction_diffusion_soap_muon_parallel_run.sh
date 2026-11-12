#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$PROJECT_ROOT/experiments/Reaction/reaction_diffusion_soap_muon.py"
OPTIMIZER=""
NU="0.01"
COEFFICIENTS=(5 10 15 20 30)
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --optimizer) OPTIMIZER="$2"; shift 2 ;;
        --coefficients)
            COEFFICIENTS=(); shift
            while [[ $# -gt 0 && "$1" != --* ]]; do COEFFICIENTS+=("$1"); shift; done
            ;;
        --nu) NU="$2"; shift 2 ;;
        --lr|--out|--log-every|--plot-every)
            PYTHON_ARGS+=("$1" "$2"); shift 2 ;;
        --upload-checkpoints)
            PYTHON_ARGS+=("$1"); shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$OPTIMIZER" != "soap" && "$OPTIMIZER" != "muon" ]]; then
    echo "Use --optimizer soap or --optimizer muon" >&2; exit 2
fi
if [[ ${#COEFFICIENTS[@]} -eq 0 ]]; then
    echo "--coefficients requires at least one rho" >&2; exit 2
fi

mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
if [[ ${#GPUS[@]} -eq 0 ]]; then echo "No CUDA devices found." >&2; exit 1; fi
cd "$PROJECT_ROOT" || exit 1

pids=(); status=0
for index in "${!COEFFICIENTS[@]}"; do
    slot=$((index % ${#GPUS[@]}))
    if [[ -n "${pids[$slot]:-}" ]]; then wait "${pids[$slot]}" || status=1; fi
    rho="${COEFFICIENTS[$index]}"
    echo "GPU ${GPUS[$slot]}: reaction-diffusion nu=$NU rho=$rho optimizer=$OPTIMIZER"
    CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" python "$SCRIPT" \
        --optimizer "$OPTIMIZER" --nu "$NU" --rho "$rho" "${PYTHON_ARGS[@]}" &
    pids[$slot]=$!
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
