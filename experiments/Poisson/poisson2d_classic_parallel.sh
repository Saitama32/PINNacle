#!/bin/bash
# Parallel launch for Poisson 2D Classic RL optimizer chain.

SCRIPT="experiments/Poisson/poisson2d_classic_chain.py"
TIMEOUT_SECONDS="${POISSON2D_CLASSIC_TIMEOUT_SECONDS:-42000}" # 11 hours 40 minutes
KILL_GRACE_SECONDS=30
PIDS=()
TIMER_PID=""
LAUNCHER_PID=$BASHPID

if ! [[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "POISSON2D_CLASSIC_TIMEOUT_SECONDS must be a positive integer, got: $TIMEOUT_SECONDS" >&2
    exit 2
fi

stop_timer() {
    if [ -n "$TIMER_PID" ]; then
        kill "$TIMER_PID" 2>/dev/null || true
        wait "$TIMER_PID" 2>/dev/null || true
        TIMER_PID=""
    fi
}

terminate_processes() {
    local signal="$1"
    local pid

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "-$signal" "$pid" 2>/dev/null || true
        fi
    done
}

wait_for_shutdown() {
    local deadline=$((SECONDS + KILL_GRACE_SECONDS))
    local pid
    local any_running

    while [ "$SECONDS" -lt "$deadline" ]; do
        any_running=0
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                any_running=1
                break
            fi
        done

        if [ "$any_running" -eq 0 ]; then
            return
        fi
        sleep 1
    done
}

shutdown_children() {
    terminate_processes TERM
    wait_for_shutdown
    terminate_processes KILL
    wait "${PIDS[@]}" 2>/dev/null || true
}

on_timeout() {
    echo "Time limit reached after ${TIMEOUT_SECONDS}s. Stopping all Poisson 2D Classic processes..."
    stop_timer
    shutdown_children
    exit 124
}

on_interrupt() {
    echo "Launcher interrupted. Stopping all Poisson 2D Classic processes..."
    stop_timer
    shutdown_children
    exit 130
}

trap on_timeout USR1
trap on_interrupt INT TERM HUP

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA device found. Exiting."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting 2 processes on GPU 0..."
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u "$SCRIPT" &
    PIDS+=("$!")
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u "$SCRIPT" &
    PIDS+=("$!")
elif [ "$NUM_GPUS" -ge 2 ]; then
    echo "Starting 1 process on each of the first 2 GPUs..."
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u "$SCRIPT" &
    PIDS+=("$!")
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python -u "$SCRIPT" &
    PIDS+=("$!")
else
    echo "Unexpected GPU count: $NUM_GPUS"
    exit 1
fi

echo "Poisson 2D Classic processes: ${PIDS[*]}"
echo "Forced shutdown scheduled in 11 hours 40 minutes (${TIMEOUT_SECONDS}s)."

(
    sleeper_pid=""
    trap 'kill "$sleeper_pid" 2>/dev/null || true; wait "$sleeper_pid" 2>/dev/null || true; exit 0' TERM
    sleep "$TIMEOUT_SECONDS" &
    sleeper_pid="$!"
    wait "$sleeper_pid"
    kill -USR1 "$LAUNCHER_PID" 2>/dev/null || true
) &
TIMER_PID="$!"

exit_status=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || exit_status="$?"
done

stop_timer
trap - USR1 INT TERM HUP

echo "All processes finished with status ${exit_status}."
exit "$exit_status"
