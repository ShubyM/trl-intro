#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"
if [[ -f "$VENV_PATH" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_PATH"
fi

TRAIN_SCRIPT=${TRAIN_SCRIPT:-"countdown_trl_vllm.py"}
TRAIN_FILE="$SCRIPT_DIR/$TRAIN_SCRIPT"
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Training script '$TRAIN_FILE' not found" >&2
    exit 1
fi

# Default to all eight GPUs on a single H100 node. Override TRAIN_GPUS to pin.
TRAIN_GPUS=${TRAIN_GPUS:-"0,1,2,3,4,5,6,7"}
IFS=',' read -r -a TRAIN_GPU_LIST <<< "$TRAIN_GPUS"
if [[ ${#TRAIN_GPU_LIST[@]} -eq 0 ]]; then
    echo "TRAIN_GPUS is empty; provide at least one device" >&2
    exit 1
fi

if [[ -z "${TRAIN_NPROC:-}" ]]; then
    TRAIN_NPROC=${#TRAIN_GPU_LIST[@]}
fi

export VLLM_SERVER_URL=${VLLM_SERVER_URL:-"http://127.0.0.1:8000"}

if command -v curl >/dev/null 2>&1; then
    if curl -fsS -o /dev/null "${VLLM_SERVER_URL}/health"; then
        echo "vLLM server reachable at ${VLLM_SERVER_URL}"
    else
        echo "Warning: unable to reach ${VLLM_SERVER_URL}/health; ensure trl vllm-serve is running" >&2
    fi
fi

echo "Starting distributed training on GPUs ${TRAIN_GPUS} (nproc ${TRAIN_NPROC}) using ${TRAIN_SCRIPT}"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun \
    --nproc_per_node=$TRAIN_NPROC \
    "$TRAIN_FILE" "$@"

echo "Training complete."
