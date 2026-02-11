#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
TRAIN_GPUS="0,1,2,3"

IFS=',' read -r -a TRAIN_GPU_LIST <<< "$TRAIN_GPUS"
if [[ -z "${TRAIN_NPROC:-}" ]]; then
    TRAIN_NPROC=${#TRAIN_GPU_LIST[@]}
fi

echo "Starting colocated training on GPUs $TRAIN_GPUS ..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun \
    --nproc_per_node=$TRAIN_NPROC \
    countdown.py

echo "Training complete."
