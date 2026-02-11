#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
VLLM_GPUS="0,1"
TRAIN_GPUS="2"

IFS=',' read -r -a TRAIN_GPU_LIST <<< "$TRAIN_GPUS"
if [[ -z "${TRAIN_NPROC:-}" ]]; then
    TRAIN_NPROC=${#TRAIN_GPU_LIST[@]}
fi

echo "Starting vLLM server on GPUs $VLLM_GPUS ..."
CUDA_VISIBLE_DEVICES=$VLLM_GPUS vllm serve \
    "$MODEL" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 1024 \
    --max-num-seqs 32 &
VLLM_PID=$!

cleanup() { kill $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for vLLM server to be ready ..."
until curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; do
    sleep 5
done
echo "vLLM server is up."

# Point the trainer at the freshly started server.
export VLLM_API_BASE=${VLLM_API_BASE:-http://127.0.0.1:8000}
export VLLM_API_TIMEOUT=${VLLM_API_TIMEOUT:-600}
export VLLM_API_KEY=${VLLM_API_KEY:-}

echo "Starting training on GPUs $TRAIN_GPUS ..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun \
    --nproc_per_node=$TRAIN_NPROC \
    countdown.py

echo "Training complete. Shutting down vLLM server."
