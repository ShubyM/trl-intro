#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
TRAIN_NPROC=2

echo "Starting vLLM server on GPUs $VLLM_GPUS ..."
CUDA_VISIBLE_DEVICES=$VLLM_GPUS trl vllm-serve \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 &
VLLM_PID=$!

cleanup() { kill $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for vLLM server to be ready ..."
until curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; do
    sleep 5
done
echo "vLLM server is up."

echo "Starting training on GPUs $TRAIN_GPUS ..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun \
    --nproc_per_node=$TRAIN_NPROC \
    countdown.py

echo "Training complete. Shutting down vLLM server."
