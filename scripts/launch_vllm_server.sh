#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export CUDA_VISIBLE_DEVICES

IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_LIST[@]}
if [[ $GPU_COUNT -eq 0 ]]; then
    echo "No GPUs specified in CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

TP_SIZE=${TP_SIZE:-$GPU_COUNT}
if [[ $TP_SIZE -ne $GPU_COUNT ]]; then
    echo "Warning: tensor parallel size ($TP_SIZE) does not match GPUs ($GPU_COUNT). Ensure this is intentional." >&2
fi

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-65536}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}

cat <<MSG
Launching trl vllm-serve with:
  model:                 $MODEL
  GPUs:                  $CUDA_VISIBLE_DEVICES (count=$GPU_COUNT)
  tensor-parallel-size:  $TP_SIZE
  host:                  $HOST
  port:                  $PORT
  max-model-len:         $MAX_MODEL_LEN
  max-num-seqs:          $MAX_NUM_SEQS
  max-num-batched-tokens:$MAX_BATCHED_TOKENS
  gpu-mem-utilization:   $GPU_MEMORY_UTILIZATION
MSG

exec trl vllm-serve \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --swap-space 16 \
    --enable-chunked-prefill \
    "$@"
