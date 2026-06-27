#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -euo pipefail

# 20Q doctor model
# MODEL_DIR="${MODEL_DIR:-/mnt/data/fujiaqi/ckpt/Qwen3.5-9B}"
MODEL_DIR="${MODEL_DIR:-/mnt/data/fujiaqi/ckpt/Qwen3.6-27B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/fujiaqi/miniconda3/envs/vllm-serve/bin/python}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

CUDA_VISIBLE_DEVICES=0,1,2,3 "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --served-model-name "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 2345 \
  --tensor-parallel-size 4 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
