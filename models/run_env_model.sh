#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
set -euo pipefail

# 20Q environment/patient model
MODEL_DIR="${MODEL_DIR:-/mnt/data/fujiaqi/ckpt/gpt-oss-120b/}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-120b}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/fujiaqi/miniconda3/envs/vllm-serve/bin/python}"

CUDA_VISIBLE_DEVICES=4,5,6,7 "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --served-model-name "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 1234 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
