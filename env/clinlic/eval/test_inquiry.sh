#!/bin/bash

# 默认参数值
PATIENT_API_BASE="${PATIENT_API_BASE:-http://localhost:1234/v1}"
PATIENT_API_KEY="${PATIENT_API_KEY:-$API_KEY}"
PATIENT_MODEL="${PATIENT_MODEL:-gpt-oss-120b}"
DOCTOR_API_BASE="${DOCTOR_API_BASE:-http://localhost:2345/v1}"
DOCTOR_API_KEY="${DOCTOR_API_KEY:-$API_KEY}"
DOCTOR_MODEL="${DOCTOR_MODEL:-Qwen3.6-27B}"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://localhost:1234/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-$API_KEY}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-120b}"
MAX_TURN="${MAX_TURN:-10}"
MIN_TURN="${MIN_TURN:-0}"
MAX_WORKERS="${MAX_WORKERS:-64}"
INPUT_FILE="${INPUT_FILE:-/root/wat/AgentClinic/agentclinic_medqa_extended.jsonl}"
STATIC="${STATIC:-false}"
PATIENT_TYPE="${PATIENT_TYPE:-basic}"
DOCTOR_TYPE="${DOCTOR_TYPE:-basic}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
TEMPERATURE="${TEMPERATURE:-0.5}"
MAX_SAMPLE="${MAX_SAMPLE:-10000}"
RUN_NUMBER="${RUN_NUMBER:-1}"
GAMMA="${GAMMA:-1}"
SPEC_COT="${SPEC_COT:-false}"
SPEC_BRANCH_NUM="${SPEC_BRANCH_NUM:-}"

API_KEY="sk-RbwokGPXdz64KSXn54CiIm13znDmjpFIibfApsjORpavOoaT"
# export PATIENT_API_BASE="https://yeysai.com/v1"
# export PATIENT_API_KEY=$API_KEY
# export PATIENT_MODEL="deepseek-v3.2"
# export DOCTOR_API_BASE="https://yeysai.com/v1"
# export DOCTOR_API_KEY=$API_KEY
# export DOCTOR_MODEL="deepseek-v3.2"
# export DOCTOR_API_BASE="https://yeysai.com/v1"
# export DOCTOR_API_KEY=$API_KEY
# export DOCTOR_MODEL="gpt-4.1"
# export JUDGE_API_BASE="https://yeysai.com/v1"
# export JUDGE_API_KEY=$API_KEY
# export JUDGE_MODEL="deepseek-v3.2"
# API_KEY="sk-8c75c5a7d93d44de8c4b8e2f9efe6e19"
# export PATIENT_API_BASE="https://api.deepseek.com/v1"
# export PATIENT_API_KEY=$API_KEY
# export PATIENT_MODEL="deepseek-chat"
# export DOCTOR_API_BASE="https://api.deepseek.com/v1"
# export DOCTOR_API_KEY=$API_KEY
# export DOCTOR_MODEL="deepseek-chat"
# export JUDGE_API_BASE="https://api.deepseek.com/v1"
# export JUDGE_API_KEY=$API_KEY
# export JUDGE_MODEL="deepseek-chat"

# 打印参数（用于验证）
echo "=== 当前配置参数 ==="
echo "PATIENT_API_BASE: $PATIENT_API_BASE"
echo "PATIENT_MODEL: $PATIENT_MODEL"
echo "DOCTOR_API_BASE: $DOCTOR_API_BASE"
echo "DOCTOR_MODEL: $DOCTOR_MODEL"
echo "JUDGE_API_BASE: $JUDGE_API_BASE"
echo "JUDGE_MODEL: $JUDGE_MODEL"
echo "MAX_TURN: $MAX_TURN"
echo "MIN_TURN: $MIN_TURN"
echo "MAX_WORKERS: $MAX_WORKERS"
echo "INPUT_FILE: $INPUT_FILE"
echo "STATIC: $STATIC"
echo "PATIENT_TYPE: $PATIENT_TYPE"
echo "DOCTOR_TYPE: $DOCTOR_TYPE"
echo "ENABLE_THINKING: $ENABLE_THINKING"
echo "TEMPERATURE: $TEMPERATURE"
echo "MAX_SAMPLE: $MAX_SAMPLE"
echo "RUN_NUMBER: $RUN_NUMBER"
echo "GAMMA: $GAMMA"
echo "SPEC_COT: $SPEC_COT"
echo "SPEC_BRANCH_NUM: ${SPEC_BRANCH_NUM:-default}"
echo "===================="

SPEC_BRANCH_ARGS=()
if [ -n "$SPEC_BRANCH_NUM" ]; then
        SPEC_BRANCH_ARGS=(--spec-branch-num "$SPEC_BRANCH_NUM")
fi

# if [ ! -f "$OUTPUT_FILE" ]; then
python3 main.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --patient-api-base "$PATIENT_API_BASE" \
        --patient-api-key "$PATIENT_API_KEY" \
        --patient-model "$PATIENT_MODEL" \
        --doctor-api-base "$DOCTOR_API_BASE" \
        --doctor-api-key "$DOCTOR_API_KEY" \
        --doctor-model "$DOCTOR_MODEL" \
        --judge-api-base "$JUDGE_API_BASE" \
        --judge-api-key "$JUDGE_API_KEY" \
        --judge-model "$JUDGE_MODEL" \
        --max-turn "$MAX_TURN" \
        --min-turn "$MIN_TURN" \
        --max-workers "$MAX_WORKERS" \
        --max-sample "$MAX_SAMPLE" \
        --enable-thinking $ENABLE_THINKING \
        --temperature $TEMPERATURE \
        --static-strategy $STATIC \
        --patient-type $PATIENT_TYPE \
        --doctor-type $DOCTOR_TYPE \
        --run-number $RUN_NUMBER \
        --gamma $GAMMA \
        --spec-cot $SPEC_COT \
        "${SPEC_BRANCH_ARGS[@]}"
# fi
