#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/env/tau2"

METHOD="${METHOD:?METHOD is required}"
DOMAIN="${DOMAIN:-retail}"
TASK_SET_NAME="${TASK_SET_NAME:-}"
TASK_SPLIT_NAME="${TASK_SPLIT_NAME:-base}"
TASK_IDS="${TASK_IDS:-}"
NUM_TASKS="${NUM_TASKS:-}"

cd "$WORK_DIR"

PATIENT_API_BASE="${PATIENT_API_BASE:-http://localhost:1234/v1}"
PATIENT_API_KEY="${PATIENT_API_KEY:-EMPTY}"
PATIENT_MODEL="${PATIENT_MODEL:-gpt-oss-120b}"

DOCTOR_API_BASE="${DOCTOR_API_BASE:-http://localhost:2345/v1}"
DOCTOR_API_KEY="${DOCTOR_API_KEY:-EMPTY}"
DOCTOR_MODEL="${DOCTOR_MODEL:-Qwen3.6-27B}"
DOCTOR_CONTEXT_WINDOW="${DOCTOR_CONTEXT_WINDOW:-65536}"

JUDGE_API_BASE="${JUDGE_API_BASE:-http://localhost:1234/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-120b}"

MAX_TURN="${MAX_TURN:-200}"
MIN_TURN="${MIN_TURN:-0}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MAX_SAMPLE="${MAX_SAMPLE:-1000000}"
RUN_NUMBER="${RUN_NUMBER:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
TEMPERATURE="${TEMPERATURE:-0.8}"
STATIC="${STATIC:-false}"
PATIENT_TYPE="${PATIENT_TYPE:-basic}"
DOCTOR_TYPE="${DOCTOR_TYPE:-basic}"
GAMMA="${GAMMA:-5.0}"
SPEC_COT="${SPEC_COT:-false}"
INPUT_FILE="${INPUT_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/outputs}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/tau2_${METHOD}.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-$OUTPUT_DIR/tau2_${METHOD}.txt}"

mkdir -p "$OUTPUT_DIR"
export TAU2_DOCTOR_CONTEXT_WINDOW="$DOCTOR_CONTEXT_WINDOW"

CMD=(
  python3 main.py
  --input "$INPUT_FILE"
  --output "$OUTPUT_FILE"
  --patient-api-base "$PATIENT_API_BASE"
  --patient-api-key "$PATIENT_API_KEY"
  --patient-model "$PATIENT_MODEL"
  --doctor-api-base "$DOCTOR_API_BASE"
  --doctor-api-key "$DOCTOR_API_KEY"
  --doctor-model "$DOCTOR_MODEL"
  --judge-api-base "$JUDGE_API_BASE"
  --judge-api-key "$JUDGE_API_KEY"
  --judge-model "$JUDGE_MODEL"
  --max-turn "$MAX_TURN"
  --min-turn "$MIN_TURN"
  --max-workers "$MAX_WORKERS"
  --max-sample "$MAX_SAMPLE"
  --enable-thinking "$ENABLE_THINKING"
  --temperature "$TEMPERATURE"
  --static-strategy "$STATIC"
  --patient-type "$PATIENT_TYPE"
  --doctor-type "$DOCTOR_TYPE"
  --run-number "$RUN_NUMBER"
  --gamma "$GAMMA"
  --spec-cot "$SPEC_COT"
  --domain "$DOMAIN"
  --task-split-name "$TASK_SPLIT_NAME"
)

if [[ -n "$TASK_SET_NAME" ]]; then
  CMD+=(--task-set-name "$TASK_SET_NAME")
fi

if [[ -n "$TASK_IDS" ]]; then
  # shellcheck disable=SC2206
  TASK_ID_ARRAY=($TASK_IDS)
  CMD+=(--task-ids "${TASK_ID_ARRAY[@]}")
fi

if [[ -n "$NUM_TASKS" ]]; then
  CMD+=(--num-tasks "$NUM_TASKS")
fi

"${CMD[@]}"
python3 "$ROOT_DIR/scripts/compute_score.py" "$OUTPUT_FILE" | tee "$SUMMARY_FILE"
