#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/outputs}"
SPEC_COT="${SPEC_COT:-false}"
GAMMAS="${GAMMAS:-1.0 1.5 2.0 3.0 5.0}"

mkdir -p "$OUTPUT_DIR"

for gamma in $GAMMAS
do
  output_stem="20q_speculative_proactive_gamma${gamma}_spec${SPEC_COT}"
  ENV_NAME=20q \
  METHOD=speculative_proactive \
  DOCTOR_TYPE=proactive \
  GAMMA="$gamma" \
  SPEC_COT="$SPEC_COT" \
  OUTPUT_FILE="$OUTPUT_DIR/${output_stem}.jsonl" \
  SUMMARY_FILE="$OUTPUT_DIR/${output_stem}.txt" \
  bash "$ROOT_DIR/results/performance/run_experiment.sh"
done
