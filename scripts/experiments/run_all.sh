#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/results/performance/run_clinlic_no_cot.sh"
bash "$ROOT_DIR/results/performance/run_clinlic_reactive.sh"
bash "$ROOT_DIR/results/performance/run_clinlic_speculative_proactive.sh"
bash "$ROOT_DIR/results/performance/run_20q_no_cot.sh"
bash "$ROOT_DIR/results/performance/run_20q_reactive.sh"
bash "$ROOT_DIR/results/performance/run_20q_speculative_proactive.sh"