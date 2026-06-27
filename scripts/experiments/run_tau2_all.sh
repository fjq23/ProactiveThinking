#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$ROOT_DIR/results/performance/run_tau2_no_cot.sh"
bash "$ROOT_DIR/results/performance/run_tau2_reactive.sh"
bash "$ROOT_DIR/results/performance/run_tau2_speculative_proactive.sh"
