#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD=speculative_proactive DOCTOR_TYPE=proactive SPEC_COT=false bash "$ROOT_DIR/results/performance/run_tau2_experiment.sh"
