#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD=no_cot DOCTOR_TYPE=basic SPEC_COT=false bash "$ROOT_DIR/results/performance/run_tau2_experiment.sh"
