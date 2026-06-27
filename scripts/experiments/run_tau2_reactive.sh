#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD=reactive DOCTOR_TYPE=reactive SPEC_COT=false bash "$ROOT_DIR/results/performance/run_tau2_experiment.sh"
