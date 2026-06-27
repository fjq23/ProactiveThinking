#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME=20q METHOD=no_cot DOCTOR_TYPE=basic SPEC_COT=false bash "$ROOT_DIR/results/performance/run_experiment.sh"
