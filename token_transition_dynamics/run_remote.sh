#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/ProcessBench/reasoning_error_detection/llama31_8b}"
DOMAINS="${DOMAINS:-gsm8k,math,olympiadbench,omnimath}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEMO_ROOT}/outputs/token_transition_dynamics}"

export PYTHONPATH="${DEMO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DEMO_ROOT}"

case "${MODE}" in
  preflight)
    "${PYTHON_BIN}" -m token_transition_dynamics.main preflight \
      --data-root "${DATA_ROOT}" \
      --domains "${DOMAINS}"
    ;;
  smoke)
    "${PYTHON_BIN}" -m token_transition_dynamics.main run \
      --data-root "${DATA_ROOT}" \
      --domains "${DOMAINS}" \
      --output-dir "${OUTPUT_ROOT}/smoke" \
      --max-records-per-domain 128 \
      --window-size 16 \
      --neighbors 10 \
      --tle-centers 4 \
      --train-windows-per-chain 2 \
      --calibration-windows-per-chain 2 \
      --max-test-windows-per-chain 6 \
      --position-bins 2 \
      --min-baseline-samples 4 \
      --bootstrap-samples 200
    ;;
  full)
    "${PYTHON_BIN}" -m token_transition_dynamics.main run \
      --data-root "${DATA_ROOT}" \
      --domains "${DOMAINS}" \
      --output-dir "${OUTPUT_ROOT}/full" \
      --window-size 24 \
      --neighbors 20 \
      --tle-centers 6 \
      --train-windows-per-chain 4 \
      --calibration-windows-per-chain 4 \
      --max-test-windows-per-chain 12 \
      --position-bins 4 \
      --min-baseline-samples 8 \
      --bootstrap-samples 1000
    ;;
  *)
    echo "usage: bash token_transition_dynamics/run_remote.sh preflight|smoke|full" >&2
    exit 2
    ;;
esac
