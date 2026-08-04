#!/usr/bin/env bash
set -euo pipefail

# User-editable remote paths and experiment settings.
PYTHON_BIN="${PYTHON_BIN:-python}"
PROCESSBENCH_ROOT="${PROCESSBENCH_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/demo/data/exact/processbench_observer_llama31_full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/ProcessBench/reasoning_error_detection/llama31_8b}"
SUBSETS="${SUBSETS:-gsm8k math olympiadbench omnimath}"
LAYERS="${LAYERS:-8,10,12,14,16,18,20,22}"
PROJECTION_DIM="${PROJECTION_DIM:-16}"
PROJECTION_BATCH_SIZE="${PROJECTION_BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda:0}"
FOLDS="${FOLDS:-5}"
LOGISTIC_C="${LOGISTIC_C:-0.25}"
SEED="${SEED:-17}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/3] Verifying the active Python environment"
"$PYTHON_BIN" - <<'PY'
import sys
import numpy
import scipy
import sklearn
import torch
import reasoning_error_detection

print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("torch:", torch.__version__)
print("reasoning_error_detection imports: OK")
PY

echo "[2/3] Running focused tests when pytest is available"
if "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pytest tests/test_reasoning_error_detection.py -q
else
    echo "pytest is not installed; focused tests skipped after runtime import verification"
fi

echo "[3/3] Detecting ProcessBench first-error steps"
for subset in $SUBSETS; do
    input_path="$PROCESSBENCH_ROOT/${subset}/geometry/trace.npz"
    output_dir="$OUTPUT_ROOT/${subset}"
    if [[ ! -f "$input_path" ]]; then
        echo "ProcessBench geometry manifest is missing: $input_path" >&2
        exit 1
    fi
    echo "--- $subset ---"
    "$PYTHON_BIN" -m reasoning_error_detection.main \
        --input "$input_path" \
        --output_dir "$output_dir" \
        --layers "$LAYERS" \
        --projection_dim "$PROJECTION_DIM" \
        --projection_batch_size "$PROJECTION_BATCH_SIZE" \
        --device "$DEVICE" \
        --folds "$FOLDS" \
        --logistic_c "$LOGISTIC_C" \
        --seed "$SEED"
done

echo "Completed. Reports: $OUTPUT_ROOT/<subset>/summary.json"
