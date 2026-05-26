#!/bin/bash
# ============================================================
# Q1 Pilot: AFC Hypothesis Verification on ProcessBench
# ============================================================
# Usage:  bash run_q1.sh
# GPU:    1x A100/H100/4090 (16GB+ VRAM)
# Time:   ~30-60 min for gsm8k (400 examples)
# ============================================================

set -e

# ── Configuration ──────────────────────────────────────────
MODEL_PATH="gz-data/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
SPLITS="gsm8k"                  # Start with gsm8k; add math,olympiadbench,omnimath later
MAX_EXAMPLES=-1                 # -1 = all; set to 50 for quick smoke test
MAX_SEQ_LEN=2048
DTYPE="float16"                 # bfloat16 if A100/H100

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
# ───────────────────────────────────────────────────────────

echo "============================================"
echo "Q1 Pilot: AFC Hypothesis Verification"
echo "============================================"
echo "Model:  ${MODEL_PATH}"
echo "Splits: ${SPLITS}"
echo "Work:   ${WORK_DIR}"
echo ""

# ── Step 1: Install dependencies ──────────────────────────
echo "[1/3] Installing dependencies..."
pip install -q torch transformers accelerate scipy scikit-learn numpy 2>&1 | tail -3
echo "Done."

# ── Step 2: Check data ────────────────────────────────────
echo ""
echo "[2/3] Checking ProcessBench data..."
for s in gsm8k math olympiadbench omnimath; do
    if [ -f "${DATA_DIR}/${s}.jsonl" ]; then
        echo "  ${s}.jsonl: OK ($(wc -l < "${DATA_DIR}/${s}.jsonl") lines)"
    else
        echo "  ${s}.jsonl: MISSING! Data should be in ${DATA_DIR}/"
        exit 1
    fi
done

# ── Step 3: Extract AFC metrics ───────────────────────────
echo ""
echo "[3/3] Extracting AFC metrics (this takes a while)..."
mkdir -p "${RESULTS_DIR}"

python3 "${WORK_DIR}/01_extract_afc.py" \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${RESULTS_DIR}" \
    --splits "${SPLITS}" \
    --max_examples ${MAX_EXAMPLES} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --dtype "${DTYPE}"

# ── Step 4: Evaluate ──────────────────────────────────────
echo ""
echo "Evaluating AFC metrics..."
python3 "${WORK_DIR}/02_evaluate.py" \
    --results_dir "${RESULTS_DIR}" \
    --splits "${SPLITS}" \
    --output "${RESULTS_DIR}/q1_evaluation.json"

echo ""
echo "============================================"
echo "Q1 Pilot Complete!"
echo "Results: ${RESULTS_DIR}/q1_evaluation.json"
echo "============================================"
