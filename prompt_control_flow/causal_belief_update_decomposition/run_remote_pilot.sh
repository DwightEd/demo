#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(cd -- "${PROJECT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
MODEL_DIR="${MODEL_DIR:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
TRACE_PATH="${TRACE_PATH:-${DEMO_ROOT}/data/causal_belief_routing/alias_pilot_200_trace.npz}"
CHARTS_PATH="${CHARTS_PATH:-${DEMO_ROOT}/outputs/causal_belief_routing/alias_pilot_200/representation/layer_charts.npz}"
UPDATE_PATH="${UPDATE_PATH:-${DEMO_ROOT}/data/causal_belief_update_decomposition/alias_pilot_200_updates.npz}"
REPORT_DIR="${REPORT_DIR:-${DEMO_ROOT}/outputs/causal_belief_update_decomposition/alias_pilot_200/updates}"
PRIMARY_LAYER="${PRIMARY_LAYER:-16}"

[[ -x "${PYTHON_BIN}" ]] || { echo "python is not executable: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${MODEL_DIR}" ]] || { echo "model directory is missing: ${MODEL_DIR}" >&2; exit 2; }
[[ -f "${TRACE_PATH}" ]] || { echo "trace is missing: ${TRACE_PATH}" >&2; exit 2; }
[[ -f "${CHARTS_PATH}" ]] || { echo "layer charts are missing: ${CHARTS_PATH}" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${DEMO_ROOT}"

echo "[1/3] Verifying the renamed package and update method"
"${PYTHON_BIN}" -m pytest \
  tests/test_causal_belief_updates.py \
  tests/test_causal_belief_routing.py \
  -q

echo "[2/3] Extracting target-token attention/MLP/block updates"
"${PYTHON_BIN}" extract_causal_belief_updates.py \
  --trace "${TRACE_PATH}" \
  --charts "${CHARTS_PATH}" \
  --model "${MODEL_DIR}" \
  --output "${UPDATE_PATH}" \
  --batch_size 8 \
  --max_batch_tokens 4096 \
  --device cuda \
  --dtype bfloat16

echo "[3/3] Auditing the preregistered layer"
"${PYTHON_BIN}" audit_causal_belief_updates.py \
  --input "${UPDATE_PATH}" \
  --output_dir "${REPORT_DIR}" \
  --primary_layer "${PRIMARY_LAYER}" \
  --bootstrap 2000

echo "CBUD pilot complete: ${REPORT_DIR}/summary.json"
