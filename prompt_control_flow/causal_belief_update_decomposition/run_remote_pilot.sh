#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(cd -- "${PROJECT_DIR}/../.." && pwd)"

# User-editable remote configuration. Change these defaults when the server,
# model, GPU, or persistent data location changes. Environment variables remain
# available for one-off overrides. `python` is used directly from the active environment.
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/CBUD/finite_field_predictive_alias/llama31_8b/pilot_200}"
PRIMARY_LAYER="${PRIMARY_LAYER:-16}"

# Artifact paths are derived from DATA_ROOT so a storage move requires one edit.
ALIAS_PATH="${ALIAS_PATH:-${DATA_ROOT}/source/predictive_alias_pairs_200.jsonl}"
TRACE_PATH="${TRACE_PATH:-${DATA_ROOT}/extractions/boundary_states_and_logit_sketch_200.npz}"
CHARTS_PATH="${CHARTS_PATH:-${DATA_ROOT}/derived/representation/layer_charts.npz}"
UPDATE_PATH="${UPDATE_PATH:-${DATA_ROOT}/extractions/attention_mlp_block_updates_200.npz}"
REPORT_DIR="${REPORT_DIR:-${DATA_ROOT}/results/update_audit}"

[[ -d "${MODEL_DIR}" ]] || { echo "model directory is missing: ${MODEL_DIR}" >&2; exit 2; }

mkdir -p \
  "$(dirname -- "${ALIAS_PATH}")" \
  "$(dirname -- "${TRACE_PATH}")" \
  "$(dirname -- "${CHARTS_PATH}")" \
  "$(dirname -- "${UPDATE_PATH}")" \
  "${REPORT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONUNBUFFERED=1
cd "${DEMO_ROOT}"

echo "[1/6] Verifying the renamed package and update method"
"${PYTHON_BIN}" - <<'PY'
import sys

import torch
import transformers

from prompt_control_flow.causal_belief_update_decomposition import (
    routing_extraction,
    update_audit,
    update_extraction,
    world,
)

assert routing_extraction and update_audit and update_extraction and world
print(
    "CBUD runtime imports: OK",
    f"python={sys.executable}",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
)
PY

if "${PYTHON_BIN}" -c \
  'import importlib.util; raise SystemExit(importlib.util.find_spec("pytest") is None)'
then
  "${PYTHON_BIN}" -m pytest \
    tests/test_causal_belief_updates.py \
    tests/test_causal_belief_routing.py \
    -q
else
  echo "pytest is not installed; focused unit tests skipped after runtime import verification"
fi

echo "[2/6] Preparing predictive-alias observations"
if [[ ! -f "${ALIAS_PATH}" ]]; then
  "${PYTHON_BIN}" build_predictive_aliases.py \
    --output "${ALIAS_PATH}" \
    --num_pairs 200 \
    --modulus 3 \
    --num_variables 4 \
    --common_rank 2 \
    --template_families 3 \
    --seed 17
else
  echo "reusing aliases: ${ALIAS_PATH}"
fi

echo "[3/6] Preparing exact boundary-state trace"
if [[ ! -f "${TRACE_PATH}" ]]; then
  "${PYTHON_BIN}" extract_causal_belief_states.py \
    --input "${ALIAS_PATH}" \
    --model "${MODEL_DIR}" \
    --output "${TRACE_PATH}" \
    --layers 0,4,8,12,16,20,24,28,32 \
    --batch_size 16 \
    --max_batch_tokens 4096 \
    --logit_sketch_dim 256 \
    --device cuda \
    --dtype bfloat16
else
  echo "reusing trace: ${TRACE_PATH}"
fi

echo "[4/6] Preparing the cross-fitted representation chart"
if [[ ! -f "${CHARTS_PATH}" ]]; then
  "${PYTHON_BIN}" audit_causal_belief_routing.py \
    --input "${TRACE_PATH}" \
    --output_dir "$(dirname -- "${CHARTS_PATH}")" \
    --folds 5 \
    --projection_dim 64 \
    --ridge_alpha 10 \
    --bootstrap 2000 \
    --compute_device cuda
else
  echo "reusing layer charts: ${CHARTS_PATH}"
fi

echo "[5/6] Extracting target-token attention/MLP/block updates"
"${PYTHON_BIN}" extract_causal_belief_updates.py \
  --trace "${TRACE_PATH}" \
  --charts "${CHARTS_PATH}" \
  --model "${MODEL_DIR}" \
  --output "${UPDATE_PATH}" \
  --batch_size 8 \
  --max_batch_tokens 4096 \
  --device cuda \
  --dtype bfloat16

echo "[6/6] Auditing the preregistered layer"
"${PYTHON_BIN}" audit_causal_belief_updates.py \
  --input "${UPDATE_PATH}" \
  --output_dir "${REPORT_DIR}" \
  --primary_layer "${PRIMARY_LAYER}" \
  --bootstrap 2000

echo "CBUD pilot complete: ${REPORT_DIR}/summary.json"
