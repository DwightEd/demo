#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODELS_DIR="/gz-data/models"
export HF_DATASETS_CACHE="${PROJ_ROOT}/data/hf_datasets"

# ---------- 1. 下载数据集 ----------
pip install -q datasets huggingface_hub
huggingface-cli download --repo-type dataset Qwen/ProcessBench --local-dir "${HF_DATASETS_CACHE}/ProcessBench"

# ---------- 2. 运行实验 ----------
cd "$PROJ_ROOT"
MODEL_DIR="${MODELS_DIR}/LLM-Research/Meta-Llama-3___1-8B-Instruct"
SUBSET="gsm8k"
GEOM_NPZ="data/${SUBSET}_geom.npz"

python 01_extract_spectral_field.py \
    --model "$MODEL_DIR" \
    --subset "$SUBSET" \
    --n_correct 50 --n_error 50 \
    --store_geometry --geom_k 4 \
    --output "$GEOM_NPZ"

python 05_geometry_analysis.py \
    --input "$GEOM_NPZ" \
    --layer_band deep

echo "Done."
