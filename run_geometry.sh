#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Geometry pipeline runner: extract per-step geometry then analyse it.
#
# Stages:
#   1) 01_extract_spectral_field.py  with --store_geometry --geom_k 4
#      -> data/<subset>_geom.npz
#   2) 05_geometry_analysis.py       with --layer_band deep
#      -> data/<subset>_geom_analysis.npz
#
# Usage (recommended inside tmux/screen):
#   chmod +x run_geometry.sh   # one-off
#   ./run_geometry.sh
# -----------------------------------------------------------------------------

set -euo pipefail

# ---- Configuration -----------------------------------------------------------
MODEL_DIR="/gz-data/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
SUBSET="gsm8k"
N_CORRECT=50
N_ERROR=50
GEOM_K=4
LAYER_BAND="deep"

# ---- HuggingFace endpoint mirror + cache location ---------------------------
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_DATASETS_CACHE="${SCRIPT_DIR}/data/hf_datasets"

# ---- Sanity checks -----------------------------------------------------------
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model dir not found: $MODEL_DIR" >&2
    exit 1
fi

mkdir -p data logs "$HF_DATASETS_CACHE"

echo "MODEL_DIR          = $MODEL_DIR"
echo "HF_ENDPOINT        = $HF_ENDPOINT"
echo "HF_DATASETS_CACHE  = $HF_DATASETS_CACHE"
echo "SUBSET             = $SUBSET"
echo "GEOM_K             = $GEOM_K"
echo "LAYER_BAND         = $LAYER_BAND"

GEOM_NPZ="data/${SUBSET}_geom.npz"
GEOM_ANALYSIS_NPZ="data/${SUBSET}_geom_analysis.npz"

# ---- 1) Extract per-step geometry (GPU stage) -------------------------------
echo
echo "============================================================"
echo "[1/2] Extract geometry  (subset=$SUBSET  n_correct=$N_CORRECT  n_error=$N_ERROR)"
echo "============================================================"
python 01_extract_spectral_field.py \
    --model "$MODEL_DIR" \
    --subset "$SUBSET" \
    --n_correct "$N_CORRECT" \
    --n_error "$N_ERROR" \
    --store_geometry \
    --geom_k "$GEOM_K" \
    --output "$GEOM_NPZ" \
    2>&1 | tee "logs/${SUBSET}_geom_extract.log"

# ---- 2) Geometry analysis (CPU stage) ---------------------------------------
echo
echo "============================================================"
echo "[2/2] Geometry analysis  (layer_band=$LAYER_BAND)"
echo "============================================================"
python 05_geometry_analysis.py \
    --input "$GEOM_NPZ" \
    --layer_band "$LAYER_BAND" \
    --output "$GEOM_ANALYSIS_NPZ" \
    2>&1 | tee "logs/${SUBSET}_geom_analysis.log"

echo
echo "============================================================"
echo "Done."
echo "  geometry npz : $GEOM_NPZ"
echo "  analysis npz : $GEOM_ANALYSIS_NPZ"
echo "  logs         : logs/${SUBSET}_geom_extract.log, logs/${SUBSET}_geom_analysis.log"
echo "============================================================"
