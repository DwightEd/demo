#!/usr/bin/env bash
# =============================================================================
# From-scratch setup + run on gpugeek (RTX A5000 24 GB, CUDA 12.1, conda3).
#
# What this does, on a freshly-rented machine with NOTHING installed beyond
# the base conda3 image + NVIDIA driver:
#   1) create a persistent conda env (python 3.10) under /gz-data
#   2) install torch (cu121) + transformers/datasets + modelscope
#   3) download Llama-3.1-8B-Instruct (~16 GB) from ModelScope into /gz-data
#   4) pre-fetch ProcessBench (gsm8k subset) into /gz-data
#   5) run 01_extract_spectral_field.py  (--store_geometry --geom_k 4)
#   6) run 05_geometry_analysis.py        (--layer_band deep)
#
# Re-running the script is safe (each step is idempotent / skipped if done).
#
# Usage:
#   bash from_scratch.sh
#
# Assumptions:
#   * /gz-data is mounted and writable (gpugeek persistent disk)
#   * `conda` is on PATH (base conda3 image)
#   * NVIDIA driver works and exposes CUDA 12.x (script aborts if nvidia-smi fails)
#   * this script sits inside the `demo/` directory next to the .py files
# =============================================================================

set -euo pipefail

# ---- Paths (persistent locations all under /gz-data) ------------------------
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODELS_DIR="/gz-data/models"
ENV_DIR="/gz-data/envs/research"        # conda env lives here (prefix-based)
PY_VER="3.10"
export HF_DATASETS_CACHE="${PROJ_ROOT}/data/hf_datasets"
export HF_HOME="/gz-data/hf_cache"

# ---- Run params -------------------------------------------------------------
SUBSET="gsm8k"
N_CORRECT=50
N_ERROR=50
GEOM_K=4
LAYER_BAND="deep"

# ---- HF / ModelScope mirror -------------------------------------------------
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# ---- Sanity ----------------------------------------------------------------
if [ ! -d "/gz-data" ]; then
    echo "ERROR: /gz-data is not mounted. Mount the persistent disk first." >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH (expected conda3 base image)" >&2
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed — no GPU visible to container" >&2
    exit 1
fi
# NOTE: `nvidia-smi | head` under `set -o pipefail` kills the script via SIGPIPE.
# Just dump the full table — it's ~20 lines.
nvidia-smi

mkdir -p "$MODELS_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" \
         "${PROJ_ROOT}/data" "${PROJ_ROOT}/logs" \
         "$(dirname "$ENV_DIR")"

echo "PROJ_ROOT          = $PROJ_ROOT"
echo "ENV_DIR            = $ENV_DIR  (python $PY_VER)"
echo "MODELS_DIR         = $MODELS_DIR"
echo "HF_HOME            = $HF_HOME"
echo "HF_DATASETS_CACHE  = $HF_DATASETS_CACHE"

# ---- 1) conda env ----------------------------------------------------------
# Source conda's shell hooks so `conda activate` works in a non-interactive shell.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo
    echo "============================================================"
    echo "[1/6] create conda env at $ENV_DIR  (python $PY_VER)"
    echo "============================================================"
    conda create -p "$ENV_DIR" "python=${PY_VER}" -y
fi
conda activate "$ENV_DIR"
python -m pip install --upgrade pip wheel setuptools

# ---- 2) deps ---------------------------------------------------------------
echo
echo "============================================================"
echo "[2/6] install dependencies"
echo "============================================================"
# Torch wheel matching the container's CUDA 12.1.0 (A5000 is Ampere → fully supported).
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch
pip install \
    "transformers>=4.40" \
    "datasets>=2.18" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "matplotlib>=3.7" \
    "tqdm>=4.65" \
    modelscope huggingface_hub

python - <<'PY'
import torch, transformers, datasets
print("torch        =", torch.__version__, "cuda?", torch.cuda.is_available())
print("transformers =", transformers.__version__)
print("datasets     =", datasets.__version__)
assert torch.cuda.is_available(), "CUDA not available from torch — check driver/wheel match"
PY

# ---- 3) model --------------------------------------------------------------
EXPECTED_MODEL_DIR="${MODELS_DIR}/LLM-Research/Meta-Llama-3___1-8B-Instruct"
echo
echo "============================================================"
echo "[3/6] download Llama-3.1-8B-Instruct from ModelScope (~16 GB)"
echo "      target: $EXPECTED_MODEL_DIR"
echo "============================================================"

# snapshot_download is file-level resumable: failed shards are retried,
# completed shards are skipped. We still need a bash-level loop because
# the python call hard-fails when *any* shard fails after its own retries.
MAX_TRIES=6
MODEL_DIR=""
for try in $(seq 1 "$MAX_TRIES"); do
    echo "  attempt ${try}/${MAX_TRIES} ..."
    if MODEL_DIR=$(python - <<'PY' 2>&1
import os, sys
from modelscope import snapshot_download
try:
    p = snapshot_download(
        "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        cache_dir=os.environ["MODELS_DIR"],
    )
    print(p)
except Exception as e:
    print(f"DOWNLOAD_FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY
    ); then
        # MODEL_DIR captured stdout. The success path prints exactly the local path.
        if [ -d "$MODEL_DIR" ] && [ -f "${MODEL_DIR}/config.json" ]; then
            echo "  ok -> $MODEL_DIR"
            break
        fi
    fi
    echo "  attempt ${try} failed; sleeping 10s then retrying..." >&2
    MODEL_DIR=""
    sleep 10
done

if [ -z "$MODEL_DIR" ] || [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "ERROR: model download failed after ${MAX_TRIES} attempts." >&2
    echo "  check network, or rerun this script — partial files are kept." >&2
    exit 1
fi
echo "MODEL_DIR = $MODEL_DIR"

# ---- 4) dataset ------------------------------------------------------------
echo
echo "============================================================"
echo "[4/6] pre-fetch ProcessBench / ${SUBSET} via hf-mirror"
echo "============================================================"
python - <<PY
import os
from datasets import load_dataset
ds = load_dataset(
    "Qwen/ProcessBench",
    "${SUBSET}",
    cache_dir=os.environ["HF_DATASETS_CACHE"],
)
print("dataset splits:", {k: len(v) for k, v in ds.items()})
PY

# ---- 5) extract geometry ---------------------------------------------------
cd "$PROJ_ROOT"
GEOM_NPZ="data/${SUBSET}_geom.npz"
GEOM_ANALYSIS_NPZ="data/${SUBSET}_geom_analysis.npz"

echo
echo "============================================================"
echo "[5/6] 01_extract_spectral_field.py  (--store_geometry --geom_k ${GEOM_K})"
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

# ---- 6) geometry analysis --------------------------------------------------
echo
echo "============================================================"
echo "[6/6] 05_geometry_analysis.py  (--layer_band ${LAYER_BAND})"
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
echo "  logs         : logs/${SUBSET}_geom_*.log"
echo "  conda env    : $ENV_DIR   (reactivate with: conda activate $ENV_DIR)"
echo "  model        : $MODEL_DIR"
echo "============================================================"
