#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Phase 2: same-problem multi-sampling -> within-problem (difficulty-controlled)
# test of whether activation participation PREDICTS failure (vs just tracks
# problem difficulty).
#
#   1) 10_sample_and_extract.py : sample K solutions/problem from GSM8K, judge by
#      gold answer, extract per-(step,layer) participation (reuses 01's pipeline).
#   2) 11_within_problem_analysis.py : within-problem AUROC + paired Wilcoxon +
#      early-window + per-position curve.
#
# Usage (inside tmux/screen):
#   chmod +x run_multisample_pipeline.sh
#   ./run_multisample_pipeline.sh                       # 300 problems x K=8
#   N_PROBLEMS=20 K=8 ./run_multisample_pipeline.sh     # quick smoke test
#   N_PROBLEMS=500 K=12 TEMP=1.0 ./run_multisample_pipeline.sh
# -----------------------------------------------------------------------------
set -euo pipefail

MODEL_DIR="/gz-data/models/Meta-Llama-3.1-8B-Instruct"
# Use the LOCAL ProcessBench (gsm8k subset): problem = GSM8K question, gold is
# derived from correct (label==-1) solutions. No external dataset needed.
DATASET_FORMAT=${DATASET_FORMAT:-processbench}
DATASET=${DATASET:-/gz-data/research/demo/data/hf_datasets/ProcessBench}
SUBSET=${SUBSET:-gsm8k}

N_PROBLEMS=${N_PROBLEMS:-300}
K=${K:-8}
TEMP=${TEMP:-0.8}
TOP_P=${TOP_P:-0.95}
MAX_NEW=${MAX_NEW:-512}
SEED=${SEED:-42}

BANDS=(${BANDS:-deep all mid})
METRICS=(${METRICS:-ae pr})
MODE=${MODE:-step_exp}
EARLY=${EARLY:-3}

FORCE=${FORCE:-0}

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_DATASETS_CACHE="${SCRIPT_DIR}/data/hf_datasets"

if [ ! -d "$MODEL_DIR" ]; then echo "ERROR: model dir not found: $MODEL_DIR" >&2; exit 1; fi
if [ ! -f "10_sample_and_extract.py" ] || [ ! -f "11_within_problem_analysis.py" ]; then
    echo "ERROR: run from the demo/ project root." >&2; exit 1
fi
mkdir -p data output logs "$HF_DATASETS_CACHE"

npz="data/gsm8k_multisample_sv.npz"

echo "MODEL_DIR  = $MODEL_DIR"
echo "DATASET    = $DATASET_FORMAT:$DATASET/$SUBSET"
echo "N_PROBLEMS = $N_PROBLEMS   K = $K   TEMP = $TEMP"

# ---- 1) sample + extract (GPU) ----
if [ -f "$npz" ] && [ "$FORCE" != "1" ]; then
    echo "[sample] $npz exists, skipping (FORCE=1 to redo)."
else
    python 10_sample_and_extract.py \
        --model "$MODEL_DIR" \
        --dataset_format "$DATASET_FORMAT" --dataset "$DATASET" --subset "$SUBSET" \
        --n_problems "$N_PROBLEMS" --k_samples "$K" \
        --temperature "$TEMP" --top_p "$TOP_P" --max_new_tokens "$MAX_NEW" \
        --seed "$SEED" --output "$npz" \
        2>&1 | tee "logs/gsm8k_multisample_extract.log"
fi

# ---- 2) within-problem analysis (CPU) ----
for band in "${BANDS[@]}"; do
    for metric in "${METRICS[@]}"; do
        echo
        echo "--- within-problem: band=$band metric=$metric mode=$MODE ---"
        python 11_within_problem_analysis.py \
            --input "$npz" --layer_band "$band" --metric "$metric" \
            --mode "$MODE" --early_window "$EARLY" \
            --output "data/within_${band}_${metric}.npz" \
            2>&1 | tee "logs/within_${band}_${metric}.log"
    done
done

echo
echo "============================================================"
echo "GATE: read section (A) within-problem AUROC and (B) Wilcoxon p."
echo "  >0.55 & p<0.05  -> participation predicts failure (difficulty controlled)"
echo "  ~0.50           -> cross-problem signal was difficulty; pivot"
echo "============================================================"
