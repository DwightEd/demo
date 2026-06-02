#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# CIM-faithful metrics pipeline (TLE/MLE intrinsic dimension + log-det info
# volume) on single reasoning trajectories, evaluated under the length gate.
#
# What it does, per subset:
#   1) Re-extract the (step x layer) field WITH --cim_metrics, which additionally
#      stores M_Dtle (nonlinear intrinsic dim) and M_Vld (log-det info volume)
#      next to the original M_D / M_V / M_C.  (A fresh extraction is required:
#      these two quantities need the raw token cloud, which the old npz did not
#      keep.)
#   2) Run 07_cim_metrics_analysis.py over several layer bands, comparing the
#      CIM-faithful metrics and the joint "low-dim + non-degenerate" anchors
#      against the linear mean_D = 0.62 baseline, all under the length gate.
#
# Usage (recommended inside tmux/screen):
#   chmod +x run_cim_pipeline.sh
#   ./run_cim_pipeline.sh                 # default: gsm8k only (cheapest)
#   SUBSETS="gsm8k math" ./run_cim_pipeline.sh
#   SUBSETS="gsm8k math olympiadbench omnimath" ./run_cim_pipeline.sh
#
# Outputs:
#   data/<subset>_cim.npz                 — field + M_Dtle + M_Vld
#   data/<subset>_cim_analysis_<band>.npz — length-gated analysis per band
#   logs/<subset>_cim_extract.log         — extraction stdout
#   logs/<subset>_cim_analysis_<band>.log — analysis stdout
# -----------------------------------------------------------------------------

set -euo pipefail

# ---- Configuration (matches run_pipeline.sh) --------------------------------
MODEL_DIR="/gz-data/models/Meta-Llama-3.1-8B-Instruct"
DATASET="/gz-data/research/demo/data/hf_datasets/ProcessBench"
# Default to gsm8k only (the cheapest go/no-go gate). Override via env:
#   SUBSETS="gsm8k math" ./run_cim_pipeline.sh
SUBSETS=(${SUBSETS:-gsm8k})
N_CORRECT=${N_CORRECT:-50}
N_ERROR=${N_ERROR:-50}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
SEED=${SEED:-42}

# Layer bands to analyse. "all" and "deep" first (cheapest insight); add mid/early
# if you want to localize the signal.
BANDS=(${BANDS:-all deep mid})

# Match-subset bins + analysis seed for the length gate.
N_MATCH_BINS=${N_MATCH_BINS:-8}
ANALYSIS_SEED=${ANALYSIS_SEED:-0}

# Skip re-extraction if the *_cim.npz already exists (set FORCE=1 to redo).
FORCE=${FORCE:-0}

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
if [ ! -f "01_extract_spectral_field.py" ] || [ ! -f "07_cim_metrics_analysis.py" ]; then
    echo "ERROR: run this from the demo/ project root (01_*.py and 07_*.py must be here)." >&2
    exit 1
fi

mkdir -p data output logs "$HF_DATASETS_CACHE"

echo "MODEL_DIR          = $MODEL_DIR"
echo "SUBSETS            = ${SUBSETS[*]}"
echo "BANDS              = ${BANDS[*]}"
echo "HF_ENDPOINT        = $HF_ENDPOINT"
echo "HF_DATASETS_CACHE  = $HF_DATASETS_CACHE"

TIMINGS=()

for subset in "${SUBSETS[@]}"; do
    t_start=$(date +%s)
    npz="data/${subset}_cim.npz"

    echo
    echo "============================================================"
    echo "subset = $subset   (n_correct=$N_CORRECT  n_error=$N_ERROR)"
    echo "============================================================"

    # ---- 1) Extract WITH cim_metrics (GPU stage) ----
    if [ -f "$npz" ] && [ "$FORCE" != "1" ]; then
        echo "[extract] $npz already exists, skipping (set FORCE=1 to redo)."
    else
        echo "[extract] -> $npz"
        python 01_extract_spectral_field.py \
            --model "$MODEL_DIR" \
            --dataset "$DATASET" \
            --subset "$subset" \
            --n_correct "$N_CORRECT" \
            --n_error "$N_ERROR" \
            --layers all \
            --max_seq_len "$MAX_SEQ_LEN" \
            --seed "$SEED" \
            --cim_metrics \
            --output "$npz" \
            2>&1 | tee "logs/${subset}_cim_extract.log"
    fi

    # ---- 2) Length-gated analysis per band (CPU stage, seconds) ----
    for band in "${BANDS[@]}"; do
        echo
        echo "--- analysis: subset=$subset  band=$band ---"
        python 07_cim_metrics_analysis.py \
            --input "$npz" \
            --layer_band "$band" \
            --n_match_bins "$N_MATCH_BINS" \
            --seed "$ANALYSIS_SEED" \
            --output "data/${subset}_cim_analysis_${band}.npz" \
            2>&1 | tee "logs/${subset}_cim_analysis_${band}.log"
    done

    t_end=$(date +%s)
    TIMINGS+=("$subset: $((t_end - t_start))s")
done

# ---- 3) Summary: pull the VERDICT + key rows from each analysis log ----------
echo
echo "============================================================"
echo "SUMMARY  (matched-AUROC is the number that matters)"
echo "============================================================"
for subset in "${SUBSETS[@]}"; do
    for band in "${BANDS[@]}"; do
        log="logs/${subset}_cim_analysis_${band}.log"
        echo
        echo "--- subset=$subset  band=$band ---"
        if [ -f "$log" ]; then
            # show the length baseline, the key CIM/anchor rows, and the verdict
            grep -E "AUROC\(n_steps\)|mean_D\(linear\)|mean_Dtle|mean_Vld|Vld_over_Dtle|Vld/exp|Vld-|best feature|BEATS|No CIM" "$log" \
                || echo "(no matching lines)"
        else
            echo "(log missing: $log)"
        fi
    done
done

echo
echo "============================================================"
echo "Timings:"
for t in "${TIMINGS[@]}"; do echo "  $t"; done
echo "============================================================"
echo "Done.  Per-band npz: data/<subset>_cim_analysis_<band>.npz"
echo "Read the matchAUROC column: does any CIM-faithful metric or joint anchor"
echo "(Vld_over_Dtle, Vld/exp(0.1*Dtle), mean_Vld) beat linear mean_D (~0.62)?"