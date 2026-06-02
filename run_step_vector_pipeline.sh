#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Step-vector activation-participation pipeline.
#
# Implements Streaming-HD's optimal step-time-exponential weighting (and 3
# baseline weightings) to turn each reasoning step into ONE vector, then measures
# how many dimensions that vector activates (participation ratio / activation
# entropy) -- the "uncertain reasoning -> more active dims" hypothesis.
#
# Per subset:
#   1) Re-extract WITH --step_vectors (stores per-(step,layer) PR & AE for each
#      weighting mode + per-step output-token entropy). Also turns on
#      --cim_metrics so the same forward pass yields the CIM metrics too (free).
#   2) Run 08_step_vector_analysis.py over several layer bands x metrics,
#      length-gated, and report the PR-vs-output-entropy correlation.
#
# Usage (inside tmux/screen):
#   chmod +x run_step_vector_pipeline.sh
#   ./run_step_vector_pipeline.sh                 # gsm8k only (cheapest)
#   SUBSETS="gsm8k math" ./run_step_vector_pipeline.sh
#   BANDS="mid all deep" METRICS="pr ae" ./run_step_vector_pipeline.sh
#
# Outputs:
#   data/<subset>_sv.npz                              field + step-vector payload
#   data/<subset>_sv_analysis_<band>_<metric>.npz     per-config analysis
#   logs/<subset>_sv_extract.log
#   logs/<subset>_sv_analysis_<band>_<metric>.log
# -----------------------------------------------------------------------------

set -euo pipefail

# ---- Configuration (your real paths) ----------------------------------------
MODEL_DIR="/gz-data/models/Meta-Llama-3.1-8B-Instruct"
DATASET="/gz-data/research/demo/data/hf_datasets/ProcessBench"

SUBSETS=(${SUBSETS:-gsm8k})
N_CORRECT=${N_CORRECT:-50}
N_ERROR=${N_ERROR:-50}
SEED=${SEED:-42}

# step-vector weighting modes to compare (step_exp = Streaming-HD optimum)
SV_MODES=${SV_MODES:-last,mean,linear,step_exp}

# analysis sweep. Paper says mid layers strongest -> mid first.
BANDS=(${BANDS:-mid all deep})
METRICS=(${METRICS:-pr ae})

N_MATCH_BINS=${N_MATCH_BINS:-8}
ANALYSIS_SEED=${ANALYSIS_SEED:-0}

# also compute CIM metrics in the same forward pass (1) or not (0)
WITH_CIM=${WITH_CIM:-1}

# skip re-extraction if *_sv.npz exists (FORCE=1 to redo)
FORCE=${FORCE:-0}

# ---- HF env (harmless with a local dataset; keep for offline safety) --------
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_DATASETS_CACHE="${SCRIPT_DIR}/data/hf_datasets"

# ---- Sanity checks -----------------------------------------------------------
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model dir not found: $MODEL_DIR" >&2; exit 1
fi
if [ ! -e "$DATASET" ]; then
    echo "ERROR: dataset not found: $DATASET" >&2; exit 1
fi
if [ ! -f "01_extract_spectral_field.py" ] || [ ! -f "08_step_vector_analysis.py" ]; then
    echo "ERROR: run from the demo/ project root (01_*.py and 08_*.py must be here)." >&2; exit 1
fi
if [ ! -f "utils/step_vector.py" ]; then
    echo "ERROR: utils/step_vector.py missing. Place it under utils/." >&2; exit 1
fi

mkdir -p data output logs "$HF_DATASETS_CACHE"

CIM_FLAG=""
[ "$WITH_CIM" = "1" ] && CIM_FLAG="--cim_metrics"

echo "MODEL_DIR   = $MODEL_DIR"
echo "DATASET     = $DATASET"
echo "SUBSETS     = ${SUBSETS[*]}"
echo "SV_MODES    = $SV_MODES"
echo "BANDS       = ${BANDS[*]}    METRICS = ${METRICS[*]}"
echo "WITH_CIM    = $WITH_CIM"

TIMINGS=()

for subset in "${SUBSETS[@]}"; do
    t_start=$(date +%s)
    npz="data/${subset}_sv.npz"

    echo
    echo "============================================================"
    echo "subset = $subset   (n_correct=$N_CORRECT  n_error=$N_ERROR)"
    echo "============================================================"

    # ---- 1) Extract WITH step_vectors (GPU stage) ----
    if [ -f "$npz" ] && [ "$FORCE" != "1" ]; then
        echo "[extract] $npz exists, skipping (FORCE=1 to redo)."
    else
        echo "[extract] -> $npz"
        python 01_extract_spectral_field.py \
            --model "$MODEL_DIR" \
            --dataset "$DATASET" \
            --subset "$subset" \
            --n_correct "$N_CORRECT" \
            --n_error "$N_ERROR" \
            --layers all \
            --seed "$SEED" \
            --step_vectors \
            --sv_modes "$SV_MODES" \
            $CIM_FLAG \
            --output "$npz" \
            2>&1 | tee "logs/${subset}_sv_extract.log"
    fi

    # ---- 2) Analysis sweep (CPU stage, seconds) ----
    for band in "${BANDS[@]}"; do
        for metric in "${METRICS[@]}"; do
            echo
            echo "--- analysis: subset=$subset  band=$band  metric=$metric ---"
            python 08_step_vector_analysis.py \
                --input "$npz" \
                --layer_band "$band" \
                --metric "$metric" \
                --n_match_bins "$N_MATCH_BINS" \
                --seed "$ANALYSIS_SEED" \
                --output "data/${subset}_sv_analysis_${band}_${metric}.npz" \
                2>&1 | tee "logs/${subset}_sv_analysis_${band}_${metric}.log"
        done
    done

    t_end=$(date +%s)
    TIMINGS+=("$subset: $((t_end - t_start))s")
done

# ---- 3) Summary: pull verdict + key rows ------------------------------------
echo
echo "============================================================"
echo "SUMMARY  (matchAUROC = the number that matters;"
echo "          rho = participation-vs-output-entropy mechanism check)"
echo "============================================================"
for subset in "${SUBSETS[@]}"; do
    for band in "${BANDS[@]}"; do
        for metric in "${METRICS[@]}"; do
            log="logs/${subset}_sv_analysis_${band}_${metric}.log"
            echo
            echo "--- subset=$subset  band=$band  metric=$metric ---"
            if [ -f "$log" ]; then
                grep -E "AUROC\(n_steps\)|last|mean|linear|step_exp|best weighting|rho\(|carries a length|does not clearly" "$log" \
                    || echo "(no matching lines)"
            else
                echo "(log missing)"
            fi
        done
    done
done

echo
echo "============================================================"
echo "Timings:"
for t in "${TIMINGS[@]}"; do echo "  $t"; done
echo "============================================================"
echo "Done."
echo "Read: (a) step_exp matchAUROC vs length baseline (does the optimal weighting carry signal?)"
echo "      (b) rho(participation, output_entropy) sign (does 'more active dims = more uncertain' hold?)"