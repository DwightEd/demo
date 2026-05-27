#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Full (step × layer) low-rank pipeline runner.
#
# Assumes:
#   * Python deps already installed (this script does NOT pip install).
#   * Llama-3.1-8B-Instruct already present locally at $MODEL_DIR.
#   * ProcessBench is downloaded via the hf-mirror.com endpoint (set below);
#     re-downloads on first run if the local HF cache is empty.
#
# Usage (recommended inside tmux/screen):
#   chmod +x run_pipeline.sh   # one-off
#   ./run_pipeline.sh
#
# Outputs:
#   data/<subset>_spectral.npz       — (T, L) spectral field per trajectory
#   data/<subset>_analysis_<X>.npz   — low-rank analysis on channel D/V/C
#   output/<subset>/fig*.png         — plots for channel D
#   logs/<subset>_*.log              — captured stdout per stage
# -----------------------------------------------------------------------------

set -euo pipefail

# ---- Configuration -----------------------------------------------------------
MODEL_DIR="/gz-data/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DATASET="Qwen/ProcessBench"
SUBSETS=(gsm8k math olympiadbench omnimath)
N_CORRECT=50
N_ERROR=50
MAX_SEQ_LEN=4096
SEED=42

# Channels to analyse (D = effective rank, V = spectral energy, C = top conc.)
CHANNELS=(D V C)

# Skip the smoke test (set to 1 to skip)
SKIP_SMOKE=${SKIP_SMOKE:-0}

# ---- HuggingFace endpoint mirror --------------------------------------------
# Route HF Hub traffic (datasets + tokenizer files if missing) through the
# hf-mirror.com mirror. The local model directory is used directly, so the
# model itself is not downloaded again.
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# Optional cache override (uncomment if /gz-data is faster than /root)
# export HF_HOME="/gz-data/hf-cache"

# ---- Sanity checks -----------------------------------------------------------
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model dir not found: $MODEL_DIR" >&2
    exit 1
fi

mkdir -p data output logs

# ---- 0) Smoke test -----------------------------------------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
    echo "============================================================"
    echo "[0/2] Smoke test (synthetic data, no GPU needed)"
    echo "============================================================"
    python smoke_test.py 2>&1 | tee logs/smoke_test.log
fi

# ---- 1) Per-subset pipeline --------------------------------------------------
SUBSET_TIMINGS=()

for subset in "${SUBSETS[@]}"; do
    t_start=$(date +%s)

    echo
    echo "============================================================"
    echo "[1/2] subset = $subset   (n_correct=$N_CORRECT  n_error=$N_ERROR)"
    echo "============================================================"

    # ---- a) Extract (T, L) spectral field (GPU stage) ----
    python 01_extract_spectral_field.py \
        --model "$MODEL_DIR" \
        --dataset "$DATASET" \
        --subset "$subset" \
        --n_correct "$N_CORRECT" \
        --n_error "$N_ERROR" \
        --layers all \
        --max_seq_len "$MAX_SEQ_LEN" \
        --seed "$SEED" \
        --output "data/${subset}_spectral.npz" \
        2>&1 | tee "logs/${subset}_extract.log"

    # ---- b) Low-rank analysis on each channel (CPU stage, seconds) ----
    for channel in "${CHANNELS[@]}"; do
        python 02_lowrank_analysis.py \
            --input "data/${subset}_spectral.npz" \
            --channel "$channel" \
            --rank_k 1 \
            --output "data/${subset}_analysis_${channel}.npz" \
            2>&1 | tee "logs/${subset}_analysis_${channel}.log"
    done

    # ---- c) Figures (only for D channel; V/C inspected via logs) ----
    python 03_plot_results.py \
        --spectral "data/${subset}_spectral.npz" \
        --analysis "data/${subset}_analysis_D.npz" \
        --outdir "output/${subset}/" \
        2>&1 | tee "logs/${subset}_plot.log"

    t_end=$(date +%s)
    SUBSET_TIMINGS+=("$subset: $((t_end - t_start))s")
done

# ---- 2) Summary --------------------------------------------------------------
echo
echo "============================================================"
echo "[2/2] AUROC summary across subsets × channels"
echo "============================================================"
for subset in "${SUBSETS[@]}"; do
    for channel in "${CHANNELS[@]}"; do
        log="logs/${subset}_analysis_${channel}.log"
        echo
        echo "--- subset=$subset  channel=$channel ---"
        if [ -f "$log" ]; then
            grep -E "AUROC|lowrank_k=" "$log" || echo "(no AUROC lines found)"
        else
            echo "(log missing: $log)"
        fi
    done
done

echo
echo "============================================================"
echo "Timings:"
for t in "${SUBSET_TIMINGS[@]}"; do
    echo "  $t"
done
echo "============================================================"
echo "Done.  Figures: output/<subset>/fig*.png   Logs: logs/"
