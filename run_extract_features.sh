#!/usr/bin/env bash
# Teacher-forcing feature extraction (refactor 2026-06-09).
# Extracts the three paper uncertainty channels (U_D/U_C/U_E, per token) + our
# raw activation-degree geometry (per-step exp-pooled + per-token) for:
#   (1) all labeled ProcessBench gsm8k solutions (gold step labels), and
#   (2) the previously sampled K=12 responses (an existing 10 npz).
#
# Runs on the GPU box (Llama-3.1-8B). U_E is one backward / token -- the slow
# part; tune UE_STRIDE / UE_LAYERS_FROM if it is too slow or OOMs.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
MODEL="${MODEL:-/gz-data/models/Meta-Llama-3.1-8B-Instruct}"
LAYERS="${LAYERS:-8,16,24,31}"          # hidden_states indices (0=embeddings)
OUTDIR="${OUTDIR:-data/features}"
UE_STRIDE="${UE_STRIDE:-1}"             # 1 = U_E at every token (faithful)
UE_LAYERS_FROM="${UE_LAYERS_FROM:-}"   # e.g. 16 -> only layers>=16 carry grad (faster)
SAMPLED_NPZ="${SAMPLED_NPZ:-data/gsm8k_multisample_sv.npz}"
SAMPLED_SUBSET="${SAMPLED_SUBSET:-gsm8k}"
N_PROBLEMS="${N_PROBLEMS:-300}"

mkdir -p "$OUTDIR"
UE_ARG=""
[ -n "$UE_LAYERS_FROM" ] && UE_ARG="--ue_layers_from $UE_LAYERS_FROM"

echo "==================================================================="
echo "[0/2] smoke test (tiny CPU model, no GPU) -- verifies wiring"
echo "==================================================================="
$PY extract_features.py --source processbench --pb_subset "$SAMPLED_SUBSET" \
    --smoke --layers all --limit 5 --ue_stride 4 \
    --output "$OUTDIR/_smoke.npz"

echo "==================================================================="
echo "[1/2] ProcessBench $SAMPLED_SUBSET  (gold step labels, teacher-forced)"
echo "==================================================================="
$PY extract_features.py \
    --source processbench \
    --pb_path data/processbench --pb_subset "$SAMPLED_SUBSET" \
    --model "$MODEL" --layers "$LAYERS" \
    --ue_stride "$UE_STRIDE" $UE_ARG \
    --output "$OUTDIR/processbench_${SAMPLED_SUBSET}_features.npz"

echo "==================================================================="
echo "[2/2] Sampled K=12 responses  ($SAMPLED_NPZ)"
echo "==================================================================="
if [ -f "$SAMPLED_NPZ" ]; then
  $PY extract_features.py \
      --source sampled --sampled_npz "$SAMPLED_NPZ" \
      --dataset_format processbench --dataset data/processbench \
      --subset "$SAMPLED_SUBSET" --n_problems "$N_PROBLEMS" \
      --model "$MODEL" --layers "$LAYERS" \
      --ue_stride "$UE_STRIDE" $UE_ARG \
      --output "$OUTDIR/sampled_${SAMPLED_SUBSET}_features.npz"
else
  echo "  SKIP: $SAMPLED_NPZ not found. Set SAMPLED_NPZ=... to the stored"
  echo "        10_sample_and_extract npz (the K=12 responses) and re-run."
fi

echo "Done. Outputs in $OUTDIR/"
