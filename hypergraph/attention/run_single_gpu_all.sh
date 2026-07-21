#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
GPU="${GPU:-0}"
LAYER="${LAYER:-14}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
DATASETS=(gsm8k math olympiadbench omnimath)

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

for dataset in "${DATASETS[@]}"; do
  traces="data/cct_traces/${dataset}_layer${LAYER}"
  results="results/cct_hg/${dataset}_layer${LAYER}"
  echo "===== ${dataset}: layer ${LAYER} ====="

  if [[ "$SKIP_EXTRACT" != "1" ]]; then
    python -m hypergraph.attention.cct extract \
      --input "data/hf_datasets/ProcessBench/${dataset}.json" \
      --model "$MODEL" \
      --output "$traces" \
      --layer "$LAYER" \
      --device cuda \
      --dtype bfloat16 \
      --skip-invalid
  fi

  python -m hypergraph.attention.cct inspect --traces "$traces"
  python -m hypergraph.attention.cct benchmark \
    --traces "$traces" \
    --output "$results" \
    --epochs 100 \
    --batch-size 16 \
    --device cuda
done
