#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/share/home/tm902089733300000/a903202310/lys/research/demo/data/exact/processbench_observer_llama31}"
OUTPUT_ROOT="${2:-outputs/token_residual_dispersion_legacy_selected_pilot}"
MAX_TRACES="${MAX_TRACES:-0}"

for subset in gsm8k math olympiadbench omnimath; do
  manifest="${DATA_ROOT}/${subset}/selected/trace.npz"
  if [[ ! -f "${manifest}" ]]; then
    echo "missing manifest: ${manifest}" >&2
    exit 2
  fi
  extra_args=()
  if [[ "${MAX_TRACES}" -gt 0 ]]; then
    extra_args+=(--max-traces "${MAX_TRACES}")
  fi
  python -m token_residual_dispersion.cli \
    --input "${manifest}" \
    --output-dir "${OUTPUT_ROOT}/${subset}" \
    --windows 4,8,16,32 \
    --min-tokens 3 \
    --rank-stride 4 \
    --legacy-sparse-pilot \
    --progress-every 100 \
    "${extra_args[@]}"
done
