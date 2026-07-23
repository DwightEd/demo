#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEMO_ROOT="${DEMO_ROOT:-$(dirname "${PROJECT_ROOT}")}"
DATA_ROOT="${DATA_ROOT:-${DEMO_ROOT}/data/exact/processbench_observer_llama31_full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/hidden_state_geometry}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

echo "project_root=${PROJECT_ROOT}"
echo "data_root=${DATA_ROOT}"
for domain in gsm8k math olympiadbench omnimath; do
  manifest="${DATA_ROOT}/${domain}/selected/trace.raw_residual_stream.npz"
  aligned_trace="${DATA_ROOT}/${domain}/selected/trace.npz"
  echo "${domain}: residual_manifest=${manifest}"
  echo "${domain}: aligned_trace=${aligned_trace}"
  if [[ ! -f "${manifest}" ]]; then
    echo "missing raw residual manifest: ${manifest}" >&2
    echo "trace.npz alone is insufficient; extract response-token hidden-state shards first." >&2
    exit 3
  fi
done

"${PYTHON_BIN}" -c 'import sys, sklearn; print(f"python={sys.executable}"); print(f"sklearn={sklearn.__version__}"); print(f"sklearn_path={sklearn.__file__}")'

common=(
  --data-root "${DATA_ROOT}"
  --domains gsm8k,math,olympiadbench,omnimath
  --response-generator llama3.1-8b
  --observer-model llama3.1-8b
  --acquisition-mode observer_teacher_forcing_replay
  --output-features token_entropy,token_nll
  --seed 17
)

case "${MODE}" in
  preflight)
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli \
      preflight "${common[@]}" --max-records-per-domain 0
    ;;
  smoke)
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${common[@]}" --tasks whole_chain,strict_prefix \
      --method raw_functional_probe --max-records-per-domain 32 \
      --pca-dim 8 --positions-per-chain 16 --time-basis 3 --layer-basis 3 \
      --l2 1.0 --restarts 1 --max-iter 150 --null-repeats 2 --bootstrap 200 \
      --output-dir "${OUTPUT_ROOT}/smoke_${RUN_TAG}"
    ;;
  full)
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${common[@]}" --tasks whole_chain,strict_prefix \
      --method raw_functional_probe --max-records-per-domain 0 \
      --pca-dim 16 --positions-per-chain 32 --time-basis 3 --layer-basis 3 \
      --l2 1.0 --restarts 3 --max-iter 500 --null-repeats 3 --bootstrap 2000 \
      --output-dir "${OUTPUT_ROOT}/full_${RUN_TAG}"
    ;;
  *)
    echo "usage: $0 preflight|smoke|full" >&2
    exit 2
    ;;
esac
