#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-canonical-preflight}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXACT_MANIFEST_NAME="${EXACT_MANIFEST_NAME:-trace.raw_residual_stream.npz}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/reasoning_activation_divergence/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -c 'import os, sys; import sklearn; print(f"python={sys.executable}"); print("conda_env=" + os.environ.get("CONDA_DEFAULT_ENV", "<inactive>")); print(f"sklearn={sklearn.__version__}"); print(f"sklearn_path={sklearn.__file__}")'

run_raw() {
  "${PYTHON_BIN}" -m functional_divergence.raw_residual_experiment "$@"
}

case "${MODE}" in
  canonical-preflight)
    run_raw \
      --input "${REPO_ROOT}/data/features/full_gsm8k.npz" \
      --hidden-dir "${REPO_ROOT}/data/hidden/gsm8k" \
      --preflight
    ;;
  canonical-pilot)
    run_raw \
      --input "${REPO_ROOT}/data/features/full_gsm8k.npz" \
      --hidden-dir "${REPO_ROOT}/data/hidden/gsm8k" \
      --output-dir "${REPO_ROOT}/outputs/raw_layer_time/canonical_gsm8k_pilot20" \
      --offsets=-2,-1,0,1 --layers 10,14,18,22 \
      --max-pairs 20 --rank 8 --folds 2 --bootstrap 200 --seed 17
    ;;
  canonical-full)
    run_raw \
      --input "${REPO_ROOT}/data/features/full_gsm8k.npz" \
      --hidden-dir "${REPO_ROOT}/data/hidden/gsm8k" \
      --output-dir "${REPO_ROOT}/outputs/raw_layer_time/canonical_gsm8k_full" \
      --offsets=-2,-1,0,1 --layers 10,14,18,22 \
      --rank 16 --folds 5 --bootstrap 2000 --seed 17
    ;;
  exact-pilot|exact-full)
    data_root="${REPO_ROOT}/data/exact/processbench_observer_llama31_full"
    if [[ "${MODE}" == "exact-pilot" ]]; then
      output_root="${REPO_ROOT}/outputs/raw_layer_time/exact_pilot"
      extra=(--max-pairs 20 --rank 8 --folds 2 --bootstrap 200)
    else
      output_root="${REPO_ROOT}/outputs/raw_layer_time/exact_full"
      extra=(--rank 16 --folds 5 --bootstrap 2000)
    fi
    for subset in gsm8k math olympiadbench omnimath; do
      echo "[$(date '+%F %T')] dataset=${subset} mode=${MODE}"
      manifest="${data_root}/${subset}/selected/${EXACT_MANIFEST_NAME}"
      if [[ ! -f "${manifest}" ]]; then
        echo "missing verified raw-residual manifest: ${manifest}" >&2
        exit 2
      fi
      run_raw --input "${manifest}" --response-generator llama3.1-8b --preflight
      run_raw \
        --input "${manifest}" \
        --output-dir "${output_root}/${subset}" \
        --response-generator llama3.1-8b \
        --offsets=-2,-1,0,1 --layers all --seed 17 \
        "${extra[@]}"
    done
    ;;
  *)
    echo "usage: $0 canonical-preflight|canonical-pilot|canonical-full|exact-pilot|exact-full" >&2
    exit 2
    ;;
esac
