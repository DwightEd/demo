#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEMO_ROOT="${DEMO_ROOT:-$(dirname "${PROJECT_ROOT}")}"
DATA_ROOT="${DATA_ROOT:-${DEMO_ROOT}/data/exact/processbench_observer_llama31_full}"
MODEL_DIR="${MODEL_DIR:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-auto}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${MODEL_DIR}}"
TOKENIZER_NAME="${TOKENIZER_NAME:-${MODEL_NAME}}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-${MODEL_REVISION}}"
GPU_ID="${GPU_ID:-0}"
COMPONENT_LAYERS="${COMPONENT_LAYERS:-8,12,16,20,24,28}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/hidden_state_geometry}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${PROJECT_ROOT}"

echo "project_root=${PROJECT_ROOT}"
echo "data_root=${DATA_ROOT}"
echo "model_dir=${MODEL_DIR}"
echo "tokenizer_dir=${TOKENIZER_DIR}"
echo "gpu_id=${GPU_ID}"
echo "component_layers=${COMPONENT_LAYERS}"
echo "output_root=${OUTPUT_ROOT}"
for domain in gsm8k math olympiadbench omnimath; do
  manifest="${DATA_ROOT}/${domain}/selected/trace.raw_residual_stream.npz"
  aligned_trace="${DATA_ROOT}/${domain}/selected/trace.npz"
  component_dir="${DATA_ROOT}/${domain}/selected/component_step_v1"
  echo "${domain}: residual_manifest=${manifest}"
  echo "${domain}: aligned_trace=${aligned_trace}"
  echo "${domain}: component_dir=${component_dir}"
  if [[ ! -f "${manifest}" ]]; then
    echo "missing raw residual manifest: ${manifest}" >&2
    echo "trace.npz alone is insufficient; extract response-token hidden-state shards first." >&2
    exit 3
  fi
  if [[ ! -f "${aligned_trace}" ]]; then
    echo "missing aligned trace with full token IDs: ${aligned_trace}" >&2
    exit 3
  fi
done

"${PYTHON_BIN}" -c 'import sys, sklearn; print(f"python={sys.executable}"); print(f"sklearn={sklearn.__version__}"); print(f"sklearn_path={sklearn.__file__}")'

component_common=(
  --data-root "${DATA_ROOT}"
  --domains gsm8k,math,olympiadbench,omnimath
  --response-generator llama3.1-8b
  --observer-model llama3.1-8b
  --acquisition-mode observer_teacher_forcing_replay
  --output-features token_entropy,token_nll
  --seed 17
  --model-dir "${MODEL_DIR}"
  --tokenizer-dir "${TOKENIZER_DIR}"
  --model-name "${MODEL_NAME}"
  --model-revision "${MODEL_REVISION}"
  --tokenizer-name "${TOKENIZER_NAME}"
  --tokenizer-revision "${TOKENIZER_REVISION}"
  --device cuda
  --dtype bfloat16
  --component-layers "${COMPONENT_LAYERS}"
)

require_component_runtime() {
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "missing model directory: ${MODEL_DIR}" >&2
    exit 4
  fi
  if [[ ! -d "${TOKENIZER_DIR}" ]]; then
    echo "missing tokenizer directory: ${TOKENIZER_DIR}" >&2
    exit 4
  fi
  "${PYTHON_BIN}" -c 'import sys
try:
    import torch, transformers
except ModuleNotFoundError as exc:
    print(f"missing component dependency: {exc.name}", file=sys.stderr)
    raise SystemExit(5)
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"component_python={sys.executable}")'
}

run_component_pytest_if_available() {
  if "${PYTHON_BIN}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("pytest") else 1)'; then
    "${PYTHON_BIN}" -m pytest \
      tests/hidden_state_geometry/test_component_contract.py \
      tests/hidden_state_geometry/test_component_features.py \
      tests/hidden_state_geometry/test_component_extraction.py \
      tests/hidden_state_geometry/test_component_resolved_hazard.py \
      tests/hidden_state_geometry/test_cli.py \
      tests/test_remote_runner.py
  else
    echo "pytest is not installed in ${PYTHON_BIN}; skipping focused component tests"
  fi
}

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
  ridge-smoke)
    ridge_config='{"pca_dim":8,"time_basis":3,"layer_basis":3,"positions_per_chain":16,"l2_grid":[0.0001,0.001,0.01,0.1],"max_iter":2000}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${common[@]}" --tasks whole_chain,strict_prefix \
      --method full_tensor_ridge --method-config-json "${ridge_config}" \
      --max-records-per-domain 32 --bootstrap 200 \
      --output-dir "${OUTPUT_ROOT}/ridge_smoke_${RUN_TAG}"
    ;;
  ridge-full)
    ridge_config='{"pca_dim":16,"time_basis":3,"layer_basis":3,"positions_per_chain":32,"l2_grid":[0.00001,0.0001,0.001,0.01,0.1,1.0],"max_iter":2000}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${common[@]}" --tasks whole_chain,strict_prefix \
      --method full_tensor_ridge --method-config-json "${ridge_config}" \
      --max-records-per-domain 0 --bootstrap 2000 \
      --output-dir "${OUTPUT_ROOT}/ridge_full_${RUN_TAG}"
    ;;
  innovation-smoke)
    innovation_config='{"source_layer":14,"destination_layer":16,"rank":4,"normal_ridge_alpha":10.0,"covariance_shrinkage":0.1,"l2":0.1,"max_iter":2000}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${common[@]}" --tasks strict_prefix \
      --method innovation_hazard --method-config-json "${innovation_config}" \
      --max-records-per-domain 32 --bootstrap 200 \
      --output-dir "${OUTPUT_ROOT}/innovation_smoke_${RUN_TAG}"
    ;;
  innovation-full)
    innovation_config='{"source_layer":14,"destination_layer":16,"rank":8,"normal_ridge_alpha":10.0,"covariance_shrinkage":0.1,"l2":0.1,"max_iter":2000}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${common[@]}" --tasks strict_prefix \
      --method innovation_hazard --method-config-json "${innovation_config}" \
      --max-records-per-domain 0 --bootstrap 2000 \
      --output-dir "${OUTPUT_ROOT}/innovation_full_${RUN_TAG}"
    ;;
  component-smoke)
    run_component_pytest_if_available
    require_component_runtime
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.component_extract_cli \
      "${component_common[@]}" --max-records-per-domain 32
    component_config='{"attention_rank":2,"mlp_rank":2,"propagation_rank":2,"pre_context_rank":2,"normal_ridge_alpha":10.0,"l2":0.1,"max_iter":1000}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${common[@]}" --tasks post_step \
      --method component_resolved_hazard --method-config-json "${component_config}" \
      --max-records-per-domain 32 --bootstrap 200 \
      --output-dir "${OUTPUT_ROOT}/component_smoke_${RUN_TAG}"
    ;;
  component-full)
    require_component_runtime
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.component_extract_cli \
      "${component_common[@]}" --max-records-per-domain 0
    component_config='{"attention_rank":4,"mlp_rank":4,"propagation_rank":4,"pre_context_rank":4,"normal_ridge_alpha":10.0,"l2":0.1,"max_iter":2000}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${common[@]}" --tasks post_step \
      --method component_resolved_hazard --method-config-json "${component_config}" \
      --max-records-per-domain 0 --bootstrap 2000 \
      --output-dir "${OUTPUT_ROOT}/component_full_${RUN_TAG}"
    ;;
  *)
    echo "usage: $0 preflight|smoke|full|ridge-smoke|ridge-full|innovation-smoke|innovation-full|component-smoke|component-full" >&2
    exit 2
    ;;
esac
