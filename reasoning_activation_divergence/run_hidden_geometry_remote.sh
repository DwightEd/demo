#!/usr/bin/env bash
set -uo pipefail
set -o errtrace

stop_on_failure() {
  local status=$?
  trap - ERR EXIT
  if (( status != 0 )); then
    printf '\nRun failed with exit code %d. The traceback above is the original error.\n' "${status}" >&2
    if [[ -t 0 && -t 1 ]]; then
      read -r -p "Press Enter to return to the terminal..." || true
    fi
  fi
  exit "${status}"
}

trap stop_on_failure ERR EXIT

MODE="${1:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEMO_ROOT="${DEMO_ROOT:-$(dirname "${PROJECT_ROOT}")}"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/ProcessBench/reasoning_error_detection/llama31_8b}"
MODEL_DIR="${MODEL_DIR:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-auto}"
TOKENIZER_NAME="${TOKENIZER_NAME:-${MODEL_NAME}}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-${MODEL_REVISION}}"
GPU_ID="${GPU_ID:-0}"
CAUSAL_DOMAINS="${CAUSAL_DOMAINS:-gsm8k,math,olympiadbench,omnimath}"
CAUSAL_LAYERS="${CAUSAL_LAYERS:-8,12,16,20,24,28}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/hidden_state_geometry}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
STATE_SEEDS="${STATE_SEEDS:-17 29 41}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${PROJECT_ROOT}"

echo "project_root=${PROJECT_ROOT}"
echo "data_root=${DATA_ROOT}"
echo "model_dir=${MODEL_DIR}"
echo "gpu_id=${GPU_ID}"
echo "causal_domains=${CAUSAL_DOMAINS}"
echo "causal_layers=${CAUSAL_LAYERS}"
echo "output_root=${OUTPUT_ROOT}"
IFS=',' read -r -a causal_domains <<< "${CAUSAL_DOMAINS}"
if [[ "${MODE}" == causal-* ]]; then
  inspected_domains=("${causal_domains[@]}")
else
  inspected_domains=(gsm8k math olympiadbench omnimath)
fi
for domain in "${inspected_domains[@]}"; do
  manifest="${DATA_ROOT}/${domain}/selected/trace.raw_residual_stream.npz"
  aligned_trace="${DATA_ROOT}/${domain}/selected/trace.npz"
  geometry_trace="${DATA_ROOT}/${domain}/geometry/trace.npz"
  pair_file="${DATA_ROOT}/${domain}/selected/causal_first_error_v1/onset_pairs_v1.jsonl"
  echo "${domain}: residual_manifest=${manifest}"
  echo "${domain}: aligned_trace=${aligned_trace}"
  if [[ "${MODE}" == causal-monitor-* ]]; then
    echo "${domain}: geometry_trace=${geometry_trace}"
  fi
  if [[ "${MODE}" == causal-* ]]; then
    echo "${domain}: causal_pair_file=${pair_file}"
  fi
  if [[ "${MODE}" != causal-* && ! -f "${manifest}" ]]; then
    echo "missing raw residual manifest: ${manifest}" >&2
    echo "trace.npz alone is insufficient; extract response-token hidden-state shards first." >&2
    exit 3
  fi
  if [[ ! -f "${aligned_trace}" ]]; then
    echo "missing aligned trace with full token IDs: ${aligned_trace}" >&2
    exit 3
  fi
  if [[ "${MODE}" == causal-monitor-* && ! -f "${geometry_trace}" ]]; then
    echo "missing exact pre-step geometry trace: ${geometry_trace}" >&2
    exit 3
  fi
done

if [[ "${MODE}" == causal-* ]]; then
  "${PYTHON_BIN}" -c 'import sys, numpy; print(f"python={sys.executable}"); print(f"numpy={numpy.__version__}")'
else
  "${PYTHON_BIN}" -c 'import sys, sklearn; print(f"python={sys.executable}"); print(f"sklearn={sklearn.__version__}"); print(f"sklearn_path={sklearn.__file__}")'
fi

causal_common=(
  --data-root "${DATA_ROOT}"
  --domains "${CAUSAL_DOMAINS}"
  --model-dir "${MODEL_DIR}"
  --model-name "${MODEL_NAME}"
  --model-revision "${MODEL_REVISION}"
  --tokenizer-name "${TOKENIZER_NAME}"
  --tokenizer-revision "${TOKENIZER_REVISION}"
  --device cuda
  --dtype bfloat16
  --layers "${CAUSAL_LAYERS}"
)

require_causal_runtime() {
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "missing model directory: ${MODEL_DIR}" >&2
    exit 4
  fi
  "${PYTHON_BIN}" -c 'import sys
try:
    import torch, transformers
except ModuleNotFoundError as exc:
    print(f"missing causal dependency: {exc.name}", file=sys.stderr)
    raise SystemExit(5)
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"causal_python={sys.executable}")'
}

require_monitor_runtime() {
  "${PYTHON_BIN}" -c 'import sys
try:
    import torch, sklearn, tqdm
except ModuleNotFoundError as exc:
    print(f"missing monitor dependency: {exc.name}", file=sys.stderr)
    raise SystemExit(5)
print(f"torch={torch.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"monitor_python={sys.executable}")'
}

run_causal_pytest_if_available() {
  if "${PYTHON_BIN}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("pytest") else 1)'; then
    "${PYTHON_BIN}" -m pytest \
      tests/causal_first_error_attribution \
      tests/test_remote_runner.py
  else
    echo "pytest is not installed in ${PYTHON_BIN}; skipping focused causal tests"
  fi
}

run_predictive_state_pytest_if_available() {
  if "${PYTHON_BIN}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("pytest") else 1)'; then
    "${PYTHON_BIN}" -m pytest \
      tests/hidden_state_geometry/test_predictive_state_monitor.py \
      tests/hidden_state_geometry/test_data.py \
      tests/hidden_state_geometry/test_tasks.py \
      tests/test_remote_runner.py
  else
    echo "pytest is not installed in ${PYTHON_BIN}; skipping focused predictive-state tests"
  fi
}

run_token_markov_pytest_if_available() {
  if "${PYTHON_BIN}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("pytest") else 1)'; then
    "${PYTHON_BIN}" -m pytest \
      tests/hidden_state_geometry/test_token_predictive_state.py \
      tests/hidden_state_geometry/test_data.py \
      tests/hidden_state_geometry/test_tasks.py \
      tests/test_remote_runner.py
  else
    echo "pytest is not installed in ${PYTHON_BIN}; skipping focused token-Markov tests"
  fi
}

base_common=(
  --data-root "${DATA_ROOT}"
  --domains gsm8k,math,olympiadbench,omnimath
  --response-generator llama3.1-8b
  --observer-model llama3.1-8b
  --acquisition-mode observer_teacher_forcing_replay
  --output-features token_entropy,token_nll
)

common=(
  "${base_common[@]}"
  --seed 17
)

case "${MODE}" in
  causal-audit)
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main audit \
      --data-root "${DATA_ROOT}" \
      --domains "${CAUSAL_DOMAINS}" \
      --output "${OUTPUT_ROOT}/causal_audit_${RUN_TAG}.json"
    ;;
  causal-extract-smoke)
    run_causal_pytest_if_available
    require_causal_runtime
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main extract \
      "${causal_common[@]}" --max-cases-per-domain 4 --topk 20
    ;;
  causal-intervene-smoke)
    require_causal_runtime
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main intervene \
      "${causal_common[@]}" --max-cases-per-domain 2
    ;;
  causal-summarize-smoke)
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main summarize \
      --data-root "${DATA_ROOT}" \
      --domains "${CAUSAL_DOMAINS}" \
      --output "${OUTPUT_ROOT}/causal_summary_${RUN_TAG}.json"
    ;;
  causal-monitor-smoke)
    run_causal_pytest_if_available
    require_monitor_runtime
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main train-monitor \
      --data-root "${DATA_ROOT}" --domains "${CAUSAL_DOMAINS}" \
      --output-dir "${OUTPUT_ROOT}/causal_monitor_smoke_${RUN_TAG}" \
      --max-chains-per-domain 32 --width 32 \
      --epochs 3 --patience 2 --batch-size 16 --bootstrap 200 \
      --device cuda
    ;;
  causal-monitor-full)
    run_causal_pytest_if_available
    require_monitor_runtime
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main train-monitor \
      --data-root "${DATA_ROOT}" --domains "${CAUSAL_DOMAINS}" \
      --output-dir "${OUTPUT_ROOT}/causal_monitor_full_${RUN_TAG}" \
      --max-chains-per-domain 0 --width 64 \
      --epochs 20 --patience 4 --batch-size 32 --bootstrap 2000 \
      --device cuda
    ;;
  causal-full)
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main audit \
      --data-root "${DATA_ROOT}" \
      --domains "${CAUSAL_DOMAINS}" \
      --output "${OUTPUT_ROOT}/causal_audit_${RUN_TAG}.json"
    run_causal_pytest_if_available
    require_causal_runtime
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main extract \
      "${causal_common[@]}" --topk 20
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main intervene \
      "${causal_common[@]}"
    "${PYTHON_BIN}" -m functional_divergence.causal_first_error_attribution.main summarize \
      --data-root "${DATA_ROOT}" \
      --domains "${CAUSAL_DOMAINS}" \
      --output "${OUTPUT_ROOT}/causal_summary_${RUN_TAG}.json"
    ;;
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
  predictive-state-smoke)
    require_monitor_runtime
    run_predictive_state_pytest_if_available
    predictive_state_config='{"pca_dim":4,"positions_per_chain":8,"width":16,"epochs":3,"patience":2,"batch_size":32,"learning_rate":0.0003,"weight_decay":0.0001,"validation_fraction":0.2,"device":"cuda","show_progress":true}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${base_common[@]}" --seed 17 --tasks strict_prefix \
      --method predictive_state_monitor --method-config-json "${predictive_state_config}" \
      --max-records-per-domain 32 --bootstrap 200 \
      --output-dir "${OUTPUT_ROOT}/predictive_state_smoke_${RUN_TAG}"
    ;;
  predictive-state-full)
    require_monitor_runtime
    run_predictive_state_pytest_if_available
    predictive_state_config='{"pca_dim":8,"positions_per_chain":16,"width":32,"epochs":20,"patience":4,"batch_size":64,"learning_rate":0.0003,"weight_decay":0.0001,"validation_fraction":0.2,"device":"cuda","show_progress":true}'
    read -r -a state_seeds <<< "${STATE_SEEDS}"
    for state_seed in "${state_seeds[@]}"; do
      "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
        "${base_common[@]}" --seed "${state_seed}" --tasks strict_prefix \
        --method predictive_state_monitor --method-config-json "${predictive_state_config}" \
        --max-records-per-domain 0 --bootstrap 2000 \
        --output-dir "${OUTPUT_ROOT}/predictive_state_full_seed${state_seed}_${RUN_TAG}"
    done
    ;;
  predictive-token-smoke)
    require_monitor_runtime
    run_predictive_state_pytest_if_available
    predictive_token_config='{"pca_dim":4,"positions_per_chain":8,"sequence_unit":"token","sequence_encoder":"attention_pool","attention_heads":4,"attention_queries":4,"width":16,"epochs":3,"patience":2,"batch_size":32,"learning_rate":0.0003,"weight_decay":0.0001,"validation_fraction":0.2,"device":"cuda","show_progress":true}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${base_common[@]}" --seed 17 --tasks strict_prefix \
      --method predictive_state_monitor --method-config-json "${predictive_token_config}" \
      --max-records-per-domain 32 --bootstrap 200 \
      --output-dir "${OUTPUT_ROOT}/predictive_token_smoke_${RUN_TAG}"
    ;;
  predictive-token-full)
    require_monitor_runtime
    run_predictive_state_pytest_if_available
    predictive_token_config='{"pca_dim":8,"positions_per_chain":16,"sequence_unit":"token","sequence_encoder":"attention_pool","attention_heads":4,"attention_queries":4,"width":32,"epochs":20,"patience":4,"batch_size":64,"learning_rate":0.0003,"weight_decay":0.0001,"validation_fraction":0.2,"device":"cuda","show_progress":true}'
    read -r -a state_seeds <<< "${STATE_SEEDS}"
    for state_seed in "${state_seeds[@]}"; do
      "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
        "${base_common[@]}" --seed "${state_seed}" --tasks strict_prefix \
        --method predictive_state_monitor --method-config-json "${predictive_token_config}" \
        --max-records-per-domain 0 --bootstrap 2000 \
        --output-dir "${OUTPUT_ROOT}/predictive_token_full_seed${state_seed}_${RUN_TAG}"
    done
    ;;
  token-markov-smoke)
    run_token_markov_pytest_if_available
    token_markov_config='{"pca_dim":4,"positions_per_chain":8,"history_order":4,"transitions_per_chain":32,"recent_window":8,"dynamics_ridge_alpha":10.0,"hazard_l2":0.1,"hazard_max_iter":2000}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${base_common[@]}" --seed 17 --tasks strict_prefix \
      --method token_predictive_state --method-config-json "${token_markov_config}" \
      --max-records-per-domain 32 --bootstrap 200 \
      --output-dir "${OUTPUT_ROOT}/token_markov_smoke_${RUN_TAG}"
    ;;
  token-markov-full)
    run_token_markov_pytest_if_available
    token_markov_config='{"pca_dim":4,"positions_per_chain":16,"history_order":4,"transitions_per_chain":64,"recent_window":8,"dynamics_ridge_alpha":10.0,"hazard_l2":0.1,"hazard_max_iter":2000}'
    "${PYTHON_BIN}" -m functional_divergence.hidden_state_geometry.cli run \
      "${base_common[@]}" --seed 17 --tasks strict_prefix \
      --method token_predictive_state --method-config-json "${token_markov_config}" \
      --max-records-per-domain 0 --bootstrap 2000 \
      --output-dir "${OUTPUT_ROOT}/token_markov_full_${RUN_TAG}"
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
  *)
    echo "usage: $0 causal-audit|causal-extract-smoke|causal-intervene-smoke|causal-summarize-smoke|causal-monitor-smoke|causal-monitor-full|causal-full|preflight|smoke|full|ridge-smoke|ridge-full|predictive-state-smoke|predictive-state-full|predictive-token-smoke|predictive-token-full|token-markov-smoke|token-markov-full|innovation-smoke|innovation-full" >&2
    exit 2
    ;;
esac
