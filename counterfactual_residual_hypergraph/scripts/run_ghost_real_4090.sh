#!/usr/bin/env bash
set -Eeuo pipefail

# Real GHOST-inspired ProcessBench evaluation for one RTX 4090:
#   select an auditable cohort -> one teacher-forced observer pass per trace ->
#   fit only train-normal geometry -> calibrate only validation-normal scores ->
#   evaluate frozen mid-layer fusion and the final-layer control on held-out test.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRWH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEMO_ROOT="${DEMO_ROOT:-$(cd -- "${CRWH_ROOT}/.." && pwd)}"

MODEL_PATH="${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
PROCESSBENCH_ROOT="${PROCESSBENCH_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/demo/data/hf_datasets/ProcessBench}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:-0}"
PROFILE="${PROFILE:-smoke}"
SUBSET="${SUBSET:-gsm8k}"
MAX_TOKENS="${MAX_TOKENS:-768}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
MIN_GPU_TOTAL_MIB="${MIN_GPU_TOTAL_MIB:-22000}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-22000}"
MIN_DISK_GIB="${MIN_DISK_GIB:-5}"
SELECTION_SEED="${SELECTION_SEED:-17}"
SPLIT_SEED="${SPLIT_SEED:-17}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.2}"
TEST_RATIO="${TEST_RATIO:-0.2}"
THRESHOLD_QUANTILE="${THRESHOLD_QUANTILE:-0.95}"
SHRINKAGE="${SHRINKAGE:-0.1}"
REGULARIZATION="${REGULARIZATION:-1e-8}"

case "${PROFILE}" in
  smoke)
    SELECTION_MODE="${SELECTION_MODE:-balanced_unique}"
    LIMIT="${LIMIT:-48}"
    BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-200}"
    ;;
  pilot)
    SELECTION_MODE="${SELECTION_MODE:-balanced_unique}"
    LIMIT="${LIMIT:-160}"
    BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
    ;;
  full)
    SELECTION_MODE="${SELECTION_MODE:-all_eligible}"
    LIMIT="${LIMIT:-0}"
    BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
    ;;
  *)
    echo "PROFILE must be smoke, pilot, or full; got ${PROFILE}" >&2
    exit 2
    ;;
esac

require_absolute_path() {
  local variable_name="$1"
  local value="${!variable_name}"
  [[ "${value}" == /* ]] || {
    echo "${variable_name} must be an absolute path, got: ${value}" >&2
    exit 2
  }
}

require_absolute_path "DEMO_ROOT"
require_absolute_path "MODEL_PATH"
require_absolute_path "PROCESSBENCH_ROOT"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CRWH_ROOT}/outputs/ghost_remote_4090}"
require_absolute_path "OUTPUT_ROOT"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd -- "${OUTPUT_ROOT}" && pwd -P)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}_${PROFILE}_${SUBSET}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

on_exit() {
  local status=$?
  trap - EXIT
  if ((status == 0)); then
    printf "success\n" > "${RUN_DIR}/_SUCCESS"
  else
    printf "exit_status=%s\n" "${status}" > "${RUN_DIR}/_FAILED"
  fi
  exit "${status}"
}
trap on_exit EXIT
exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

stage() {
  printf "\n===== %s =====\n" "$1"
}

stage "configuration"
cat <<EOF
DEMO_ROOT=${DEMO_ROOT}
CRWH_ROOT=${CRWH_ROOT}
MODEL_PATH=${MODEL_PATH}
PROCESSBENCH_ROOT=${PROCESSBENCH_ROOT}
PROFILE=${PROFILE}
SUBSET=${SUBSET}
SELECTION_MODE=${SELECTION_MODE}
LIMIT=${LIMIT}
MAX_TOKENS=${MAX_TOKENS}
GPU=${GPU}
RUN_DIR=${RUN_DIR}
EOF

stage "host_preflight"
command -v -- "${PYTHON_BIN}" >/dev/null || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
}
PYTHON_CANDIDATE="$(command -v -- "${PYTHON_BIN}")"
PYTHON_DIRECTORY="$(cd -- "$(dirname -- "${PYTHON_CANDIDATE}")" && pwd -P)"
PYTHON_BIN="${PYTHON_DIRECTORY}/$(basename -- "${PYTHON_CANDIDATE}")"
command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is required" >&2
  exit 2
}
[[ -d "${MODEL_PATH}" ]] || {
  echo "Observer model directory does not exist: ${MODEL_PATH}" >&2
  exit 2
}
[[ -d "${PROCESSBENCH_ROOT}" ]] || {
  echo "ProcessBench directory does not exist: ${PROCESSBENCH_ROOT}" >&2
  exit 2
}
[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "Observer model config.json is missing" >&2
  exit 2
}

PB_INPUT=""
for candidate in \
  "${PROCESSBENCH_ROOT}/${SUBSET}.json" \
  "${PROCESSBENCH_ROOT}/${SUBSET}.jsonl"; do
  if [[ -f "${candidate}" ]]; then
    PB_INPUT="${candidate}"
    break
  fi
done
[[ -n "${PB_INPUT}" ]] || {
  echo "No ${SUBSET}.json or ${SUBSET}.jsonl under ${PROCESSBENCH_ROOT}" >&2
  find "${PROCESSBENCH_ROOT}" -maxdepth 1 -type f -printf '  %f\n' >&2
  exit 2
}

GPU_LINE="$(
  nvidia-smi \
    --query-gpu=index,memory.total,memory.free \
    --format=csv,noheader,nounits |
    awk -F',' -v requested="${GPU}" '
      {
        gsub(/[[:space:]]/, "", $1);
        gsub(/[[:space:]]/, "", $2);
        gsub(/[[:space:]]/, "", $3);
        if ($1 == requested) {
          print $1 " " $2 " " $3
        }
      }
    '
)"
[[ -n "${GPU_LINE}" ]] || {
  echo "GPU index ${GPU} was not reported by nvidia-smi" >&2
  exit 2
}
read -r GPU_INDEX GPU_TOTAL_MIB GPU_FREE_MIB <<< "${GPU_LINE}"
if ((GPU_TOTAL_MIB < MIN_GPU_TOTAL_MIB)); then
  echo "GPU ${GPU_INDEX} has only ${GPU_TOTAL_MIB} MiB total" >&2
  exit 2
fi
if ((GPU_FREE_MIB < MIN_GPU_FREE_MIB)); then
  echo "GPU ${GPU_INDEX} has only ${GPU_FREE_MIB} MiB free" >&2
  exit 2
fi
nvidia-smi > "${RUN_DIR}/gpu-before.txt"

DISK_AVAILABLE_KIB="$(df -Pk "${RUN_DIR}" | awk 'END {print $4}')"
if [[ ! "${DISK_AVAILABLE_KIB}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine free disk space" >&2
  exit 2
fi
if ((DISK_AVAILABLE_KIB < MIN_DISK_GIB * 1024 * 1024)); then
  echo "Less than ${MIN_DISK_GIB} GiB is free under ${OUTPUT_ROOT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="${CRWH_ROOT}/src:${DEMO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

stage "dependencies"
if [[ "${INSTALL_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r "${CRWH_ROOT}/requirements-remote.txt"
elif [[ "${INSTALL_DEPS}" != "0" ]]; then
  echo "INSTALL_DEPS must be 0 or 1" >&2
  exit 2
fi
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" - <<'PY'
import importlib
import json

required = ("numpy", "torch", "transformers", "accelerate", "pytest")
versions = {}
for name in required:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")
print(json.dumps(versions, indent=2, sort_keys=True))
PY
"${PYTHON_BIN}" -m pip freeze > "${RUN_DIR}/pip-freeze.txt"

stage "cuda_and_model_contract"
MODEL_PATH="${MODEL_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("torch.cuda.is_bf16_supported() is false")
properties = torch.cuda.get_device_properties(0)
if properties.total_memory < 22_000 * 1024 * 1024:
    raise SystemExit(f"visible GPU has only {properties.total_memory} bytes")
model_path = Path(os.environ["MODEL_PATH"])
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
if config.model_type != "llama":
    raise SystemExit(f"expected model_type=llama, got {config.model_type}")
if int(config.num_hidden_layers) != 32:
    raise SystemExit(
        f"expected 32 hidden layers for the registered probes, "
        f"got {config.num_hidden_layers}"
    )
contract = {
    "model_path": str(model_path.resolve()),
    "model_type": config.model_type,
    "num_hidden_layers": int(config.num_hidden_layers),
    "hidden_size": int(config.hidden_size),
    "visible_gpu": properties.name,
    "visible_gpu_total_bytes": properties.total_memory,
    "dtype": "bfloat16",
}
print(json.dumps(contract, indent=2))
PY

stage "source_witness"
{
  sha256sum \
    "${CRWH_ROOT}/src/crwh/ghost.py" \
    "${CRWH_ROOT}/src/crwh/ghost_hf.py" \
    "${CRWH_ROOT}/src/crwh/ghost_eval.py" \
    "${CRWH_ROOT}/src/crwh/ghost_cli.py" \
    "${DEMO_ROOT}/hypergraph/attention/splitting.py" \
    "${DEMO_ROOT}/hypergraph/attention/cct/processbench.py"
} > "${RUN_DIR}/source-files.sha256"
sha256sum "${RUN_DIR}/source-files.sha256" |
  awk '{print $1}' > "${RUN_DIR}/source-tree.sha256"
git -c safe.directory="${DEMO_ROOT}" -C "${DEMO_ROOT}" rev-parse HEAD \
  > "${RUN_DIR}/git-head.txt" 2>/dev/null || true
git -c safe.directory="${DEMO_ROOT}" -C "${DEMO_ROOT}" status --short \
  -- counterfactual_residual_hypergraph/src/crwh/ghost.py \
  counterfactual_residual_hypergraph/src/crwh/ghost_hf.py \
  counterfactual_residual_hypergraph/src/crwh/ghost_eval.py \
  counterfactual_residual_hypergraph/src/crwh/ghost_cli.py \
  counterfactual_residual_hypergraph/scripts/run_ghost_real_4090.sh \
  > "${RUN_DIR}/git-status.txt" 2>/dev/null || true

stage "ghost_tests"
"${PYTHON_BIN}" -m pytest -q \
  -p no:cacheprovider \
  --basetemp "${RUN_DIR}/pytest-tmp" \
  "${CRWH_ROOT}/tests/test_ghost_geometry.py" \
  "${CRWH_ROOT}/tests/test_ghost_real.py" \
  "${CRWH_ROOT}/tests/test_ghost_cli.py" \
  "${CRWH_ROOT}/tests/test_ghost_remote_script.py"

stage "select_processbench_cohort"
SELECTED_INPUT="${RUN_DIR}/${SUBSET}-selected.jsonl"
COHORT_MANIFEST="${RUN_DIR}/cohort-manifest.json"
"${PYTHON_BIN}" -m crwh.ghost_cli select \
  --input "${PB_INPUT}" \
  --model "${MODEL_PATH}" \
  --output "${SELECTED_INPUT}" \
  --manifest "${COHORT_MANIFEST}" \
  --mode "${SELECTION_MODE}" \
  --limit "${LIMIT}" \
  --max-tokens "${MAX_TOKENS}" \
  --seed "${SELECTION_SEED}"

stage "one_pass_mid_layer_extraction"
EMBEDDINGS="${RUN_DIR}/ghost-embeddings.npz"
EXTRACTION_MANIFEST="${RUN_DIR}/extraction-manifest.json"
"${PYTHON_BIN}" -m crwh.ghost_cli extract \
  --input "${SELECTED_INPUT}" \
  --model "${MODEL_PATH}" \
  --output "${EMBEDDINGS}" \
  --manifest "${EXTRACTION_MANIFEST}" \
  --mid-depths auto \
  --mid-start 0.25 \
  --mid-end 0.50 \
  --probe-count 6 \
  --max-tokens "${MAX_TOKENS}" \
  --dtype bfloat16 \
  --device cuda \
  --attention-implementation sdpa
nvidia-smi > "${RUN_DIR}/gpu-after-extraction.txt"

stage "normal_reference_fit_calibration_and_test"
EVALUATION_DIR="${RUN_DIR}/evaluation"
"${PYTHON_BIN}" -m crwh.ghost_cli evaluate \
  --embeddings "${EMBEDDINGS}" \
  --output-dir "${EVALUATION_DIR}" \
  --split-seed "${SPLIT_SEED}" \
  --validation-ratio "${VALIDATION_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --threshold-quantile "${THRESHOLD_QUANTILE}" \
  --shrinkage "${SHRINKAGE}" \
  --regularization "${REGULARIZATION}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}"

stage "artifact_audit"
for artifact in \
  "${COHORT_MANIFEST}" \
  "${EXTRACTION_MANIFEST}" \
  "${EMBEDDINGS}" \
  "${EVALUATION_DIR}/split.json" \
  "${EVALUATION_DIR}/fit-audit.json" \
  "${EVALUATION_DIR}/anomaly-scores-test.csv" \
  "${EVALUATION_DIR}/scores-test.csv" \
  "${EVALUATION_DIR}/metrics.json" \
  "${EVALUATION_DIR}/reference-model-response_mean.npz" \
  "${EVALUATION_DIR}/reference-model-response_last.npz" \
  "${EVALUATION_DIR}/summary.json"; do
  [[ -s "${artifact}" ]] || {
    echo "Required artifact is missing or empty: ${artifact}" >&2
    exit 2
  }
done
cp "${EVALUATION_DIR}/summary.json" "${RUN_DIR}/summary.json"
nvidia-smi > "${RUN_DIR}/gpu-after.txt"

stage "result"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

summary = json.loads(
    Path("${RUN_DIR}/summary.json").read_text(encoding="utf-8")
)
print(json.dumps({
    "status": summary["status"],
    "result_kind": summary["result_kind"],
    "method": summary["method"],
    "fit_normal_traces": summary["fit_normal_traces"],
    "calibration_normal_traces": summary["calibration_normal_traces"],
    "test_traces": summary["test_traces"],
    "mid_fused": summary["primary_test_mid_fused"],
    "final_layer": summary["primary_test_final_layer"],
    "run_dir": "${RUN_DIR}",
}, indent=2, ensure_ascii=False))
PY

echo
echo "GHOST-inspired real ProcessBench run completed."
echo "Run directory: ${RUN_DIR}"
echo "Main summary: ${RUN_DIR}/summary.json"
echo "All metrics: ${EVALUATION_DIR}/metrics.json"
echo "Predictions: ${EVALUATION_DIR}/scores-test.csv"
echo "This is a normal-reference one-class result with zero training epochs."
echo "It is an abstract-level reimplementation, not an exact-paper claim."
