#!/usr/bin/env bash
# Run strict single-layer attention extraction and response-level HyperCHARM.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash hypergraph/attention/scripts/run_single_layer_response_pipeline.sh \
    [--layer 14] [--dataset omnimath] [--seed 17] \
    [--generator-model Llama-3.1-8B-Instruct] [--extract-only]

The pipeline performs:
  1. strict one-layer, all-head prompt+response attention extraction;
  2. trace-cohort audit and graph preflight;
  3. one deterministic problem-disjoint train/validation/test split;
  4. validation-selected HyperCHARM training and one final held-out test.

High-level options:
  --input PATH       ProcessBench JSON/JSONL source; defaults to
                     data/hf_datasets/ProcessBench/<dataset>.json
  --model PATH       local observer model; defaults to
                     /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
  --layer ID         zero-based Transformer block id (default: 14)
  --dataset NAME     output tag (default: omnimath)
  --seed N           model initialization seed (default: 17)
  --seeds N          compatibility alias for --seed; exactly one value is required
  --generator-model TAG
                     exact ProcessBench generator tag selected before training;
                     a complete cache is reused, otherwise only matches are forwarded
  --limit N          pilot extraction limit; output gets a _pilotN suffix
  --mode MODE        model_parallel or data_parallel (default: model_parallel)
  --extract-only     stop after strict manifest validation and shard audit
  --help             show this message

Method environment variables and defaults:
  THRESHOLD=0.05                 SOURCE_SELECTION=threshold_fallback_topk
  TOP_K=16                       MIN_SOURCES=2
  SOURCE_SCOPE=all_past
  TOPOLOGY_HEADS=0               node features still retain all extracted heads
  PROPAGATION_MODE=symmetric     INCIDENCE_WEIGHT_MODE=uniform
  EDGE_ATTR_MODE=faithful        NODE_FEATURE_MODE=attention_diagonal
  MESSAGE_OPERATOR=hypergraph    PREPROCESSING=per_graph_zscore
  POOLING=mean                   MODEL_LAYERS=2
  HIDDEN_DIM=128                 EPOCHS=50
  PATIENCE=5                     SPLIT_MODE=fixed_holdout
  SPLIT_SEED=17                  VAL_RATIO=0.1 TEST_RATIO=0.2
  LEARNING_RATE=3e-4             WEIGHT_DECAY=1e-3
  DROPOUT=0.25                   MONITOR=aupr

Runtime environment variables:
  PYTHON_BIN=python              GPU0=0 GPU1=1 TRAIN_GPUS=0,1
  QUERY_CHUNK_SIZE=0             STORAGE_DTYPE=float32
  TRACE_EQUIVALENCE_THRESHOLD=0.01  cache-validation setting, not graph tau
  MAX_SEQ_LEN=0                 0 disables the user cap; model context still applies
  MAX_ATTENTION_GIB=24          dense tensor allocation guard, not a token cap
  REPLAY_MODE=observer           PROMPT_STYLE=plain
  GENERATOR_MODEL=             empty selects the all-generator observer cohort
  TRACE_EXTRACTION_LIMIT=      override extraction-side limit for explicit caches
  REUSE_TRACES=1                 REUSE_RUNS=1 OVERWRITE_RUNS=0
  TRACE_ROOT=/custom/path        RUN_ROOT=/custom/path

Important semantics:
  - every prompt and response token is a node;
  - only response tokens are hyperedge receivers;
  - SOURCE_SCOPE=all_past allows prompt and earlier response tokens as sources;
  - response token logits are mean-pooled and trained with response_bce;
  - extracting only --layer ID also restricts attention-diagonal node features
    to that layer. Training-time --selected-layers alone is not strict single-layer.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off|'') return 1 ;;
    *) die "expected a boolean value, got: $1" ;;
  esac
}

guard_config() {
  local config_path="$1"
  shift
  "${PYTHON_BIN}" - "${config_path}" "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
request = {}
for item in sys.argv[2:]:
    if "=" not in item:
        raise SystemExit(f"invalid configuration entry: {item!r}")
    key, value = item.split("=", 1)
    request[key] = value

if path.is_file():
    current = json.loads(path.read_text(encoding="utf-8"))
    if current != request:
        keys = sorted(set(current) | set(request))
        differences = [
            f"  {key}: stored={current.get(key)!r}, requested={request.get(key)!r}"
            for key in keys
            if current.get(key) != request.get(key)
        ]
        raise SystemExit(
            f"configuration mismatch for {path}:\n" + "\n".join(differences)
        )
    print("configuration gate passed:", path)
else:
    existing = [] if not path.parent.exists() else list(path.parent.iterdir())
    unexpected = [item for item in existing if item.name != "preflight.json"]
    if unexpected:
        raise SystemExit(
            f"output directory has no configuration gate and contains unexpected "
            f"artifacts: {[item.name for item in unexpected[:10]]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(request, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)
    print("wrote configuration gate:", path)
PY
}

INPUT="${INPUT:-}"
MODEL="${MODEL:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
LAYER="${LAYER:-14}"
DATASET_TAG="${DATASET_TAG:-omnimath}"
SEEDS="${SEEDS:-17}"
LIMIT="${LIMIT:-}"
MODE="${MODE:-model_parallel}"
EXTRACT_ONLY="${EXTRACT_ONLY:-0}"
GENERATOR_MODEL="${GENERATOR_MODEL:-}"

while (($#)); do
  case "$1" in
    --input) INPUT="${2:?--input requires a path}"; shift 2 ;;
    --model) MODEL="${2:?--model requires a path}"; shift 2 ;;
    --layer) LAYER="${2:?--layer requires an integer}"; shift 2 ;;
    --dataset) DATASET_TAG="${2:?--dataset requires a name}"; shift 2 ;;
    --folds) die "--folds was removed: this entrypoint now performs one fixed held-out test" ;;
    --seed) SEEDS="${2:?--seed requires an integer}"; shift 2 ;;
    --seeds) SEEDS="${2:?--seeds requires a list}"; shift 2 ;;
    --generator-model) GENERATOR_MODEL="${2:?--generator-model requires a tag}"; shift 2 ;;
    --limit) LIMIT="${2:?--limit requires an integer}"; shift 2 ;;
    --mode) MODE="${2:?--mode requires a value}"; shift 2 ;;
    --extract-only) EXTRACT_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (run with --help)" ;;
  esac
done

INPUT="${INPUT:-${REPO_ROOT}/data/hf_datasets/ProcessBench/${DATASET_TAG}.json}"
[[ "${LAYER}" =~ ^[0-9]+$ ]] || die "--layer must be a non-negative integer"
[[ -z "${LIMIT}" || "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || die "--limit must be positive"
[[ "${DATASET_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--dataset contains unsafe characters"
[[ -z "${GENERATOR_MODEL}" || "${GENERATOR_MODEL}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "--generator-model contains unsafe characters"
[[ "${MODE}" == "data_parallel" || "${MODE}" == "model_parallel" ]] || \
  die "--mode must be data_parallel or model_parallel"

command -v realpath >/dev/null 2>&1 || die "realpath is required"
[[ -f "${INPUT}" ]] || die "input does not exist: ${INPUT}"
[[ -d "${MODEL}" ]] || die "model directory does not exist: ${MODEL}"
INPUT="$(realpath "${INPUT}")"
MODEL="$(realpath "${MODEL}")"
SOURCE_INPUT="${INPUT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-${TRAIN_GPU:-${GPU0},${GPU1}}}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-0}"
TRACE_EQUIVALENCE_THRESHOLD="${TRACE_EQUIVALENCE_THRESHOLD:-0.01}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
DTYPE="${DTYPE:-auto}"
ARCHIVE_COMPRESSION="${ARCHIVE_COMPRESSION:-none}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-0}"
MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB:-24}"
REPLAY_MODE="${REPLAY_MODE:-observer}"
PROMPT_STYLE="${PROMPT_STYLE:-plain}"
[[ "${REPLAY_MODE}" == "observer" ]] || \
  die "this audited pipeline supports REPLAY_MODE=observer only"

THRESHOLD="${THRESHOLD:-0.05}"
SOURCE_SELECTION="${SOURCE_SELECTION:-threshold_fallback_topk}"
TOP_K="${TOP_K:-16}"
SOURCE_SCOPE="${SOURCE_SCOPE:-all_past}"
MIN_SOURCES="${MIN_SOURCES:-2}"
TOPOLOGY_HEADS="${TOPOLOGY_HEADS:-0}"
PROPAGATION_MODE="${PROPAGATION_MODE:-symmetric}"
INCIDENCE_WEIGHT_MODE="${INCIDENCE_WEIGHT_MODE:-uniform}"
EDGE_ATTR_MODE="${EDGE_ATTR_MODE:-faithful}"
NODE_FEATURE_MODE="${NODE_FEATURE_MODE:-attention_diagonal}"
MESSAGE_OPERATOR="${MESSAGE_OPERATOR:-hypergraph}"
PREPROCESSING="${PREPROCESSING:-per_graph_zscore}"
POOLING="${POOLING:-mean}"
MODEL_LAYERS="${MODEL_LAYERS:-2}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
SPLIT_MODE="${SPLIT_MODE:-fixed_holdout}"
SPLIT_SEED="${SPLIT_SEED:-17}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.2}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
DROPOUT="${DROPOUT:-0.25}"
MONITOR="${MONITOR:-aupr}"
REUSE_TRACES="${REUSE_TRACES:-1}"
REUSE_RUNS="${REUSE_RUNS:-1}"
OVERWRITE_RUNS="${OVERWRITE_RUNS:-0}"
ALLOW_RESPLIT_OFFICIAL_DATA="${ALLOW_RESPLIT_OFFICIAL_DATA:-0}"

read -r -a TRAIN_GPU_VALUES <<< "${TRAIN_GPUS//,/ }"
((${#TRAIN_GPU_VALUES[@]})) || die "TRAIN_GPUS resolved to an empty list"
for index in "${!TRAIN_GPU_VALUES[@]}"; do
  [[ -n "${TRAIN_GPU_VALUES[$index]}" ]] || die "TRAIN_GPUS contains an empty device"
  for ((previous = 0; previous < index; previous++)); do
    [[ "${TRAIN_GPU_VALUES[$index]}" != "${TRAIN_GPU_VALUES[$previous]}" ]] || \
      die "TRAIN_GPUS contains duplicate device: ${TRAIN_GPU_VALUES[$index]}"
  done
done
read -r -a SEED_VALUES <<< "${SEEDS//,/ }"
((${#SEED_VALUES[@]})) || die "--seeds resolved to an empty list"
(( ${#SEED_VALUES[@]} == 1 )) || \
  die "the fixed-test entrypoint requires exactly one model seed"
for index in "${!SEED_VALUES[@]}"; do
  seed="${SEED_VALUES[$index]}"
  [[ "${seed}" =~ ^(0|[1-9][0-9]*)$ ]] || \
    die "seed must be a canonical non-negative integer: ${seed}"
  for ((previous = 0; previous < index; previous++)); do
    [[ "${seed}" != "${SEED_VALUES[$previous]}" ]] || \
      die "--seeds contains a duplicate value: ${seed}"
  done
done

[[ "${SOURCE_SELECTION}" == "threshold_fallback_topk" ]] || \
  die "the original-aligned entrypoint requires SOURCE_SELECTION=threshold_fallback_topk"
[[ "${TOP_K}" =~ ^[1-9][0-9]*$ ]] || die "TOP_K must be a positive integer"
[[ "${SPLIT_MODE}" == "fixed_holdout" ]] || \
  die "this entrypoint requires SPLIT_MODE=fixed_holdout; use train.py directly for CV diagnostics"
[[ "${SPLIT_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] || \
  die "SPLIT_SEED must be a canonical non-negative integer"
[[ "${VAL_RATIO}" =~ ^0?\.[0-9]+$ ]] || die "VAL_RATIO must be a decimal in (0,1)"
[[ "${TEST_RATIO}" =~ ^0?\.[0-9]+$ ]] || die "TEST_RATIO must be a decimal in (0,1)"
[[ "${TOPOLOGY_HEADS}" == "all" || "${TOPOLOGY_HEADS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
  die "TOPOLOGY_HEADS must be all or a comma-separated list of non-negative head ids"
[[ "${NODE_FEATURE_MODE}" == "attention_diagonal" ]] || \
  die "this attention-only entrypoint requires NODE_FEATURE_MODE=attention_diagonal"
[[ "${QUERY_CHUNK_SIZE}" =~ ^[0-9]+$ ]] || \
  die "QUERY_CHUNK_SIZE must be a non-negative integer"
[[ "${MAX_SEQ_LEN}" =~ ^[0-9]+$ ]] || \
  die "MAX_SEQ_LEN must be a non-negative integer (0 disables the user cap)"
[[ "${QUERY_CHUNK_SIZE}" == "0" ]] || \
  die "the strict pipeline requires QUERY_CHUNK_SIZE=0; cached chunks changed the real-model threshold topology"

cd "${REPO_ROOT}"
if [[ -n "${GENERATOR_MODEL}" ]]; then
  "${PYTHON_BIN}" - "${GENERATOR_MODEL}" "${MODEL}" <<'PY'
import sys

from hypergraph.attention.pipeline_guard import dataset_generator_matches_observer

if not dataset_generator_matches_observer(sys.argv[1], sys.argv[2]):
    raise SystemExit(
        "--generator-model does not identify the observer checkpoint; "
        "use the all-generator observer run or invoke train.py manually for a source-specific analysis"
    )
PY
fi
LIMIT_SUFFIX=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_SUFFIX="_pilot${LIMIT}"
fi
COHORT_SUFFIX="_observer_all"
if [[ -n "${GENERATOR_MODEL}" ]]; then
  GENERATOR_SLUG="${GENERATOR_MODEL//\//-}"
  COHORT_SUFFIX="_matched_${GENERATOR_SLUG}"
fi
FULL_DATASET_TRACE_ROOT="${REPO_ROOT}/outputs/attention_traces/${DATASET_TAG}_llama31_layer${LAYER}"
DEFAULT_TRACE_ROOT="${FULL_DATASET_TRACE_ROOT}${LIMIT_SUFFIX}"
TRACE_EXTRACTION_LIMIT="${TRACE_EXTRACTION_LIMIT-${LIMIT}}"
[[ -z "${TRACE_EXTRACTION_LIMIT}" || "${TRACE_EXTRACTION_LIMIT}" =~ ^[1-9][0-9]*$ ]] || \
  die "TRACE_EXTRACTION_LIMIT must be empty or a positive integer"
if [[ -z "${TRACE_ROOT:-}" && "${TRACE_EXTRACTION_LIMIT}" != "${LIMIT}" ]]; then
  die "TRACE_EXTRACTION_LIMIT may differ from --limit only with an explicit TRACE_ROOT"
fi
TRACE_INPUT_MODE="full_dataset_cache"
COHORT_REPORT_PATH=""
if [[ -n "${TRACE_ROOT:-}" ]]; then
  if [[ "${MODE}" == "data_parallel" && -n "${GENERATOR_MODEL}" && -n "${LIMIT}" ]]; then
    die "explicit data_parallel traces cannot prove global generator-before-limit order; use model_parallel or omit --limit"
  fi
  TRACE_INPUT_MODE="explicit_trace_root"
elif [[ -n "${GENERATOR_MODEL}" ]]; then
  if [[ -f "${FULL_DATASET_TRACE_ROOT}/shard_audit.json" ]] \
      && [[ "${MODE}" != "data_parallel" || -z "${LIMIT}" ]]; then
    TRACE_ROOT="${FULL_DATASET_TRACE_ROOT}"
    TRACE_EXTRACTION_LIMIT=""
  else
    COHORT_ROOT="${COHORT_ROOT:-${REPO_ROOT}/outputs/attention_cohorts}"
    COHORT_INPUT="${COHORT_ROOT}/${DATASET_TAG}_matched_${GENERATOR_SLUG}.json"
    COHORT_REPORT_PATH="${COHORT_INPUT}.report.json"
    "${PYTHON_BIN}" -m hypergraph.attention.cohort \
      --input "${SOURCE_INPUT}" \
      --output "${COHORT_INPUT}" \
      --report "${COHORT_REPORT_PATH}" \
      --generator-model "${GENERATOR_MODEL}" >/dev/null
    INPUT="$(realpath "${COHORT_INPUT}")"
    TRACE_ROOT="${REPO_ROOT}/outputs/attention_traces/${DATASET_TAG}_llama31_layer${LAYER}${LIMIT_SUFFIX}${COHORT_SUFFIX}"
    TRACE_INPUT_MODE="materialized_matched_generator"
  fi
else
  TRACE_ROOT="${DEFAULT_TRACE_ROOT}"
fi
PROTOCOL_SUFFIX="_fixed_original"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/attention_hypergraph/${DATASET_TAG}_response_layer${LAYER}${LIMIT_SUFFIX}${COHORT_SUFFIX}${PROTOCOL_SUFFIX}}"

read -r SOURCE_INPUT_SHA256 INPUT_SHA256 EXTRACTION_CODE_SHA256 TRAINING_CODE_SHA256 < <("${PYTHON_BIN}" - "${SOURCE_INPUT}" "${INPUT}" <<'PY'
import hashlib
import sys
from pathlib import Path

source_input_path = Path(sys.argv[1])
input_path = Path(sys.argv[2])

def digest(paths):
    value = hashlib.sha256()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"required pipeline source is missing: {path}")
        value.update(str(path).encode("utf-8"))
        value.update(path.read_bytes())
    return value.hexdigest()

input_digest = hashlib.sha256()
with input_path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        input_digest.update(block)
extraction_paths = [
    Path("hypergraph/attention/extract.py"),
    Path("hypergraph/attention/trace_contract.py"),
    Path("utils/step_boundaries.py"),
    Path("hypergraph/attention/scripts/extract_dual_gpu.sh"),
]
training_paths = [
    Path("hypergraph/attention/aggregate_fixed.py"),
    Path("hypergraph/attention/trace_contract.py"),
    Path("hypergraph/attention/data.py"),
    Path("hypergraph/attention/schema.py"),
    Path("hypergraph/attention/construction.py"),
    Path("hypergraph/attention/model.py"),
    Path("hypergraph/attention/objectives.py"),
    Path("hypergraph/attention/shards.py"),
    Path("hypergraph/attention/train.py"),
    Path("hypergraph/attention/pipeline_guard.py"),
    Path("hypergraph/attention/scripts/run_single_layer_response_pipeline.sh"),
]
source_digest = input_digest if source_input_path == input_path else hashlib.sha256()
if source_input_path != input_path:
    with source_input_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            source_digest.update(block)
print(
    source_digest.hexdigest(),
    input_digest.hexdigest(),
    digest(extraction_paths),
    digest(training_paths),
)
PY
)
COHORT_REPORT_SHA256=""
if [[ -n "${COHORT_REPORT_PATH}" ]]; then
  COHORT_REPORT_SHA256="$("${PYTHON_BIN}" - "${COHORT_REPORT_PATH}" <<'PY'
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"
fi
"${PYTHON_BIN}" - <<'PY'
import torch
import transformers
print("runtime:", "torch", torch.__version__, "transformers", transformers.__version__)
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable to PyTorch; install a wheel compatible with the host driver"
    )
if torch.cuda.device_count() < 2:
    raise SystemExit("the dual-GPU pipeline requires at least two visible CUDA devices")
PY

printf '\n===== Single-layer response pipeline =====\n'
printf 'repo:                    %s\n' "${REPO_ROOT}"
printf 'source input:            %s\n' "${SOURCE_INPUT}"
printf 'trace input:             %s (%s)\n' "${INPUT}" "${TRACE_INPUT_MODE}"
printf 'model:                   %s\n' "${MODEL}"
printf 'dataset/layer:           %s / %s (zero-based block id)\n' "${DATASET_TAG}" "${LAYER}"
printf 'trace root:              %s\n' "${TRACE_ROOT}"
printf 'run root:                %s\n' "${RUN_ROOT}"
printf 'generator cohort:        %s\n' "${GENERATOR_MODEL:-all generators}"
printf 'source/receiver:         %s / response-only receivers\n' "${SOURCE_SCOPE}"
printf 'objective/pooling:       response_bce / %s\n' "${POOLING}"
printf 'topology:                %s, threshold=%s, fallback_top_k=%s, min_sources=%s, heads=%s\n' \
  "${SOURCE_SELECTION}" "${THRESHOLD}" "${TOP_K}" "${MIN_SOURCES}" "${TOPOLOGY_HEADS}"
printf 'propagation/incidence:   %s / %s\n' "${PROPAGATION_MODE}" "${INCIDENCE_WEIGHT_MODE}"
printf 'node/operator:           %s / %s\n' "${NODE_FEATURE_MODE}" "${MESSAGE_OPERATOR}"
printf 'split:                   %s seed=%s train/val/test=%.3f/%.3f/%.3f\n' \
  "${SPLIT_MODE}" "${SPLIT_SEED}" \
  "$("${PYTHON_BIN}" -c "print(1-float('${VAL_RATIO}')-float('${TEST_RATIO}'))")" \
  "${VAL_RATIO}" "${TEST_RATIO}"
printf 'model seed:              %s\n\n' "${SEEDS}"
if [[ "${PROPAGATION_MODE}" == "symmetric" || "${PREPROCESSING}" == "per_graph_zscore" ]]; then
  printf '%s\n\n' \
    'scope:                   offline full-response (explicitly recorded by train.py)'
fi

TRACE_CONFIG_PATH="${TRACE_ROOT}/pipeline_request.json"
TRACE_SOURCE_ARGS=()
if [[ -n "${COHORT_REPORT_PATH}" ]]; then
  TRACE_SOURCE_ARGS+=(
    "source_input=${SOURCE_INPUT}"
    "source_input_sha256=${SOURCE_INPUT_SHA256}"
    "cohort_report=${COHORT_REPORT_PATH}"
    "cohort_report_sha256=${COHORT_REPORT_SHA256}"
    "generator_model=${GENERATOR_MODEL}"
  )
fi
read -r TRACE_GATE_MODE TRACE_REQUEST_SHA256 LEGACY_METHOD_CODE_SHA256 < <(
  "${PYTHON_BIN}" -m hypergraph.attention.pipeline_guard \
  --path "${TRACE_CONFIG_PATH}" \
  --extraction-code-sha256 "${EXTRACTION_CODE_SHA256}" \
  --shell \
  "input=${INPUT}" \
  "input_sha256=${INPUT_SHA256}" \
  "model=${MODEL}" \
  "layer=${LAYER}" \
  "limit=${TRACE_EXTRACTION_LIMIT}" \
  "mode=${MODE}" \
  "query_chunk_size=${QUERY_CHUNK_SIZE}" \
  "storage_dtype=${STORAGE_DTYPE}" \
  "dtype=${DTYPE}" \
  "archive_compression=${ARCHIVE_COMPRESSION}" \
  "max_seq_len=${MAX_SEQ_LEN}" \
  "replay_mode=${REPLAY_MODE}" \
  "prompt_style=${PROMPT_STYLE}" \
  "chunk_equivalence_threshold=${TRACE_EQUIVALENCE_THRESHOLD}" \
  "${TRACE_SOURCE_ARGS[@]}"
)
printf 'trace request gate:       %s (%s)\n' \
  "${TRACE_GATE_MODE}" "${TRACE_REQUEST_SHA256}"

if [[ -f "${TRACE_ROOT}/shard_audit.json" ]] && is_true "${REUSE_TRACES}"; then
  printf 'Reusing audited traces: %s\n' "${TRACE_ROOT}"
elif find "${TRACE_ROOT}" -mindepth 1 -maxdepth 1 \
    ! -name "$(basename "${TRACE_CONFIG_PATH}")" -print -quit | grep -q .; then
  die "trace directory exists without a reusable shard audit: ${TRACE_ROOT}"
else
  INPUT="${INPUT}" \
  MODEL="${MODEL}" \
  OUTPUT_ROOT="${TRACE_ROOT}" \
  MODE="${MODE}" \
  QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE}" \
  STORAGE_DTYPE="${STORAGE_DTYPE}" \
  DTYPE="${DTYPE}" \
  ARCHIVE_COMPRESSION="${ARCHIVE_COMPRESSION}" \
  MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
  MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB}" \
  REPLAY_MODE="${REPLAY_MODE}" \
  PROMPT_STYLE="${PROMPT_STYLE}" \
  GPU0="${GPU0}" GPU1="${GPU1}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LIMIT="${TRACE_EXTRACTION_LIMIT}" \
    bash "${SCRIPT_DIR}/extract_dual_gpu.sh" \
      --attention_layers "${LAYER}" \
      --attention_heads all \
      --chunk-equivalence-threshold "${TRACE_EQUIVALENCE_THRESHOLD}"
fi

if [[ "${MODE}" == "data_parallel" ]]; then
  TRACE_INPUTS=("${TRACE_ROOT}/shard0" "${TRACE_ROOT}/shard1")
else
  TRACE_INPUTS=("${TRACE_ROOT}/balanced")
fi
for trace_input in "${TRACE_INPUTS[@]}"; do
  [[ -d "${trace_input}" ]] || die "missing extracted trace directory: ${trace_input}"
done
APPLY_TRAIN_SELECTION_LIMIT=1
if [[ "${MODE}" == "data_parallel" && -n "${LIMIT}" ]]; then
  [[ "${TRACE_EXTRACTION_LIMIT}" == "${LIMIT}" ]] || \
    die "data_parallel --limit must be enforced during extraction; set TRACE_EXTRACTION_LIMIT equal to --limit"
  # Extraction applied LIMIT globally before the rows were sharded. Reapplying
  # it while traversing shard0 then shard1 would make the cohort storage-order dependent.
  APPLY_TRAIN_SELECTION_LIMIT=0
fi

printf 'Re-auditing extraction manifests before cache use...\n'
"${PYTHON_BIN}" -m hypergraph.attention.shards "${TRACE_INPUTS[@]}" >/dev/null
printf 'fresh shard/manifest audit passed\n'

"${PYTHON_BIN}" - "${LAYER}" "${TRACE_INPUTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

expected = [int(sys.argv[1])]
for directory in map(Path, sys.argv[2:]):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing extraction manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = manifest.get("extraction_config", {}).get("attention_layers")
    if observed != expected:
        raise SystemExit(
            f"not a strict single-layer trace: {manifest_path} has {observed}, expected {expected}"
        )
print("strict single-layer manifest gate passed:", expected)
PY

if is_true "${EXTRACT_ONLY}"; then
  [[ -f "${TRACE_ROOT}/shard_audit.json" ]] || \
    die "extraction completed without shard audit: ${TRACE_ROOT}/shard_audit.json"
  printf '\nExtraction-only pipeline complete. Audited traces: %s\n' "${TRACE_ROOT}"
  exit 0
fi

GRAPH_ARGS=(
  --selected-layers "${LAYER}"
  --selected-heads "${TOPOLOGY_HEADS}"
  --threshold "${THRESHOLD}"
  --top-k "${TOP_K}"
  --source-selection "${SOURCE_SELECTION}"
  --source-scope "${SOURCE_SCOPE}"
  --min-sources "${MIN_SOURCES}"
  --include-center
  --propagation-mode "${PROPAGATION_MODE}"
  --incidence-weight-mode "${INCIDENCE_WEIGHT_MODE}"
  --edge-attr-mode "${EDGE_ATTR_MODE}"
  --node-feature-mode "${NODE_FEATURE_MODE}"
)

COHORT_ARGS=()
if [[ -n "${GENERATOR_MODEL}" ]]; then
  COHORT_ARGS+=(--generator-model "${GENERATOR_MODEL}")
fi
if [[ -n "${LIMIT}" ]] && is_true "${APPLY_TRAIN_SELECTION_LIMIT}"; then
  COHORT_ARGS+=(--limit "${LIMIT}")
fi
AUDIT_ARGS=()
if [[ "${REPLAY_MODE}" == "observer" ]]; then
  AUDIT_ARGS+=(--allow-observer-traces)
fi

RUN_PARENT="$(dirname "${RUN_ROOT}")"
mkdir -p "${RUN_PARENT}"
PREFLIGHT_CANDIDATE="$(mktemp "${RUN_PARENT}/.attention-preflight.XXXXXX")"
cleanup_preflight_candidate() {
  if [[ -n "${PREFLIGHT_CANDIDATE:-}" && -f "${PREFLIGHT_CANDIDATE}" ]]; then
    rm -f -- "${PREFLIGHT_CANDIDATE}"
  fi
}
trap cleanup_preflight_candidate INT TERM EXIT

"${PYTHON_BIN}" -m hypergraph.attention.train inspect \
  "${TRACE_INPUTS[@]}" \
  "${GRAPH_ARGS[@]}" \
  "${COHORT_ARGS[@]}" \
  "${AUDIT_ARGS[@]}" \
  --objective response_bce \
  --output "${PREFLIGHT_CANDIDATE}"

PREFLIGHT_SHA256="$("${PYTHON_BIN}" - "${PREFLIGHT_CANDIDATE}" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
case "${TRACE_GATE_MODE}" in
  initialized_v2|validated_v2) TRACE_REQUEST_KIND="v2" ;;
  validated_legacy_without_rewrite) TRACE_REQUEST_KIND="legacy" ;;
  *) die "unknown trace request gate mode: ${TRACE_GATE_MODE}" ;;
esac
RUN_CONFIG_PATH="${RUN_ROOT}/pipeline_request.json"
guard_config "${RUN_CONFIG_PATH}" \
  "trace_root=${TRACE_ROOT}" \
  "trace_input_mode=${TRACE_INPUT_MODE}" \
  "source_input=${SOURCE_INPUT}" \
  "source_input_sha256=${SOURCE_INPUT_SHA256}" \
  "cohort_report=${COHORT_REPORT_PATH}" \
  "cohort_report_sha256=${COHORT_REPORT_SHA256}" \
  "input_sha256=${INPUT_SHA256}" \
  "trace_request_kind=${TRACE_REQUEST_KIND}" \
  "trace_request_sha256=${TRACE_REQUEST_SHA256}" \
  "legacy_monolithic_method_code_sha256=${LEGACY_METHOD_CODE_SHA256}" \
  "current_extraction_validation_code_sha256=${EXTRACTION_CODE_SHA256}" \
  "preflight_sha256=${PREFLIGHT_SHA256}" \
  "training_code_sha256=${TRAINING_CODE_SHA256}" \
  "layer=${LAYER}" \
  "generator_model=${GENERATOR_MODEL}" \
  "cohort_suffix=${COHORT_SUFFIX}" \
  "selection_limit=${LIMIT}" \
  "trace_extraction_limit=${TRACE_EXTRACTION_LIMIT}" \
  "threshold=${THRESHOLD}" \
  "top_k=${TOP_K}" \
  "source_selection=${SOURCE_SELECTION}" \
  "source_scope=${SOURCE_SCOPE}" \
  "min_sources=${MIN_SOURCES}" \
  "topology_heads=${TOPOLOGY_HEADS}" \
  "propagation_mode=${PROPAGATION_MODE}" \
  "incidence_weight_mode=${INCIDENCE_WEIGHT_MODE}" \
  "edge_attr_mode=${EDGE_ATTR_MODE}" \
  "node_feature_mode=${NODE_FEATURE_MODE}" \
  "message_operator=${MESSAGE_OPERATOR}" \
  "preprocessing=${PREPROCESSING}" \
  "objective=response_bce" \
  "pooling=${POOLING}" \
  "model_layers=${MODEL_LAYERS}" \
  "hidden_dim=${HIDDEN_DIM}" \
  "dropout=${DROPOUT}" \
  "learning_rate=${LEARNING_RATE}" \
  "weight_decay=${WEIGHT_DECAY}" \
  "epochs=${EPOCHS}" \
  "patience=${PATIENCE}" \
  "monitor=${MONITOR}" \
  "split_mode=${SPLIT_MODE}" \
  "split_seed=${SPLIT_SEED}" \
  "val_ratio=${VAL_RATIO}" \
  "test_ratio=${TEST_RATIO}" \
  "allow_resplit_official_data=${ALLOW_RESPLIT_OFFICIAL_DATA}" \
  "seeds=${SEEDS}"

"${PYTHON_BIN}" - "${PREFLIGHT_CANDIDATE}" "${RUN_ROOT}/preflight.json" <<'PY'
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
os.replace(source, destination)
PY
PREFLIGHT_CANDIDATE=""
trap - INT TERM EXIT

mkdir -p "${RUN_ROOT}/logs"
seed="${SEED_VALUES[0]}"
RUN_DIR="${RUN_ROOT}/fixed_seed${seed}"
LOG_FILE="${RUN_ROOT}/logs/fixed_seed${seed}.log"
if [[ -f "${RUN_DIR}/results.json" ]] && is_true "${REUSE_RUNS}"; then
  printf 'Reusing completed fixed-holdout run: %s\n' "${RUN_DIR}"
else
  TRAIN_ARGS=(
    train
    "${TRACE_INPUTS[@]}"
    --objective response_bce
    "${GRAPH_ARGS[@]}"
    "${COHORT_ARGS[@]}"
    --message-operator "${MESSAGE_OPERATOR}"
    --preprocessing "${PREPROCESSING}"
    --pooling "${POOLING}"
    --split-mode fixed_holdout
    --split-seed "${SPLIT_SEED}"
    --val-ratio "${VAL_RATIO}"
    --test-ratio "${TEST_RATIO}"
    --model-layers "${MODEL_LAYERS}"
    --hidden-dim "${HIDDEN_DIM}"
    --dropout "${DROPOUT}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --monitor "${MONITOR}"
    --seed "${seed}"
    --device cuda:0
    --output "${RUN_DIR}"
  )
  TRAIN_ARGS+=("${AUDIT_ARGS[@]}")
  if [[ "${PROPAGATION_MODE}" == "symmetric" || "${PREPROCESSING}" == "per_graph_zscore" ]]; then
    TRAIN_ARGS+=(--allow-offline-full-context)
  fi
  if is_true "${ALLOW_RESPLIT_OFFICIAL_DATA}"; then
    TRAIN_ARGS+=(--allow-resplit-official-data)
  fi
  if is_true "${OVERWRITE_RUNS}"; then
    TRAIN_ARGS+=(--overwrite)
  elif [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    die "partial run directory exists; set OVERWRITE_RUNS=1 or remove it: ${RUN_DIR}"
  fi
  TRAIN_DEVICE="${TRAIN_GPU_VALUES[0]}"
  printf '\nTraining fixed split seed=%s model seed=%s on physical GPU %s -> %s\n' \
    "${SPLIT_SEED}" "${seed}" "${TRAIN_DEVICE}" "${RUN_DIR}"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${TRAIN_DEVICE}" "${PYTHON_BIN}" \
    -m hypergraph.attention.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
fi

"${PYTHON_BIN}" -m hypergraph.attention.aggregate_fixed \
  --root "${RUN_ROOT}" \
  --run "${RUN_DIR}"

printf '\nPipeline complete. Results: %s\n' "${RUN_ROOT}"
