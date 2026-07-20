#!/usr/bin/env bash
# Run strict single-layer attention extraction and response-level HyperCHARM.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash hypergraph/attention/scripts/run_single_layer_response_pipeline.sh \
    [--layer 14] [--dataset omnimath] [--folds 5] [--seeds 17] [--extract-only]

The pipeline performs:
  1. strict one-layer, all-head prompt+response attention extraction;
  2. complementary-shard audit and graph preflight;
  3. response_bce HyperCHARM training over problem-disjoint folds;
  4. aggregation of held-out response metrics.

High-level options:
  --input PATH       ProcessBench JSON/JSONL source; defaults to
                     data/hf_datasets/ProcessBench/<dataset>.json
  --model PATH       local observer model; defaults to
                     /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
  --layer ID         zero-based Transformer block id (default: 14)
  --dataset NAME     output tag (default: omnimath)
  --folds N          group-CV folds (default: 5)
  --seeds LIST       comma/space-separated seeds (default: 17)
  --limit N          pilot extraction limit; output gets a _pilotN suffix
  --mode MODE        data_parallel or model_parallel (default: data_parallel)
  --extract-only     stop after strict manifest validation and shard audit
  --help             show this message

Method environment variables and defaults:
  THRESHOLD=0.01                 SOURCE_SELECTION=threshold
  SOURCE_SCOPE=all_past          MIN_SOURCES=1
  PROPAGATION_MODE=symmetric     INCIDENCE_WEIGHT_MODE=uniform
  EDGE_ATTR_MODE=faithful        NODE_FEATURE_MODE=attention_diagonal
  MESSAGE_OPERATOR=hypergraph    PREPROCESSING=per_graph_zscore
  POOLING=mean                   MODEL_LAYERS=2
  HIDDEN_DIM=128                 EPOCHS=50
  PATIENCE=5                     SPLIT_MODE=group_cv
  LEARNING_RATE=3e-4             WEIGHT_DECAY=1e-3
  DROPOUT=0.25                   MONITOR=aupr

Runtime environment variables:
  PYTHON_BIN=python              GPU0=0 GPU1=1 TRAIN_GPUS=0,1
  QUERY_CHUNK_SIZE=64            STORAGE_DTYPE=float32
  REPLAY_MODE=observer           PROMPT_STYLE=plain
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
    if path.parent.exists() and any(path.parent.iterdir()):
        raise SystemExit(
            f"non-empty output directory has no configuration gate: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(request, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("wrote configuration gate:", path)
PY
}

INPUT="${INPUT:-}"
MODEL="${MODEL:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
LAYER="${LAYER:-14}"
DATASET_TAG="${DATASET_TAG:-omnimath}"
FOLDS="${FOLDS:-5}"
SEEDS="${SEEDS:-17}"
LIMIT="${LIMIT:-}"
MODE="${MODE:-data_parallel}"
EXTRACT_ONLY="${EXTRACT_ONLY:-0}"

while (($#)); do
  case "$1" in
    --input) INPUT="${2:?--input requires a path}"; shift 2 ;;
    --model) MODEL="${2:?--model requires a path}"; shift 2 ;;
    --layer) LAYER="${2:?--layer requires an integer}"; shift 2 ;;
    --dataset) DATASET_TAG="${2:?--dataset requires a name}"; shift 2 ;;
    --folds) FOLDS="${2:?--folds requires an integer}"; shift 2 ;;
    --seeds) SEEDS="${2:?--seeds requires a list}"; shift 2 ;;
    --limit) LIMIT="${2:?--limit requires an integer}"; shift 2 ;;
    --mode) MODE="${2:?--mode requires a value}"; shift 2 ;;
    --extract-only) EXTRACT_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (run with --help)" ;;
  esac
done

INPUT="${INPUT:-${REPO_ROOT}/data/hf_datasets/ProcessBench/${DATASET_TAG}.json}"
[[ "${LAYER}" =~ ^[0-9]+$ ]] || die "--layer must be a non-negative integer"
[[ "${FOLDS}" =~ ^[0-9]+$ ]] && ((FOLDS >= 3)) || \
  die "--folds must be an integer >= 3"
[[ -z "${LIMIT}" || "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || die "--limit must be positive"
[[ "${DATASET_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--dataset contains unsafe characters"
[[ "${MODE}" == "data_parallel" || "${MODE}" == "model_parallel" ]] || \
  die "--mode must be data_parallel or model_parallel"

command -v realpath >/dev/null 2>&1 || die "realpath is required"
[[ -f "${INPUT}" ]] || die "input does not exist: ${INPUT}"
[[ -d "${MODEL}" ]] || die "model directory does not exist: ${MODEL}"
INPUT="$(realpath "${INPUT}")"
MODEL="$(realpath "${MODEL}")"

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-${TRAIN_GPU:-${GPU0},${GPU1}}}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-64}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
DTYPE="${DTYPE:-auto}"
ARCHIVE_COMPRESSION="${ARCHIVE_COMPRESSION:-none}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB:-24}"
REPLAY_MODE="${REPLAY_MODE:-observer}"
PROMPT_STYLE="${PROMPT_STYLE:-plain}"

THRESHOLD="${THRESHOLD:-0.01}"
SOURCE_SELECTION="${SOURCE_SELECTION:-threshold}"
SOURCE_SCOPE="${SOURCE_SCOPE:-all_past}"
MIN_SOURCES="${MIN_SOURCES:-1}"
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
SPLIT_MODE="${SPLIT_MODE:-group_cv}"
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

[[ "${SOURCE_SELECTION}" == "threshold" ]] || \
  die "this audited entrypoint supports SOURCE_SELECTION=threshold only"
[[ "${NODE_FEATURE_MODE}" == "attention_diagonal" ]] || \
  die "this attention-only entrypoint requires NODE_FEATURE_MODE=attention_diagonal"

LIMIT_SUFFIX=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_SUFFIX="_pilot${LIMIT}"
fi
TRACE_ROOT="${TRACE_ROOT:-${REPO_ROOT}/outputs/attention_traces/${DATASET_TAG}_llama31_layer${LAYER}${LIMIT_SUFFIX}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/attention_hypergraph/${DATASET_TAG}_response_layer${LAYER}${LIMIT_SUFFIX}}"

cd "${REPO_ROOT}"
read -r INPUT_SHA256 METHOD_CODE_SHA256 < <("${PYTHON_BIN}" - "${INPUT}" <<'PY'
import hashlib
import sys
from pathlib import Path

input_path = Path(sys.argv[1])

def digest(paths):
    value = hashlib.sha256()
    for path in paths:
        path = Path(path)
        value.update(str(path).encode("utf-8"))
        value.update(path.read_bytes())
    return value.hexdigest()

input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
method_paths = sorted(Path("hypergraph/attention").glob("*.py"))
method_paths.extend(
    [
        Path("hypergraph/attention/scripts/extract_dual_gpu.sh"),
        Path("hypergraph/attention/scripts/run_single_layer_response_pipeline.sh"),
    ]
)
method_hash = digest(method_paths)
print(input_hash, method_hash)
PY
)
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
printf 'input:                   %s\n' "${INPUT}"
printf 'model:                   %s\n' "${MODEL}"
printf 'dataset/layer:           %s / %s (zero-based block id)\n' "${DATASET_TAG}" "${LAYER}"
printf 'trace root:              %s\n' "${TRACE_ROOT}"
printf 'run root:                %s\n' "${RUN_ROOT}"
printf 'source/receiver:         %s / response-only receivers\n' "${SOURCE_SCOPE}"
printf 'objective/pooling:       response_bce / %s\n' "${POOLING}"
printf 'topology:                %s, threshold=%s, min_sources=%s\n' \
  "${SOURCE_SELECTION}" "${THRESHOLD}" "${MIN_SOURCES}"
printf 'propagation/incidence:   %s / %s\n' "${PROPAGATION_MODE}" "${INCIDENCE_WEIGHT_MODE}"
printf 'node/operator:           %s / %s\n' "${NODE_FEATURE_MODE}" "${MESSAGE_OPERATOR}"
printf 'split/folds/seeds:       %s / %s / %s\n\n' "${SPLIT_MODE}" "${FOLDS}" "${SEEDS}"
if [[ "${PROPAGATION_MODE}" == "symmetric" || "${PREPROCESSING}" == "per_graph_zscore" ]]; then
  printf '%s\n\n' \
    'scope:                   offline full-response (explicitly recorded by train.py)'
fi

TRACE_CONFIG_PATH="${TRACE_ROOT}/pipeline_request.json"
guard_config "${TRACE_CONFIG_PATH}" \
  "input=${INPUT}" \
  "input_sha256=${INPUT_SHA256}" \
  "model=${MODEL}" \
  "method_code_sha256=${METHOD_CODE_SHA256}" \
  "layer=${LAYER}" \
  "limit=${LIMIT}" \
  "mode=${MODE}" \
  "query_chunk_size=${QUERY_CHUNK_SIZE}" \
  "storage_dtype=${STORAGE_DTYPE}" \
  "dtype=${DTYPE}" \
  "archive_compression=${ARCHIVE_COMPRESSION}" \
  "max_seq_len=${MAX_SEQ_LEN}" \
  "replay_mode=${REPLAY_MODE}" \
  "prompt_style=${PROMPT_STYLE}" \
  "chunk_equivalence_threshold=${THRESHOLD}"

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
  LIMIT="${LIMIT}" \
    bash "${SCRIPT_DIR}/extract_dual_gpu.sh" \
      --attention_layers "${LAYER}" \
      --attention_heads all \
      --chunk-equivalence-threshold "${THRESHOLD}"
fi

if [[ "${MODE}" == "data_parallel" ]]; then
  TRACE_INPUTS=("${TRACE_ROOT}/shard0" "${TRACE_ROOT}/shard1")
else
  TRACE_INPUTS=("${TRACE_ROOT}/balanced")
fi
for trace_input in "${TRACE_INPUTS[@]}"; do
  [[ -d "${trace_input}" ]] || die "missing extracted trace directory: ${trace_input}"
done

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

RUN_CONFIG_PATH="${RUN_ROOT}/pipeline_request.json"
guard_config "${RUN_CONFIG_PATH}" \
  "trace_root=${TRACE_ROOT}" \
  "input_sha256=${INPUT_SHA256}" \
  "method_code_sha256=${METHOD_CODE_SHA256}" \
  "layer=${LAYER}" \
  "threshold=${THRESHOLD}" \
  "source_selection=${SOURCE_SELECTION}" \
  "source_scope=${SOURCE_SCOPE}" \
  "min_sources=${MIN_SOURCES}" \
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
  "folds=${FOLDS}" \
  "seeds=${SEEDS}"

mkdir -p "${RUN_ROOT}/logs"

GRAPH_ARGS=(
  --selected-layers "${LAYER}"
  --selected-heads all
  --threshold "${THRESHOLD}"
  --source-selection "${SOURCE_SELECTION}"
  --source-scope "${SOURCE_SCOPE}"
  --min-sources "${MIN_SOURCES}"
  --include-center
  --propagation-mode "${PROPAGATION_MODE}"
  --incidence-weight-mode "${INCIDENCE_WEIGHT_MODE}"
  --edge-attr-mode "${EDGE_ATTR_MODE}"
  --node-feature-mode "${NODE_FEATURE_MODE}"
)

"${PYTHON_BIN}" -m hypergraph.attention.train inspect \
  "${TRACE_INPUTS[@]}" \
  "${GRAPH_ARGS[@]}" \
  --output "${RUN_ROOT}/preflight.json"

read -r -a SEED_VALUES <<< "${SEEDS//,/ }"
((${#SEED_VALUES[@]})) || die "--seeds resolved to an empty list"
train_pids=()
train_labels=()
train_logs=()
training_job_index=0

cleanup_training_jobs() {
  local pid
  for pid in "${train_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_training_jobs INT TERM EXIT

wait_for_training_wave() {
  local status=0
  local index
  for index in "${!train_pids[@]}"; do
    if wait "${train_pids[$index]}"; then
      printf 'Completed %s\n' "${train_labels[$index]}"
    else
      printf 'FAILED %s; tail of %s:\n' \
        "${train_labels[$index]}" "${train_logs[$index]}" >&2
      tail -n 40 "${train_logs[$index]}" >&2 || true
      status=1
    fi
  done
  train_pids=()
  train_labels=()
  train_logs=()
  [[ "${status}" -eq 0 ]] || die "at least one parallel fold training job failed"
}

for seed in "${SEED_VALUES[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "seed must be a non-negative integer: ${seed}"
  for ((fold = 0; fold < FOLDS; fold++)); do
    RUN_DIR="${RUN_ROOT}/fold${fold}_seed${seed}"
    LOG_FILE="${RUN_ROOT}/logs/fold${fold}_seed${seed}.log"
    if [[ -f "${RUN_DIR}/results.json" ]] && is_true "${REUSE_RUNS}"; then
      printf 'Reusing completed run: %s\n' "${RUN_DIR}"
      continue
    fi
    TRAIN_ARGS=(
      train
      "${TRACE_INPUTS[@]}"
      --objective response_bce
      "${GRAPH_ARGS[@]}"
      --message-operator "${MESSAGE_OPERATOR}"
      --preprocessing "${PREPROCESSING}"
      --pooling "${POOLING}"
      --split-mode "${SPLIT_MODE}"
      --folds "${FOLDS}"
      --fold-index "${fold}"
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
    if [[ "${REPLAY_MODE}" == "observer" ]]; then
      TRAIN_ARGS+=(--allow-observer-traces)
    fi
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
    TRAIN_DEVICE="${TRAIN_GPU_VALUES[$((training_job_index % ${#TRAIN_GPU_VALUES[@]}))]}"
    printf '\nTraining fold=%s seed=%s on physical GPU %s -> %s\n' \
      "${fold}" "${seed}" "${TRAIN_DEVICE}" "${RUN_DIR}"
    CUDA_VISIBLE_DEVICES="${TRAIN_DEVICE}" "${PYTHON_BIN}" \
      -m hypergraph.attention.train "${TRAIN_ARGS[@]}" \
      >"${LOG_FILE}" 2>&1 &
    train_pids+=("$!")
    train_labels+=("fold=${fold} seed=${seed} gpu=${TRAIN_DEVICE}")
    train_logs+=("${LOG_FILE}")
    ((training_job_index += 1))
    if ((${#train_pids[@]} >= ${#TRAIN_GPU_VALUES[@]})); then
      wait_for_training_wave
    fi
  done
done
if ((${#train_pids[@]})); then
  wait_for_training_wave
fi
trap - INT TERM EXIT

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in sorted(root.glob("fold*_seed*/results.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload.get("metrics", {}).get("test", {})
    records.append(
        {
            "run": path.parent.name,
            "best_epoch": payload.get("best_epoch"),
            "auroc": test.get("auroc"),
            "aupr": test.get("aupr"),
            "accuracy_0.5": test.get("accuracy_0.5"),
        }
    )
if not records:
    raise SystemExit("no completed results.json files found")

aggregate = {}
for key in ("auroc", "aupr", "accuracy_0.5"):
    values = [float(row[key]) for row in records if row[key] is not None and math.isfinite(float(row[key]))]
    aggregate[key] = {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
summary = {"num_runs": len(records), "runs": records, "test_aggregate": aggregate}
destination = root / "aggregate_results.json"
destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("aggregate results:", destination)
PY

printf '\nPipeline complete. Results: %s\n' "${RUN_ROOT}"
