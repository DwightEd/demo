#!/usr/bin/env bash
# Two-GPU attention extraction for an 8B decoder on 24 GiB cards.
#
# Required: INPUT=/absolute/dataset.json
# Optional: MODEL, OUTPUT_ROOT, MODE, QUERY_CHUNK_SIZE, DTYPE, STORAGE_DTYPE,
#           ARCHIVE_COMPRESSION, LIMIT, GPU0, GPU1, PYTHON_BIN.
# Any command-line arguments are appended to the extractor invocation.

set -Eeuo pipefail

: "${INPUT:?Set INPUT to an absolute ProcessBench JSON or JSONL path}"

MODEL="${MODEL:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/attention_traces/llama31_8b_observer}"
MODE="${MODE:-model_parallel}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-0}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
DTYPE="${DTYPE:-auto}"
ARCHIVE_COMPRESSION="${ARCHIVE_COMPRESSION:-none}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB:-24}"
REPLAY_MODE="${REPLAY_MODE:-observer}"
PROMPT_STYLE="${PROMPT_STYLE:-plain}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU_MEMORY="${GPU_MEMORY:-22GiB}"
CPU_MEMORY="${CPU_MEMORY:-64GiB}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${INPUT}" ]]; then
  printf 'Input file does not exist: %s\n' "${INPUT}" >&2
  exit 2
fi
if [[ "${INPUT}" != /* ]]; then
  printf 'INPUT must be an absolute path: %s\n' "${INPUT}" >&2
  exit 2
fi
if [[ ! -d "${MODEL}" ]]; then
  printf 'Model directory does not exist: %s\n' "${MODEL}" >&2
  exit 2
fi
if ! [[ "${QUERY_CHUNK_SIZE}" =~ ^[0-9]+$ ]]; then
  printf 'QUERY_CHUNK_SIZE must be a non-negative integer\n' >&2
  exit 2
fi
if [[ "${GPU0}" == "${GPU1}" ]]; then
  printf 'GPU0 and GPU1 must name two different physical devices\n' >&2
  exit 2
fi
for argument in "$@"; do
  case "${argument}" in
    --input|--input=*|--output_dir|--output_dir=*|--model|--model=*|\
    --device|--device=*|--device_map|--device_map=*|--device-map|--device-map=*|\
    --max_memory|--max_memory=*|--max-memory|--max-memory=*|\
    --num_shards|--num_shards=*|--num-shards|--num-shards=*|\
    --shard_index|--shard_index=*|--shard-index|--shard-index=*|\
    --query_chunk_size|--query_chunk_size=*|--query-chunk-size|--query-chunk-size=*|\
    --verify_chunked_equivalence|--no-verify_chunked_equivalence|\
    --verify-chunked-equivalence|--no-verify-chunked-equivalence)
      printf 'Protected extractor option must be configured through the script: %s\n' \
        "${argument}" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}/logs"
export TOKENIZERS_PARALLELISM=false

common_args=(
  --input "${INPUT}"
  --model "${MODEL}"
  --model_class base
  --replay_mode "${REPLAY_MODE}"
  --prompt_style "${PROMPT_STYLE}"
  --dtype "${DTYPE}"
  --storage_dtype "${STORAGE_DTYPE}"
  --query_chunk_size "${QUERY_CHUNK_SIZE}"
  --verify_chunked_equivalence
  --archive_compression "${ARCHIVE_COMPRESSION}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --max_attention_gib "${MAX_ATTENTION_GIB}"
)
if [[ -n "${LIMIT:-}" ]]; then
  common_args+=(--limit "${LIMIT}")
fi

case "${MODE}" in
  data_parallel)
    pids=()
    cleanup() {
      for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
      done
    }
    trap cleanup INT TERM

    (
      set -o pipefail
      PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU0}" \
        "${PYTHON_BIN}" -m hypergraph.attention.extract \
        "${common_args[@]}" \
        "$@" \
        --output_dir "${OUTPUT_ROOT}/shard0" \
        --device cuda:0 --num-shards 2 --shard-index 0 \
        2>&1 | tee "${OUTPUT_ROOT}/logs/shard0.log"
    ) &
    pids+=("$!")

    (
      set -o pipefail
      PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU1}" \
        "${PYTHON_BIN}" -m hypergraph.attention.extract \
        "${common_args[@]}" \
        "$@" \
        --output_dir "${OUTPUT_ROOT}/shard1" \
        --device cuda:0 --num-shards 2 --shard-index 1 \
        2>&1 | tee "${OUTPUT_ROOT}/logs/shard1.log"
    ) &
    pids+=("$!")

    status=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    trap - INT TERM
    if [[ "${status}" -ne 0 ]]; then
      printf 'At least one shard failed; inspect %s/logs/shard*.log\n' "${OUTPUT_ROOT}" >&2
      exit "${status}"
    fi
    "${PYTHON_BIN}" -m hypergraph.attention.shards \
      "${OUTPUT_ROOT}/shard0" "${OUTPUT_ROOT}/shard1" \
      --output "${OUTPUT_ROOT}/shard_audit.json"
    printf 'Completed and audited complementary shards: %s/shard0 and %s/shard1\n' \
      "${OUTPUT_ROOT}" "${OUTPUT_ROOT}"
    ;;

  model_parallel)
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU0},${GPU1}" "${PYTHON_BIN}" \
      -m hypergraph.attention.extract \
      "${common_args[@]}" \
      "$@" \
      --output_dir "${OUTPUT_ROOT}/balanced" \
      --device_map balanced \
      --max_memory "0=${GPU_MEMORY},1=${GPU_MEMORY},cpu=${CPU_MEMORY}" \
      2>&1 | tee "${OUTPUT_ROOT}/logs/balanced.log"
    "${PYTHON_BIN}" -m hypergraph.attention.shards \
      "${OUTPUT_ROOT}/balanced" \
      --output "${OUTPUT_ROOT}/shard_audit.json"
    ;;

  *)
    printf 'MODE must be data_parallel or model_parallel, got: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
