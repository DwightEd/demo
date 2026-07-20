#!/usr/bin/env bash
# Run the strict single-layer response pipeline on every ProcessBench subset.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SINGLE_PIPELINE="${SCRIPT_DIR}/run_single_layer_response_pipeline.sh"

usage() {
  cat <<'EOF'
Usage:
  bash hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh \
    [--layer 14] [--folds 5] [--seeds 17]

Defaults:
  datasets: gsm8k, math, olympiadbench, omnimath
  extraction: two complementary data-parallel shards on GPU 0 and GPU 1
  training: concurrent folds scheduled across GPU 0 and GPU 1

Options:
  --datasets LIST   comma/space-separated ProcessBench subset names
  --model PATH      local observer model path
  --layer ID        zero-based Transformer block id (default: 14)
  --folds N         problem-disjoint group-CV folds (default: 5)
  --seeds LIST      comma/space-separated seeds (default: 17)
  --limit N         pilot limit applied independently to every subset
  --help            show this message

Runtime environment variables:
  PYTHON_BIN=python  GPU0=0  GPU1=1  TRAIN_GPUS=0,1
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

DATASETS="${DATASETS:-gsm8k,math,olympiadbench,omnimath}"
MODEL="${MODEL:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
LAYER="${LAYER:-14}"
FOLDS="${FOLDS:-5}"
SEEDS="${SEEDS:-17}"
LIMIT="${LIMIT:-}"

while (($#)); do
  case "$1" in
    --datasets) DATASETS="${2:?--datasets requires a list}"; shift 2 ;;
    --model) MODEL="${2:?--model requires a path}"; shift 2 ;;
    --layer) LAYER="${2:?--layer requires an integer}"; shift 2 ;;
    --folds) FOLDS="${2:?--folds requires an integer}"; shift 2 ;;
    --seeds) SEEDS="${2:?--seeds requires a list}"; shift 2 ;;
    --limit) LIMIT="${2:?--limit requires an integer}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (run with --help)" ;;
  esac
done

[[ -x "${SINGLE_PIPELINE}" || -f "${SINGLE_PIPELINE}" ]] || \
  die "single-dataset pipeline does not exist: ${SINGLE_PIPELINE}"
[[ "${LAYER}" =~ ^[0-9]+$ ]] || die "--layer must be a non-negative integer"
[[ "${FOLDS}" =~ ^[0-9]+$ ]] && ((FOLDS >= 3)) || \
  die "--folds must be an integer >= 3"
[[ -z "${LIMIT}" || "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || \
  die "--limit must be a positive integer"

read -r -a DATASET_VALUES <<< "${DATASETS//,/ }"
((${#DATASET_VALUES[@]})) || die "--datasets resolved to an empty list"
for dataset in "${DATASET_VALUES[@]}"; do
  [[ "${dataset}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "unsafe dataset name: ${dataset}"
  input="${REPO_ROOT}/data/hf_datasets/ProcessBench/${dataset}.json"
  [[ -f "${input}" ]] || die "ProcessBench input does not exist: ${input}"
done
[[ -d "${MODEL}" ]] || die "model directory does not exist: ${MODEL}"

cd "${REPO_ROOT}"
printf '\n===== All-ProcessBench response pipeline =====\n'
printf 'datasets:       %s\n' "${DATASET_VALUES[*]}"
printf 'layer:          %s\n' "${LAYER}"
printf 'folds/seeds:    %s / %s\n' "${FOLDS}" "${SEEDS}"
printf 'extract GPUs:   %s, %s\n' "${GPU0:-0}" "${GPU1:-1}"
printf 'training GPUs:  %s\n\n' "${TRAIN_GPUS:-${GPU0:-0},${GPU1:-1}}"

for dataset in "${DATASET_VALUES[@]}"; do
  printf '\n===== Dataset: %s =====\n' "${dataset}"
  args=(
    --model "${MODEL}"
    --layer "${LAYER}"
    --dataset "${dataset}"
    --folds "${FOLDS}"
    --seeds "${SEEDS}"
    --mode data_parallel
  )
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi
  bash "${SINGLE_PIPELINE}" "${args[@]}"
done

limit_suffix=""
if [[ -n "${LIMIT}" ]]; then
  limit_suffix="_pilot${LIMIT}"
fi
SUMMARY_ROOT="${REPO_ROOT}/outputs/attention_hypergraph/all_processbench_response_layer${LAYER}${limit_suffix}"
mkdir -p "${SUMMARY_ROOT}"

"${PYTHON_BIN:-python}" - "${REPO_ROOT}" "${SUMMARY_ROOT}" "${LAYER}" "${limit_suffix}" "${DATASET_VALUES[@]}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

repo = Path(sys.argv[1])
output = Path(sys.argv[2])
layer = int(sys.argv[3])
suffix = sys.argv[4]
datasets = sys.argv[5:]
metrics = ("auroc", "aupr", "accuracy_0.5")

per_dataset = {}
for dataset in datasets:
    path = (
        repo
        / "outputs"
        / "attention_hypergraph"
        / f"{dataset}_response_layer{layer}{suffix}"
        / "aggregate_results.json"
    )
    if not path.is_file():
        raise SystemExit(f"missing dataset aggregate: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_dataset[dataset] = {
        "source": str(path),
        "num_runs": payload.get("num_runs"),
        "test_aggregate": payload.get("test_aggregate", {}),
    }

macro = {}
for metric in metrics:
    values = []
    for dataset in datasets:
        value = (
            per_dataset[dataset]
            .get("test_aggregate", {})
            .get(metric, {})
            .get("mean")
        )
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    macro[metric] = {
        "n_datasets": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }

summary = {
    "schema": "all_processbench_response_macro_v1",
    "layer": layer,
    "datasets": datasets,
    "per_dataset": per_dataset,
    "macro_dataset_mean": macro,
    "note": "Macro values average held-out dataset-level means; they are not pooled predictions.",
}
json_path = output / "aggregate_results.json"
json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

rows = [
    "# All-ProcessBench Response Results",
    "",
    f"- Layer: `{layer}`",
    "- Aggregation: unweighted macro mean over dataset-level held-out results",
    "",
    "| dataset | AUROC | AUPRC | accuracy@0.5 |",
    "|---|---:|---:|---:|",
]
for dataset in datasets:
    aggregate = per_dataset[dataset]["test_aggregate"]
    values = [aggregate.get(metric, {}).get("mean") for metric in metrics]
    rendered = ["NA" if value is None else f"{float(value):.4f}" for value in values]
    rows.append(f"| {dataset} | {rendered[0]} | {rendered[1]} | {rendered[2]} |")
rows.extend(["", "## Macro Dataset Mean", ""])
for metric in metrics:
    value = macro[metric]["mean"]
    rows.append(f"- `{metric}`: " + ("NA" if value is None else f"{float(value):.4f}"))
(output / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("all-dataset aggregate:", json_path)
PY

printf '\nAll ProcessBench datasets complete. Summary: %s\n' "${SUMMARY_ROOT}"
