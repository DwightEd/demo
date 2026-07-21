#!/usr/bin/env bash
# Run the fixed-holdout, original-aligned response pipeline on ProcessBench.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
ALL_PIPELINE="${SCRIPT_DIR}/run_all_processbench_response_pipeline.sh"
SINGLE_PIPELINE="${SCRIPT_DIR}/run_single_layer_response_pipeline.sh"

usage() {
  cat <<'EOF'
Usage:
  bash hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh \
    [--layer 14] [--seed 17] [--mode model_parallel] \
    [--generator-model Llama-3.1-8B-Instruct]

Defaults:
  datasets: gsm8k, math, olympiadbench, omnimath
  extraction: exact full forward balanced over both GPUs
  evaluation: one persisted problem-disjoint train/validation/test split per dataset
  selection: validation AUPRC; final test is evaluated exactly once

Options:
  --datasets LIST   comma/space-separated ProcessBench subset names
  --model PATH      local observer model path
  --layer ID        zero-based Transformer block id (default: 14)
  --seed N          model initialization seed (default: 17)
  --seeds N         compatibility alias for --seed; exactly one value is required
  --generator-model TAG
                     exact ProcessBench generator tag selected before training
  --mode MODE       model_parallel (default) or data_parallel
  --limit N         pilot limit applied independently to every subset
  --help            show this message

Method/runtime environment variables:
  SPLIT_SEED=17 VAL_RATIO=0.1 TEST_RATIO=0.2
  THRESHOLD=0.05 SOURCE_SELECTION=threshold_fallback_topk TOP_K=16 MIN_SOURCES=2
  TOPOLOGY_HEADS=0
  NODE_FEATURE_MODE=attention_diagonal ACTIVATION_LAYER=
  MAX_SEQ_LEN=0  # no user-imposed token cap
  PYTHON_BIN=python GPU0=0 GPU1=1 TRAIN_GPUS=0,1

The original code's three values are hyperedge attributes
(attention mean, attention max, normalized head id). Node features are the
self-attention diagonal over every extracted head; selecting TOPOLOGY_HEADS=0
only restricts which attention rows create hyperedges.
Set NODE_FEATURE_MODE=activation_only or diagonal_plus_activation to add
post-block hidden states to token nodes. The 3-D hyperedge attributes do not change.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

DATASETS="${DATASETS:-gsm8k,math,olympiadbench,omnimath}"
MODEL="${MODEL:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
LAYER="${LAYER:-14}"
SEEDS="${SEEDS:-17}"
LIMIT="${LIMIT:-}"
MODE="${MODE:-model_parallel}"
GENERATOR_MODEL="${GENERATOR_MODEL:-}"
SPLIT_SEED="${SPLIT_SEED:-17}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.2}"
THRESHOLD="${THRESHOLD:-0.05}"
SOURCE_SELECTION="${SOURCE_SELECTION:-threshold_fallback_topk}"
TOP_K="${TOP_K:-16}"
MIN_SOURCES="${MIN_SOURCES:-2}"
TOPOLOGY_HEADS="${TOPOLOGY_HEADS:-0}"
NODE_FEATURE_MODE="${NODE_FEATURE_MODE:-attention_diagonal}"
ACTIVATION_LAYER="${ACTIVATION_LAYER:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-0}"

while (($#)); do
  case "$1" in
    --datasets) DATASETS="${2:?--datasets requires a list}"; shift 2 ;;
    --model) MODEL="${2:?--model requires a path}"; shift 2 ;;
    --layer) LAYER="${2:?--layer requires an integer}"; shift 2 ;;
    --folds) die "--folds was removed: this entrypoint performs one fixed held-out test" ;;
    --seed) SEEDS="${2:?--seed requires an integer}"; shift 2 ;;
    --seeds) SEEDS="${2:?--seeds requires one integer}"; shift 2 ;;
    --generator-model) GENERATOR_MODEL="${2:?--generator-model requires a tag}"; shift 2 ;;
    --mode) MODE="${2:?--mode requires a value}"; shift 2 ;;
    --limit) LIMIT="${2:?--limit requires an integer}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (run with --help)" ;;
  esac
done

[[ -f "${SINGLE_PIPELINE}" ]] || die "single-dataset pipeline does not exist: ${SINGLE_PIPELINE}"
command -v realpath >/dev/null 2>&1 || die "realpath is required"
[[ "${LAYER}" =~ ^[0-9]+$ ]] || die "--layer must be a non-negative integer"
[[ "${SEEDS}" =~ ^(0|[1-9][0-9]*)$ ]] || die "exactly one canonical non-negative --seed is required"
[[ "${SPLIT_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] || die "SPLIT_SEED must be non-negative"
[[ -z "${LIMIT}" || "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || die "--limit must be positive"
[[ "${MODE}" == "model_parallel" || "${MODE}" == "data_parallel" ]] || \
  die "--mode must be model_parallel or data_parallel"
[[ "${MAX_SEQ_LEN}" =~ ^[0-9]+$ ]] || \
  die "MAX_SEQ_LEN must be a non-negative integer"
[[ -z "${GENERATOR_MODEL}" || "${GENERATOR_MODEL}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "--generator-model contains unsafe characters"
[[ -z "${TRACE_ROOT:-}" && -z "${RUN_ROOT:-}" ]] || \
  die "TRACE_ROOT/RUN_ROOT are ambiguous across datasets; use the single-dataset wrapper"

read -r -a DATASET_VALUES <<< "${DATASETS//,/ }"
((${#DATASET_VALUES[@]})) || die "--datasets resolved to an empty list"
for dataset in "${DATASET_VALUES[@]}"; do
  [[ "${dataset}" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe dataset name: ${dataset}"
  input="${REPO_ROOT}/data/hf_datasets/ProcessBench/${dataset}.json"
  [[ -f "${input}" ]] || die "ProcessBench input does not exist: ${input}"
done
for index in "${!DATASET_VALUES[@]}"; do
  for ((previous = 0; previous < index; previous++)); do
    [[ "${DATASET_VALUES[$index],,}" != "${DATASET_VALUES[$previous],,}" ]] || \
      die "--datasets contains duplicate subset: ${DATASET_VALUES[$index]}"
  done
done
[[ -d "${MODEL}" ]] || die "model directory does not exist: ${MODEL}"

case "${NODE_FEATURE_MODE}" in
  attention_diagonal)
    [[ -z "${ACTIVATION_LAYER}" ]] || \
      die "ACTIVATION_LAYER must be empty for NODE_FEATURE_MODE=attention_diagonal"
    node_variant_suffix="_node_attention"
    ;;
  activation_only|diagonal_plus_activation)
    ACTIVATION_LAYER="${ACTIVATION_LAYER:-$((LAYER + 1))}"
    [[ "${ACTIVATION_LAYER}" =~ ^[1-9][0-9]*$ ]] || \
      die "ACTIVATION_LAYER must be a positive hidden_states index"
    if [[ "${NODE_FEATURE_MODE}" == "activation_only" ]]; then
      node_variant_suffix="_node_hidden_hs${ACTIVATION_LAYER}"
    else
      node_variant_suffix="_node_attention_hidden_hs${ACTIVATION_LAYER}"
    fi
    ;;
  *)
    die "NODE_FEATURE_MODE must be attention_diagonal, activation_only, or diagonal_plus_activation"
    ;;
esac
if [[ "${MAX_SEQ_LEN}" == "0" ]]; then
  seq_policy_suffix="_nocap"
else
  seq_policy_suffix="_maxseq${MAX_SEQ_LEN}"
fi

cd "${REPO_ROOT}"
MODEL="$(realpath "${MODEL}")"
limit_suffix=""
if [[ -n "${LIMIT}" ]]; then
  limit_suffix="_pilot${LIMIT}"
fi
cohort_suffix="_observer_all"
if [[ -n "${GENERATOR_MODEL}" ]]; then
  generator_slug="${GENERATOR_MODEL//\//-}"
  cohort_suffix="_matched_${generator_slug}"
fi
run_suffix="${limit_suffix}${cohort_suffix}${node_variant_suffix}${seq_policy_suffix}_fixed_original"
SUMMARY_ROOT="${REPO_ROOT}/outputs/attention_hypergraph/all_processbench_response_layer${LAYER}${run_suffix}"

"${PYTHON_BIN:-python}" - "${SUMMARY_ROOT}" "${LAYER}" "${SEEDS}" \
  "${SPLIT_SEED}" "${VAL_RATIO}" "${TEST_RATIO}" "${THRESHOLD}" \
  "${SOURCE_SELECTION}" "${TOP_K}" "${MIN_SOURCES}" "${TOPOLOGY_HEADS}" \
  "${NODE_FEATURE_MODE}" "${ACTIVATION_LAYER}" "${MAX_SEQ_LEN}" \
  "${MODE}" "${LIMIT}" "${GENERATOR_MODEL}" "${MODEL}" \
  "${ALL_PIPELINE}" "${SINGLE_PIPELINE}" "${DATASET_VALUES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
request = {
    "schema": "all_processbench_fixed_holdout_request_v1",
    "layer": sys.argv[2],
    "model_seed": sys.argv[3],
    "split_seed": sys.argv[4],
    "val_ratio": sys.argv[5],
    "test_ratio": sys.argv[6],
    "threshold": sys.argv[7],
    "source_selection": sys.argv[8],
    "top_k": sys.argv[9],
    "min_sources": sys.argv[10],
    "topology_heads": sys.argv[11],
    "node_feature_mode": sys.argv[12],
    "activation_layer": sys.argv[13],
    "max_seq_len": sys.argv[14],
    "mode": sys.argv[15],
    "limit": sys.argv[16],
    "generator_model": sys.argv[17],
    "observer_model": sys.argv[18],
    "all_wrapper_sha256": hashlib.sha256(Path(sys.argv[19]).read_bytes()).hexdigest(),
    "single_wrapper_sha256": hashlib.sha256(Path(sys.argv[20]).read_bytes()).hexdigest(),
    "datasets": sys.argv[21:],
}
path = root / "pipeline_request.json"
root.mkdir(parents=True, exist_ok=True)
if path.is_file():
    stored = json.loads(path.read_text(encoding="utf-8"))
    provenance = {"all_wrapper_sha256", "single_wrapper_sha256"}
    differences = {
        key: {"stored": stored.get(key), "requested": request.get(key)}
        for key in sorted((set(stored) | set(request)) - provenance)
        if stored.get(key) != request.get(key)
    }
    if differences:
        raise SystemExit(f"cross-dataset experiment request mismatch for {path}: {differences}")
    print("cross-dataset fixed-holdout gate passed:", path)
else:
    unexpected = [item.name for item in root.iterdir()]
    if unexpected:
        raise SystemExit(f"summary directory has no request gate and is non-empty: {unexpected[:10]}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
PY

printf '\n===== All-ProcessBench fixed-test pipeline =====\n'
printf 'datasets:        %s\n' "${DATASET_VALUES[*]}"
printf 'layer:           %s\n' "${LAYER}"
printf 'model seed:      %s\n' "${SEEDS}"
printf 'split:           seed=%s val=%s test=%s\n' "${SPLIT_SEED}" "${VAL_RATIO}" "${TEST_RATIO}"
printf 'graph:           %s tau=%s fallback_top_k=%s min_sources=%s topology_heads=%s\n' \
  "${SOURCE_SELECTION}" "${THRESHOLD}" "${TOP_K}" "${MIN_SOURCES}" "${TOPOLOGY_HEADS}"
printf 'nodes:           %s activation_layer=%s\n' \
  "${NODE_FEATURE_MODE}" "${ACTIVATION_LAYER:-disabled}"
printf 'sequence:        max_seq_len=%s (%s)\n' \
  "${MAX_SEQ_LEN}" "${seq_policy_suffix#_}"
printf 'generator:       %s\n' "${GENERATOR_MODEL:-all generators}"
printf 'extraction:      %s, exact full forward\n' "${MODE}"
printf 'extract GPUs:    %s, %s\n\n' "${GPU0:-0}" "${GPU1:-1}"

for dataset in "${DATASET_VALUES[@]}"; do
  printf '\n===== Dataset: %s =====\n' "${dataset}"
  args=(
    --model "${MODEL}"
    --layer "${LAYER}"
    --dataset "${dataset}"
    --seed "${SEEDS}"
    --mode "${MODE}"
  )
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${GENERATOR_MODEL}" ]]; then
    args+=(--generator-model "${GENERATOR_MODEL}")
  fi
  SPLIT_SEED="${SPLIT_SEED}" VAL_RATIO="${VAL_RATIO}" TEST_RATIO="${TEST_RATIO}" \
  THRESHOLD="${THRESHOLD}" SOURCE_SELECTION="${SOURCE_SELECTION}" \
  TOP_K="${TOP_K}" MIN_SOURCES="${MIN_SOURCES}" TOPOLOGY_HEADS="${TOPOLOGY_HEADS}" \
  NODE_FEATURE_MODE="${NODE_FEATURE_MODE}" ACTIVATION_LAYER="${ACTIVATION_LAYER}" \
  MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
    bash "${SINGLE_PIPELINE}" "${args[@]}"
done

"${PYTHON_BIN:-python}" - "${REPO_ROOT}" "${SUMMARY_ROOT}" "${LAYER}" \
  "${run_suffix}" "${cohort_suffix}" "${GENERATOR_MODEL}" "${SEEDS}" \
  "${SPLIT_SEED}" "${VAL_RATIO}" "${TEST_RATIO}" "${THRESHOLD}" \
  "${SOURCE_SELECTION}" "${TOP_K}" "${MIN_SOURCES}" "${TOPOLOGY_HEADS}" \
  "${NODE_FEATURE_MODE}" "${ACTIVATION_LAYER}" "${MAX_SEQ_LEN}" \
  "${LIMIT}" "${DATASET_VALUES[@]}" <<'PY'
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

repo = Path(sys.argv[1])
output = Path(sys.argv[2])
layer = int(sys.argv[3])
suffix = sys.argv[4]
cohort_suffix = sys.argv[5]
generator_model = sys.argv[6] or None
model_seed = sys.argv[7]
split_seed = sys.argv[8]
val_ratio = sys.argv[9]
test_ratio = sys.argv[10]
threshold = sys.argv[11]
source_selection = sys.argv[12]
top_k = sys.argv[13]
min_sources = sys.argv[14]
topology_heads = sys.argv[15]
node_feature_mode = sys.argv[16]
activation_layer = sys.argv[17]
max_seq_len = sys.argv[18]
limit = sys.argv[19]
datasets = sys.argv[20:]
metric_names = ("auroc", "aupr", "accuracy_0.5")

per_dataset = {}
compatibility_records = {}
for dataset in datasets:
    root = repo / "outputs" / "attention_hypergraph" / f"{dataset}_response_layer{layer}{suffix}"
    aggregate_path = root / "aggregate_results.json"
    preflight_path = root / "preflight.json"
    gate_path = root / "pipeline_request.json"
    prediction_path = root / "predictions_test.csv"
    for required in (aggregate_path, preflight_path, gate_path, prediction_path):
        if not required.is_file():
            raise SystemExit(f"missing completed fixed-test artifact: {required}")
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "fixed_holdout_response_test_v1":
        raise SystemExit(f"dataset {dataset} is not a fixed-holdout result: {aggregate_path}")
    final = payload.get("final_test")
    if not isinstance(final, dict):
        raise SystemExit(f"dataset {dataset} lacks final_test")
    for metric in metric_names:
        value = final.get(metric)
        if value is None or not math.isfinite(float(value)):
            raise SystemExit(f"dataset {dataset} has undefined/non-finite final test {metric}")
    split = payload.get("split") or {}
    if split.get("mode") != "fixed_holdout" or str(split.get("split_seed")) != split_seed:
        raise SystemExit(f"dataset {dataset} has incompatible split manifest: {split}")

    preflight_bytes = preflight_path.read_bytes()
    preflight_sha256 = hashlib.sha256(preflight_bytes).hexdigest()
    preflight = json.loads(preflight_bytes.decode("utf-8"))
    gate_bytes = gate_path.read_bytes()
    gate = json.loads(gate_bytes.decode("utf-8"))
    expected_gate = {
        "layer": str(layer),
        "generator_model": generator_model or "",
        "cohort_suffix": cohort_suffix,
        "selection_limit": limit,
        "split_mode": "fixed_holdout",
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "threshold": threshold,
        "source_selection": source_selection,
        "top_k": top_k,
        "min_sources": min_sources,
        "topology_heads": topology_heads,
        "node_feature_mode": node_feature_mode,
        "activation_layer": activation_layer,
        "max_seq_len": max_seq_len,
        "seeds": model_seed,
        "preflight_sha256": preflight_sha256,
    }
    differences = {
        key: {"stored": gate.get(key), "expected": expected}
        for key, expected in expected_gate.items()
        if gate.get(key) != expected
    }
    if differences:
        raise SystemExit(f"dataset run/preflight gate mismatch for {dataset}: {differences}")
    if preflight.get("cohort_gate_passed", preflight.get("training_gate_passed")) is not True:
        raise SystemExit(f"dataset preflight did not pass its cohort gate: {preflight_path}")
    cohort_audit = preflight.get("cohort_audit") or {}
    representation = cohort_audit.get("representation_provenance")
    axis_contract = cohort_audit.get("attention_axis_contract")
    if not isinstance(representation, dict) or not isinstance(axis_contract, dict):
        raise SystemExit(f"dataset preflight lacks representation audit: {preflight_path}")
    compatibility = {
        "representation": representation,
        "attention_axis_contract": axis_contract,
        "graph_config": preflight.get("graph_config"),
    }
    compatibility_records[dataset] = compatibility
    per_dataset[dataset] = {
        "source": str(aggregate_path),
        "predictions_test": str(prediction_path),
        "preflight_sha256": preflight_sha256,
        "partition_sizes": payload.get("partition_sizes"),
        "split": split,
        "final_test": final,
        "generator_final_test": payload.get("generator_final_test", {}),
        "representation_fingerprint": cohort_audit.get("representation_fingerprint"),
        "selection": preflight.get("selection"),
    }

reference_dataset = datasets[0]
reference = compatibility_records[reference_dataset]
for dataset in datasets[1:]:
    if compatibility_records[dataset] != reference:
        raise SystemExit(
            "cross-dataset observer/template/axis/graph compatibility mismatch: "
            f"{reference_dataset!r} != {dataset!r}"
        )

macro = {}
for metric in metric_names:
    values = [float(per_dataset[name]["final_test"][metric]) for name in datasets]
    macro[metric] = {
        "n_datasets": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }

summary = {
    "schema": "all_processbench_fixed_holdout_macro_v1",
    "protocol": "single_problem_disjoint_fixed_holdout_per_dataset",
    "layer": layer,
    "model_seed": int(model_seed),
    "split_seed": int(split_seed),
    "generator_model": generator_model,
    "method_contract": {
        "node_feature_mode": node_feature_mode,
        "activation_layer": None if not activation_layer else int(activation_layer),
        "node_features": (
            "self-attention diagonal over all extracted heads"
            if node_feature_mode == "attention_diagonal"
            else "post-block hidden state"
            if node_feature_mode == "activation_only"
            else "self-attention diagonal concatenated with post-block hidden state"
        ),
        "edge_attributes": ["attention_mean", "attention_max", "flattened_head_normalized"],
        "source_selection": source_selection,
        "fallback_top_k": int(top_k),
        "min_sources_before_center": int(min_sources),
        "topology_heads": topology_heads,
        "threshold": float(threshold),
        "max_seq_len": int(max_seq_len),
    },
    "datasets": datasets,
    "per_dataset": per_dataset,
    "macro_final_test": macro,
    "cross_dataset_representation_compatibility": reference,
    "note": (
        "Each ProcessBench subset uses one persisted problem-disjoint internal holdout. "
        "This mirrors the original train/validation/test protocol, but it is not the "
        "original paper's external RAGTruth test set."
    ),
}
json_path = output / "aggregate_results.json"
markdown_path = output / "summary.md"
rows = [
    "# All-ProcessBench Fixed Held-Out Test",
    "",
    f"- Layer: `{layer}`",
    f"- Node features: `{node_feature_mode}`; hidden_states index: "
    f"`{activation_layer or 'disabled'}`",
    f"- Sequence policy: `max_seq_len={max_seq_len}`",
    f"- Generator cohort: `{generator_model or 'all generators'}`",
    f"- Split seed: `{split_seed}`; model seed: `{model_seed}`",
    "- Model selection uses validation AUPRC; each held-out test is evaluated once.",
    "",
    "| dataset | test n | positives | AUROC | AUPRC | accuracy@0.5 |",
    "|---|---:|---:|---:|---:|---:|",
]
for dataset in datasets:
    final = per_dataset[dataset]["final_test"]
    rows.append(
        f"| {dataset} | {int(final['n'])} | {int(final['positives'])} | "
        f"{float(final['auroc']):.4f} | {float(final['aupr']):.4f} | "
        f"{float(final['accuracy_0.5']):.4f} |"
    )
rows.extend(["", "## Unweighted Macro", ""])
for metric in metric_names:
    rows.append(f"- `{metric}`: {macro[metric]['mean']:.4f} +/- {macro[metric]['std']:.4f}")

json_tmp = json_path.with_suffix(".json.tmp")
markdown_tmp = markdown_path.with_suffix(".md.tmp")
json_tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
markdown_tmp.write_text("\n".join(rows) + "\n", encoding="utf-8")
os.replace(json_tmp, json_path)
os.replace(markdown_tmp, markdown_path)

print("===== Final fixed held-out tests =====")
for dataset in datasets:
    final = per_dataset[dataset]["final_test"]
    print(
        f"{dataset}: n={int(final['n'])} positives={int(final['positives'])} "
        f"AUROC={float(final['auroc']):.6f} AUPRC={float(final['aupr']):.6f}"
    )
print(f"macro AUROC={macro['auroc']['mean']:.6f} AUPRC={macro['aupr']['mean']:.6f}")
print("all-dataset aggregate:", json_path)
print("human-readable summary:", markdown_path)
PY

printf '\nAll ProcessBench datasets complete. Summary: %s\n' "${SUMMARY_ROOT}"
