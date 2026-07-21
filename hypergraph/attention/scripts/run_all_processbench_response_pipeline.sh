#!/usr/bin/env bash
# Run the strict single-layer response pipeline on every ProcessBench subset.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
ALL_PIPELINE="${SCRIPT_DIR}/run_all_processbench_response_pipeline.sh"
SINGLE_PIPELINE="${SCRIPT_DIR}/run_single_layer_response_pipeline.sh"

usage() {
  cat <<'EOF'
Usage:
  bash hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh \
    [--layer 14] [--folds 5] [--seeds 17] [--mode model_parallel] \
    [--generator-model Llama-3.1-8B-Instruct]

Defaults:
  datasets: gsm8k, math, olympiadbench, omnimath
  extraction: exact full forward with the observer model balanced over both GPUs
  training: concurrent folds scheduled across GPU 0 and GPU 1

Options:
  --datasets LIST   comma/space-separated ProcessBench subset names
  --model PATH      local observer model path
  --layer ID        zero-based Transformer block id (default: 14)
  --folds N         problem-disjoint group-CV folds (default: 5)
  --seeds LIST      comma/space-separated seeds (default: 17)
  --generator-model TAG
                     exact ProcessBench generator tag selected before training
  --mode MODE       model_parallel (default) or data_parallel
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
MODE="${MODE:-model_parallel}"
GENERATOR_MODEL="${GENERATOR_MODEL:-}"

while (($#)); do
  case "$1" in
    --datasets) DATASETS="${2:?--datasets requires a list}"; shift 2 ;;
    --model) MODEL="${2:?--model requires a path}"; shift 2 ;;
    --layer) LAYER="${2:?--layer requires an integer}"; shift 2 ;;
    --folds) FOLDS="${2:?--folds requires an integer}"; shift 2 ;;
    --seeds) SEEDS="${2:?--seeds requires a list}"; shift 2 ;;
    --generator-model) GENERATOR_MODEL="${2:?--generator-model requires a tag}"; shift 2 ;;
    --mode) MODE="${2:?--mode requires a value}"; shift 2 ;;
    --limit) LIMIT="${2:?--limit requires an integer}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (run with --help)" ;;
  esac
done

[[ -x "${SINGLE_PIPELINE}" || -f "${SINGLE_PIPELINE}" ]] || \
  die "single-dataset pipeline does not exist: ${SINGLE_PIPELINE}"
command -v realpath >/dev/null 2>&1 || die "realpath is required"
[[ "${LAYER}" =~ ^[0-9]+$ ]] || die "--layer must be a non-negative integer"
[[ "${FOLDS}" =~ ^[0-9]+$ ]] && ((FOLDS >= 3)) || \
  die "--folds must be an integer >= 3"
[[ -z "${LIMIT}" || "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || \
  die "--limit must be a positive integer"
[[ "${MODE}" == "model_parallel" || "${MODE}" == "data_parallel" ]] || \
  die "--mode must be model_parallel or data_parallel"
[[ -z "${GENERATOR_MODEL}" || "${GENERATOR_MODEL}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "--generator-model contains unsafe characters"
[[ -z "${TRACE_ROOT:-}" && -z "${RUN_ROOT:-}" ]] || \
  die "TRACE_ROOT/RUN_ROOT are ambiguous across datasets; invoke the single-dataset wrapper for custom roots"

read -r -a DATASET_VALUES <<< "${DATASETS//,/ }"
((${#DATASET_VALUES[@]})) || die "--datasets resolved to an empty list"
for dataset in "${DATASET_VALUES[@]}"; do
  [[ "${dataset}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "unsafe dataset name: ${dataset}"
  input="${REPO_ROOT}/data/hf_datasets/ProcessBench/${dataset}.json"
  [[ -f "${input}" ]] || die "ProcessBench input does not exist: ${input}"
done
for index in "${!DATASET_VALUES[@]}"; do
  for ((previous = 0; previous < index; previous++)); do
    [[ "${DATASET_VALUES[$index],,}" != "${DATASET_VALUES[$previous],,}" ]] || \
      die "--datasets contains duplicate subset: ${DATASET_VALUES[$index]}"
  done
done
read -r -a SEED_VALUES <<< "${SEEDS//,/ }"
((${#SEED_VALUES[@]})) || die "--seeds resolved to an empty list"
for index in "${!SEED_VALUES[@]}"; do
  seed="${SEED_VALUES[$index]}"
  [[ "${seed}" =~ ^(0|[1-9][0-9]*)$ ]] || \
    die "seed must be a canonical non-negative integer: ${seed}"
  for ((previous = 0; previous < index; previous++)); do
    [[ "${seed}" != "${SEED_VALUES[$previous]}" ]] || \
      die "--seeds contains a duplicate value: ${seed}"
  done
done
[[ -d "${MODEL}" ]] || die "model directory does not exist: ${MODEL}"

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
run_suffix="${limit_suffix}${cohort_suffix}"
SUMMARY_ROOT="${REPO_ROOT}/outputs/attention_hypergraph/all_processbench_response_layer${LAYER}${run_suffix}"

"${PYTHON_BIN:-python}" - "${SUMMARY_ROOT}" "${LAYER}" "${FOLDS}" "${SEEDS}" \
  "${MODE}" "${LIMIT}" "${GENERATOR_MODEL}" "${MODEL}" "${ALL_PIPELINE}" \
  "${SINGLE_PIPELINE}" "${DATASET_VALUES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
request = {
    "schema": "all_processbench_pipeline_request_v2",
    "layer": sys.argv[2],
    "folds": sys.argv[3],
    "seeds": sys.argv[4],
    "mode": sys.argv[5],
    "limit": sys.argv[6],
    "generator_model": sys.argv[7],
    "observer_model": sys.argv[8],
    "all_wrapper_sha256": hashlib.sha256(Path(sys.argv[9]).read_bytes()).hexdigest(),
    "single_wrapper_sha256": hashlib.sha256(Path(sys.argv[10]).read_bytes()).hexdigest(),
    "datasets": sys.argv[11:],
}
path = root / "pipeline_request.json"
root.mkdir(parents=True, exist_ok=True)
if path.is_file():
    stored = json.loads(path.read_text(encoding="utf-8"))
    # Wrapper hashes are provenance, not experiment identity.  Reporting-only
    # changes (for example adding pooled OOF aggregation) must not invalidate
    # completed extraction/training artifacts.  Per-dataset gates separately
    # verify the extraction method, training code, cohort, and preflight hash.
    semantic_keys = (
        "layer",
        "folds",
        "seeds",
        "mode",
        "limit",
        "generator_model",
        "observer_model",
        "datasets",
    )
    semantic_differences = {
        key: {"stored": stored.get(key), "requested": request.get(key)}
        for key in semantic_keys
        if stored.get(key) != request.get(key)
    }
    if semantic_differences:
        raise SystemExit(
            f"cross-dataset experiment request mismatch for {path}: "
            f"{semantic_differences}"
        )
    wrapper_differences = {
        key: {"stored": stored.get(key), "current": request.get(key)}
        for key in ("all_wrapper_sha256", "single_wrapper_sha256")
        if stored.get(key) != request.get(key)
    }
    if wrapper_differences:
        print(
            "experiment gate passed; orchestration/reporting code changed and "
            f"will be revalidated by per-dataset gates: {wrapper_differences}"
        )
    else:
        print("cross-dataset experiment gate passed:", path)
else:
    unexpected = [item.name for item in root.iterdir()]
    if unexpected:
        raise SystemExit(
            f"summary directory has no request gate and is non-empty: {unexpected[:10]}"
        )
    rendered = json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)
PY

printf '\n===== All-ProcessBench response pipeline =====\n'
printf 'datasets:       %s\n' "${DATASET_VALUES[*]}"
printf 'layer:          %s\n' "${LAYER}"
printf 'folds/seeds:    %s / %s\n' "${FOLDS}" "${SEEDS}"
printf 'generator:      %s\n' "${GENERATOR_MODEL:-all generators}"
printf 'extraction:     %s, exact full forward\n' "${MODE}"
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
    --mode "${MODE}"
  )
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${GENERATOR_MODEL}" ]]; then
    args+=(--generator-model "${GENERATOR_MODEL}")
  fi
  bash "${SINGLE_PIPELINE}" "${args[@]}"
done

"${PYTHON_BIN:-python}" - "${REPO_ROOT}" "${SUMMARY_ROOT}" "${LAYER}" \
  "${run_suffix}" "${cohort_suffix}" "${GENERATOR_MODEL}" "${FOLDS}" \
  "${SEEDS}" "${LIMIT}" "${DATASET_VALUES[@]}" <<'PY'
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
folds = int(sys.argv[7])
seeds_raw = sys.argv[8]
limit = sys.argv[9]
datasets = sys.argv[10:]
metrics = ("auroc", "aupr", "accuracy_0.5")
seed_values = [int(value) for value in seeds_raw.replace(",", " ").split()]
expected_num_runs = folds * len(seed_values)

per_dataset = {}
compatibility_records = {}
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
    preflight_path = path.parent / "preflight.json"
    run_gate_path = path.parent / "pipeline_request.json"
    if not preflight_path.is_file():
        raise SystemExit(f"missing strict dataset preflight: {preflight_path}")
    if not run_gate_path.is_file():
        raise SystemExit(f"missing dataset run gate: {run_gate_path}")
    preflight_bytes = preflight_path.read_bytes()
    preflight_sha256 = hashlib.sha256(preflight_bytes).hexdigest()
    preflight = json.loads(preflight_bytes.decode("utf-8"))
    run_gate_bytes = run_gate_path.read_bytes()
    run_gate = json.loads(run_gate_bytes.decode("utf-8"))
    expected_gate_fields = {
        "layer": str(layer),
        "generator_model": generator_model or "",
        "cohort_suffix": cohort_suffix,
        "selection_limit": limit,
        "folds": str(folds),
        "seeds": seeds_raw,
        "preflight_sha256": preflight_sha256,
    }
    gate_differences = {
        key: {"stored": run_gate.get(key), "expected": value}
        for key, value in expected_gate_fields.items()
        if run_gate.get(key) != value
    }
    if gate_differences:
        raise SystemExit(
            f"dataset run/preflight gate mismatch for {dataset}: {gate_differences}"
        )
    if payload.get("num_runs") != expected_num_runs:
        raise SystemExit(
            f"dataset {dataset} has {payload.get('num_runs')} runs; "
            f"expected {expected_num_runs}"
        )
    pooled_oof = payload.get("pooled_oof_test")
    if not isinstance(pooled_oof, dict):
        raise SystemExit(
            f"dataset {dataset} lacks pooled_oof_test; run aggregate_oof.py on "
            f"{path.parent} or rerun the updated single-dataset aggregation"
        )
    pooled_final = pooled_oof.get("seed_ensemble")
    if not isinstance(pooled_final, dict):
        raise SystemExit(f"dataset {dataset} has no pooled OOF seed ensemble")
    for metric in metrics:
        value = pooled_final.get(metric)
        if value is None or not math.isfinite(float(value)):
            raise SystemExit(
                f"dataset {dataset} has undefined/non-finite pooled OOF {metric}"
            )
    cohort_gate_passed = preflight.get(
        "cohort_gate_passed", preflight.get("training_gate_passed")
    )
    if cohort_gate_passed is not True:
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
        "current_validation": {
            key: run_gate.get(key)
            for key in (
                "current_extraction_validation_code_sha256",
                "training_code_sha256",
            )
        },
    }
    compatibility_records[dataset] = compatibility
    per_dataset[dataset] = {
        "source": str(path),
        "preflight": str(preflight_path),
        "preflight_sha256": preflight_sha256,
        "run_gate": str(run_gate_path),
        "run_gate_sha256": hashlib.sha256(run_gate_bytes).hexdigest(),
        "trace_request_provenance": {
            key: run_gate.get(key)
            for key in (
                "trace_request_kind",
                "trace_request_sha256",
                "legacy_monolithic_method_code_sha256",
            )
        },
        "num_runs": payload.get("num_runs"),
        "test_aggregate": payload.get("test_aggregate", {}),
        "pooled_oof_test": pooled_oof,
        "generator_test_aggregate": payload.get("generator_test_aggregate", {}),
        "representation_fingerprint": cohort_audit.get(
            "representation_fingerprint"
        ),
        "source_provenance": cohort_audit.get("source_provenance"),
        "selection": preflight.get("selection"),
    }

reference_dataset = datasets[0]
reference_compatibility = compatibility_records[reference_dataset]
for dataset in datasets[1:]:
    if compatibility_records[dataset] != reference_compatibility:
        raise SystemExit(
            "cross-dataset observer/template/axis/graph compatibility mismatch: "
            f"{reference_dataset!r} != {dataset!r}"
        )

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
        if value is None or not math.isfinite(float(value)):
            raise SystemExit(f"dataset {dataset} has undefined/non-finite {metric}")
        values.append(float(value))
    if len(values) != len(datasets):
        raise SystemExit(f"metric {metric} is incomplete across datasets")
    macro[metric] = {
        "n_datasets": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }

pooled_macro = {}
for metric in metrics:
    values = [
        float(per_dataset[dataset]["pooled_oof_test"]["seed_ensemble"][metric])
        for dataset in datasets
    ]
    pooled_macro[metric] = {
        "n_datasets": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }

summary = {
    "schema": "all_processbench_response_macro_v3",
    "layer": layer,
    "generator_model": generator_model,
    "datasets": datasets,
    "per_dataset": per_dataset,
    "cross_dataset_representation_compatibility": reference_compatibility,
    "macro_fold_metric_mean_legacy": macro,
    "macro_pooled_oof_test": pooled_macro,
    "note": (
        "Primary values are unweighted macro means of each dataset's final pooled OOF "
        "test metric. Fold metric means are retained only as variability diagnostics."
    ),
}
json_path = output / "aggregate_results.json"

rows = [
    "# All-ProcessBench Response Results",
    "",
    f"- Layer: `{layer}`",
    f"- Generator cohort: `{generator_model or 'all generators'}`",
    "- Primary aggregation: final pooled OOF test metric per dataset, then unweighted macro mean",
    "- Diagnostic aggregation: mean and standard deviation of individual held-out folds",
    "",
    "| dataset | final OOF AUROC | fold AUROC mean +/- std | final OOF AUPRC | final n |",
    "|---|---:|---:|---:|---:|",
]
for dataset in datasets:
    diagnostic = per_dataset[dataset]["test_aggregate"]["auroc"]
    final = per_dataset[dataset]["pooled_oof_test"]["seed_ensemble"]
    rows.append(
        f"| {dataset} | {float(final['auroc']):.4f} | "
        f"{float(diagnostic['mean']):.4f} +/- {float(diagnostic['std']):.4f} | "
        f"{float(final['aupr']):.4f} | {int(final['n'])} |"
    )
rows.extend(["", "## Macro Final Pooled OOF Test", ""])
for metric in metrics:
    value = pooled_macro[metric]["mean"]
    rows.append(f"- `{metric}`: " + ("NA" if value is None else f"{float(value):.4f}"))
summary_path = output / "summary.md"
json_temporary = json_path.with_suffix(".json.tmp")
summary_temporary = summary_path.with_suffix(".md.tmp")
json_temporary.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
summary_temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
os.replace(json_temporary, json_path)
os.replace(summary_temporary, summary_path)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("all-dataset aggregate:", json_path)
PY

printf '\nAll ProcessBench datasets complete. Summary: %s\n' "${SUMMARY_ROOT}"
