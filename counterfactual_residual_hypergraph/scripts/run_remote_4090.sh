#!/usr/bin/env bash
set -Eeuo pipefail

# One audited server smoke:
#   1. validate the declared model, ProcessBench source, Python, and free GPU;
#   2. run the CRWH tests and deterministic synthetic pipeline;
#   3. extract two real Llama/ProcessBench CCT traces and inspect them.
#
# This script does not run a real paired-view CRWH ProcessBench experiment.
# CRWH still needs a paired-view, per-head residual-write extractor before that
# claim is valid. The real-model CCT stage is an extractor/GPU compatibility
# witness, not a CRWH result.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CRWH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEMO_ROOT="${DEMO_ROOT:-$(cd -- "${CRWH_ROOT}/.." && pwd)}"

MODEL_PATH="${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
PROCESSBENCH_ROOT="${PROCESSBENCH_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/demo/data/hf_datasets/ProcessBench}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:-0}"
PROFILE="${PROFILE:-smoke}"
SUBSET="${SUBSET:-gsm8k}"
LAYER="${LAYER:-14}"
MAX_TOKENS="${MAX_TOKENS:-768}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
MIN_GPU_TOTAL_MIB="${MIN_GPU_TOTAL_MIB:-22000}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-22000}"
MIN_DISK_GIB="${MIN_DISK_GIB:-5}"

require_absolute_path() {
  local variable_name="$1"
  local value="${!variable_name}"
  if [[ "${value}" != /* ]]; then
    echo "${variable_name} must be an absolute path, got: ${value}" >&2
    exit 2
  fi
}

case "${PROFILE}" in
  smoke)
    LIMIT="${LIMIT:-2}"
    TOP_SOURCES="${TOP_SOURCES:-1}"
    NODE_DIM="${NODE_DIM:-32}"
    ;;
  pilot)
    LIMIT="${LIMIT:-8}"
    TOP_SOURCES="${TOP_SOURCES:-2}"
    NODE_DIM="${NODE_DIM:-64}"
    ;;
  *)
    echo "PROFILE must be 'smoke' or 'pilot', got: ${PROFILE}" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CRWH_ROOT}/outputs/remote_4090}"
require_absolute_path "DEMO_ROOT"
require_absolute_path "MODEL_PATH"
require_absolute_path "PROCESSBENCH_ROOT"
require_absolute_path "OUTPUT_ROOT"
CCT_ROOT="${DEMO_ROOT}/hypergraph/attention/cct"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd -- "${OUTPUT_ROOT}" && pwd -P)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}_${PROFILE}_${SUBSET}_l${LAYER}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "Exactly one preallocated CUDA device is required; got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
  fi
  GPU_SELECTOR="${CUDA_VISIBLE_DEVICES}"
  GPU_SOURCE="preexisting CUDA_VISIBLE_DEVICES"
else
  GPU_SELECTOR="${GPU}"
  GPU_SOURCE="GPU"
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

LOG_FILE="${RUN_DIR}/run.log"
CURRENT_STAGE="bootstrap"
START_EPOCH="$(date +%s)"

finish() {
  local exit_code=$?
  local end_epoch
  local elapsed
  local success_tmp="${RUN_DIR}/._SUCCESS.$$"
  local failed_tmp="${RUN_DIR}/._FAILED.$$"
  trap - EXIT
  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - START_EPOCH))
  printf "[finalizing] stage=%s exit_code=%s elapsed_seconds=%s\n" \
    "${CURRENT_STAGE}" "${exit_code}" "${elapsed}"

  exec 1>&3 2>&4
  if ! wait "${TEE_PID}"; then
    printf "Log writer failed: %s\n" "${LOG_FILE}" >&2
    exit_code=1
  fi

  if [[ ${exit_code} -eq 0 ]] &&
    ! { printf "success\n" > "${success_tmp}" &&
      mv -- "${success_tmp}" "${RUN_DIR}/_SUCCESS"; }; then
    printf "Could not write the success marker in %s\n" "${RUN_DIR}" >&2
    exit_code=1
  fi

  if [[ ${exit_code} -ne 0 ]]; then
    if ! {
      printf "status=failed\n"
      printf "exit_code=%s\n" "${exit_code}"
      printf "failed_stage=%s\n" "${CURRENT_STAGE}"
      printf "elapsed_seconds=%s\n" "${elapsed}"
    } > "${failed_tmp}" ||
      ! mv -- "${failed_tmp}" "${RUN_DIR}/_FAILED"; then
      printf "Could not write the failure marker in %s\n" "${RUN_DIR}" >&2
      exit_code=1
    fi
    printf "[done] status=failed stage=%s exit_code=%s elapsed_seconds=%s\n" \
      "${CURRENT_STAGE}" "${exit_code}" "${elapsed}" >&2
  else
    printf "[done] status=success elapsed_seconds=%s\n" "${elapsed}"
  fi
  exit "${exit_code}"
}

exec 3>&1 4>&2
exec > >(tee -a "${LOG_FILE}" >&3) 2>&1
TEE_PID="$!"
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

stage() {
  CURRENT_STAGE="$1"
  printf "\n===== %s =====\n" "${CURRENT_STAGE}"
}

stage "configuration"
cat <<EOF
CRWH_ROOT=${CRWH_ROOT}
DEMO_ROOT=${DEMO_ROOT}
CCT_ROOT=${CCT_ROOT}
MODEL_PATH=${MODEL_PATH}
PROCESSBENCH_ROOT=${PROCESSBENCH_ROOT}
GPU_REQUEST=${GPU}
GPU_SELECTOR=${GPU_SELECTOR}
GPU_SOURCE=${GPU_SOURCE}
PROFILE=${PROFILE}
SUBSET=${SUBSET}
LAYER=${LAYER}
LIMIT=${LIMIT}
TOP_SOURCES=${TOP_SOURCES}
NODE_DIM=${NODE_DIM}
MAX_TOKENS=${MAX_TOKENS}
MIN_GPU_TOTAL_MIB=${MIN_GPU_TOTAL_MIB}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB}
MIN_DISK_GIB=${MIN_DISK_GIB}
RUN_DIR=${RUN_DIR}
EOF

stage "path_preflight"
[[ -d "${CRWH_ROOT}/src/crwh" ]] || {
  echo "CRWH source directory is missing: ${CRWH_ROOT}/src/crwh" >&2
  exit 2
}
[[ -d "${CCT_ROOT}" ]] || {
  echo "CCT source directory is missing: ${CCT_ROOT}" >&2
  exit 2
}
[[ -d "${MODEL_PATH}" ]] || {
  echo "Model directory is missing: ${MODEL_PATH}" >&2
  exit 2
}
[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "Model config is missing: ${MODEL_PATH}/config.json" >&2
  exit 2
}
find -L "${MODEL_PATH}" -maxdepth 1 -type f -name '*.safetensors' -print -quit |
  grep -q . || {
    echo "No model safetensors were found in ${MODEL_PATH}" >&2
    exit 2
  }
[[ -d "${PROCESSBENCH_ROOT}" ]] || {
  echo "ProcessBench directory is missing: ${PROCESSBENCH_ROOT}" >&2
  exit 2
}
[[ -w "${RUN_DIR}" ]] || {
  echo "Run directory is not writable: ${RUN_DIR}" >&2
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
  echo "Available top-level files:" >&2
  find "${PROCESSBENCH_ROOT}" -maxdepth 1 -type f -printf '  %f\n' >&2
  exit 2
}
printf "PB_INPUT=%s\n" "${PB_INPUT}"
df -h "${RUN_DIR}"
DISK_AVAILABLE_KIB="$(df -Pk "${RUN_DIR}" | awk 'END {print $4}')"
if [[ ! "${DISK_AVAILABLE_KIB}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine free disk space for ${RUN_DIR}" >&2
  exit 2
fi
if ((DISK_AVAILABLE_KIB < MIN_DISK_GIB * 1024 * 1024)); then
  echo "At least ${MIN_DISK_GIB} GiB free disk is required for run artifacts." >&2
  exit 2
fi

stage "python_preflight"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
}
PYTHON_CANDIDATE="$(command -v -- "${PYTHON_BIN}")"
PYTHON_DIRECTORY="$(
  cd -- "$(dirname -- "${PYTHON_CANDIDATE}")" && pwd -P
)"
PYTHON_BIN="${PYTHON_DIRECTORY}/$(basename -- "${PYTHON_CANDIDATE}")"
printf "PYTHON_BIN_RESOLVED=%s\n" "${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import sys

print("python", sys.version)
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required")
PY

stage "model_contract"
MODEL_PATH="${MODEL_PATH}" \
RUN_DIR="${RUN_DIR}" \
LAYER="${LAYER}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

model_path = Path(os.environ["MODEL_PATH"])
config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
if config.get("model_type") != "llama":
    raise SystemExit(
        f"Expected a Llama model, got model_type={config.get('model_type')!r}"
    )
layer = int(os.environ["LAYER"])
num_layers = int(config.get("num_hidden_layers", 0))
if not 0 <= layer < num_layers:
    raise SystemExit(f"Layer {layer} is invalid for a {num_layers}-layer model")
manifest = {
    "model_path": str(model_path),
    "model_type": config["model_type"],
    "architectures": config.get("architectures", []),
    "num_hidden_layers": num_layers,
    "hidden_size": config.get("hidden_size"),
    "num_attention_heads": config.get("num_attention_heads"),
    "num_key_value_heads": config.get("num_key_value_heads"),
    "selected_layer": layer,
}
destination = Path(os.environ["RUN_DIR"]) / "model-contract.json"
destination.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

gpu_snapshot() {
  local label="$1"
  nvidia-smi \
    --id="${GPU_SELECTOR}" \
    --query-gpu=index,uuid,name,driver_version,memory.used,memory.free,memory.total \
    --format=csv,noheader |
    tee "${RUN_DIR}/gpu-${label}.csv"
  GPU_USED_MIB="$(
    nvidia-smi --id="${GPU_SELECTOR}" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  GPU_FREE_MIB="$(
    nvidia-smi --id="${GPU_SELECTOR}" --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  GPU_TOTAL_MIB="$(
    nvidia-smi --id="${GPU_SELECTOR}" --query-gpu=memory.total \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  for value in "${GPU_USED_MIB}" "${GPU_FREE_MIB}" "${GPU_TOTAL_MIB}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
      echo "Could not parse nvidia-smi memory values for ${GPU_SELECTOR}" >&2
      exit 2
    fi
  done
  if ((GPU_TOTAL_MIB < MIN_GPU_TOTAL_MIB)); then
    echo "GPU has ${GPU_TOTAL_MIB} MiB total; need at least ${MIN_GPU_TOTAL_MIB}." >&2
    exit 2
  fi
  if [[ "${ALLOW_BUSY_GPU}" != "1" ]] &&
    ((GPU_USED_MIB >= 500 || GPU_FREE_MIB < MIN_GPU_FREE_MIB)); then
    echo "GPU ${GPU_SELECTOR}: used=${GPU_USED_MIB} MiB, free=${GPU_FREE_MIB} MiB." >&2
    echo "Wait for a free GPU or rerun explicitly with ALLOW_BUSY_GPU=1." >&2
    exit 2
  fi
}

stage "gpu_preflight"
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is unavailable" >&2
  exit 2
}
gpu_snapshot "initial"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="${CRWH_ROOT}/src:${DEMO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

stage "cuda_torch_contract"
"${PYTHON_BIN}" - <<'PY'
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("The active torch installation cannot use CUDA")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Expected one visible GPU after CUDA_VISIBLE_DEVICES, got "
        f"{torch.cuda.device_count()}"
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The visible GPU/torch build does not support BF16")
free_bytes, total_bytes = torch.cuda.mem_get_info()
print("cuda_free_mib", free_bytes // (1024 * 1024))
print("cuda_total_mib", total_bytes // (1024 * 1024))
torch.manual_seed(0)
x = torch.randn(32, 32, device="cuda")
y = x @ x
print("WITNESS", tuple(y.shape), torch.cuda.get_device_name(0), float(y[0, 0]))
PY
TORCH_VERSION_BEFORE="$(
  "${PYTHON_BIN}" -c 'import torch; print(torch.__version__)'
)"
TORCH_CONSTRAINT="${RUN_DIR}/torch-constraint.txt"
printf "torch==%s\n" "${TORCH_VERSION_BEFORE}" > "${TORCH_CONSTRAINT}"

stage "dependency_setup"
if [[ "${INSTALL_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install \
    --constraint "${TORCH_CONSTRAINT}" \
    -r "${CRWH_ROOT}/requirements-remote.txt"
  "${PYTHON_BIN}" -m pip install -e "${CRWH_ROOT}" --no-deps
  "${PYTHON_BIN}" -m pip check
else
  echo "INSTALL_DEPS=0: reusing the active Python environment."
  echo "Use INSTALL_DEPS=1 if the non-Torch dependencies are missing."
fi

EXPECTED_TORCH_VERSION="${TORCH_VERSION_BEFORE}" "${PYTHON_BIN}" - <<'PY'
import os

import numpy
import pytest
import torch
import transformers

print("numpy", numpy.__version__)
print("pytest", pytest.__version__)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.__version__ != os.environ["EXPECTED_TORCH_VERSION"]:
    raise SystemExit(
        "Dependency setup changed torch from "
        f"{os.environ['EXPECTED_TORCH_VERSION']} to {torch.__version__}"
    )
if not torch.cuda.is_available():
    raise SystemExit("The active torch installation cannot use CUDA")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Expected one visible GPU after CUDA_VISIBLE_DEVICES, got "
        f"{torch.cuda.device_count()}"
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 support disappeared after dependency setup")
PY
"${PYTHON_BIN}" -m pip freeze > "${RUN_DIR}/pip-freeze.txt"

stage "source_witness"
CRWH_ROOT="${CRWH_ROOT}" \
CCT_ROOT="${CCT_ROOT}" \
RUN_DIR="${RUN_DIR}" \
"${PYTHON_BIN}" - <<'PY'
import hashlib
import os
from pathlib import Path

crwh_root = Path(os.environ["CRWH_ROOT"])
cct_root = Path(os.environ["CCT_ROOT"])
run_dir = Path(os.environ["RUN_DIR"])
crwh_paths = []
for relative in ("src", "tests", "configs", "scripts"):
    base = crwh_root / relative
    if base.exists():
        crwh_paths.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
for name in (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-remote.txt",
):
    path = crwh_root / name
    if path.is_file():
        crwh_paths.append(path)

def tree_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


cct_paths = [
    path
    for path in cct_root.rglob("*")
    if path.is_file()
    and "__pycache__" not in path.parts
    and path.suffix != ".pyc"
]
crwh_digest = tree_digest(crwh_root, crwh_paths)
cct_digest = tree_digest(cct_root, cct_paths)
combined = hashlib.sha256(
    f"crwh:{crwh_digest}\ncct:{cct_digest}\n".encode("ascii")
).hexdigest()
(run_dir / "crwh-source-tree.sha256").write_text(
    crwh_digest + "\n", encoding="utf-8"
)
(run_dir / "cct-source-tree.sha256").write_text(cct_digest + "\n", encoding="utf-8")
(run_dir / "source-tree.sha256").write_text(
    combined + "\n", encoding="utf-8"
)
print("crwh_source_tree_sha256", crwh_digest)
print("cct_source_tree_sha256", cct_digest)
print("combined_source_tree_sha256", combined)
PY
if command -v git >/dev/null 2>&1 &&
  git -c safe.directory="${DEMO_ROOT}" -C "${DEMO_ROOT}" rev-parse HEAD \
    > "${RUN_DIR}/git-head.txt" 2>/dev/null; then
  git -c safe.directory="${DEMO_ROOT}" -C "${DEMO_ROOT}" \
    status --short -- counterfactual_residual_hypergraph hypergraph/attention/cct \
    > "${RUN_DIR}/git-status.txt"
else
  echo "Git witness unavailable; source-tree.sha256 remains authoritative." |
    tee "${RUN_DIR}/git-status.txt"
fi

stage "crwh_tests"
"${PYTHON_BIN}" -m pytest -q \
  -p no:cacheprovider \
  --basetemp "${RUN_DIR}/pytest-tmp" \
  "${CRWH_ROOT}/tests"

stage "crwh_synthetic_cpu"
SYNTHETIC_DIR="${RUN_DIR}/crwh-synthetic"
"${PYTHON_BIN}" -m crwh synthetic \
  --output-dir "${SYNTHETIC_DIR}" \
  --examples 16 \
  --epochs 3 \
  --seed 42
SYNTHETIC_DIR="${SYNTHETIC_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SYNTHETIC_DIR"])
required = (
    "config.json",
    "metrics.json",
    "scores.json",
    "review_queue.json",
)
for name in required:
    path = root / name
    if not path.is_file():
        raise SystemExit(f"Missing synthetic artifact: {path}")
    json.loads(path.read_text(encoding="utf-8"))
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
if metrics.get("finite_scores") is not True:
    raise SystemExit("Synthetic pipeline produced non-finite scores")
print("synthetic_metrics", json.dumps(metrics, sort_keys=True))
PY

stage "select_short_processbench_records"
SELECTED_INPUT="${RUN_DIR}/${SUBSET}_selected_${LIMIT}.jsonl"
PB_INPUT="${PB_INPUT}" \
MODEL_PATH="${MODEL_PATH}" \
SELECTED_INPUT="${SELECTED_INPUT}" \
LIMIT="${LIMIT}" \
MAX_TOKENS="${MAX_TOKENS}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os

from transformers import AutoTokenizer

from hypergraph.attention.cct.processbench import (
    PlainReasoningRenderer,
    ProcessBenchReader,
    TokenizerAligner,
)

source = os.environ["PB_INPUT"]
destination = os.environ["SELECTED_INPUT"]
limit = int(os.environ["LIMIT"])
max_tokens = int(os.environ["MAX_TOKENS"])
tokenizer = AutoTokenizer.from_pretrained(
    os.environ["MODEL_PATH"],
    use_fast=True,
    local_files_only=True,
)
renderer = PlainReasoningRenderer()
aligner = TokenizerAligner()
selected = []
for record in ProcessBenchReader(source).records():
    tokenized = aligner.tokenize(tokenizer, renderer.render(record))
    token_count = len(tokenized.input_ids)
    if token_count > max_tokens:
        continue
    selected.append(
        {
            "id": record.trace_id,
            "problem_id": record.problem_id,
            "problem": record.question,
            "steps": list(record.steps),
            "label": record.labels.first_error,
            "generator": record.generator_model,
            "_token_count": token_count,
        }
    )
    print(
        "selected",
        record.trace_id,
        "tokens",
        token_count,
        "steps",
        len(record.steps),
    )
    if len(selected) == limit:
        break
if len(selected) != limit:
    raise SystemExit(
        f"Only found {len(selected)} records with <= {max_tokens} tokens; "
        f"need {limit}"
    )
with open(destination, "w", encoding="utf-8") as stream:
    for row in selected:
        row.pop("_token_count")
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print("selected_input", destination)
PY

stage "real_llama_cct_extraction"
gpu_snapshot "before-real-extraction"
CCT_TRACES="${RUN_DIR}/cct-traces"
cd "${DEMO_ROOT}"
"${PYTHON_BIN}" -m hypergraph.attention.cct extract \
  --input "${SELECTED_INPUT}" \
  --model "${MODEL_PATH}" \
  --output "${CCT_TRACES}" \
  --layer "${LAYER}" \
  --top-sources "${TOP_SOURCES}" \
  --node-dim "${NODE_DIM}" \
  --projection-seed 17 \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --device cuda \
  --skip-invalid

TRACE_COUNT="$(
  find "${CCT_TRACES}" -maxdepth 1 -type f -name '*.npz' |
    wc -l |
    tr -d '[:space:]'
)"
if ((TRACE_COUNT != LIMIT)); then
  echo "Expected ${LIMIT} real traces, found ${TRACE_COUNT}." >&2
  cat "${CCT_TRACES}"/extraction_failures_*.json >&2
  exit 1
fi
"${PYTHON_BIN}" -m hypergraph.attention.cct inspect --traces "${CCT_TRACES}" |
  tee "${RUN_DIR}/cct-inspect.json"
cp "${CCT_TRACES}"/extraction_config_*.json "${RUN_DIR}/"
cp "${CCT_TRACES}"/extraction_failures_*.json "${RUN_DIR}/"

stage "summary"
RUN_DIR="${RUN_DIR}" \
MODEL_PATH="${MODEL_PATH}" \
PB_INPUT="${PB_INPUT}" \
PROFILE="${PROFILE}" \
TRACE_COUNT="${TRACE_COUNT}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
summary = {
    "status": "success",
    "scope": [
        "CRWH tests",
        "CRWH synthetic CPU smoke",
        "real Llama ProcessBench CCT extraction smoke",
    ],
    "not_claimed": "real paired-view CRWH ProcessBench experiment",
    "profile": os.environ["PROFILE"],
    "model_path": os.environ["MODEL_PATH"],
    "processbench_input": os.environ["PB_INPUT"],
    "real_cct_traces": int(os.environ["TRACE_COUNT"]),
    "run_dir": str(run_dir),
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo
echo "All supported smoke stages passed."
echo "Artifacts: ${RUN_DIR}"
echo "Scientific boundary: no real paired-view CRWH ProcessBench result was run."
