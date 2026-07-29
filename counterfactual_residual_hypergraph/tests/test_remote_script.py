from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_remote_4090.sh"
REMOTE_REQUIREMENTS = PROJECT_ROOT / "requirements-remote.txt"


def test_remote_script_is_fail_closed_and_uses_the_declared_server_paths() -> None:
    raw = SCRIPT.read_bytes()
    text = raw.decode("utf-8")

    assert raw.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r\n" not in raw
    assert "set -Eeuo pipefail" in text
    assert (
        "/share/home/tm902089733300000/a903202310/lys/"
        "models/Meta-Llama-3.1-8B-Instruct"
    ) in text
    assert (
        "/share/home/tm902089733300000/a903202310/lys/"
        "research/demo/data/hf_datasets/ProcessBench"
    ) in text
    assert "nvidia-smi" in text
    assert "torch.cuda.is_available()" in text
    assert "torch.cuda.is_bf16_supported()" in text
    assert "WITNESS" in text
    assert 'MAX_TOKENS="${MAX_TOKENS:-768}"' in text
    assert 'find -L "${MODEL_PATH}"' in text
    assert 'require_absolute_path "MODEL_PATH"' in text
    assert 'require_absolute_path "PROCESSBENCH_ROOT"' in text
    assert 'require_absolute_path "OUTPUT_ROOT"' in text
    assert "realpath" not in text
    assert 'PYTHON_CANDIDATE="$(command -v -- "${PYTHON_BIN}")"' in text
    assert 'PYTHON_BIN="${PYTHON_DIRECTORY}/$(basename -- "${PYTHON_CANDIDATE}")"' in text
    assert "model_type" in text
    assert '"llama"' in text
    assert "torch-constraint.txt" in text
    assert "-m pip check" in text
    assert 'gpu_snapshot "before-real-extraction"' in text
    assert 'TEE_PID="$!"' in text
    assert 'wait "${TEE_PID}"' in text
    assert "trap - EXIT" in text
    assert "-m pytest" in text
    assert "-m crwh synthetic" in text
    assert "-m hypergraph.attention.cct extract" in text
    assert "-m hypergraph.attention.cct inspect" in text
    assert "does not run a real paired-view CRWH ProcessBench experiment" in text
    assert 'CCT_ROOT="${DEMO_ROOT}/hypergraph/attention/cct"' in text
    assert 'CCT_ROOT="${CCT_ROOT}"' in text
    assert "cct_source_tree_sha256" in text
    assert "hypergraph/attention/cct" in text
    assert "--overwrite" not in text
    assert "bitsandbytes" not in text.lower()


def test_remote_requirements_do_not_replace_the_servers_cuda_torch() -> None:
    requirements = [
        line.strip()
        for line in REMOTE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirements
    assert not any(
        requirement.lower().split("=", 1)[0].strip() == "torch"
        for requirement in requirements
    )
    assert any(requirement.startswith("transformers") for requirement in requirements)
    assert any(requirement.startswith("pytest") for requirement in requirements)
