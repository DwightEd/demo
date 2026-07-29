from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_remote_4090.sh"
REMOTE_REQUIREMENTS = PROJECT_ROOT / "requirements-remote.txt"
REAL_CCT_AUDIT = PROJECT_ROOT / "src" / "crwh" / "real_cct_audit.py"


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


def test_remote_script_defaults_to_an_evaluable_smoke_cohort() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    smoke_block = text.partition("  smoke)")[2].partition("    ;;")[0]

    assert smoke_block
    assert 'LIMIT="${LIMIT:-24}"' in smoke_block
    assert 'TOP_SOURCES="${TOP_SOURCES:-2}"' in smoke_block


def test_remote_script_trains_and_audits_binary_holdout_partitions() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    audit_text = REAL_CCT_AUDIT.read_text(encoding="utf-8")

    assert "-m hypergraph.attention.cct train" in text
    assert "response_class_counts" in audit_text
    assert "validation" in audit_text
    assert "test" in audit_text
    assert "both response classes" in audit_text


def test_remote_script_validates_real_baseline_artifacts_and_scope() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    audit_text = REAL_CCT_AUDIT.read_text(encoding="utf-8")

    for artifact in (
        "metrics.json",
        "model.pt",
        "model.safetensors",
        "checkpoint.json",
        "normalizer.npz",
        "predictions_validation.csv",
        "predictions_test.csv",
        "split.json",
    ):
        assert artifact in audit_text
    assert (
        '"not_claimed": "real paired-view CRWH ProcessBench experiment"'
        in text
    )


def test_remote_script_delegates_real_business_audits_to_a_testable_module() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert text.count("-m crwh.real_cct_audit") >= 2


def test_real_cct_audit_classifies_hg_pair_and_no_edge_results() -> None:
    from crwh.real_cct_audit import classify_result_kind

    assert (
        classify_result_kind(
            {"hyper": 4, "pair": 2},
            hypergraph_gate_passed=True,
        )
        == "real_processbench_cct_hg_baseline"
    )
    assert (
        classify_result_kind(
            {"pair": 3},
            hypergraph_gate_passed=False,
        )
        == "real_processbench_cct_pair_graph_witness"
    )
    assert (
        classify_result_kind({}, hypergraph_gate_passed=False)
        == "real_processbench_cct_no_edge_witness"
    )


def test_remote_script_configures_profile_specific_hypergraph_gates() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    smoke_block = text.partition("  smoke)")[2].partition("    ;;")[0]
    pilot_block = text.partition("  pilot)")[2].partition("    ;;")[0]

    assert (
        'MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE="${'
        'MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE:-0.25}"'
    ) in smoke_block
    assert 'MIN_TOTAL_HYPEREDGES="${MIN_TOTAL_HYPEREDGES:-4}"' in smoke_block
    assert (
        'MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE="${'
        'MIN_TRAIN_HYPEREDGE_TRACE_COVERAGE:-0.25}"'
    ) in pilot_block
    assert 'MIN_TOTAL_HYPEREDGES="${MIN_TOTAL_HYPEREDGES:-12}"' in pilot_block
    assert "--min-train-hyperedge-trace-coverage" in text
    assert "--min-total-hyperedges" in text


def test_source_witness_covers_shared_split_and_evaluation_code() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    witness_block = text.partition('stage "source_witness"')[2].partition(
        'stage "crwh_tests"'
    )[0]

    assert "splitting.py" in witness_block
    assert "evaluation.py" in witness_block


def test_real_training_checkpoint_is_bound_to_the_source_witness() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    training_block = text.partition(
        'stage "real_processbench_cct_graph_training"'
    )[2].partition('stage "real_training_artifact_audit"')[0]

    assert (
        '--source-tree-sha256-file "${RUN_DIR}/source-tree.sha256"'
        in training_block
    )
