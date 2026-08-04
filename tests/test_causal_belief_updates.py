from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prompt_control_flow.causal_belief_update_decomposition.update_audit import (
    BeliefUpdateAuditConfig,
    run_belief_update_audit,
)
from prompt_control_flow.causal_belief_update_decomposition.update_metrics import (
    component_update_metrics,
    component_reconstruction_error,
)
from prompt_control_flow.causal_belief_update_decomposition.update_schema import (
    BELIEF_UPDATE_SCHEMA,
    BeliefUpdateTrace,
)


def test_block_component_capture_records_boundary_attention_mlp_and_delta() -> None:
    torch = pytest.importorskip("torch")
    from prompt_control_flow.causal_belief_update_decomposition.update_extraction import (
        _BlockComponentCapture,
    )

    class FakeAttention(torch.nn.Module):
        def forward(self, hidden):
            return (2.0 * hidden, None)

    class FakeMlp(torch.nn.Module):
        def forward(self, hidden):
            return 0.5 * hidden

    class FakeBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttention()
            self.mlp = FakeMlp()

        def forward(self, hidden):
            after_attention = hidden + self.self_attn(hidden)[0]
            return (after_attention + self.mlp(after_attention),)

    block = FakeBlock()
    hidden = torch.arange(1 * 3 * 2, dtype=torch.float32).reshape(1, 3, 2)
    target_indices = torch.tensor([2])

    with _BlockComponentCapture([block], target_indices) as capture:
        block(hidden)
    attention, mlp, block_delta = capture.components(0)

    boundary = hidden[:, 2]
    assert torch.allclose(attention, 2.0 * boundary)
    assert torch.allclose(mlp, 1.5 * boundary)
    assert torch.allclose(block_delta, attention + mlp)


def test_component_metrics_measure_signed_target_progress_and_error() -> None:
    writes = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    true_updates = np.asarray([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    opposite_updates = np.asarray([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])

    metrics = component_update_metrics(writes, true_updates, opposite_updates)

    assert np.allclose(metrics["target_progress"], [0.5, 0.5, 0.0])
    assert np.allclose(metrics["target_error"], [0.5, 0.5, 1.0])
    assert np.allclose(metrics["alignment_margin"], [1.0, 1.0, 0.0])


def test_component_metrics_reject_zero_target_updates() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        component_update_metrics(
            np.ones((2, 3)),
            np.zeros((2, 3)),
            np.ones((2, 3)),
        )


def test_component_metrics_broadcast_row_targets_over_components() -> None:
    writes = np.asarray(
        [
            [[1.0, 0.0], [0.5, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.5], [0.0, 0.0]],
        ]
    )
    true_updates = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    opposite_updates = np.asarray([[0.0, 1.0], [1.0, 0.0]])

    metrics = component_update_metrics(writes, true_updates, opposite_updates)

    assert np.allclose(metrics["target_progress"], [[1.0, 0.5, 0.0]] * 2)


def test_component_reconstruction_error_detects_hook_semantic_mismatch() -> None:
    attention = np.asarray([[1.0, 2.0], [2.0, 1.0]])
    mlp = np.asarray([[0.5, -0.5], [1.0, 1.0]])
    exact_block = attention + mlp
    mismatched_block = exact_block.copy()
    mismatched_block[1] += np.asarray([3.0, 4.0])

    exact_error = component_reconstruction_error(attention, mlp, exact_block)
    mismatch_error = component_reconstruction_error(attention, mlp, mismatched_block)

    assert exact_error[0] == pytest.approx(0.0)
    assert mismatch_error[1] > 0.5


def _synthetic_update_trace() -> BeliefUpdateTrace:
    rng = np.random.default_rng(211)
    pair_ids = np.repeat(np.arange(40), 2)
    branches = np.tile(np.asarray([0, 1]), 40)
    shape = (len(pair_ids), 2)
    attention_margin = 0.20 + 0.01 * rng.normal(size=shape)
    mlp_margin = 0.15 + 0.01 * rng.normal(size=shape)
    block_margin = 0.75 + 0.01 * rng.normal(size=shape)
    attention_error = 0.80 + 0.01 * rng.normal(size=shape)
    mlp_error = 0.90 + 0.01 * rng.normal(size=shape)
    block_error = 0.25 + 0.01 * rng.normal(size=shape)
    return BeliefUpdateTrace(
        row_indices=np.arange(len(pair_ids)),
        pair_ids=pair_ids,
        branches=branches,
        layers=np.asarray([8, 16]),
        attention_alignment_true=attention_margin.astype(np.float32),
        attention_alignment_opposite=np.zeros(shape, dtype=np.float32),
        mlp_alignment_true=mlp_margin.astype(np.float32),
        mlp_alignment_opposite=np.zeros(shape, dtype=np.float32),
        block_alignment_true=block_margin.astype(np.float32),
        block_alignment_opposite=np.zeros(shape, dtype=np.float32),
        attention_target_progress=np.full(shape, 0.25, dtype=np.float32),
        mlp_target_progress=np.full(shape, 0.20, dtype=np.float32),
        block_target_progress=np.full(shape, 0.85, dtype=np.float32),
        attention_target_error=attention_error.astype(np.float32),
        mlp_target_error=mlp_error.astype(np.float32),
        block_target_error=block_error.astype(np.float32),
        attention_write_norm=np.full(shape, 2.0, dtype=np.float32),
        mlp_write_norm=np.full(shape, 1.5, dtype=np.float32),
        block_write_norm=np.full(shape, 2.5, dtype=np.float32),
        reconstruction_relative_error=np.full(shape, 1e-6, dtype=np.float32),
        state_replay_relative_error=np.full(shape, 1e-6, dtype=np.float32),
        metadata={
            "schema": BELIEF_UPDATE_SCHEMA,
            "representation_gate": {"ready_for_routing_analysis": True},
        },
    )


def test_belief_update_trace_roundtrip_preserves_component_metrics(tmp_path) -> None:
    trace = _synthetic_update_trace()
    path = tmp_path / "belief_updates.npz"

    trace.save(path)
    loaded = BeliefUpdateTrace.load(path)

    assert np.array_equal(loaded.layers, np.asarray([8, 16]))
    assert np.allclose(loaded.mlp_margin_gain, trace.block_margin - trace.attention_margin)
    assert np.allclose(
        loaded.mlp_target_error_reduction,
        trace.attention_target_error - trace.block_target_error,
    )


def test_belief_update_audit_requires_preregistered_primary_layer(tmp_path) -> None:
    trace = _synthetic_update_trace()
    path = tmp_path / "belief_updates.npz"
    trace.save(path)

    with pytest.raises(ValueError, match="primary layer"):
        run_belief_update_audit(
            path,
            tmp_path / "audit",
            BeliefUpdateAuditConfig(primary_layer=12, bootstrap=20, seed=223),
        )


def test_belief_update_audit_separates_attention_and_mlp_signatures(tmp_path) -> None:
    trace = _synthetic_update_trace()
    path = tmp_path / "belief_updates.npz"
    trace.save(path)

    report = run_belief_update_audit(
        path,
        tmp_path / "audit",
        BeliefUpdateAuditConfig(
            primary_layer=16,
            bootstrap=100,
            seed=227,
            max_reconstruction_p95=1e-4,
            max_state_replay_p95=1e-4,
        ),
    )

    assert report["tests"]["attention_target_progress_above_zero"]["ci_low"] > 0.0
    assert report["tests"]["mlp_incremental_direction_gain"]["ci_low"] > 0.0
    assert report["tests"]["mlp_target_error_reduction"]["ci_low"] > 0.0
    assert report["decision_gate"]["decomposition_valid"] is True
    assert report["decision_gate"]["attention_update_signature"] is True
    assert report["decision_gate"]["mlp_update_signature"] is True
    assert report["decision_gate"]["ready_for_factorial_patching"] is True
    assert (tmp_path / "audit" / "summary.json").exists()
    assert (tmp_path / "audit" / "row_scores.csv").exists()


def test_belief_update_audit_fails_closed_on_state_replay_mismatch(tmp_path) -> None:
    trace = _synthetic_update_trace()
    trace.state_replay_relative_error[:] = 0.2
    path = tmp_path / "belief_updates.npz"
    trace.save(path)

    report = run_belief_update_audit(
        path,
        tmp_path / "audit",
        BeliefUpdateAuditConfig(
            primary_layer=16,
            bootstrap=20,
            seed=229,
            max_state_replay_p95=1e-3,
        ),
    )

    assert report["decision_gate"]["conditions"]["state_replay_within_threshold"] is False
    assert report["decision_gate"]["decomposition_valid"] is False


def test_belief_update_audit_cli_requires_explicit_primary_layer() -> None:
    from prompt_control_flow.cli.audit_causal_belief_updates import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "updates.npz", "--output_dir", "audit"])


def test_remote_runner_builds_missing_cbud_prerequisites() -> None:
    demo_root = Path(__file__).resolve().parents[1]
    script = (
        demo_root
        / "prompt_control_flow"
        / "causal_belief_update_decomposition"
        / "run_remote_pilot.sh"
    ).read_text(encoding="utf-8")

    assert 'if [[ ! -f "${ALIAS_PATH}" ]]' in script
    assert '"${PYTHON_BIN}" build_predictive_aliases.py' in script
    assert 'if [[ ! -f "${TRACE_PATH}" ]]' in script
    assert '"${PYTHON_BIN}" extract_causal_belief_states.py' in script
    assert 'if [[ ! -f "${CHARTS_PATH}" ]]' in script
    assert '"${PYTHON_BIN}" audit_causal_belief_routing.py' in script
    assert "trace is missing" not in script


def test_remote_runner_centralizes_remote_data_and_gpu_configuration() -> None:
    demo_root = Path(__file__).resolve().parents[1]
    script = (
        demo_root
        / "prompt_control_flow"
        / "causal_belief_update_decomposition"
        / "run_remote_pilot.sh"
    ).read_text(encoding="utf-8")

    assert 'GPU_ID="${GPU_ID:-0}"' in script
    assert (
        'DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/'
        'a903202310/lys/data/CBUD/finite_field_predictive_alias/'
        'llama31_8b/pilot_200}"'
    ) in script
    assert 'ALIAS_PATH="${ALIAS_PATH:-${DATA_ROOT}/source/' in script
    assert 'TRACE_PATH="${TRACE_PATH:-${DATA_ROOT}/extractions/' in script
    assert 'CHARTS_PATH="${CHARTS_PATH:-${DATA_ROOT}/derived/representation/' in script
    assert 'UPDATE_PATH="${UPDATE_PATH:-${DATA_ROOT}/extractions/' in script
    assert 'REPORT_DIR="${REPORT_DIR:-${DATA_ROOT}/results/update_audit}"' in script
    assert 'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"' in script
    assert "memory.used" not in script
    assert 'mkdir -p \\' in script


def test_remote_runner_does_not_require_pytest_at_runtime() -> None:
    demo_root = Path(__file__).resolve().parents[1]
    script = (
        demo_root
        / "prompt_control_flow"
        / "causal_belief_update_decomposition"
        / "run_remote_pilot.sh"
    ).read_text(encoding="utf-8")

    assert "CBUD runtime imports: OK" in script
    assert 'importlib.util.find_spec("pytest") is None' in script
    assert "pytest is not installed; focused unit tests skipped" in script


def test_remote_runner_uses_active_python_and_checks_model_runtime() -> None:
    demo_root = Path(__file__).resolve().parents[1]
    script = (
        demo_root
        / "prompt_control_flow"
        / "causal_belief_update_decomposition"
        / "run_remote_pilot.sh"
    ).read_text(encoding="utf-8")

    assert 'PYTHON_BIN="${PYTHON_BIN:-python}"' in script
    assert "/opt/conda/bin/python" not in script
    assert 'command -v "${PYTHON_BIN}"' not in script
    assert "RESOLVED_PYTHON" not in script
    assert "import torch" in script
    assert "import transformers" in script
