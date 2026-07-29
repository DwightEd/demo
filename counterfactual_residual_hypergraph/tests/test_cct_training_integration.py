from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT.parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from hypergraph.attention.cct.cli import main as cct_main
from hypergraph.attention.cct.contracts import CausalHypergraph, FirstErrorLabels
from hypergraph.attention.cct.data import CausalTrace, TraceRepository


def _balanced_trace(index: int) -> CausalTrace:
    has_error = index % 2
    signal = 1.0 if has_error else -1.0
    node_features = np.zeros((3, 9), dtype=np.float32)
    node_features[:, 0] = signal
    node_features[:, 1] = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    node_features[2, 2:8] = np.asarray(
        [0.8, 0.2, signal, 0.1, 0.05, 1.0],
        dtype=np.float32,
    )
    node_features[2, -1] = 1.0
    return CausalTrace(
        trace_id=f"integration-trace-{index:03d}",
        problem_id=f"integration-problem-{index:03d}",
        generator_model="integration-generator",
        observer_model="integration-observer",
        layer_id=1,
        prompt_tokens=2,
        response_tokens=1,
        graph=CausalHypergraph(
            node_features=node_features,
            incidence=np.asarray([[0, 2], [0, 0]], dtype=np.int64),
            receivers=np.asarray([2], dtype=np.int64),
            edge_features=np.asarray(
                [[signal, 1.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            edge_kind=np.asarray(["pair"]),
            response_nodes=np.asarray([2], dtype=np.int64),
        ),
        labels=FirstErrorLabels(
            num_steps=1,
            first_error=0 if has_error else -1,
        ),
    )


def test_trace_repository_contains_unsafe_trace_ids_inside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "traces"
    repository = TraceRepository(root)
    trace = replace(_balanced_trace(0), trace_id="../escape")

    artifact = repository.save(trace)

    assert artifact.is_file()
    assert artifact.resolve().is_relative_to(root.resolve())
    assert ".." not in artifact.name
    assert "/" not in artifact.name
    assert "\\" not in artifact.name
    assert "escape" not in artifact.name


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("trace_id", " trace-id"),
        ("trace_id", "trace-id "),
        ("trace_id", "trace\x00id"),
        ("problem_id", " problem-id"),
        ("problem_id", "problem-id "),
        ("problem_id", "problem\x00id"),
    ),
)
def test_causal_trace_rejects_unsafe_identifiers(
    field: str,
    invalid_value: str,
) -> None:
    with pytest.raises(ValueError):
        replace(_balanced_trace(0), **{field: invalid_value})


def test_cct_train_cli_writes_auditable_held_out_results(tmp_path: Path) -> None:
    traces_dir = tmp_path / "traces"
    repository = TraceRepository(traces_dir)
    for index in range(24):
        repository.save(_balanced_trace(index))

    output_dir = tmp_path / "training"
    source_tree_sha256 = "ab" * 32
    source_witness = tmp_path / "source-tree.sha256"
    source_witness.write_text(source_tree_sha256 + "\n", encoding="ascii")
    exit_code = cct_main(
        [
            "train",
            "--traces",
            str(traces_dir),
            "--output",
            str(output_dir),
            "--hidden-dim",
            "4",
            "--model-layers",
            "1",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--batch-size",
            "8",
            "--seed",
            "17",
            "--split-seed",
            "17",
            "--validation-ratio",
            "0.2",
            "--test-ratio",
            "0.2",
            "--device",
            "cpu",
            "--bootstrap-replicates",
            "8",
            "--bootstrap-confidence",
            "0.8",
            "--source-tree-sha256-file",
            str(source_witness),
        ]
    )

    assert exit_code == 0
    expected_artifacts = (
        "metrics.json",
        "model.pt",
        "model.safetensors",
        "checkpoint.json",
        "normalizer.npz",
        "predictions_validation.csv",
        "predictions_test.csv",
        "split.json",
    )
    for name in expected_artifacts:
        artifact = output_dir / name
        assert artifact.is_file(), name
        assert artifact.stat().st_size > 0, name

    metrics = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    for partition in ("validation", "test"):
        response = metrics[partition]["response"]
        assert response["auroc"] is not None
        assert response["aupr"] is not None
        assert 0 < response["positives"] < response["n"]

    split = json.loads((output_dir / "split.json").read_text(encoding="utf-8"))
    for partition in ("validation", "test"):
        balance = split["partition_balance"][partition]
        assert balance["positive_traces"] > 0
        assert balance["negative_traces"] > 0

    checkpoint = json.loads(
        (output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["schema_version"] == 1
    assert checkpoint["model"] == {
        "node_dim": 9,
        "edge_dim": 4,
        "hidden_dim": 4,
        "num_layers": 1,
    }
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        checkpoint["trace_cohort_sha256"],
    )
    assert checkpoint["split_sha256"] == hashlib.sha256(
        (output_dir / "split.json").read_bytes()
    ).hexdigest()
    assert checkpoint["source_tree_sha256"] == source_tree_sha256

    from hypergraph.attention.cct.training import load_training_checkpoint

    model, normalizer = load_training_checkpoint(output_dir, device="cpu")
    fixed_trace = normalizer.transform(_balanced_trace(100))
    model.eval()
    import torch

    with torch.no_grad():
        logits = model(fixed_trace.graph)
    assert tuple(logits.shape) == (1,)
    assert bool(torch.isfinite(logits).all())
