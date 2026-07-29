from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT.parent
for import_root in (PROJECT_ROOT / "src", DEMO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from crwh.real_cct_audit import audit_training_artifacts
from hypergraph.attention.cct.cli import main as cct_main
from hypergraph.attention.cct.contracts import CausalHypergraph, FirstErrorLabels
from hypergraph.attention.cct.data import CausalTrace, TraceRepository
from hypergraph.attention.cct.training import (
    trace_cohort_sha256,
    trace_sidecar_sha256,
)
from hypergraph.attention.splitting import FixedHoldoutConfig, FixedHoldoutSplitter


@dataclass(frozen=True)
class _AuditFixture:
    training_dir: Path
    traces_dir: Path
    preflight_path: Path


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
        trace_id=f"audit-trace-{index:03d}",
        problem_id=f"audit-problem-{index:03d}",
        generator_model="audit-generator",
        observer_model="audit-observer",
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


def _write_preflight(traces_dir: Path, destination: Path) -> None:
    traces = list(TraceRepository(traces_dir).traces())
    split = FixedHoldoutSplitter(
        FixedHoldoutConfig(
            seed=17,
            validation_ratio=0.2,
            test_ratio=0.2,
        )
    ).split([trace.split_record() for trace in traces])
    partitions: dict[str, object] = {}
    for partition in (split.train, split.validation, split.test):
        labels = [traces[index].response_label for index in partition.indices]
        partitions[partition.name] = {
            "traces": len(labels),
            "response_class_counts": {
                str(label): sum(value == label for value in labels)
                for label in (0, 1)
            },
            "problem_ids": list(partition.group_ids),
            "trace_ids": list(partition.trace_ids),
        }
    destination.write_text(
        json.dumps(
            {
                "split_seed": 17,
                "validation_ratio": 0.2,
                "test_ratio": 0.2,
                "source_provenance": {
                    "trace_cohort_sha256": trace_cohort_sha256(traces_dir),
                    "trace_sidecar_sha256": trace_sidecar_sha256(traces_dir),
                },
                "partitions": partitions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def audited_training(tmp_path_factory: pytest.TempPathFactory) -> _AuditFixture:
    root = tmp_path_factory.mktemp("real-cct-artifact-audit")
    traces_dir = root / "traces"
    repository = TraceRepository(traces_dir)
    for index in range(24):
        repository.save(_balanced_trace(index))
    (traces_dir / "extraction_config_test.json").write_text(
        json.dumps(
            {
                "model": "audit-observer",
                "layer": 1,
                "projection_seed": 17,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    preflight_path = root / "cct-split-preflight.json"
    _write_preflight(traces_dir, preflight_path)
    training_dir = root / "training"
    exit_code = cct_main(
        [
            "train",
            "--traces",
            str(traces_dir),
            "--output",
            str(training_dir),
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
        ]
    )
    assert exit_code == 0
    return _AuditFixture(training_dir, traces_dir, preflight_path)


def _training_copy(
    tmp_path: Path,
    fixture: _AuditFixture,
) -> Path:
    destination = tmp_path / "training"
    shutil.copytree(fixture.training_dir, destination)
    return destination


def _prediction_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_prediction_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_real_artifact_audit_accepts_consistent_outputs(
    audited_training: _AuditFixture,
) -> None:
    summary = audit_training_artifacts(
        audited_training.training_dir,
        audited_training.traces_dir,
        audited_training.preflight_path,
    )

    assert summary["best_epoch"] in (1, 2)
    assert set(summary["partitions"]) == {"validation", "test"}
    for partition in ("validation", "test"):
        response = summary["partitions"][partition]["response"]
        assert response["auroc"] is not None
        assert response["aupr"] is not None
        assert summary["partitions"][partition]["prediction_rows"] == response["n"]


def test_real_artifact_audit_rejects_unknown_validation_trace_id(
    tmp_path: Path,
    audited_training: _AuditFixture,
) -> None:
    training_dir = _training_copy(tmp_path, audited_training)
    predictions = training_dir / "predictions_validation.csv"
    fieldnames, rows = _prediction_rows(predictions)
    rows[0]["trace_id"] = "unknown-validation-trace"
    _write_prediction_rows(predictions, fieldnames, rows)

    with pytest.raises(ValueError):
        audit_training_artifacts(
            training_dir,
            audited_training.traces_dir,
            audited_training.preflight_path,
        )


def test_real_artifact_audit_rejects_duplicate_validation_trace_id(
    tmp_path: Path,
    audited_training: _AuditFixture,
) -> None:
    training_dir = _training_copy(tmp_path, audited_training)
    predictions = training_dir / "predictions_validation.csv"
    fieldnames, rows = _prediction_rows(predictions)
    assert len(rows) >= 2
    rows[1]["trace_id"] = rows[0]["trace_id"]
    _write_prediction_rows(predictions, fieldnames, rows)

    with pytest.raises(ValueError):
        audit_training_artifacts(
            training_dir,
            audited_training.traces_dir,
            audited_training.preflight_path,
        )


def test_real_artifact_audit_rejects_validation_label_tampering(
    tmp_path: Path,
    audited_training: _AuditFixture,
) -> None:
    training_dir = _training_copy(tmp_path, audited_training)
    predictions = training_dir / "predictions_validation.csv"
    fieldnames, rows = _prediction_rows(predictions)
    rows[0]["label"] = str(1 - int(rows[0]["label"]))
    _write_prediction_rows(predictions, fieldnames, rows)

    with pytest.raises(ValueError):
        audit_training_artifacts(
            training_dir,
            audited_training.traces_dir,
            audited_training.preflight_path,
        )


def test_real_artifact_audit_rejects_split_assignment_preflight_mismatch(
    tmp_path: Path,
    audited_training: _AuditFixture,
) -> None:
    training_dir = _training_copy(tmp_path, audited_training)
    split_path = training_dir / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation = split["partition_trace_ids"]["validation"]
    test = split["partition_trace_ids"]["test"]
    validation[0], test[0] = test[0], validation[0]
    split_path.write_text(
        json.dumps(split, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        audit_training_artifacts(
            training_dir,
            audited_training.traces_dir,
            audited_training.preflight_path,
        )


@pytest.mark.parametrize(
    ("section", "metric", "replacement"),
    (
        ("response", "auroc", 0.314159),
        ("response", "brier", 0.271828),
        ("response", "ece", 0.161803),
        ("localization", "mean_rank", 1.75),
    ),
)
def test_real_artifact_audit_rejects_metric_tampering(
    tmp_path: Path,
    audited_training: _AuditFixture,
    section: str,
    metric: str,
    replacement: float,
) -> None:
    training_dir = _training_copy(tmp_path, audited_training)
    metrics_path = training_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    original = metrics["validation"][section][metric]
    assert original != replacement
    metrics["validation"][section][metric] = replacement
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        audit_training_artifacts(
            training_dir,
            audited_training.traces_dir,
            audited_training.preflight_path,
        )


def test_real_artifact_audit_rejects_predicted_step_tampering(
    tmp_path: Path,
    audited_training: _AuditFixture,
) -> None:
    training_dir = _training_copy(tmp_path, audited_training)
    predictions = training_dir / "predictions_validation.csv"
    fieldnames, rows = _prediction_rows(predictions)
    assert rows[0]["predicted_step"] == "0"
    rows[0]["predicted_step"] = "-1"
    _write_prediction_rows(predictions, fieldnames, rows)

    with pytest.raises(ValueError):
        audit_training_artifacts(
            training_dir,
            audited_training.traces_dir,
            audited_training.preflight_path,
        )


def test_real_artifact_audit_rejects_preflight_source_provenance_tampering(
    tmp_path: Path,
    audited_training: _AuditFixture,
) -> None:
    preflight_path = tmp_path / "cct-split-preflight.json"
    preflight = json.loads(
        audited_training.preflight_path.read_text(encoding="utf-8")
    )
    provenance = preflight["source_provenance"]
    assert provenance["trace_cohort_sha256"] != "0" * 64
    provenance["trace_cohort_sha256"] = "0" * 64
    preflight_path.write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        audit_training_artifacts(
            audited_training.training_dir,
            audited_training.traces_dir,
            preflight_path,
        )


def test_real_artifact_audit_rejects_checkpoint_source_provenance_tampering(
    tmp_path: Path,
    audited_training: _AuditFixture,
) -> None:
    training_dir = _training_copy(tmp_path, audited_training)
    checkpoint_path = training_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["trace_cohort_sha256"] != "0" * 64
    checkpoint["trace_cohort_sha256"] = "0" * 64
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        audit_training_artifacts(
            training_dir,
            audited_training.traces_dir,
            audited_training.preflight_path,
        )
