from __future__ import annotations

import pytest

from functional_divergence.config import RunConfig, SourceConfig
from functional_divergence.domain import (
    CohortSummary,
    DatasetMetadata,
    DatasetResult,
    ExperimentResult,
    SourceProvenance,
)
from functional_divergence.progress import RecordingProgress


def test_run_config_validates_the_joint_grid_once() -> None:
    config = RunConfig(offsets=(-2, -1, 0, 1), layers=(8, 10, 12), folds=3)

    assert config.offsets == (-2, -1, 0, 1)
    assert config.layers == (8, 10, 12)
    assert RunConfig(layers="8,10,12").layers == (8, 10, 12)

    with pytest.raises(ValueError, match="consecutive"):
        RunConfig(offsets=(-2, 0, 1))
    with pytest.raises(ValueError, match="at least two layers"):
        RunConfig(layers=(8,))


def test_source_config_owns_paths_and_generator_selection(tmp_path) -> None:
    source = SourceConfig(
        manifest=tmp_path / "trace.npz",
        hidden_dir=tmp_path / "hidden",
        response_generator="llama3.1-8b",
    )

    assert source.manifest.name == "trace.npz"
    assert source.response_generator == "llama3.1-8b"


def test_metadata_serializes_only_at_the_reporting_boundary() -> None:
    provenance = SourceProvenance(
        manifest_path="/data/trace.npz",
        source_format="exact_response_state_manifest_v1",
        snapshot_kind="raw_residual_stream",
        representation_scope="raw_residual_stream",
        axis_kind="token",
        problem_group_field="problem_group_id",
        generator_field="response_generator",
        generator_filter="llama3.1-8b",
        response_generators=("Llama-3.1-8B-Instruct",),
    )
    cohort = CohortSummary(
        manifest_records=400,
        selected_records=61,
        error_records=38,
        correct_records=23,
        candidate_pairs=38,
        retained_pairs=35,
        dropped_boundary_pairs=3,
        components=23,
    )
    metadata = DatasetMetadata(
        provenance=provenance,
        cohort=cohort,
        depth_semantics="sparse_depth_interval",
        component_grouping="matched_rows_plus_problem_ids",
    )

    encoded = metadata.to_dict()

    assert encoded["source_path"] == "/data/trace.npz"
    assert encoded["n_manifest_records"] == 400
    assert encoded["n_retained_pairs"] == 35
    assert encoded["response_generators"] == ["Llama-3.1-8B-Instruct"]


def test_recording_progress_exposes_stage_and_loop_events() -> None:
    progress = RecordingProgress()
    progress.stage("load", "gsm8k")
    values = list(progress.track(range(3), total=3, description="matched pairs"))

    assert values == [0, 1, 2]
    assert progress.events == [
        ("stage", "load", "gsm8k"),
        ("start", "matched pairs", 3),
        ("finish", "matched pairs", 3),
    ]


def test_experiment_result_owns_output_serialization() -> None:
    provenance = SourceProvenance(
        manifest_path="/data/trace.npz",
        source_format="exact_response_state_manifest_v1",
        snapshot_kind="raw_residual_stream",
        representation_scope="raw_residual_stream",
        axis_kind="token",
        problem_group_field="problem_group_id",
        generator_field="response_generator",
        generator_filter="llama3.1-8b",
        response_generators=("Llama-3.1-8B-Instruct",),
    )
    metadata = DatasetMetadata(
        provenance=provenance,
        cohort=CohortSummary(400, 61, 38, 23, 38, 35, 3, 23),
        depth_semantics="sparse_depth_interval",
        component_grouping="matched_rows_plus_problem_ids",
    )
    dataset = DatasetResult(
        metadata=metadata,
        pairs=35,
        time_offsets=(-2, -1, 0, 1),
        layer_ids=(8, 10),
        hidden_dim=4096,
        diagnostics={"projection_rank": 16},
        metrics={"radial_edge_change": {"paired_auroc": 0.6}},
        comparisons={},
    )
    result = ExperimentResult(dataset=dataset, seed=17, bootstrap=2000, rank=16, ridge_alpha=1.0)

    encoded = result.to_dict()

    assert encoded["dataset"]["n_retained_pairs"] == 35
    assert encoded["dataset"]["hidden_dim"] == 4096
    assert encoded["requested_rank"] == 16
