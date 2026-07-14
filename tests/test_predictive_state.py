from __future__ import annotations

from pathlib import Path

import numpy as np

from prompt_control_flow.predictive_state_audit import (
    PredictiveStateAuditConfig,
    run_predictive_state_audit,
    write_predictive_state_outputs,
)
from prompt_control_flow.predictive_state_data import (
    ProjectionConfig,
    WindowConfig,
    build_transition_bundle,
    build_window_observations,
    inspect_predictive_state_source,
    load_predictive_state_dataset,
)
from prompt_control_flow.predictive_state_model import PredictiveModelConfig


def _object_array(values: list[np.ndarray]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def _rotation_sequence(length: int, phase: float, hidden: int) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) + phase
    columns = []
    for frequency in (0.17, 0.31, 0.47, 0.71):
        columns.extend([np.sin(frequency * time), np.cos(frequency * time)])
    return np.stack(columns[:hidden], axis=1).astype(np.float32)


def _write_predictive_synthetic(path: Path, n_problems: int = 12) -> None:
    rng = np.random.default_rng(91)
    hidden = 8
    n_tokens = 48
    window = 4
    layers = np.asarray([16], dtype=np.int64)
    trajectories: list[np.ndarray] = []
    clouds: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    input_ids: list[np.ndarray] = []
    problem_ids: list[int] = []
    sample_idx: list[int] = []
    is_correct: list[int] = []
    responses: list[str] = []

    token_pattern = np.tile(np.asarray([10, 11, 12, 13], dtype=np.int64), n_tokens // window)
    step_sizes = np.asarray([16, 16, 16], dtype=np.int64)
    step_ranges = np.asarray([[0, 15], [16, 31], [32, 47]], dtype=np.int64)
    for problem in range(n_problems):
        for sample in range(4):
            correct = sample < 2
            sequence = _rotation_sequence(
                n_tokens,
                phase=problem * 0.37 + sample * 0.11,
                hidden=hidden,
            )
            sequence += rng.normal(0.0, 0.01, size=sequence.shape).astype(np.float32)
            if not correct:
                blocks = sequence.reshape(-1, window, hidden).copy()
                permutation = rng.permutation(blocks.shape[0])
                sequence = blocks[permutation].reshape(n_tokens, hidden)
            trajectories.append(
                rng.normal(0.0, 1.0, size=(3, 1, hidden)).astype(np.float32)
            )
            clouds.append(sequence[:, None, :].astype(np.float32))
            sizes.append(step_sizes.copy())
            ranges.append(step_ranges.copy())
            input_ids.append(token_pattern.copy())
            problem_ids.append(problem)
            sample_idx.append(sample)
            is_correct.append(int(correct))
            responses.append("synthetic response " + "x" * 48)

    labels = np.asarray(is_correct, dtype=np.int64)
    np.savez(
        path,
        sv_vec_step_exp=_object_array(trajectories),
        sv_clouds=_object_array(clouds),
        cloud_sizes=_object_array(sizes),
        input_ids=_object_array(input_ids),
        time_axis_token_ranges=_object_array(ranges),
        problem_ids=np.asarray(problem_ids, dtype=np.int64),
        sample_idx=np.asarray(sample_idx, dtype=np.int64),
        is_correct=labels,
        is_correct_strict=labels,
        format_ok=np.ones(labels.size, dtype=bool),
        responses=np.asarray(responses, dtype=object),
        layers_used=layers,
        cloud_layers=layers,
        model_name=np.asarray("synthetic-predictive-state"),
    )


def test_predictive_state_loader_requires_exact_cloud_token_alignment(tmp_path: Path) -> None:
    source = tmp_path / "predictive.npz"
    _write_predictive_synthetic(source, n_problems=4)
    schema = inspect_predictive_state_source(source)
    assert schema["exact_token_alignment"] is True
    assert schema["token_range_key"] == "time_axis_token_ranges"
    assert schema["cloud_layers"] == [16]

    z = np.load(source, allow_pickle=True)
    payload = {key: z[key] for key in z.files}
    broken = payload["time_axis_token_ranges"].copy()
    broken[0] = np.asarray([[0, 14], [16, 31], [32, 47]], dtype=np.int64)
    payload["time_axis_token_ranges"] = broken
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, **payload)
    try:
        load_predictive_state_dataset(invalid)
    except ValueError as exc:
        assert "does not equal cloud size" in str(exc)
    else:
        raise AssertionError("misaligned token ranges must fail preflight")


def test_legacy_multisample_runs_state_only_and_cannot_pass_full_gate(
    tmp_path: Path,
) -> None:
    exact = tmp_path / "exact.npz"
    _write_predictive_synthetic(exact)
    z = np.load(exact, allow_pickle=True)
    payload = {
        key: z[key]
        for key in z.files
        if key not in {"input_ids", "time_axis_token_ranges"}
    }
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **payload)

    schema = inspect_predictive_state_source(legacy)
    assert schema["alignment_mode"] == "legacy_cloud_order"
    assert schema["exact_token_alignment"] is False
    assert schema["confirmatory_ready"] is False
    dataset = load_predictive_state_dataset(legacy)
    assert dataset.token_ids is None
    assert np.array_equal(dataset.token_positions[0], np.arange(48))

    report, packed = run_predictive_state_audit(
        dataset,
        ProjectionConfig(
            projection_dim=8,
            batch_size=32,
            max_batch_tokens=2048,
            seed=7,
            compute_device="cpu",
        ),
        WindowConfig(
            window_tokens=4,
            window_stride=4,
            max_skipped_tokens=0,
            window_batch_size=512,
            compute_device="cpu",
        ),
        PredictiveModelConfig(
            latent_dim=4,
            ridge=1e-2,
            covariance_shrinkage=0.15,
            tangent_variance=0.8,
        ),
        PredictiveStateAuditConfig(
            folds=3,
            horizons=(1,),
            bootstrap=20,
            permutations=9,
            length_match_ratio=1.0,
            seed=5,
            verbose=False,
        ),
    )

    assert report["meta"]["analysis_tier"] == "exploratory_legacy_state_only"
    assert report["meta"]["primary_channel"] == "raw"
    assert report["decision_gate"]["checks"]["exact_lexical_control_available"] is False
    assert report["decision_gate"]["passes"] is False
    names = packed["score_names"].tolist()
    token_residual = packed["scores"][:, names.index(
        "predictive.token_residual.mahalanobis_mean"
    )]
    assert not np.any(np.isfinite(token_residual))
    assert np.any(np.isfinite(packed["scores"][:, names.index(
        "predictive.raw.mahalanobis_mean"
    )]))


def test_transition_builder_rejects_unobserved_token_gap() -> None:
    sequence = np.arange(16, dtype=np.float32).reshape(8, 2)
    token_positions = np.asarray([0, 1, 2, 3, 20, 21, 22, 23], dtype=np.int64)
    observations, ranges, _ = build_window_observations(
        [sequence],
        [token_positions],
        WindowConfig(
            window_tokens=4,
            window_stride=4,
            max_skipped_tokens=0,
            compute_device="cpu",
        ),
    )

    transitions = build_transition_bundle(
        observations,
        ranges,
        [0],
        np.asarray([7], dtype=np.int64),
        horizon=1,
        max_transition_gap=0,
    )

    assert observations[0].shape[0] == 2
    assert transitions.x.shape[0] == 0


def test_predictive_state_audit_recovers_order_only_failure(tmp_path: Path) -> None:
    source = tmp_path / "predictive.npz"
    _write_predictive_synthetic(source)
    dataset = load_predictive_state_dataset(source)
    report, packed = run_predictive_state_audit(
        dataset,
        ProjectionConfig(
            projection_dim=8,
            batch_size=32,
            max_batch_tokens=2048,
            seed=7,
            compute_device="cpu",
        ),
        WindowConfig(
            window_tokens=4,
            window_stride=4,
            max_skipped_tokens=0,
            window_batch_size=512,
            compute_device="cpu",
        ),
        PredictiveModelConfig(
            latent_dim=4,
            ridge=1e-2,
            covariance_shrinkage=0.15,
            tangent_variance=0.8,
        ),
        PredictiveStateAuditConfig(
            folds=3,
            horizons=(1,),
            context_windows=1,
            min_token_count=2,
            bootstrap=40,
            permutations=19,
            length_match_ratio=1.0,
            seed=5,
            verbose=False,
        ),
    )
    scores = {row["name"]: row for row in report["scores"]}
    primary = scores["predictive.token_residual.mahalanobis_mean"]
    static = scores["static.token_residual.mahalanobis_mean"]
    consensus = scores["baseline.fixed_window_consensus"]
    assert primary["same_problem_auroc"] > 0.80
    assert primary["same_problem_auroc"] > static["same_problem_auroc"] + 0.15
    assert primary["same_problem_auroc"] > consensus["same_problem_auroc"] + 0.15
    assert report["auc_deltas"]["ordered_minus_shuffle"]["point"] > 0.0
    assert packed["scores"].shape[0] == dataset.n_samples
    assert np.all(packed["valid_windows"] == 12)

    paths = write_predictive_state_outputs(
        report,
        packed,
        output=tmp_path / "scores.npz",
        output_dir=tmp_path / "audit",
        render_plots=False,
    )
    for value in paths.values():
        assert Path(value).exists()
