from __future__ import annotations

from pathlib import Path

import numpy as np

from prompt_control_flow.conditional_tangent import (
    ConditionalTangentConfig,
    load_conditional_tangent_dataset,
    run_conditional_tangent_audit,
)
from prompt_control_flow.conditional_tangent_report import (
    ConditionalTangentValidationConfig,
    build_validation_summary,
    write_validation_report,
)


def _objects(values: list[np.ndarray]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    output[:] = values
    return output


def _write_synthetic(path: Path, *, cotangent: bool) -> None:
    trajectories: list[np.ndarray] = []
    qvecs: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    stepcloud: list[np.ndarray] = []
    response_clouds: list[np.ndarray] = []
    cotangents: list[np.ndarray] = []
    gold: list[int] = []
    correct: list[int] = []
    problems: list[int] = []
    for row in range(18):
        family = row % 2
        is_error = row >= 12
        feasible = np.asarray(
            [1.0, 0.0, 0.0, 0.0]
            if family == 0
            else [0.0, 1.0, 0.0, 0.0],
            dtype=np.float32,
        )
        normal = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        question = feasible[None, :]
        updates = np.stack(
            [
                feasible,
                feasible,
                normal if is_error else feasible,
                normal if is_error else feasible,
            ]
        )[:, None, :]
        states = np.cumsum(updates, axis=0) + question[None, :, :]
        trajectories.append(states.astype(np.float32))
        qvecs.append(question)
        ranges.append(
            np.asarray([[0, 2], [3, 5], [6, 8], [9, 11]], dtype=np.int64)
        )
        resultant = np.full((4, 1, 1), 0.95, dtype=np.float32)
        if is_error:
            resultant[2:, 0, 0] = 0.25
        stepcloud.append(resultant)
        tokens = []
        gradients = np.zeros((4, 1, 4), dtype=np.float32)
        for step in range(4):
            direction = normal if is_error and step >= 2 else feasible
            tokens.extend(
                [
                    direction,
                    direction + np.asarray([0.0, 0.0, 0.0, 0.01], dtype=np.float32),
                    direction,
                ]
            )
            gradients[step, 0] = direction
        response_clouds.append(np.asarray(tokens, dtype=np.float32)[:, None, :])
        cotangents.append(gradients)
        gold.append(2 if is_error else -1)
        correct.append(0 if is_error else 1)
        problems.append(row)
    payload = {
        "ids": np.asarray([f"chain-{row}" for row in range(18)], dtype=object),
        "problem_ids": np.asarray(problems, dtype=np.int64),
        "gold_error_step": np.asarray(gold, dtype=np.int64),
        "is_correct": np.asarray(correct, dtype=np.int64),
        "stepvec": _objects(trajectories),
        "qvec": _objects(qvecs),
        "sv_layers": np.asarray([8], dtype=np.int64),
        "step_token_ranges": _objects(ranges),
        "stepcloud": _objects(stepcloud),
        "cloud_feature_names": np.asarray(["resultant"], dtype=object),
        "layers_used": np.asarray([8], dtype=np.int64),
        "respcloud": _objects(response_clouds),
        "clouds_stored": np.asarray(True),
        "cloud_store_layers": np.asarray([8], dtype=np.int64),
    }
    if cotangent:
        payload.update(
            {
                "step_output_cotangent": _objects(cotangents),
                "step_output_cotangent_layers": np.asarray([8], dtype=np.int64),
                "step_output_cotangent_kind": np.asarray(
                    "exact_downstream_cotangent", dtype=object
                ),
            }
        )
    np.savez_compressed(path, **payload)


def _run(path: Path):
    dataset = load_conditional_tangent_dataset(path, layers="8")
    return run_conditional_tangent_audit(
        dataset,
        ConditionalTangentConfig(
            device="cpu",
            batch_size=16,
            folds=3,
            neighbors=4,
            search_multiplier=3,
            tangent_rank=1,
            q_temperature=0.05,
            phase_sigma=0.15,
            persistence_window=2,
            global_reference_cap=64,
            random_seed=5,
        ),
    )


def test_question_conditioned_tangent_recovers_injected_normal_escape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "full_synthetic.npz"
    _write_synthetic(path, cotangent=True)
    result = _run(path)
    metric = result.metric_names.index("qpt_escape_ratio")
    output_metric = result.metric_names.index("output_transverse_energy")
    alignment_metric = result.metric_names.index("output_normal_alignment")
    error_values = np.asarray(
        [
            result.fields[row][2, 0, metric]
            for row, gold in enumerate(result.dataset.source.gold_error_step)
            if gold >= 0
        ]
    )
    correct_values = np.asarray(
        [
            result.fields[row][2, 0, metric]
            for row, gold in enumerate(result.dataset.source.gold_error_step)
            if gold < 0
        ]
    )
    assert np.nanmean(error_values) > 0.90
    assert np.nanmean(correct_values) < 0.05
    assert np.nanmean(
        [
            result.fields[row][2, 0, output_metric]
            for row in range(12, 18)
        ]
    ) > 0.90
    assert np.nanmean(
        [
            result.fields[row][2, 0, alignment_metric]
            for row in range(12, 18)
        ]
    ) > 0.90
    assert result.metadata["output_gate"] == "available"
    summary = build_validation_summary(
        result,
        ConditionalTangentValidationConfig(
            folds=3,
            bootstrap=20,
            length_bins=3,
            event_offsets=(-1, 0, 1),
            min_coverage=0.50,
            random_seed=11,
        ),
    )
    output_gate = summary["hypothesis_gates"]["output_sensitivity"]
    assert output_gate["status"] == "tested_exact_cotangent"
    assert output_gate["best_first_error_event"] is not None
    assert output_gate["best_response_row"] is not None
    assert "output_beyond_escape" in output_gate["best_response_row"]


def test_summary_audits_length_response_and_missing_output_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "full_synthetic.npz"
    _write_synthetic(path, cotangent=False)
    result = _run(path)
    cfg = ConditionalTangentValidationConfig(
        folds=3,
        bootstrap=20,
        length_bins=3,
        event_offsets=(-1, 0, 1),
        min_coverage=0.50,
        random_seed=7,
    )
    summary = build_validation_summary(result, cfg)
    assert summary["hypothesis_gates"]["output_sensitivity"]["status"] == (
        "not_tested_missing_cotangent"
    )
    question_gate = summary["hypothesis_gates"]["question_conditioning"]
    assert question_gate["correct_chains"] == 12
    assert "equal-weight" in question_gate["unit_of_analysis"]
    qpt_rows = [
        row
        for row in summary["step_metrics"]
        if row["metric"] == "qpt_escape_ratio"
        and row["variant"] == "raw"
        and row["layer"] == 8
    ]
    assert len(qpt_rows) == 1
    assert qpt_rows[0]["signed_auroc"] > 0.90
    assert np.isfinite(qpt_rows[0]["length_bucket_auroc"])
    response_rows = [
        row
        for row in summary["response_metrics"]
        if row["metric"] == "qpt_escape_ratio"
        and row["variant"] == "nuisance_residual"
        and row["aggregation"] == "max"
    ]
    assert response_rows
    persistence_rows = [
        row
        for row in summary["response_metrics"]
        if row["metric"] == "coherent_normal_drift"
        and row["variant"] == "raw"
        and row["aggregation"] == "mean"
    ]
    assert len(persistence_rows) == 1
    assert isinstance(
        persistence_rows[0].get("persistence_beyond_instantaneous"), dict
    )

    _, paths = write_validation_report(
        result,
        tmp_path / "report",
        cfg,
        render_plots=False,
        score_output=tmp_path / "report" / "scores.npz",
        include_normal_vectors=True,
    )
    for path_value in paths.values():
        assert Path(path_value).exists()
