from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from functional_divergence.causal_first_error_attribution.monitor_data import (
    MonitorBoundaryRow,
)
from functional_divergence.causal_first_error_attribution.monitor_experiment import (
    MonitorExperimentConfig,
    ProcessBenchMonitorExperiment,
    correct_chain_threshold,
    evaluate_boundary_scores,
    paired_problem_bootstrap,
)
from functional_divergence.causal_first_error_attribution.monitor_training import (
    MonitorTrainingConfig,
)


def _row(
    chain: str,
    problem: str,
    domain: str,
    step: int,
    gold: int,
) -> MonitorBoundaryRow:
    return MonitorBoundaryRow(
        chain_id=chain,
        problem_hash=f"{domain}::{problem}",
        sibling_group=f"{domain}::{problem}",
        domain=domain,
        candidate_step=step,
        first_error_step=gold,
        decision_position=step + 2,
        label=int(step == gold),
        nuisance=np.zeros(5, dtype=np.float32),
        output_context=np.zeros(5, dtype=np.float32),
        store_index=0,
        state_index=0,
    )


def test_false_alarm_threshold_is_selected_on_correct_chain_maxima() -> None:
    rows = [
        _row("correct-a", "p1", "gsm8k", 0, -1),
        _row("correct-a", "p1", "gsm8k", 1, -1),
        _row("correct-b", "p2", "gsm8k", 0, -1),
        _row("correct-b", "p2", "gsm8k", 1, -1),
        _row("error", "p3", "gsm8k", 0, 1),
        _row("error", "p3", "gsm8k", 1, 1),
    ]
    scores = np.asarray([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])

    threshold = correct_chain_threshold(
        rows, np.arange(len(rows)), scores, target_false_alarm_rate=0.5
    )

    assert threshold == 0.8


def test_false_alarm_threshold_requires_correct_validation_chains() -> None:
    rows = [
        _row("error", "p1", "gsm8k", 0, 1),
        _row("error", "p1", "gsm8k", 1, 1),
    ]

    with pytest.raises(ValueError, match="fully-correct validation chain"):
        correct_chain_threshold(
            rows,
            np.arange(2),
            np.asarray([0.1, 0.9]),
            target_false_alarm_rate=0.1,
        )


def test_boundary_evaluation_reports_localization_and_problem_balanced_rows() -> None:
    rows = [
        _row("error-a", "p1", "math", 0, 1),
        _row("error-a", "p1", "math", 1, 1),
        _row("correct-a", "p2", "math", 0, -1),
        _row("correct-a", "p2", "math", 1, -1),
    ]
    scores = np.asarray([0.1, 0.9, 0.1, 0.2])

    report = evaluate_boundary_scores(
        rows, np.arange(len(rows)), scores, false_alarm_threshold=0.5
    )

    assert report["top1_localization"] == 1.0
    assert report["mrr"] == 1.0
    assert report["correct_chain_false_alarm_rate"] == 0.0
    assert report["row_auroc"] == 1.0
    assert report["problem_groups"] == 2


def test_boundary_nll_balances_chains_before_risk_set_rows() -> None:
    rows = [
        _row("long", "shared", "math", 0, 2),
        _row("long", "shared", "math", 1, 2),
        _row("long", "shared", "math", 2, 2),
        _row("short", "shared", "math", 0, 0),
    ]
    scores = np.asarray([0.5, 0.5, 0.5, 0.9])

    report = evaluate_boundary_scores(
        rows, np.arange(len(rows)), scores, false_alarm_threshold=0.5
    )

    expected = (-np.log(0.5) - np.log(0.9)) / 2.0
    assert report["row_nll"] == pytest.approx(expected)


def test_paired_bootstrap_tests_candidate_against_a_structural_control() -> None:
    rows = []
    candidate = []
    baseline = []
    for domain in ("gsm8k", "math"):
        for problem in range(4):
            rows.extend(
                [
                    _row(f"{domain}-{problem}", f"p{problem}", domain, 0, 1),
                    _row(f"{domain}-{problem}", f"p{problem}", domain, 1, 1),
                ]
            )
            candidate.extend([0.1, 0.9])
            baseline.extend([0.9, 0.1])
    indices = np.arange(len(rows))

    report = paired_problem_bootstrap(
        rows,
        indices,
        candidate_scores=np.asarray(candidate),
        baseline_scores=np.asarray(baseline),
        candidate_thresholds={"gsm8k": 0.5, "math": 0.5},
        baseline_thresholds={"gsm8k": 0.5, "math": 0.5},
        repeats=100,
        seed=11,
    )

    assert report["top1_gain"]["point"] == 1.0
    assert report["top1_gain"]["ci_low"] > 0.0
    assert report["nll_improvement"]["ci_low"] > 0.0
    assert set(report["domain_points"]) == {"gsm8k", "math"}


def test_localization_is_problem_balanced_when_one_problem_has_more_chains() -> None:
    rows = []
    scores = []
    # Two failed localizations from p-many and one successful localization
    # from p-single must average to 0.5 by problem, not 1/3 by chain.
    for chain in ("many-a", "many-b"):
        rows.extend(
            [
                _row(chain, "p-many", "math", 0, 1),
                _row(chain, "p-many", "math", 1, 1),
            ]
        )
        scores.extend([0.9, 0.1])
    rows.extend(
        [
            _row("single", "p-single", "math", 0, 1),
            _row("single", "p-single", "math", 1, 1),
        ]
    )
    scores.extend([0.1, 0.9])

    report = evaluate_boundary_scores(
        rows,
        np.arange(len(rows)),
        np.asarray(scores),
        false_alarm_threshold=0.5,
    )

    assert report["top1_localization"] == 0.5
    assert report["mrr"] == 0.75


def _write_integration_domain(root: Path, domain: str) -> None:
    geometry = root / domain / "geometry"
    geometry.mkdir(parents=True)
    selected = root / domain / "selected"
    selected.mkdir(parents=True)
    states = np.arange(8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4) / 100.0
    np.save(geometry / "states.npy", states)
    ranges = np.asarray([[[3, 4], [5, 6]]] * 4, dtype=np.int64)
    scores = np.asarray([[[0.1, 0.2], [0.3, 0.4]]] * 4, dtype=np.float32)
    np.savez_compressed(
        selected / "trace.npz",
        chain_idx=np.arange(4, dtype=np.int64),
        gold_error_step=np.asarray([1, 1, 1, -1], dtype=np.int64),
        n_steps=np.asarray([2, 2, 2, 2], dtype=np.int64),
        step_token_ranges=ranges,
        step_scores=scores,
        step_score_names=np.asarray(["token_entropy", "token_nll"]),
        dataset=np.asarray([domain] * 4),
    )
    np.savez_compressed(
        geometry / "trace.npz",
        chain_idx=np.arange(4, dtype=np.int64),
        gold_error_step=np.asarray([1, 1, 1, -1], dtype=np.int64),
        n_steps=np.asarray([2, 2, 2, 2], dtype=np.int64),
        step_token_ranges=ranges,
        step_scores=np.empty((4, 2, 0), dtype=np.float32),
        step_score_names=np.asarray([], dtype=object),
        problem_group_id=np.asarray([f"{domain}-g{i}" for i in range(4)]),
        problem_ids=np.asarray(
            [f"problem_sha256:{domain}-{i}" for i in range(4)]
        ),
        dataset=np.asarray([domain] * 4),
        step_pre_state_memmap_path=np.asarray("states.npy", dtype=object),
        step_pre_state_memmap_count=np.asarray(8, dtype=np.int64),
        step_pre_state_vector_chain_idx=np.repeat(np.arange(4), 2),
        step_pre_state_vector_step_idx=np.tile(np.arange(2), 4),
        step_layer_state_vector_layers=np.asarray([1, 2, 3, 4]),
        state_representation_kind=np.asarray("hidden_state", dtype=object),
        hidden_state_token_semantics=np.asarray(
            "h_i_after_reading_token_i", dtype=object
        ),
        step_prediction_position_shift=np.asarray(-1, dtype=np.int8),
        metadata_json=np.asarray(
            [
                '{"step_pre_state_temporal_semantics":'
                '"causal_before_first_step_token"}'
            ]
            * 4,
            dtype=object,
        ),
    )


def test_experiment_runs_from_geometry_files_to_durable_lodo_results(tmp_path) -> None:
    domains = ("gsm8k", "math", "omnimath")
    for domain in domains:
        _write_integration_domain(tmp_path / "data", domain)
    output = tmp_path / "results"

    summary = ProcessBenchMonitorExperiment(
        MonitorExperimentConfig(
            data_root=tmp_path / "data",
            domains=domains,
            output_dir=output,
            arms=(
                "output_history",
                "static_layer_set",
                "two_boundary_bag",
                "two_boundary_innovation",
            ),
            validation_fraction=0.25,
            bootstrap_repeats=10,
            training=MonitorTrainingConfig(
                width=4,
                epochs=1,
                patience=1,
                batch_size=4,
                learning_rate=1e-3,
                device="cpu",
                show_progress=False,
            ),
        )
    ).run()

    assert summary["rows"] == 24
    assert summary["events"] == 9
    assert set(summary["folds"]) == set(domains)
    assert (
        "two_boundary_innovation_vs_two_boundary_bag"
        in summary["temporal_paired_contrasts"]
    )
    assert summary["method"] == "prefix_conditioned_two_boundary_innovation_hazard"
    hidden_parameters = {
        summary["folds"]["gsm8k"]["arms"][arm]["parameters"]
        for arm in (
            "static_layer_set",
            "two_boundary_bag",
            "two_boundary_innovation",
        )
    }
    assert len(hidden_parameters) == 1
    assert (output / "config.json").is_file()
    assert (output / "predictions.jsonl").is_file()
    assert (output / "summary.json").is_file()
