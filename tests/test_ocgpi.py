from __future__ import annotations

import numpy as np
import pytest

from prompt_control_flow.data import (
    load_chain_records,
    validate_records_against_reference,
)
from prompt_control_flow.ocgpi.metrics import (
    summarize_binary_increment,
    summarize_forecast_increment,
)
from prompt_control_flow.ocgpi.dataset import (
    BinaryTask,
    ReasoningSequence,
    SequenceCollection,
    build_forecast_task,
    build_online_response_task,
    select_output_tier,
)
from prompt_control_flow.ocgpi.models import (
    CrossFitConfig,
    FiniteStandardizer,
    binary_task_bootstrap_seed,
    crossfit_binary_increment,
    crossfit_forecast_increment,
    grouped_splits,
    length_matched_permutation,
)
from prompt_control_flow.ocgpi.schema import CompactTraceItem, TraceArtifact
from prompt_control_flow.ocgpi.gates import evaluate_claim_gate


def test_canonical_processbench_source_matches_full_artifact_conventions(
    tmp_path,
) -> None:
    source = tmp_path / "ProcessBench"
    source.mkdir()
    (source / "gsm8k.json").write_text(
        """[
          {"problem":"skip","steps":["a","b"],"label":0},
          {"problem":"kept","steps":[" s0 ","s1","s2"],"label":1},
          {"problem":"correct","steps":["c0","c1","c2","c3"],"label":-1,"final_answer_correct":true}
        ]""",
        encoding="utf-8",
    )
    rows = load_chain_records(
        source,
        input_format="processbench_source",
        subset="gsm8k",
    )
    assert [row.chain_idx for row in rows] == [0, 1]
    assert [row.problem_id for row in rows] == [0, 1]
    assert rows[0].response == "s0\ns1\ns2"
    assert rows[0].gold_error_step == 1
    assert rows[1].is_correct == 1

    reference = tmp_path / "full_gsm8k.npz"
    np.savez_compressed(
        reference,
        problem_ids=np.asarray([0, 1]),
        steps_text=np.asarray([row.steps for row in rows], dtype=object),
        responses=np.asarray([row.response for row in rows], dtype=object),
        gold_error_step=np.asarray([1, -1]),
    )
    preflight = validate_records_against_reference(rows, reference)
    assert preflight["exact_text_match"] is True
    rows[0].response = "different"
    with pytest.raises(ValueError, match="fields=response"):
        validate_records_against_reference(rows, reference)


def test_compact_logit_features_are_shift_invariant_and_chunk_exact() -> None:
    torch = pytest.importorskip("torch")
    from prompt_control_flow.ocgpi.logit_trace import (
        LogitTraceConfig,
        compact_features_from_logits,
    )

    generator = torch.Generator().manual_seed(7)
    logits = torch.randn((9, 53), generator=generator)
    targets = torch.randint(0, 53, (9,), generator=generator)
    cfg = LogitTraceConfig(top_k=20, sketch_dim=16, token_chunk_size=4)
    whole, _ = compact_features_from_logits(logits, targets, cfg)
    shifted, _ = compact_features_from_logits(
        logits + torch.linspace(-10.0, 10.0, 9)[:, None], targets, cfg
    )
    torch.testing.assert_close(whole, shifted, atol=2e-6, rtol=2e-6)

    chunks = []
    previous = None
    for start in range(0, 9, 4):
        value, previous = compact_features_from_logits(
            logits[start : start + 4],
            targets[start : start + 4],
            cfg,
            previous_logits=previous,
        )
        chunks.append(value)
    torch.testing.assert_close(whole, torch.cat(chunks), atol=2e-6, rtol=2e-6)


def test_step_aggregation_matches_schema() -> None:
    pytest.importorskip("torch")
    from prompt_control_flow.ocgpi.logit_trace import (
        LogitTraceConfig,
        aggregate_token_features_to_steps,
        step_feature_names,
        token_feature_names,
    )

    cfg = LogitTraceConfig(top_k=20, sketch_dim=8)
    rng = np.random.default_rng(3)
    token = rng.normal(size=(7, len(token_feature_names(cfg.sketch_dim)))).astype(
        np.float32
    )
    step, ranges = aggregate_token_features_to_steps(
        token,
        [(11, 12), (14, 17)],
        response_start_token=11,
    )
    assert step.shape == (2, len(step_feature_names(cfg.sketch_dim)))
    np.testing.assert_array_equal(ranges, np.asarray([[0, 1], [3, 6]], dtype=np.int32))


def test_trace_artifact_round_trip(tmp_path) -> None:
    token_names = ("out.a", "out.b")
    step_names = ("out.a.last", "out.b.last")
    items = []
    for i in range(3):
        items.append(
            CompactTraceItem(
                chain_idx=i,
                problem_id=10 + i,
                gold_error_step=-1 if i == 0 else 1,
                is_correct=int(i == 0),
                sample_idx=-1,
                dataset="synthetic",
                generator="unit-test",
                response_hash=f"hash-{i}",
                token_ids=np.asarray([1, 2, 3], dtype=np.int64),
                token_features=np.full((3, 2), i + 1, dtype=np.float32),
                step_features=np.full((2, 2), i + 1, dtype=np.float32),
                step_token_ranges=np.asarray([[0, 0], [1, 2]], dtype=np.int32),
                replay_kind="synthetic",
            )
        )
    artifact = TraceArtifact.from_items(
        items,
        token_feature_names=token_names,
        step_feature_names=step_names,
        metadata={"test": True},
    )
    path = tmp_path / "trace.npz"
    artifact.save(path)
    loaded = TraceArtifact.load(path)
    assert loaded.n_chains == 3
    assert loaded.metadata["test"] is True
    np.testing.assert_allclose(loaded.step_matrix(2), 3.0)


def test_state_geometry_is_basis_and_global_scale_invariant() -> None:
    pytest.importorskip("torch")
    from prompt_control_flow.ocgpi.geometry_features import compute_state_geometry

    rng = np.random.default_rng(11)
    states = rng.normal(size=(7, 6, 15)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(15, 15)))
    baseline = compute_state_geometry(states)
    transformed = compute_state_geometry((states @ q * 5.5).astype(np.float32))
    np.testing.assert_allclose(
        baseline, transformed, rtol=3e-5, atol=3e-5, equal_nan=True
    )


def test_batched_state_geometry_matches_individual_computation() -> None:
    pytest.importorskip("torch")
    from prompt_control_flow.ocgpi.geometry_features import compute_state_geometry

    rng = np.random.default_rng(29)
    states = rng.normal(size=(4, 7, 6, 15)).astype(np.float32)
    batched = compute_state_geometry(states)
    individual = np.stack([compute_state_geometry(value) for value in states])
    np.testing.assert_allclose(
        batched, individual, rtol=2e-6, atol=2e-6, equal_nan=True
    )


def test_grouped_splits_have_no_problem_leakage() -> None:
    y = np.asarray([0, 0, 1, 1] * 12, dtype=np.int8)
    groups = np.repeat(np.arange(24), 2)
    for train, test in grouped_splits(y, groups, n_splits=4, seed=3, stratified=True):
        assert not set(groups[train]).intersection(groups[test])


def test_weighted_standardizer_prevents_repeated_prefix_dominance() -> None:
    values = np.asarray([[0.0], [10.0], [10.0], [10.0]])
    weights = np.asarray([1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    transform = FiniteStandardizer().fit(values, sample_weight=weights)
    np.testing.assert_allclose(transform.center, np.asarray([0.0]))


def test_output_saturation_tiers_are_nested() -> None:
    names = (
        "control.length",
        "last.out.entropy_norm.mean",
        "last.out.probability_count_sketch.000.mean",
        "last.out.topk_count_sketch.000.mean",
    )
    task = BinaryTask(
        name="synthetic",
        checkpoint=1.0,
        x_output=np.ones((8, 4), dtype=np.float32),
        x_geometry=np.ones((8, 2), dtype=np.float32),
        y=np.asarray([0, 1] * 4, dtype=np.int8),
        groups=np.arange(8),
        chain_idx=np.arange(8),
        output_names=names,
        geometry_names=("g0", "g1"),
        geometry_groups=("depth", "temporal"),
        nuisance_values=np.ones((8, 1), dtype=np.float32),
    )
    assert select_output_tier(task, "scalar").x_output.shape[1] == 2
    assert select_output_tier(task, "distribution").x_output.shape[1] == 3
    assert select_output_tier(task, "full_compact").x_output.shape[1] == 4


def test_online_prefix_task_does_not_require_final_relative_position() -> None:
    sequence = ReasoningSequence(
        chain_idx=1,
        problem_id=7,
        gold_error_step=2,
        is_error=1,
        output=np.arange(12, dtype=np.float32).reshape(3, 4),
        geometry=np.arange(15, dtype=np.float32).reshape(3, 5),
        step_token_counts=np.asarray([2, 4, 3], dtype=np.float32),
    )
    collection = SequenceCollection(
        sequences=[sequence],
        output_names=("o0", "o1", "o2", "o3"),
        geometry_names=("g0", "g1", "g2", "g3", "g4"),
        geometry_groups=("depth",) * 5,
        preflight={},
    )
    task = build_online_response_task(collection)
    np.testing.assert_array_equal(task.step_idx, np.asarray([0, 1, 2]))
    assert len(task.y) == 3
    assert not any("relative" in name for name in task.output_names)


def test_online_binary_task_uses_non_negative_bootstrap_seed() -> None:
    assert binary_task_bootstrap_seed(17, -1.0) == 10_017
    assert binary_task_bootstrap_seed(17, 0.25) == 42
    with pytest.raises(ValueError, match="base_seed must be non-negative"):
        binary_task_bootstrap_seed(-1, -1.0)


def test_forecast_task_uses_only_causal_length_controls() -> None:
    sequence = ReasoningSequence(
        chain_idx=1,
        problem_id=7,
        gold_error_step=-1,
        is_error=0,
        output=np.arange(12, dtype=np.float32).reshape(3, 4),
        geometry=np.arange(15, dtype=np.float32).reshape(3, 5),
        step_token_counts=np.asarray([2, 4, 3], dtype=np.float32),
    )
    collection = SequenceCollection(
        sequences=[sequence],
        output_names=("a.last", "b.last", "c.last", "d.last"),
        geometry_names=("g0", "g1", "g2", "g3", "g4"),
        geometry_groups=("depth",) * 5,
        preflight={},
    )
    task = build_forecast_task(collection, history=2, horizon=1)
    assert task.x_output.shape[0] == 1
    assert task.output_names[:3] == (
        "control.log1p_prefix_steps",
        "control.log1p_prefix_tokens",
        "control.log1p_current_step_tokens",
    )
    assert not any("relative" in name for name in task.output_names)


def test_length_matched_null_permuted_across_problem_groups() -> None:
    n = 80
    values = np.arange(n, dtype=np.float64)[:, None]
    prefix_tokens = np.repeat(np.arange(8, dtype=np.float64), 10)
    nuisance = np.stack([np.log1p(np.arange(n)), prefix_tokens, np.zeros(n)], axis=1)
    groups = np.arange(n, dtype=np.int64)
    result = length_matched_permutation(
        values,
        nuisance,
        groups,
        rng=np.random.default_rng(41),
        bins=5,
    )
    donor = result[:, 0].astype(np.int64)
    assert np.all(donor != groups)
    quantiles = np.unique(np.quantile(prefix_tokens, np.linspace(0.0, 1.0, 6)))
    source_bin = np.digitize(prefix_tokens, quantiles[1:-1], right=True)
    donor_bin = np.digitize(prefix_tokens[donor], quantiles[1:-1], right=True)
    np.testing.assert_array_equal(source_bin, donor_bin)


def test_binary_conditional_geometry_increment_recovers_unique_signal() -> None:
    rng = np.random.default_rng(19)
    n = 320
    output = rng.normal(size=(n, 6))
    innovation = rng.normal(size=(n, 4))
    geometry = output @ rng.normal(size=(6, 9)) + innovation @ rng.normal(size=(4, 9))
    logit = 0.5 * output[:, 0] + 2.0 * innovation[:, 0] - 1.5 * innovation[:, 1]
    y = (logit + rng.logistic(size=n) * 0.5 > 0.0).astype(np.int8)
    groups = np.arange(n, dtype=np.int64)
    nuisance = np.stack([rng.normal(size=n), rng.normal(size=n)], axis=1)
    cfg = CrossFitConfig(
        outer_folds=4,
        inner_folds=3,
        logistic_c=0.5,
        chart_max_dim=8,
        adapter_l2=0.1,
        seed=5,
    )
    result = crossfit_binary_increment(output, geometry, y, groups, nuisance, cfg)
    summary = summarize_binary_increment(result, n_boot=100, seed=5)
    assert (
        summary["output_plus_geometry"]["auroc"]
        > summary["output_only"]["auroc"] + 0.08
    )
    assert summary["increment"]["conditional_usable_information"]["point_bits"] > 0.02


def test_forecast_partial_r2_recovers_output_orthogonal_geometry() -> None:
    rng = np.random.default_rng(23)
    n = 360
    output = rng.normal(size=(n, 8))
    innovation = rng.normal(size=(n, 5))
    geometry = output @ rng.normal(size=(8, 12)) + innovation @ rng.normal(size=(5, 12))
    target = output @ rng.normal(size=(8, 7)) + innovation @ rng.normal(size=(5, 7))
    target += rng.normal(scale=0.2, size=target.shape)
    groups = np.repeat(np.arange(120), 3)
    nuisance = np.stack([rng.normal(size=n), rng.normal(size=n)], axis=1)
    cfg = CrossFitConfig(
        outer_folds=4,
        inner_folds=3,
        ridge_alpha=1.0,
        geometry_ridge_alpha=1.0,
        chart_max_dim=10,
        seed=9,
    )
    result = crossfit_forecast_increment(
        output, geometry, target, groups, nuisance, cfg
    )
    summary = summarize_forecast_increment(result, n_boot=100, seed=9)
    assert summary["increment"]["partial_r2"]["point"] > 0.20
    assert summary["output_plus_geometry_mse"] < summary["output_only_mse"]


def test_decision_gate_separates_mechanism_detector_and_confirmatory_claims() -> None:
    positive = {"ci_low": 0.01}
    response = {
        "problem_groups": 120,
        "increment": {
            "conditional_usable_information": positive,
            "delta_auroc_vs_null": positive,
        },
    }
    forecast = {
        "increment": {
            "partial_r2": positive,
            "partial_r2_vs_null": positive,
        }
    }
    exploratory = evaluate_claim_gate(
        response,
        forecast,
        {
            "observer_model_match": True,
            "geometry": {"mainline_geometry_ready": False},
        },
    )
    assert exploratory["mechanism_supported"] is True
    assert exploratory["detector_increment_supported"] is True
    assert exploratory["confirmatory_ready"] is False

    confirmatory = evaluate_claim_gate(
        response,
        forecast,
        {
            "observer_model_match": True,
            "geometry": {"mainline_geometry_ready": True},
        },
    )
    assert confirmatory["confirmatory_ready"] is True
