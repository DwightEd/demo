from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.evaluation import (
    evaluate_task,
    lodo_splits,
    summarize_predictions,
)
from functional_divergence.hidden_state_geometry.method import MethodFoldResult
from functional_divergence.hidden_state_geometry.registry import (
    ContrastSpec,
    RandomizationSpec,
    register_method,
)
from functional_divergence.hidden_state_geometry.tasks import build_whole_chain_task


@register_method(
    "evaluation_dummy",
    contrasts=(
        ContrastSpec("arbitrary_gain", "control_arm", "signal_arm"),
    ),
    randomizations=(
        RandomizationSpec("arbitrary_order", "permuted_arm_r", "signal_arm"),
    ),
)
class EvaluationDummy:
    def __init__(self, config):
        self.config = config

    def fit_predict(self, fold):
        base = np.asarray(
            [0.8 if row.sample.chain_id % 2 else 0.2 for row in fold.test_examples]
        )
        return MethodFoldResult(
            probabilities={
                "control_arm": np.clip(base * 0.8 + 0.1, 0.01, 0.99),
                "signal_arm": base,
                "permuted_arm_r0": np.full(len(base), 0.5),
            },
            diagnostics={"dummy": True},
            factors={"dummy": np.ones(1)},
        )


@register_method(
    "missing_declared_arm_dummy",
    contrasts=(ContrastSpec("gain", "control_arm", "signal_arm"),),
)
class MissingDeclaredArmDummy:
    def __init__(self, config):
        self.config = config

    def fit_predict(self, fold):
        return MethodFoldResult(
            probabilities={"control_arm": np.full(len(fold.test_examples), 0.5)},
            diagnostics={},
            factors={},
        )


@register_method("invalid_probability_dummy")
class InvalidProbabilityDummy:
    def __init__(self, config):
        self.config = config

    def fit_predict(self, fold):
        return MethodFoldResult(
            probabilities={"score": np.full(len(fold.test_examples), 1.2)},
            diagnostics={},
            factors={},
        )


def _sample(domain: str, chain_id: int) -> ChainSample:
    return ChainSample(
        chain_id=chain_id,
        manifest_row=chain_id,
        problem_group=f"p{chain_id}",
        dataset=domain,
        generator="llama",
        observer_model="llama",
        state_path=None,  # Dummy method never touches a shard.
        state_count=1,
        response_start=0,
        step_ranges=np.asarray([[0, 0]]),
        layer_ids=np.asarray([8]),
        output_steps=np.zeros((1, 1)),
        output_feature_names=("entropy",),
        first_error_step=0 if chain_id % 2 else -1,
    )


def test_lodo_holds_out_one_complete_domain_and_covers_every_row_once():
    samples = tuple(
        _sample(domain, domain_index * 10 + row)
        for domain_index, domain in enumerate(("gsm8k", "math", "olympiad", "omnimath"))
        for row in range(4)
    )
    task = build_whole_chain_task(samples)

    splits = list(lodo_splits(task))
    result = evaluate_task(
        task,
        method_name="evaluation_dummy",
        method_config={},
        n_boot=30,
        seed=5,
    )

    assert len(splits) == 4
    for split in splits:
        assert len(np.unique(task.domains[split.test])) == 1
        assert not set(task.groups[split.train]).intersection(task.groups[split.test])
    assert all(np.isfinite(values).all() for values in result.probabilities.values())
    assert len(result.fold_audit) == 4
    assert set(result.fold_ids.tolist()) == {0, 1, 2, 3}
    assert "arbitrary_gain" in result.summary["increments"]
    assert "arbitrary_order" in result.summary["randomization_checks"]


def test_evaluator_fails_fast_when_plugin_omits_a_declared_arm():
    samples = tuple(
        _sample(domain, domain_index * 10 + row)
        for domain_index, domain in enumerate(("a", "b"))
        for row in range(4)
    )

    with pytest.raises(ValueError, match="declared contrast arms"):
        evaluate_task(
            build_whole_chain_task(samples),
            method_name="missing_declared_arm_dummy",
            method_config={},
            n_boot=5,
            seed=5,
        )


def test_evaluator_rejects_invalid_plugin_probabilities():
    samples = tuple(
        _sample(domain, domain_index * 10 + row)
        for domain_index, domain in enumerate(("a", "b"))
        for row in range(4)
    )

    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        evaluate_task(
            build_whole_chain_task(samples),
            method_name="invalid_probability_dummy",
            method_config={},
            n_boot=5,
            seed=5,
        )


def test_lodo_qualifies_local_group_ids_but_rejects_real_cross_domain_hash_overlap():
    local_ids = tuple(
        replace(_sample(domain, row), problem_group=f"local-{row}")
        for domain in ("a", "b")
        for row in range(4)
    )
    local_task = build_whole_chain_task(local_ids)
    assert len(lodo_splits(local_task)) == 2
    assert not set(local_task.groups[local_task.domains == "a"]).intersection(
        local_task.groups[local_task.domains == "b"]
    )

    hashed = list(local_ids)
    hashed[0] = replace(hashed[0], problem_hash="problem_sha256:duplicate")
    hashed[4] = replace(hashed[4], problem_hash="problem_sha256:duplicate")
    with pytest.raises(RuntimeError, match="problem-hash leakage"):
        lodo_splits(build_whole_chain_task(tuple(hashed)))


def test_macro_metrics_weight_domains_equally_and_hidden_increment_has_fixed_sign():
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    domains = np.asarray(["small", "small", "large", "large", "large", "large", "large", "large"])
    groups = np.asarray([f"g{i}" for i in range(len(labels))])
    output = np.asarray([0.4, 0.6, 0.45, 0.55, 0.45, 0.55, 0.45, 0.55])
    joint = np.asarray([0.1, 0.9, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8])

    summary = summarize_predictions(
        labels,
        domains,
        groups,
        {"output_only": output, "output_plus_hidden": joint},
        n_boot=50,
        seed=7,
        contrasts=(
            ContrastSpec(
                "hidden_given_output_nll", "output_only", "output_plus_hidden"
            ),
        ),
    )

    per_domain = summary["arms"]["output_plus_hidden"]["per_domain"]
    expected_macro = np.mean([values["nll_nats"] for values in per_domain.values()])
    assert summary["arms"]["output_plus_hidden"]["macro"]["nll_nats"] == expected_macro
    assert summary["increments"]["hidden_given_output_nll"]["point"] > 0


def test_randomization_summary_reports_multiple_null_seeds_without_calling_it_a_ci():
    labels = np.asarray([0, 1] * 4)
    domains = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    groups = np.asarray([f"g{i}" for i in range(8)])
    ordered = np.asarray([0.2, 0.8] * 4)
    probabilities = {
        "ordered": ordered,
        "time_null_r0": np.asarray([0.4, 0.6] * 4),
        "time_null_r1": np.asarray([0.45, 0.55] * 4),
    }

    summary = summarize_predictions(
        labels,
        domains,
        groups,
        probabilities,
        n_boot=10,
        seed=3,
        randomizations=(
            RandomizationSpec("time_order", "time_null_r", "ordered", 2),
        ),
        visible_steps=np.full(8, 2),
    )

    result = summary["randomization_checks"]["time_order"]
    assert result["n_randomizations"] == 2
    assert len(result["seed_points"]) == 2
    assert result["uncertainty_kind"] == "randomization_seed_sensitivity_not_ci"


def test_randomization_summary_marks_an_empty_identifiable_stratum_unavailable():
    labels = np.asarray([0, 1] * 2)
    domains = np.asarray(["a", "a", "b", "b"])
    groups = np.asarray([f"g{i}" for i in range(4)])
    probability = np.asarray([0.2, 0.8, 0.2, 0.8])

    summary = summarize_predictions(
        labels,
        domains,
        groups,
        {"ordered": probability, "time_null_r0": probability},
        n_boot=5,
        seed=3,
        randomizations=(
            RandomizationSpec("time_order", "time_null_r", "ordered", 3),
        ),
        visible_steps=np.full(4, 2),
    )

    assert summary["randomization_checks"]["time_order"] == {
        "status": "not_identifiable",
        "candidate": "ordered",
        "null_arms": ["time_null_r0"],
        "identifiable_rows": 0,
        "minimum_visible_steps": 3,
        "estimand_scope": "exploratory_axis_randomization_diagnostic",
        "training_scope": "all_outer_training_rows_not_stratum_refit",
        "claim_status": "sensitivity_only_not_axis_order_evidence",
    }
