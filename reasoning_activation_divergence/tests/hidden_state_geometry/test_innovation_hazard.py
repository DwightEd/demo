from __future__ import annotations

import numpy as np
import pytest
from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.method import FoldInput
from functional_divergence.hidden_state_geometry.methods import load_builtin_methods
from functional_divergence.hidden_state_geometry.methods.innovation_hazard import (
    InnovationHazardConfig,
)
from functional_divergence.hidden_state_geometry.registry import (
    create_method,
    method_spec,
)
from functional_divergence.hidden_state_geometry.tasks import (
    build_strict_prefix_task,
    build_whole_chain_task,
)
from sklearn.metrics import roc_auc_score


def _sample(
    tmp_path,
    *,
    domain: str,
    chain_id: int,
    gold: int,
    rng: np.random.Generator,
) -> ChainSample:
    n_steps = 4
    source = rng.normal(size=(n_steps, 6))
    destination = 0.75 * source + 0.15 * rng.normal(size=(n_steps, 6))
    if gold >= 1:
        destination[gold - 1] += np.asarray([8.0, -6.0, 4.0, 0.0, 0.0, 0.0])
    states = np.stack([source, destination], axis=1).astype(np.float32)
    path = tmp_path / f"{domain}_{chain_id}.npy"
    np.save(path, states)
    return ChainSample(
        chain_id=chain_id,
        manifest_row=chain_id,
        problem_group=f"problem-{chain_id}",
        dataset=domain,
        generator="llama31-8b",
        observer_model="llama31-8b",
        state_path=path,
        state_count=n_steps,
        response_start=10,
        step_ranges=np.asarray(
            [[10 + step, 10 + step] for step in range(n_steps)], dtype=np.int64
        ),
        layer_ids=np.asarray([14, 16], dtype=np.int64),
        output_steps=np.zeros((n_steps, 2), dtype=np.float32),
        output_feature_names=("token_entropy", "token_nll"),
        first_error_step=gold,
    )


def _fold(tmp_path) -> tuple[FoldInput, np.ndarray]:
    rng = np.random.default_rng(41)
    samples = []
    chain_id = 0
    for domain in ("train_a", "train_b", "train_c", "test"):
        for index in range(24):
            gold = 1 + index % 3 if index % 2 else -1
            samples.append(
                _sample(
                    tmp_path,
                    domain=domain,
                    chain_id=chain_id,
                    gold=gold,
                    rng=rng,
                )
            )
            chain_id += 1
    task = build_strict_prefix_task(tuple(samples))
    train = np.flatnonzero(task.domains != "test")
    test = np.flatnonzero(task.domains == "test")
    return (
        FoldInput(
            task_name=task.name,
            train_examples=tuple(task.examples[index] for index in train),
            train_labels=task.labels[train],
            train_groups=task.groups[train],
            test_examples=tuple(task.examples[index] for index in test),
            seed=17,
        ),
        task.labels[test],
    )


def test_innovation_hazard_is_a_registered_strict_prefix_method() -> None:
    load_builtin_methods()

    specification = method_spec("innovation_hazard")
    method = create_method(
        "innovation_hazard",
        {
            "source_layer": 14,
            "destination_layer": 16,
            "rank": 3,
            "l2": 0.01,
        },
    )

    assert isinstance(method.config, InnovationHazardConfig)
    assert set(specification.arm_definitions) == {
        "nuisance",
        "output_only",
        "hidden_only",
        "innovation_only",
        "output_plus_hidden",
        "output_plus_innovation",
    }
    assert {(item.baseline, item.candidate) for item in specification.contrasts} == {
        ("nuisance", "output_only"),
        ("output_only", "output_plus_hidden"),
        ("output_only", "output_plus_innovation"),
        ("output_plus_hidden", "output_plus_innovation"),
    }


def test_innovation_hazard_recovers_a_cross_domain_transition_anomaly(tmp_path) -> None:
    load_builtin_methods()
    fold, test_labels = _fold(tmp_path)
    method = create_method(
        "innovation_hazard",
        {
            "source_layer": 14,
            "destination_layer": 16,
            "rank": 3,
            "normal_ridge_alpha": 1.0,
            "covariance_shrinkage": 0.1,
            "l2": 0.01,
            "max_iter": 1000,
        },
    )

    result = method.fit_predict(fold)

    innovation_auc = roc_auc_score(
        test_labels, result.probabilities["output_plus_innovation"]
    )
    output_auc = roc_auc_score(test_labels, result.probabilities["output_only"])
    assert innovation_auc > 0.9
    assert innovation_auc > output_auc + 0.2
    assert result.diagnostics["target"] == "discrete_time_first_error_hazard"
    assert result.diagnostics["post_error_policy"] == "censored"
    assert result.diagnostics["normal_bank_rows"] == int(np.sum(fold.train_labels == 0))
    assert result.diagnostics["layers"] == [14, 16]
    dimensions = result.diagnostics["arm_feature_dimensions"]
    assert dimensions["hidden_only"] == dimensions["innovation_only"]
    assert dimensions["output_plus_hidden"] == dimensions["output_plus_innovation"]
    assert all(
        np.isfinite(probability).all() for probability in result.probabilities.values()
    )


def test_innovation_hazard_rejects_retrospective_whole_chain_rows(tmp_path) -> None:
    rng = np.random.default_rng(3)
    samples = tuple(
        _sample(
            tmp_path,
            domain="train" if index < 4 else "test",
            chain_id=index,
            gold=2 if index % 2 else -1,
            rng=rng,
        )
        for index in range(8)
    )
    task = build_whole_chain_task(samples)
    fold = FoldInput(
        task_name=task.name,
        train_examples=task.examples[:4],
        train_labels=task.labels[:4],
        train_groups=task.groups[:4],
        test_examples=task.examples[4:],
        seed=5,
    )

    with pytest.raises(ValueError, match="strict_prefix"):
        create_method("innovation_hazard", None).fit_predict(fold)
