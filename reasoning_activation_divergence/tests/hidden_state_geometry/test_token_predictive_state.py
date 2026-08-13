from __future__ import annotations

import numpy as np

from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.method import FoldInput
from functional_divergence.hidden_state_geometry.methods import load_builtin_methods
from functional_divergence.hidden_state_geometry.methods.token_predictive_state import (
    TokenPredictiveStateConfig,
    _lagged_transitions,
    _shuffle_older_lags,
    fit_token_dynamics,
)
from functional_divergence.hidden_state_geometry.registry import create_method
from functional_divergence.hidden_state_geometry.tasks import build_strict_prefix_task


def test_lagged_transitions_never_use_the_target_or_future_as_predictors():
    sequence = np.arange(7, dtype=np.float64)[:, None]

    lags, targets = _lagged_transitions(sequence, order=3)

    assert targets[:, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert lags[:, :, 0].tolist() == [
        [2.0, 1.0, 0.0],
        [3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0],
        [5.0, 4.0, 3.0],
    ]


def test_shuffle_control_keeps_current_state_and_only_reorders_older_history():
    lags = np.asarray(
        [
            [[10.0], [20.0], [30.0], [40.0]],
            [[11.0], [21.0], [31.0], [41.0]],
            [[12.0], [22.0], [32.0], [42.0]],
            [[13.0], [23.0], [33.0], [43.0]],
        ]
    )

    shuffled = _shuffle_older_lags(lags, seed=19)

    assert np.array_equal(shuffled[:, 0], lags[:, 0])
    for lag in range(1, lags.shape[1]):
        assert sorted(shuffled[:, lag, 0].tolist()) == sorted(
            lags[:, lag, 0].tolist()
        )
    assert not np.array_equal(shuffled[:, 1:], lags[:, 1:])


def test_ordered_finite_history_recovers_second_order_dynamics_beyond_ar1():
    rng = np.random.default_rng(7)
    sequences = []
    for _ in range(32):
        values = [rng.normal(size=2) for _ in range(3)]
        for _ in range(29):
            values.append(
                0.65 * values[-1]
                - 0.45 * values[-2]
                + 0.25 * values[-3]
                + rng.normal(scale=0.01, size=2)
            )
        sequences.append(np.asarray(values, dtype=np.float64))

    train = tuple(sequences[:24])
    test = tuple(sequences[24:])
    model = fit_token_dynamics(
        train,
        order=3,
        ridge_alpha=1e-6,
        transitions_per_chain=0,
        seed=11,
    )

    diagnostics = model.evaluate(test, transitions_per_chain=0, seed=23)

    assert diagnostics["ordered_nmse"] < diagnostics["ar1_nmse"] * 0.25
    assert diagnostics["ordered_nmse"] < diagnostics["shuffled_nmse"]
    assert diagnostics["history_gain_nmse"] > 0.0
    assert diagnostics["history_order_gain_nmse"] > 0.0


def test_prefix_features_are_unchanged_when_only_future_tokens_change():
    rng = np.random.default_rng(31)
    training = tuple(rng.normal(size=(20, 3)) for _ in range(12))
    model = fit_token_dynamics(
        training,
        order=4,
        ridge_alpha=1.0,
        transitions_per_chain=0,
        seed=5,
    )
    sequence = rng.normal(size=(18, 3))
    changed_future = sequence.copy()
    changed_future[10:] += 1000.0

    original = model.prefix_features(sequence[:10], recent_window=6)
    counterfactual = model.prefix_features(changed_future[:10], recent_window=6)

    assert np.array_equal(original, counterfactual)
    assert original.shape == (8,)
    assert np.isfinite(original).all()


def _chain_sample(tmp_path, chain_id: int, *, dataset: str, error: bool) -> ChainSample:
    rng = np.random.default_rng(chain_id)
    token_states = np.empty((15, 2, 6), dtype=np.float32)
    values = [rng.normal(size=(2, 6)) for _ in range(3)]
    for _ in range(12):
        values.append(
            0.7 * values[-1]
            - 0.4 * values[-2]
            + 0.2 * values[-3]
            + rng.normal(scale=0.02, size=(2, 6))
        )
    token_states[:] = np.asarray(values, dtype=np.float32)
    if error:
        token_states[9:, :, 0] += np.linspace(0.0, 0.5, 6)[:, None]
    path = tmp_path / f"chain_{chain_id}.npy"
    np.save(path, token_states)
    return ChainSample(
        chain_id=chain_id,
        manifest_row=chain_id,
        problem_group=f"problem_sha256:p{chain_id}",
        dataset=dataset,
        generator="llama31-8b",
        observer_model="llama31-8b",
        state_path=path,
        state_count=15,
        response_start=10,
        step_ranges=np.asarray(
            [[10 + 3 * step, 12 + 3 * step] for step in range(5)]
        ),
        layer_ids=np.asarray([8, 10]),
        output_steps=np.column_stack(
            [np.linspace(0.1, 0.5, 5), np.linspace(0.5, 0.1, 5)]
        ).astype(np.float32),
        output_feature_names=("token_entropy", "token_nll"),
        first_error_step=4 if error else -1,
    )


def test_plugin_audits_token_dynamics_then_uses_explicit_features_for_hazard(tmp_path):
    samples = tuple(
        _chain_sample(
            tmp_path,
            chain_id,
            dataset="train-a" if chain_id < 8 else "train-b" if chain_id < 16 else "test",
            error=bool(chain_id % 2),
        )
        for chain_id in range(24)
    )
    task = build_strict_prefix_task(samples)
    train = np.asarray(
        [index for index, row in enumerate(task.examples) if row.sample.dataset != "test"]
    )
    test = np.asarray(
        [index for index, row in enumerate(task.examples) if row.sample.dataset == "test"]
    )
    fold = FoldInput(
        task_name="strict_prefix",
        train_examples=tuple(task.examples[index] for index in train),
        train_labels=task.labels[train],
        train_groups=task.groups[train],
        test_examples=tuple(task.examples[index] for index in test),
        seed=17,
    )
    load_builtin_methods()
    method = create_method(
        "token_predictive_state",
        TokenPredictiveStateConfig(
            pca_dim=2,
            positions_per_chain=4,
            history_order=4,
            transitions_per_chain=0,
            recent_window=4,
        ),
    )

    result = method.fit_predict(fold)

    assert set(result.probabilities) == {
        "output_only",
        "ar1_innovation",
        "ordered_history",
        "shuffled_history",
    }
    assert all(values.shape == (len(test),) for values in result.probabilities.values())
    assert all(np.isfinite(values).all() for values in result.probabilities.values())
    assert result.diagnostics["analysis_unit"] == "token_transition"
    assert result.diagnostics["dynamics_fit_scope"] == (
        "outer_train_unique_at_risk_chains_label_free_loss"
    )
    assert result.diagnostics["censoring_uses_first_error_labels"] is True
    assert result.diagnostics["hazard_model"] == "regularized_logistic_no_neural_network"
    assert result.diagnostics["strict_markov_claim"] is False
    assert result.diagnostics["test_transition_diagnostics"]["transitions"] > 0
    assert result.diagnostics["arm_feature_dimensions"]["ordered_history"] == (
        result.diagnostics["arm_feature_dimensions"]["shuffled_history"]
    )
