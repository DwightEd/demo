from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from functional_divergence.hidden_state_geometry.methods import (
    full_tensor_ridge as ridge_module,
)
from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.method import FoldInput
from functional_divergence.hidden_state_geometry.model import RegularizedLogistic
from functional_divergence.hidden_state_geometry.methods import load_builtin_methods
from functional_divergence.hidden_state_geometry.methods.full_tensor_ridge import (
    FullTensorRidgeConfig,
    fit_ridge_path,
    inner_lodo_splits,
)
from functional_divergence.hidden_state_geometry.preprocessing import (
    FiniteStandardizer,
    domain_group_balanced_weights,
)
from functional_divergence.hidden_state_geometry.registry import (
    create_method,
    method_spec,
)
from functional_divergence.hidden_state_geometry.tasks import build_whole_chain_task


def _correlated_ridge_data() -> tuple[np.ndarray, np.ndarray]:
    """Fixed high-correlation problem that needs strong-to-weak continuation."""
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(72, 10))
    loading = rng.normal(size=(10, 96))
    values = latent @ loading + 0.05 * rng.normal(size=(72, 96))
    weights = np.ones(len(values), dtype=np.float64)
    standardized = FiniteStandardizer().fit(values, weights).transform(values)
    beta = rng.normal(size=96)
    labels = (standardized @ beta + 0.5 * rng.normal(size=72) > 0).astype(np.int8)
    return standardized, labels


def test_ridge_path_uses_strong_to_weak_warm_starts(monkeypatch):
    calls = []

    class FakeRidge:
        def __init__(self, *, l2, max_iter):
            self.l2 = l2
            self.max_iter = max_iter
            self._coefficients = np.asarray([l2, max_iter], dtype=np.float64)

        def fit(self, values, labels, sample_weight, initial_parameters=None):
            del values, labels, sample_weight
            calls.append((self.l2, initial_parameters))
            return self

        @property
        def coefficients(self):
            return self._coefficients.copy()

    monkeypatch.setattr(ridge_module, "RegularizedLogistic", FakeRidge)
    fitted = fit_ridge_path(
        np.ones((4, 2)),
        np.asarray([0, 1, 0, 1]),
        np.ones(4),
        (0.001, 0.1, 0.01),
        77,
    )

    assert tuple(fitted) == (0.1, 0.01, 0.001)
    assert [l2 for l2, _ in calls] == [0.1, 0.01, 0.001]
    assert calls[0][1] is None
    assert np.array_equal(calls[1][1], np.asarray([0.1, 77.0]))
    assert np.array_equal(calls[2][1], np.asarray([0.01, 77.0]))


def test_ridge_path_stops_after_the_first_failed_path_point(monkeypatch):
    calls = []

    class FakeRidge:
        def __init__(self, *, l2, max_iter):
            self.l2 = l2
            self._coefficients = np.asarray([l2, max_iter], dtype=np.float64)

        def fit(self, values, labels, sample_weight, initial_parameters=None):
            del values, labels, sample_weight, initial_parameters
            calls.append(self.l2)
            if self.l2 == 0.01:
                raise RuntimeError("synthetic optimizer failure")
            return self

        @property
        def coefficients(self):
            return self._coefficients.copy()

    monkeypatch.setattr(ridge_module, "RegularizedLogistic", FakeRidge)

    with pytest.raises(RuntimeError, match="ridge path failed at l2=0.01"):
        fit_ridge_path(
            np.ones((4, 2)),
            np.asarray([0, 1, 0, 1]),
            np.ones(4),
            (0.1, 0.01, 0.001),
            77,
        )

    assert calls == [0.1, 0.01]


def test_ridge_path_matches_independent_convex_fits_with_a_generous_budget():
    values, labels = _correlated_ridge_data()
    l2_path = (0.1, 0.01, 0.001)
    weights = np.ones(len(labels), dtype=np.float64)
    fitted = fit_ridge_path(values, labels, weights, l2_path, 2000)
    cold = {
        l2: RegularizedLogistic(l2=l2, max_iter=2000).fit(values, labels, weights)
        for l2 in l2_path
    }

    for l2 in l2_path:
        assert np.isclose(
            fitted[l2].objective_,
            cold[l2].objective_,
            rtol=1e-4,
            atol=1e-6,
        )
        assert np.allclose(
            fitted[l2].predict_proba(values),
            cold[l2].predict_proba(values),
            rtol=1e-5,
            atol=2e-3,
        )


def _rank_two_fold(tmp_path) -> tuple[FoldInput, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    coefficient = np.zeros((2, 2, 2), dtype=np.float64)
    coefficient[0, 0, 0] = 1.0
    coefficient[1, 1, 1] = 1.0
    samples = []
    chain_id = 0
    for domain in ("train_a", "train_b", "train_c", "test"):
        for pair in range(12):
            base = rng.normal(size=(2, 2, 2))
            margin = float(np.sum(base * coefficient))
            while abs(margin) < 0.5:
                base = rng.normal(size=(2, 2, 2))
                margin = float(np.sum(base * coefficient))
            for sign in (1.0, -1.0):
                label = int(sign * margin > 0)
                path = tmp_path / f"rank2_{chain_id}.npy"
                np.save(path, np.asarray(sign * base, dtype=np.float32))
                samples.append(
                    ChainSample(
                        chain_id=chain_id,
                        manifest_row=chain_id,
                        problem_group=f"{domain}-pair-{pair}",
                        dataset=domain,
                        generator="llama",
                        observer_model="llama",
                        state_path=path,
                        state_count=2,
                        response_start=10,
                        step_ranges=np.asarray([[10, 10], [11, 11]]),
                        layer_ids=np.asarray([8, 10]),
                        output_steps=np.zeros((2, 2), dtype=np.float32),
                        output_feature_names=("entropy", "nll"),
                        first_error_step=1 if label else -1,
                    )
                )
                chain_id += 1
    task = build_whole_chain_task(tuple(samples))
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
        coefficient,
    )


def test_full_tensor_ridge_is_an_independent_registered_method():
    load_builtin_methods()
    specification = method_spec("full_tensor_ridge")
    method = create_method(
        "full_tensor_ridge",
        {
            "pca_dim": 2,
            "time_basis": 2,
            "layer_basis": 2,
            "positions_per_chain": 2,
            "l2_grid": [1e-4, 1e-2],
            "max_iter": 300,
        },
    )

    assert isinstance(method.config, FullTensorRidgeConfig)
    assert FullTensorRidgeConfig().max_iter == 2000
    assert isinstance(
        create_method("full_tensor_ridge", None).config, FullTensorRidgeConfig
    )
    assert method.config.l2_grid == (1e-4, 1e-2)
    assert set(specification.arm_definitions) == {
        "nuisance",
        "output_only",
        "hidden_only",
        "output_plus_hidden",
    }
    assert {
        (item.name, item.baseline, item.candidate)
        for item in specification.contrasts
    } == {
        ("output_summary_given_nuisance_nll", "nuisance", "output_only"),
        ("hidden_given_nuisance_nll", "nuisance", "hidden_only"),
        (
            "hidden_given_output_summary_nll",
            "output_only",
            "output_plus_hidden",
        ),
    }
    assert specification.randomizations == ()


def test_full_tensor_config_rejects_fractional_integer_fields():
    with pytest.raises(ValueError, match="integers"):
        FullTensorRidgeConfig(pca_dim=1.5, positions_per_chain=2)


def test_inner_lodo_keeps_domains_and_problem_groups_intact(tmp_path):
    fold, _, _ = _rank_two_fold(tmp_path)

    splits = inner_lodo_splits(fold)

    assert {split.held_domain for split in splits} == {
        "train_a",
        "train_b",
        "train_c",
    }
    validation_rows = []
    domains = np.asarray(
        [example.sample.dataset for example in fold.train_examples], dtype=object
    )
    for split in splits:
        assert set(domains[split.validation]) == {split.held_domain}
        assert split.held_domain not in set(domains[split.train])
        assert set(fold.train_groups[split.train]).isdisjoint(
            fold.train_groups[split.validation]
        )
        assert set(np.unique(fold.train_labels[split.train])) == {0, 1}
        assert set(np.unique(fold.train_labels[split.validation])) == {0, 1}
        validation_rows.extend(split.validation.tolist())
    assert sorted(validation_rows) == list(range(len(fold.train_examples)))


def test_domain_group_weights_equalize_domains_and_groups():
    domains = np.asarray(["a", "a", "a", "b", "b", "b", "b"], dtype=object)
    groups = np.asarray(["a1", "a1", "a2", "b1", "b2", "b2", "b2"], dtype=object)

    weights = domain_group_balanced_weights(domains, groups)

    assert np.isclose(weights[domains == "a"].sum(), weights[domains == "b"].sum())
    for domain in ("a", "b"):
        domain_groups = np.unique(groups[domains == domain])
        totals = [
            weights[(domains == domain) & (groups == group)].sum()
            for group in domain_groups
        ]
        assert np.allclose(totals, totals[0])


def test_full_tensor_ridge_learns_a_nonseparable_functional_signal(tmp_path):
    fold, test_labels, coefficient = _rank_two_fold(tmp_path)
    method = create_method(
        "full_tensor_ridge",
        {
            "pca_dim": 2,
            "time_basis": 2,
            "layer_basis": 2,
            "positions_per_chain": 2,
            "l2_grid": [1e-4, 1e-2],
            "max_iter": 500,
        },
    )

    result = method.fit_predict(fold)

    expected = {
        "nuisance",
        "output_only",
        "hidden_only",
        "output_plus_hidden",
    }
    assert np.linalg.matrix_rank(coefficient.reshape(2, -1)) == 2
    assert set(result.probabilities) == expected
    assert all(values.shape == (24,) for values in result.probabilities.values())
    assert all(np.isfinite(values).all() for values in result.probabilities.values())
    hidden_auc = roc_auc_score(test_labels, result.probabilities["hidden_only"])
    output_auc = roc_auc_score(test_labels, result.probabilities["output_only"])
    assert hidden_auc > 0.9
    assert hidden_auc > output_auc + 0.2
    assert result.diagnostics["tensor_shape"] == [2, 2, 2]
    assert result.diagnostics["flattened_hidden_dim"] == 8
    assert set(result.diagnostics["selected_l2"]) == expected
    assert set(result.diagnostics["selected_at_grid_edge"]) == expected
    assert set(result.diagnostics["inner_cv_scores"]) == expected
    assert set(result.diagnostics["final_optimizer"]) == expected
    for diagnostics in result.diagnostics["final_optimizer"].values():
        assert diagnostics["iterations"] >= 0
        assert np.isfinite(diagnostics["objective"])
        assert np.isfinite(diagnostics["gradient_inf_norm"])
        assert isinstance(diagnostics["message"], str)
    for held_domain in result.diagnostics["selection"]["optimizer"].values():
        assert set(held_domain) == expected
        for arm_path in held_domain.values():
            assert set(arm_path) == {"0.0001", "0.01"}
            assert all(item["iterations"] >= 0 for item in arm_path.values())
    for held_domain, fitted_domains in result.diagnostics["selection"][
        "projection_train_domains"
    ].items():
        assert held_domain not in fitted_domains
    assert (
        result.diagnostics["comparison_design"]
        == "outer_lodo_inner_lodo_full_tensor_ridge"
    )
    assert (
        result.diagnostics["coefficient_coordinate_scope"]
        == "fold_local_whitened_pca_not_cross_fold_aligned"
    )
    assert result.diagnostics["axis_order_controls_in_this_method"] is False
    assert "output_plus_hidden.coefficients" in result.factors
    assert "output_plus_hidden.hidden_tensor_coefficient" in result.factors
    assert "pca_components" in result.factors
