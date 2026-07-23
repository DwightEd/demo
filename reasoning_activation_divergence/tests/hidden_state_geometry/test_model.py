from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from functional_divergence.hidden_state_geometry import model as model_module
from functional_divergence.hidden_state_geometry.model import (
    RankOneLogistic,
    RegularizedLogistic,
)


def test_regularized_logistic_rejects_invalid_warm_start_parameters():
    values = np.arange(24, dtype=np.float64).reshape(8, 3)
    labels = np.asarray([0, 1] * 4, dtype=np.int8)

    with pytest.raises(ValueError, match="initial_parameters must be finite"):
        RegularizedLogistic(l2=0.1).fit(
            values, labels, initial_parameters=np.zeros(values.shape[1])
        )
    with pytest.raises(ValueError, match="initial_parameters must be finite"):
        RegularizedLogistic(l2=0.1).fit(
            values,
            labels,
            initial_parameters=np.full(values.shape[1] + 1, np.nan),
        )


def test_regularized_logistic_exposes_convergence_diagnostics():
    rng = np.random.default_rng(29)
    values = rng.normal(size=(40, 4))
    labels = (values[:, 0] > 0).astype(np.int8)

    model = RegularizedLogistic(l2=0.1, max_iter=500).fit(values, labels)

    assert model.converged_
    assert model.iterations_ is not None and model.iterations_ >= 0
    assert model.objective_ is not None and np.isfinite(model.objective_)
    assert model.gradient_inf_norm_ is not None
    assert np.isfinite(model.gradient_inf_norm_)
    assert isinstance(model.message_, str)


def test_regularized_logistic_rejects_nonfinite_gradient_and_clears_old_fit(
    monkeypatch,
):
    values = np.arange(24, dtype=np.float64).reshape(8, 3)
    labels = np.asarray([0, 1] * 4, dtype=np.int8)
    model = RegularizedLogistic(l2=0.1, max_iter=50).fit(values, labels)

    class FakeResult:
        success = True
        fun = 0.25
        x = np.zeros(values.shape[1] + 1)
        jac = np.full(values.shape[1] + 1, np.nan)
        nit = 7
        message = "synthetic nonfinite gradient"

    monkeypatch.setattr(model_module, "minimize", lambda *args, **kwargs: FakeResult())

    with pytest.raises(RuntimeError, match="grad_inf=nan"):
        model.fit(values, labels)

    assert model.parameters_ is None
    assert model.converged_ is False
    assert model.gradient_inf_norm_ is not None
    assert np.isnan(model.gradient_inf_norm_)


def test_rank_one_probe_recovers_a_synthetic_tensor_signal():
    rng = np.random.default_rng(11)
    tensor = rng.normal(size=(240, 3, 4, 5))
    factors = (
        np.asarray([1.0, -0.5, 0.25]),
        np.asarray([0.2, 1.0, -0.7, 0.3]),
        np.asarray([0.8, -0.5, 0.2, 0.1, -0.3]),
    )
    signal = np.einsum("nabc,a,b,c->n", tensor, *factors)
    labels = (signal + rng.normal(scale=0.35, size=len(signal)) > 0).astype(np.int8)
    model = RankOneLogistic(l2=0.05, restarts=4, max_iter=500, seed=7)

    model.fit(tensor[:180], labels[:180])
    probability = model.predict_proba(tensor[180:])

    assert roc_auc_score(labels[180:], probability) > 0.9
    assert model.signal_parameters == sum(tensor.shape[1:]) - 2


def test_rank_one_probe_coefficient_tensor_is_separable():
    rng = np.random.default_rng(12)
    tensor = rng.normal(size=(80, 2, 3, 4))
    labels = (tensor[:, 1, 2, 3] > 0).astype(np.int8)
    model = RankOneLogistic(l2=0.1, restarts=3, max_iter=400, seed=9).fit(
        tensor, labels
    )

    coefficient = model.coefficient_tensor
    singular = np.linalg.svd(coefficient.reshape(2, -1), compute_uv=False)

    assert singular[0] > 0
    assert singular[1] < singular[0] * 1e-8


def test_static_baseline_matches_rank_one_objective_when_tensor_is_zero():
    rng = np.random.default_rng(19)
    static = rng.normal(size=(100, 4))
    labels = (static[:, 0] - 0.7 * static[:, 2] > 0).astype(np.int8)
    weights = np.linspace(0.5, 1.5, len(labels))
    zero = np.zeros((len(labels), 2, 2, 3), dtype=np.float64)

    baseline = RegularizedLogistic(l2=0.3, max_iter=500).fit(
        static, labels, sample_weight=weights
    )
    nested = RankOneLogistic(l2=0.3, restarts=1, max_iter=500, seed=2).fit(
        zero, labels, static=static, sample_weight=weights
    )

    assert np.allclose(
        baseline.predict_proba(static), nested.predict_proba(zero, static), atol=1e-6
    )


def test_joint_probe_contains_the_fitted_static_baseline_as_an_exact_candidate():
    rng = np.random.default_rng(23)
    static = rng.normal(size=(120, 3))
    tensor = rng.normal(size=(120, 2, 2, 4))
    labels = (static[:, 0] + 0.2 * rng.normal(size=120) > 0).astype(np.int8)
    baseline = RegularizedLogistic(l2=0.5, max_iter=500).fit(static, labels)

    joint = RankOneLogistic(l2=0.5, restarts=2, max_iter=300, seed=4).fit(
        tensor,
        labels,
        static=static,
        baseline_parameters=baseline.coefficients,
    )

    assert joint.objective_ <= joint.baseline_objective_ + 1e-12
