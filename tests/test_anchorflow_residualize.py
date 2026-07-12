import numpy as np

from anchorflow.residualize import crossfit_residualize
from anchorflow.eval import impute_from_train


def test_crossfit_residualize_removes_simple_control_trend():
    rng = np.random.default_rng(0)
    n = 120
    groups = np.arange(n)
    x = np.linspace(-1, 1, n)
    controls = np.column_stack([x, x**2])
    score = 3 * x + 0.1 * rng.normal(size=n)
    resid = crossfit_residualize(score, controls, groups, folds=5)
    m = np.isfinite(resid)
    assert m.sum() == n
    assert abs(np.corrcoef(resid[m], x[m])[0, 1]) < 0.35


def test_fold_imputation_uses_training_values_only():
    train = np.array([[1.0, np.nan], [3.0, np.nan]])
    test = np.array([[1000.0, 9.0], [np.nan, np.nan]])
    tr, te, fills = impute_from_train(train, test)
    np.testing.assert_allclose(fills, [2.0, 0.0])
    np.testing.assert_allclose(tr, [[1.0, 0.0], [3.0, 0.0]])
    # The all-missing training column is disabled even though test has a value.
    np.testing.assert_allclose(te, [[1000.0, 0.0], [2.0, 0.0]])
