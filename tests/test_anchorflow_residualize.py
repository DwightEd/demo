import numpy as np

from anchorflow.residualize import crossfit_residualize


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
