from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
except ImportError as exc:  # pragma: no cover
    raise SystemExit("anchorflow.residualize needs scikit-learn") from exc


def crossfit_residualize(score, controls, groups, *, folds: int = 5, seed: int = 0) -> np.ndarray:
    s = np.asarray(score, float)
    C = np.asarray(controls, float)
    if C.ndim == 1:
        C = C[:, None]
    groups = np.asarray(groups)
    out = np.full(len(s), np.nan)
    m = np.isfinite(s) & np.all(np.isfinite(C), axis=1)
    if m.sum() < 30:
        return out
    idx = np.where(m)[0]
    n_splits = min(int(folds), len(np.unique(groups[idx])))
    if n_splits < 2:
        out[idx] = s[idx] - np.nanmean(s[idx])
        return out
    for tr0, te0 in GroupKFold(n_splits=n_splits).split(C[idx], s[idx], groups[idx]):
        tr = idx[tr0]
        te = idx[te0]
        model = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=seed)
        model.fit(C[tr], s[tr])
        out[te] = s[te] - model.predict(C[te])
    return out
