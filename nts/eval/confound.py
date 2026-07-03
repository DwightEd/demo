# nts/eval/confound.py — cross-fit residualization, OOF logistic, cluster-bootstrap increment
import warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .metrics import auroc, bdir


def _col_means(X):
    """nanmean per column; all-NaN columns (e.g. speed when every row is t=0) -> 0
    without numpy's benign-but-noisy 'Mean of empty slice' RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(X, 0)
    mu[~np.isfinite(mu)] = 0.0
    return mu


def residualize(value, X, correct_mask, groups, folds=5):
    value = np.asarray(value, float); X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    out = np.full(len(value), np.nan)
    for tr, te in GroupKFold(folds).split(X, value, groups):
        htr = tr[correct_mask[tr] & np.isfinite(value[tr])]
        if len(htr) < 20:
            continue
        mu = _col_means(X[htr])   # impute NaN covariates (e.g. speed@t=0)
        Xtr = np.where(np.isfinite(X[htr]), X[htr], mu)
        Xte = np.where(np.isfinite(X[te]), X[te], mu)
        reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
        reg.fit(Xtr, value[htr])
        out[te] = value[te] - reg.predict(Xte)   # value[te]=NaN at t=0 keeps out NaN there
    return out


def oof_logit(X, y, g, folds=5):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, Xte = X[tr].copy(), X[te].copy()
        mu = _col_means(Xtr)
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        p.fit(Xtr, y[tr]); s[te] = p.predict_proba(Xte)[:, 1]
    return s


def cluster_boot_increment(s_full, s_base, y, groups, nboot=500, seed=0):
    rng = np.random.default_rng(seed); gids = np.unique(groups)
    by = {c: np.where(groups == c)[0] for c in gids}; d = []
    for _ in range(nboot):
        take = np.concatenate([by[c] for c in rng.choice(gids, len(gids), replace=True)])
        d.append(bdir(auroc(s_full[take], y[take])) - bdir(auroc(s_base[take], y[take])))
    d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(np.nanmean(d)), float(lo), float(hi), bool(lo > 0)
