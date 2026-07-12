from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class HazardTargets:
    target: np.ndarray
    at_risk: np.ndarray
    lengths: np.ndarray
    first_error: np.ndarray
    event_observed: np.ndarray


def make_first_error_hazard_targets(
    lengths: Sequence[int],
    first_error: Sequence[Optional[int]],
    *,
    max_steps: Optional[int] = None,
) -> HazardTargets:
    """Create discrete first-error targets with correct right-censoring.

    Correct chains use ``first_error=-1`` (or ``None``): every observed step is
    at risk with target zero, and the chain is right-censored at its final step.
    For an error chain, the first-error step has target one and every later step
    is masked because it is no longer at risk for the *first* event.
    """
    lens = np.asarray(lengths, int)
    if lens.ndim != 1 or np.any(lens < 0):
        raise ValueError("lengths must be a non-negative vector")
    raw = list(first_error)
    if len(raw) != len(lens):
        raise ValueError("lengths and first_error must have the same size")
    err = np.asarray([-1 if e is None else int(e) for e in raw], int)
    width = int(max_steps) if max_steps is not None else int(lens.max(initial=0))
    if width < int(lens.max(initial=0)):
        raise ValueError("max_steps truncates an observed chain")

    y = np.zeros((len(lens), width), dtype=float)
    risk = np.zeros((len(lens), width), dtype=bool)
    observed = err >= 0
    for i, (length, event) in enumerate(zip(lens, err)):
        if event >= length and event >= 0:
            raise ValueError(f"first_error[{i}] lies outside the observed chain")
        stop = int(event + 1) if event >= 0 else int(length)
        risk[i, :stop] = True
        if event >= 0:
            y[i, event] = 1.0
    return HazardTargets(y, risk, lens, err, observed)


def discrete_hazard_nll(
    logits: np.ndarray,
    target: np.ndarray,
    at_risk: np.ndarray,
    *,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Stable Bernoulli negative log-likelihood over at-risk positions only."""
    z = np.asarray(logits, float)
    y = np.asarray(target, float)
    m = np.asarray(at_risk, bool)
    if z.shape != y.shape or z.shape != m.shape:
        raise ValueError("logits, target and at_risk shapes must match")
    loss = np.logaddexp(0.0, z) - y * z
    if reduction == "none":
        return np.where(m, loss, np.nan)
    vals = loss[m]
    if reduction == "sum":
        return float(vals.sum())
    if reduction == "mean":
        return float(vals.mean()) if vals.size else float("nan")
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def hazard_to_event_cdf(hazard: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Convert conditional hazards to cumulative first-event probability."""
    h = np.clip(np.asarray(hazard, float), 0.0, 1.0)
    survival = np.cumprod(1.0 - h, axis=axis)
    return 1.0 - survival


class DiscreteHazardReadout:
    """Small sklearn readout trained only on first-error at-risk positions."""

    def __init__(self, *, class_weight: str | dict | None = "balanced", max_iter: int = 3000):
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.fill_: Optional[np.ndarray] = None
        self.model_ = None

    @staticmethod
    def _flatten_at_risk(feature_sequences, targets: HazardTargets):
        rows = []
        labels = []
        for i, seq in enumerate(feature_sequences):
            X = np.asarray(seq, float)
            if X.ndim == 1:
                X = X[:, None]
            if X.ndim != 2 or len(X) < targets.lengths[i]:
                raise ValueError("feature sequence does not cover its observed length")
            idx = np.where(targets.at_risk[i, : len(X)])[0]
            rows.append(X[idx])
            labels.append(targets.target[i, idx])
        return np.vstack(rows), np.concatenate(labels).astype(int)

    def fit(self, feature_sequences, lengths, first_error):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        targets = make_first_error_hazard_targets(lengths, first_error)
        X, y = self._flatten_at_risk(feature_sequences, targets)
        if len(np.unique(y)) < 2:
            raise ValueError("hazard training requires both event and non-event positions")
        fill = np.zeros(X.shape[1], float)
        for j in range(X.shape[1]):
            good = np.isfinite(X[:, j])
            fill[j] = float(np.mean(X[good, j])) if good.any() else 0.0
            X[~good, j] = fill[j]
        self.fill_ = fill
        self.model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=self.max_iter,
                class_weight=self.class_weight,
            ),
        )
        self.model_.fit(X, y)
        return self

    def predict_hazard(self, feature_sequence: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.fill_ is None:
            raise RuntimeError("fit the hazard readout before prediction")
        X = np.asarray(feature_sequence, float).copy()
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[1] != len(self.fill_):
            raise ValueError("feature width differs from fitted hazard readout")
        for j, fill in enumerate(self.fill_):
            X[~np.isfinite(X[:, j]), j] = fill
        return self.model_.predict_proba(X)[:, 1]


def grouped_oof_hazard(
    feature_sequences: Sequence[np.ndarray],
    lengths: Sequence[int],
    first_error: Sequence[Optional[int]],
    groups: Sequence[object],
    *,
    folds: int = 5,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 3000,
) -> Dict[str, object]:
    """Problem-grouped OOF hazard predictions with fold-local preprocessing.

    Splits operate on whole chains.  The readout's missing-value fills and
    standardization are fitted only on training chains, while correct test
    chains remain right-censored and post-error test positions are excluded
    from evaluation through the returned ``at_risk`` mask.
    """
    from sklearn.model_selection import GroupKFold

    seqs = [np.asarray(x, float) for x in feature_sequences]
    lens = np.asarray(lengths, int)
    errs = list(first_error)
    grp = np.asarray(groups)
    n = len(seqs)
    if not (len(lens) == len(errs) == len(grp) == n):
        raise ValueError("feature_sequences, lengths, first_error and groups must align")
    n_splits = min(int(folds), len(np.unique(grp)))
    if n_splits < 2:
        raise ValueError("grouped OOF hazard requires at least two unique groups")

    predictions: List[np.ndarray] = [np.full(int(length), np.nan) for length in lens]
    fold_id = np.full(n, -1, dtype=int)
    skipped: List[int] = []
    dummy = np.zeros(n, dtype=int)
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(dummy, dummy, grp)):
        reader = DiscreteHazardReadout(class_weight=class_weight, max_iter=max_iter)
        try:
            reader.fit(
                [seqs[i] for i in train],
                lens[train],
                [errs[i] for i in train],
            )
        except ValueError:
            skipped.append(int(fold))
            continue
        for i in test:
            predictions[int(i)] = reader.predict_hazard(seqs[int(i)][: lens[int(i)]])
            fold_id[int(i)] = int(fold)

    targets = make_first_error_hazard_targets(lens, errs)
    return {
        "hazard": predictions,
        "event_cdf": [hazard_to_event_cdf(x) for x in predictions],
        "target": targets.target,
        "at_risk": targets.at_risk,
        "fold_id": fold_id,
        "skipped_folds": skipped,
    }
