#!/usr/bin/env python3
"""Latent-constraint EM audit for same-problem multisampling data.

This is the lightweight hidden-state model version of the current geometry
story.  It does NOT explicitly reconstruct a correct reasoning tube.  Instead,
it treats "being on the constrained reasoning manifold" as a latent state and
learns that state sequence with EM.

Inputs are the signals that have already been useful in this project:

- cloud_spread / resultant, when token clouds are stored;
- sv_out_entropy / sv_out_committal;
- sv_pr_step_exp / sv_ae_step_exp as low-dimensional second-moment proxies;
- step-vector jump from sv_vec_step_exp, averaged over one layer band.

No layer-synchrony or cross-layer tensor signal is used here.

The HMM is unsupervised during EM.  Chain labels are used only after fitting on
the training split to orient latent states into a risk score.  Evaluation is
grouped by problem id and reports same-problem paired AUROC as the headline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


def finite_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def descriptive(xs: Iterable[float]) -> Dict[str, Any]:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.percentile(a, [10, 25, 50, 75, 90])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "q10": float(q10),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _avg_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    ranks = np.arange(1, sorted_vals.size + 1, dtype=np.float64)
    out = ranks.copy()
    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            out[i:j] = ranks[i:j].mean()
        i = j
    return out


def auroc_signed(err: Iterable[float], cor: Iterable[float]) -> float:
    err = np.asarray([x for x in err if np.isfinite(x)], dtype=np.float64)
    cor = np.asarray([x for x in cor if np.isfinite(x)], dtype=np.float64)
    if err.size == 0 or cor.size == 0:
        return float("nan")
    vals = np.concatenate([err, cor])
    labels = np.concatenate([np.ones(err.size), np.zeros(cor.size)])
    order = np.argsort(vals, kind="mergesort")
    ranks = _avg_ranks(vals[order])
    full = np.empty_like(ranks)
    full[order] = ranks
    sum_pos = full[labels == 1].sum()
    U = sum_pos - err.size * (err.size + 1) / 2.0
    return float(U / (err.size * cor.size))


def safe_mean(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def window_mask(T: int, window: str) -> np.ndarray:
    if T <= 0:
        return np.zeros(0, dtype=bool)
    frac = np.arange(T) / max(1, T - 1)
    if window == "late":
        m = frac >= 0.6
        return m if m.any() else frac >= frac.max()
    if window == "early":
        m = frac < 0.4
        return m if m.any() else frac <= frac.min()
    return np.ones(T, dtype=bool)


def band_cols(n_layers: int, band: str) -> np.ndarray:
    if band == "all":
        return np.arange(n_layers)
    if band == "deep":
        return np.arange(int(n_layers * 0.6), n_layers)
    if band == "mid":
        return np.arange(int(n_layers * 0.3), int(n_layers * 0.7))
    return np.array([int(x) for x in band.split(",") if x.strip()], dtype=int)


def label_policy(data: np.lib.npyio.NpzFile, policy: str) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(data["problem_ids"])
    if policy == "answer":
        return (data["is_correct"].astype(int) == 0).astype(int), np.ones(n, bool), "answer incorrect"
    if policy == "strict":
        return (data["is_correct_strict"].astype(int) == 0).astype(int), np.ones(n, bool), "strict incorrect"
    if policy == "answer_format_ok":
        return (
            (data["is_correct"].astype(int) == 0).astype(int),
            data["format_ok"].astype(bool),
            "answer incorrect among format-ok samples",
        )
    raise ValueError(policy)


def group_folds(groups: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    k = min(k, len(uniq))
    fold_of = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def same_problem_groups(problem_ids: np.ndarray, y_err: np.ndarray, mask: np.ndarray, min_per_class: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        if np.sum(y_err[idx] == 1) >= min_per_class and np.sum(y_err[idx] == 0) >= min_per_class:
            out.append(idx)
    return out


def within_pair_auroc(groups: Sequence[np.ndarray], vals: np.ndarray, y_err: np.ndarray) -> Tuple[float, int]:
    conc = 0.0
    pairs = 0
    for idx in groups:
        err = [float(vals[i]) for i in idx if y_err[i] == 1 and np.isfinite(vals[i])]
        cor = [float(vals[i]) for i in idx if y_err[i] == 0 and np.isfinite(vals[i])]
        for a in err:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        pairs += len(err) * len(cor)
    return (conc / pairs if pairs else float("nan")), int(pairs)


def paired_delta(groups: Sequence[np.ndarray], vals: np.ndarray, y_err: np.ndarray) -> Dict[str, Any]:
    ds: List[float] = []
    for idx in groups:
        err = [float(vals[i]) for i in idx if y_err[i] == 1 and np.isfinite(vals[i])]
        cor = [float(vals[i]) for i in idx if y_err[i] == 0 and np.isfinite(vals[i])]
        if err and cor:
            ds.append(float(np.mean(err) - np.mean(cor)))
    a = np.asarray(ds, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "median": float(q50), "q25": float(q25), "q75": float(q75), "fraction_positive": float((a > 0).mean())}


def cloud_step_resultant(H: np.ndarray) -> float:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return float("nan")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / np.maximum(norms, EPS)
    return float(np.linalg.norm(np.nanmean(U, axis=0)))


def add_cloud_sequences(data: np.lib.npyio.NpzFile, seqs: List[Dict[str, np.ndarray]]) -> None:
    if "sv_clouds" not in data.files or "cloud_sizes" not in data.files:
        return
    for i, (obj, size_obj) in enumerate(zip(data["sv_clouds"], data["cloud_sizes"])):
        if obj is None or size_obj is None:
            continue
        C = np.asarray(obj, dtype=np.float64)
        sizes = np.asarray(size_obj, dtype=int).reshape(-1)
        if C.ndim != 3 or C.shape[0] == 0 or sizes.size == 0:
            continue
        X = C[:, 0, :]
        cursor = 0
        resultants: List[float] = []
        tok_counts: List[float] = []
        for s in sizes:
            s = int(s)
            if s <= 0:
                continue
            H = X[cursor : cursor + s]
            cursor += s
            if H.size == 0:
                continue
            resultants.append(cloud_step_resultant(H))
            tok_counts.append(float(s))
        if resultants:
            r = np.asarray(resultants, dtype=np.float64)
            seqs[i]["cloud_resultant"] = r
            seqs[i]["cloud_spread"] = 1.0 - r
            seqs[i]["log_step_tokens"] = np.log1p(np.asarray(tok_counts, dtype=np.float64))


def add_matrix_sequences(data: np.lib.npyio.NpzFile, seqs: List[Dict[str, np.ndarray]], *, band: str) -> None:
    for metric in ("pr", "ae"):
        key = f"sv_{metric}_step_exp"
        if key not in data.files:
            continue
        raw = data[key]
        first = np.asarray(raw[0], dtype=np.float64)
        if first.ndim != 2:
            continue
        cols = band_cols(first.shape[1], band)
        for i, obj in enumerate(raw):
            M = np.asarray(obj, dtype=np.float64)
            if M.ndim != 2:
                continue
            valid = cols[cols < M.shape[1]]
            if valid.size == 0:
                valid = np.arange(M.shape[1])
            seqs[i][f"{metric}_{band}"] = np.nanmean(M[:, valid], axis=1)


def add_logit_sequences(data: np.lib.npyio.NpzFile, seqs: List[Dict[str, np.ndarray]]) -> None:
    for key, name in (
        ("sv_out_entropy", "out_entropy"),
        ("sv_out_committal", "out_committal"),
    ):
        if key not in data.files:
            continue
        for i, obj in enumerate(data[key]):
            v = np.asarray(obj, dtype=np.float64).reshape(-1)
            if v.size:
                seqs[i][name] = v


def add_vector_jump_sequences(data: np.lib.npyio.NpzFile, seqs: List[Dict[str, np.ndarray]], *, band: str, normalize: str) -> None:
    if "sv_vec_step_exp" not in data.files or not bool(data.get("sv_vectors_stored", np.array(False))):
        return
    raw = data["sv_vec_step_exp"]
    for i, obj in enumerate(raw):
        V = np.asarray(obj, dtype=np.float64)
        if V.ndim != 3 or V.shape[0] == 0:
            continue
        cols = band_cols(V.shape[1], band)
        valid = cols[cols < V.shape[1]]
        if valid.size == 0:
            valid = np.arange(V.shape[1])
        X = np.nanmean(V[:, valid, :], axis=1)
        if normalize == "l2":
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
        elif normalize == "center_chain":
            X = X - np.nanmean(X, axis=0, keepdims=True)
        elif normalize != "none":
            raise ValueError(normalize)
        jump = np.full(X.shape[0], np.nan, dtype=np.float64)
        djump = np.full(X.shape[0], np.nan, dtype=np.float64)
        if X.shape[0] >= 2:
            D = X[1:] - X[:-1]
            jump[1:] = np.linalg.norm(D, axis=1)
            U0 = X[:-1] / np.maximum(np.linalg.norm(X[:-1], axis=1, keepdims=True), EPS)
            U1 = X[1:] / np.maximum(np.linalg.norm(X[1:], axis=1, keepdims=True), EPS)
            djump[1:] = 1.0 - np.sum(U0 * U1, axis=1)
        seqs[i]["step_jump"] = jump
        seqs[i]["direction_jump"] = djump


def build_sequences(data: np.lib.npyio.NpzFile, *, band: str, normalize_vectors: str) -> List[Dict[str, np.ndarray]]:
    n = len(data["problem_ids"])
    seqs: List[Dict[str, np.ndarray]] = [dict() for _ in range(n)]
    add_cloud_sequences(data, seqs)
    add_matrix_sequences(data, seqs, band=band)
    add_logit_sequences(data, seqs)
    add_vector_jump_sequences(data, seqs, band=band, normalize=normalize_vectors)
    return seqs


def available_features(seqs: Sequence[Dict[str, np.ndarray]]) -> List[str]:
    names = sorted({k for s in seqs for k, v in s.items() if np.asarray(v).ndim == 1})
    return names


def parse_groups(spec: str, *, band: str) -> Dict[str, List[str]]:
    if spec.strip():
        out: Dict[str, List[str]] = {}
        for part in spec.split(";"):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError("feature_groups entries must be name=a,b,c")
            name, rhs = part.split("=", 1)
            out[name.strip()] = [x.strip() for x in rhs.split(",") if x.strip()]
        return out
    return {
        "spread": ["cloud_spread"],
        "spread_entropy": ["cloud_spread", "out_entropy", "out_committal"],
        "spread_moment": ["cloud_spread", f"pr_{band}", f"ae_{band}"],
        "spread_entropy_moment": ["cloud_spread", "out_entropy", "out_committal", f"pr_{band}", f"ae_{band}"],
        "all_effective": ["cloud_spread", "out_entropy", "out_committal", f"pr_{band}", f"ae_{band}", "step_jump", "direction_jump"],
    }


def align_feature_matrix(seq: Mapping[str, np.ndarray], names: Sequence[str], *, include_deltas: bool) -> np.ndarray:
    lens = [len(np.asarray(seq[n]).reshape(-1)) for n in names if n in seq]
    if not lens:
        return np.empty((0, 0), dtype=np.float64)
    T = int(min(lens))
    cols: List[np.ndarray] = []
    for n in names:
        if n not in seq:
            cols.append(np.full(T, np.nan, dtype=np.float64))
            continue
        v = np.asarray(seq[n], dtype=np.float64).reshape(-1)[:T]
        cols.append(v)
    if include_deltas:
        for c in list(cols):
            d = np.full(T, np.nan, dtype=np.float64)
            if T >= 2:
                d[1:] = c[1:] - c[:-1]
            cols.append(d)
    X = np.column_stack(cols)
    good = np.isfinite(X).any(axis=1)
    return X[good]


def robust_standardizer(mats: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rows = [X for X in mats if X.size]
    if not rows:
        return np.zeros(0, dtype=np.float64), np.ones(0, dtype=np.float64)
    M = np.vstack(rows)
    med = np.nanmedian(M, axis=0)
    q25 = np.nanpercentile(M, 25, axis=0)
    q75 = np.nanpercentile(M, 75, axis=0)
    scale = (q75 - q25) / 1.349
    sd = np.nanstd(M, axis=0)
    scale = np.where(scale > EPS, scale, np.where(sd > EPS, sd, 1.0))
    med = np.where(np.isfinite(med), med, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return med.astype(np.float64), scale.astype(np.float64)


def transform_matrix(X: np.ndarray, med: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    Y = (X - med[None, :]) / scale[None, :]
    return np.nan_to_num(Y, nan=0.0, posinf=5.0, neginf=-5.0)


def logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.maximum(np.sum(np.exp(a - m), axis=axis, keepdims=True), EPS))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


@dataclass
class DiagGaussianHMM:
    n_states: int
    max_iter: int = 50
    tol: float = 1e-4
    sticky: float = 0.85
    reg: float = 1e-3
    seed: int = 0
    pi: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    var: Optional[np.ndarray] = None
    loglik_: float = float("nan")
    n_iter_: int = 0

    def _init_params(self, seqs: Sequence[np.ndarray]) -> None:
        rng = np.random.default_rng(self.seed)
        M = np.vstack([X for X in seqs if X.size])
        N, D = M.shape
        proxy = M[:, 0] if D else rng.normal(size=N)
        qs = np.quantile(proxy, np.linspace(0, 1, self.n_states + 2)[1:-1])
        bins = np.digitize(proxy, qs)
        self.mu = np.zeros((self.n_states, D), dtype=np.float64)
        self.var = np.zeros((self.n_states, D), dtype=np.float64)
        global_mu = M.mean(axis=0)
        global_var = M.var(axis=0) + self.reg
        for k in range(self.n_states):
            mk = M[bins == k]
            if mk.shape[0] < 2:
                self.mu[k] = global_mu + rng.normal(scale=0.05, size=D)
                self.var[k] = global_var
            else:
                self.mu[k] = mk.mean(axis=0)
                self.var[k] = mk.var(axis=0) + self.reg
        self.pi = np.full(self.n_states, 1.0 / self.n_states)
        self.A = np.full((self.n_states, self.n_states), (1.0 - self.sticky) / max(1, self.n_states - 1))
        np.fill_diagonal(self.A, self.sticky)
        self.A /= self.A.sum(axis=1, keepdims=True)

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.var is not None
        X3 = X[:, None, :]
        V = np.maximum(self.var[None, :, :], self.reg)
        return -0.5 * (np.sum(np.log(2 * np.pi * V), axis=2) + np.sum((X3 - self.mu[None, :, :]) ** 2 / V, axis=2))

    def _fb(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        assert self.pi is not None and self.A is not None
        B = self._log_emission(X)
        log_pi = np.log(np.maximum(self.pi, EPS))
        log_A = np.log(np.maximum(self.A, EPS))
        T = X.shape[0]
        alpha = np.empty((T, self.n_states), dtype=np.float64)
        beta = np.empty((T, self.n_states), dtype=np.float64)
        alpha[0] = log_pi + B[0]
        for t in range(1, T):
            alpha[t] = B[t] + logsumexp(alpha[t - 1][:, None] + log_A, axis=0)
        beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(log_A + B[t + 1][None, :] + beta[t + 1][None, :], axis=1)
        ll = float(logsumexp(alpha[-1], axis=0))
        gamma = np.exp(alpha + beta - ll)
        xi_sum = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for t in range(T - 1):
            z = alpha[t][:, None] + log_A + B[t + 1][None, :] + beta[t + 1][None, :] - ll
            xi_sum += np.exp(z)
        return gamma, xi_sum, B, ll

    def fit(self, seqs: Sequence[np.ndarray]) -> "DiagGaussianHMM":
        seqs = [X for X in seqs if X.ndim == 2 and X.shape[0] >= 2 and X.shape[1] > 0]
        if not seqs:
            raise ValueError("no usable sequences for HMM")
        self._init_params(seqs)
        prev = -np.inf
        for it in range(1, self.max_iter + 1):
            assert self.mu is not None and self.var is not None and self.A is not None
            D = self.mu.shape[1]
            pi_acc = np.zeros(self.n_states)
            A_acc = np.zeros((self.n_states, self.n_states))
            w_acc = np.zeros(self.n_states)
            x_acc = np.zeros((self.n_states, D))
            x2_acc = np.zeros((self.n_states, D))
            ll_total = 0.0
            for X in seqs:
                gamma, xi, _, ll = self._fb(X)
                ll_total += ll
                pi_acc += gamma[0]
                A_acc += xi
                w_acc += gamma.sum(axis=0)
                x_acc += gamma.T @ X
                x2_acc += gamma.T @ (X ** 2)
            self.pi = (pi_acc + 1e-2)
            self.pi /= self.pi.sum()
            self.A = A_acc + 1e-2
            self.A += np.eye(self.n_states) * 1e-2
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.mu = x_acc / np.maximum(w_acc[:, None], EPS)
            self.var = x2_acc / np.maximum(w_acc[:, None], EPS) - self.mu ** 2
            self.var = np.maximum(self.var, self.reg)
            self.n_iter_ = it
            self.loglik_ = ll_total
            if np.isfinite(prev) and abs(ll_total - prev) <= self.tol * (1.0 + abs(prev)):
                break
            prev = ll_total
        return self

    def filtered(self, X: np.ndarray) -> np.ndarray:
        assert self.pi is not None and self.A is not None
        B = self._log_emission(X)
        log_pi = np.log(np.maximum(self.pi, EPS))
        log_A = np.log(np.maximum(self.A, EPS))
        T = X.shape[0]
        alpha = np.empty((T, self.n_states), dtype=np.float64)
        alpha[0] = log_pi + B[0]
        alpha[0] -= logsumexp(alpha[0], axis=0)
        for t in range(1, T):
            alpha[t] = B[t] + logsumexp(alpha[t - 1][:, None] + log_A, axis=0)
            alpha[t] -= logsumexp(alpha[t], axis=0)
        return np.exp(alpha)


def state_weights(
    model: DiagGaussianHMM,
    seqs: Sequence[np.ndarray],
    y_err: np.ndarray,
    idxs: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    K = model.n_states
    err_vals: List[List[float]] = [[] for _ in range(K)]
    cor_vals: List[List[float]] = [[] for _ in range(K)]
    for i in idxs:
        X = seqs[int(i)]
        if X.ndim != 2 or X.shape[0] < 2:
            continue
        P = model.filtered(X)
        late = window_mask(P.shape[0], "late")
        s = P[late].mean(axis=0) if late.any() else P.mean(axis=0)
        dst = err_vals if y_err[int(i)] == 1 else cor_vals
        for k in range(K):
            dst[k].append(float(s[k]))
    delta = np.zeros(K, dtype=np.float64)
    err_mean = np.zeros(K, dtype=np.float64)
    cor_mean = np.zeros(K, dtype=np.float64)
    for k in range(K):
        err_mean[k] = np.mean(err_vals[k]) if err_vals[k] else 0.0
        cor_mean[k] = np.mean(cor_vals[k]) if cor_vals[k] else 0.0
        delta[k] = err_mean[k] - cor_mean[k]
    pos = np.maximum(delta, 0.0)
    if pos.sum() <= EPS:
        pos[int(np.argmax(delta))] = 1.0
    weights = pos / max(float(pos.sum()), EPS)
    meta = {
        "state_error_mean": err_mean,
        "state_correct_mean": cor_mean,
        "state_delta": delta,
        "risk_weights": weights,
    }
    return weights, meta


def score_hmm_sequence(model: DiagGaussianHMM, X: np.ndarray, weights: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    P = model.filtered(X)
    risk = P @ weights
    late = window_mask(risk.size, "late")
    out = {
        "hmm_risk_mean": safe_mean(risk),
        "hmm_risk_late": safe_mean(risk[late]),
        "hmm_risk_max": float(np.nanmax(risk)) if np.isfinite(risk).any() else float("nan"),
        "hmm_risk_final": float(risk[-1]) if risk.size else float("nan"),
        "hmm_risk_volatility": safe_mean(np.abs(np.diff(risk))) if risk.size >= 2 else 0.0,
    }
    return out, risk


def summarize_sequence_feature(v: np.ndarray) -> Dict[str, float]:
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).any():
        return {}
    late = window_mask(x.size, "late")
    d = np.diff(x)
    return {
        "mean": safe_mean(x),
        "late": safe_mean(x[late]),
        "max": float(np.nanmax(x)),
        "volatility": safe_mean(np.abs(d)) if d.size else 0.0,
    }


def baseline_scores(seqs: Sequence[Dict[str, np.ndarray]], features: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    n = len(seqs)
    for feat in features:
        for suffix in ("mean", "late", "max", "volatility"):
            out[f"base.{feat}.{suffix}"] = np.full(n, np.nan, dtype=np.float64)
    for i, seq in enumerate(seqs):
        for feat in features:
            if feat not in seq:
                continue
            sm = summarize_sequence_feature(seq[feat])
            for suffix, val in sm.items():
                key = f"base.{feat}.{suffix}"
                if key in out:
                    out[key][i] = val
    return out


def collect_row(
    name: str,
    vals: np.ndarray,
    y_err: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    target_fpr: float,
) -> Dict[str, Any]:
    m = mask & np.isfinite(vals)
    err = vals[m & (y_err == 1)]
    cor = vals[m & (y_err == 0)]
    within, pairs = within_pair_auroc(groups, vals, y_err)
    threshold = float(np.quantile(cor, 1.0 - target_fpr)) if cor.size else float("nan")
    recall = float(np.mean(err > threshold)) if err.size and np.isfinite(threshold) else float("nan")
    return {
        "score": name,
        "n": int(m.sum()),
        "n_error": int((m & (y_err == 1)).sum()),
        "n_correct": int((m & (y_err == 0)).sum()),
        "within_pair_auroc_error_high": within,
        "within_pairs": pairs,
        "cross_auroc_error_high": auroc_signed(err, cor),
        "error": descriptive(err),
        "correct": descriptive(cor),
        "paired_delta_error_minus_correct": paired_delta(groups, vals, y_err),
        "threshold_at_target_fpr": threshold,
        "wrong_recall_at_target_fpr": recall,
        "target_fpr": float(target_fpr),
    }


def run_group_cv(
    seq_dicts: Sequence[Dict[str, np.ndarray]],
    problem_ids: np.ndarray,
    y_err: np.ndarray,
    mask: np.ndarray,
    features: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    idx_all = np.where(mask)[0]
    n = len(seq_dicts)
    out = {
        "hmm_risk_mean": np.full(n, np.nan, dtype=np.float64),
        "hmm_risk_late": np.full(n, np.nan, dtype=np.float64),
        "hmm_risk_max": np.full(n, np.nan, dtype=np.float64),
        "hmm_risk_final": np.full(n, np.nan, dtype=np.float64),
        "hmm_risk_volatility": np.full(n, np.nan, dtype=np.float64),
    }
    fold_meta: List[Dict[str, Any]] = []
    folds = group_folds(problem_ids[mask], args.folds, args.seed)
    for fold_id, (tr_rel, te_rel) in enumerate(folds, start=1):
        tr_idx = idx_all[tr_rel]
        te_idx = idx_all[te_rel]
        raw_train = [align_feature_matrix(seq_dicts[int(i)], features, include_deltas=args.include_deltas) for i in tr_idx]
        med, scale = robust_standardizer(raw_train)
        if med.size == 0:
            continue
        train_mats = [transform_matrix(X, med, scale) for X in raw_train if X.shape[0] >= args.min_steps]
        if len(train_mats) < args.min_train_sequences:
            continue
        model = DiagGaussianHMM(
            n_states=args.states,
            max_iter=args.em_iters,
            tol=args.em_tol,
            sticky=args.sticky,
            reg=args.var_reg,
            seed=args.seed + fold_id,
        ).fit(train_mats)
        # Rebuild transformed train matrices by absolute index for state orientation.
        train_by_idx: Dict[int, np.ndarray] = {}
        for i in tr_idx:
            X = align_feature_matrix(seq_dicts[int(i)], features, include_deltas=args.include_deltas)
            if X.shape[0] >= args.min_steps:
                train_by_idx[int(i)] = transform_matrix(X, med, scale)
        orient_idxs = list(train_by_idx.keys())
        orient_seqs = [train_by_idx.get(i, np.empty((0, med.size))) for i in range(n)]
        weights, wmeta = state_weights(model, orient_seqs, y_err, orient_idxs)
        scored = 0
        for i in te_idx:
            X = align_feature_matrix(seq_dicts[int(i)], features, include_deltas=args.include_deltas)
            if X.shape[0] < args.min_steps:
                continue
            Xt = transform_matrix(X, med, scale)
            s, _ = score_hmm_sequence(model, Xt, weights)
            for k, v in s.items():
                out[k][int(i)] = v
            scored += 1
        fold_meta.append({
            "fold": fold_id,
            "train_sequences": int(len(train_mats)),
            "test_sequences_scored": int(scored),
            "loglik": float(model.loglik_),
            "n_iter": int(model.n_iter_),
            "risk_state_meta": finite_json(wmeta),
        })
    return out, {"folds": fold_meta}


def high_spread_subset_mask(seqs: Sequence[Dict[str, np.ndarray]], mask: np.ndarray, q: float) -> np.ndarray:
    vals = np.full(len(seqs), np.nan, dtype=np.float64)
    for i, seq in enumerate(seqs):
        if "cloud_spread" not in seq:
            continue
        sm = summarize_sequence_feature(seq["cloud_spread"])
        vals[i] = sm.get("late", float("nan"))
    m = mask & np.isfinite(vals)
    if not m.any():
        return np.zeros(len(seqs), dtype=bool)
    thr = float(np.quantile(vals[m], q))
    return m & (vals >= thr)


def run_policy(
    data: np.lib.npyio.NpzFile,
    *,
    policy: str,
    seqs: Sequence[Dict[str, np.ndarray]],
    feature_groups: Mapping[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    problem_ids = data["problem_ids"].astype(int)
    y_err, mask, desc = label_policy(data, policy)
    groups = same_problem_groups(problem_ids, y_err, mask, args.min_per_class)
    rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {"label_policy": desc}
    feature_universe = sorted({f for fs in feature_groups.values() for f in fs})
    b = baseline_scores(seqs, feature_universe)
    for name, vals in b.items():
        rows.append(collect_row(name, vals, y_err, mask, groups, target_fpr=args.target_fpr))

    high_mask = high_spread_subset_mask(seqs, mask, args.high_spread_q)
    high_groups = same_problem_groups(problem_ids, y_err, high_mask, args.min_per_class)

    for gname, requested in feature_groups.items():
        feats = [f for f in requested if f in available_features(seqs)]
        skipped = [f for f in requested if f not in feats]
        if not feats:
            diagnostics[f"hmm.{gname}"] = {"skipped": "no requested features present", "requested": requested}
            continue
        scores, meta = run_group_cv(seqs, problem_ids, y_err, mask, feats, args)
        diagnostics[f"hmm.{gname}"] = {
            "features": feats,
            "skipped_features": skipped,
            **meta,
        }
        for sname, vals in scores.items():
            row = collect_row(f"hmm.{gname}.{sname}", vals, y_err, mask, groups, target_fpr=args.target_fpr)
            if high_groups:
                w, pairs = within_pair_auroc(high_groups, vals, y_err)
                row["high_spread_within_pair_auroc"] = w
                row["high_spread_pairs"] = pairs
            rows.append(row)

    rows.sort(
        key=lambda r: (
            np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0),
            np.nan_to_num(r["cross_auroc_error_high"], nan=-1.0),
        ),
        reverse=True,
    )
    return {
        "description": desc,
        "n_samples": int(mask.sum()),
        "n_error": int(y_err[mask].sum()),
        "n_correct": int(mask.sum() - y_err[mask].sum()),
        "n_contrastive_problems": int(len(groups)),
        "available_features": available_features(seqs),
        "feature_groups": feature_groups,
        "diagnostics": diagnostics,
        "results": rows,
    }


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    seqs = build_sequences(data, band=args.band, normalize_vectors=args.normalize_vectors)
    feature_groups = parse_groups(args.feature_groups, band=args.band)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "prompt_style": str(data["prompt_style"]) if "prompt_style" in data.files else "unknown",
            "step_split": str(data["step_split"]) if "step_split" in data.files else "unknown",
            "model": str(data["model_name"]) if "model_name" in data.files else "unknown",
            "band": args.band,
            "states": int(args.states),
            "include_deltas": bool(args.include_deltas),
            "normalize_vectors": args.normalize_vectors,
            "notes": {
                "method": "diagonal Gaussian HMM fitted by EM; chain labels only orient latent states after fitting",
                "online": "scores use filtered posterior p(z_t | x_1..x_t), not smoothed future context",
                "no_layer_sync": "uses band-averaged step-vector signals only; no layer desync/cross-layer tensor feature",
            },
        },
        "policies": {
            pol: run_policy(data, policy=pol, seqs=seqs, feature_groups=feature_groups, args=args)
            for pol in policies
        },
    }


def write_outputs(res: Mapping[str, Any], output_dir: str, top: int) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = f"latent_constraint_em_{os.path.splitext(str(res['meta']['basename']))[0]}_{res['meta']['band']}_K{res['meta']['states']}"
    jp = os.path.join(output_dir, stem + ".json")
    mp = os.path.join(output_dir, stem + ".md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, indent=2, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(f"# Latent Constraint EM Audit: {res['meta']['basename']}\n\n")
        f.write("## Result Analysis\n\n")
        f.write("- This audit treats constrained reasoning as a latent state, learned by EM with a diagonal Gaussian HMM.\n")
        f.write("- It uses only previously useful channels: spread/resultant, entropy/committal, second-moment PR/AE proxies, and step jump.\n")
        f.write("- HMM scores are online-filtered posteriors; labels orient states after EM but do not supervise emissions.\n")
        f.write("- Same-problem paired AUROC is the headline metric; cross-problem AUROC is context.\n\n")
        f.write("Metadata:\n\n")
        f.write(f"```json\n{json.dumps(finite_json(res['meta']), indent=2, ensure_ascii=False)}\n```\n\n")
        for pol, sec in res["policies"].items():
            f.write(f"### {pol}\n\n")
            f.write(
                f"{sec['n_error']} error / {sec['n_correct']} correct samples; "
                f"{sec['n_contrastive_problems']} contrastive problems.\n\n"
            )
            f.write(f"Available features: `{', '.join(sec['available_features'])}`\n\n")
            f.write("| score | within | cross | recall@FPR | err med | cor med | delta med | high-spread within |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in sec["results"][:top]:
                dlt = r["paired_delta_error_minus_correct"]
                f.write(
                    f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
                    f"{r['cross_auroc_error_high']:.3f} | "
                    f"{r['wrong_recall_at_target_fpr']:.3f} | "
                    f"{r['error'].get('median', float('nan')):.3f} | "
                    f"{r['correct'].get('median', float('nan')):.3f} | "
                    f"{dlt.get('median', float('nan')):.3f} | "
                    f"{r.get('high_spread_within_pair_auroc', float('nan')):.3f} |\n"
                )
            f.write("\n")
        f.write("## Follow-Up Research Direction\n\n")
        f.write("- If `spread_entropy_moment` beats `spread_entropy`, second-moment dynamics add a real latent-state signal.\n")
        f.write("- If HMM max/late risk beats the corresponding base features, the useful object is state persistence rather than a static scalar.\n")
        f.write("- If only base cloud_spread wins, the current latent model is not reading beyond the old spread signal.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Keep state count small first (3 or 4); larger K can overfit small same-problem data.\n")
        f.write("- Compare `answer_format_ok` and `strict` before claiming reasoning-specific failures.\n")
        f.write("- Later versions can replace diagonal Gaussian emissions with mixture or low-rank tensor emissions, but only after this audit shows an HMM gain.\n")
    return jp, mp


def print_report(res: Mapping[str, Any], top: int) -> None:
    meta = res["meta"]
    print(f"\n===== latent constraint EM | {meta['basename']} | {meta['band']} | K={meta['states']} =====")
    print(f"prompt={meta['prompt_style']} step_split={meta['step_split']} model={meta['model']}")
    for pol, sec in res["policies"].items():
        print(f"\n[{pol}] err={sec['n_error']} cor={sec['n_correct']} contrastive={sec['n_contrastive_problems']}")
        print(f"available_features={sec['available_features']}")
        for r in sec["results"][:top]:
            dlt = r["paired_delta_error_minus_correct"]
            print(
                f"  {r['score']:42s} within {r['within_pair_auroc_error_high']:.3f} "
                f"cross {r['cross_auroc_error_high']:.3f} "
                f"recall@fpr {r['wrong_recall_at_target_fpr']:.3f} "
                f"err_med {r['error'].get('median', float('nan')):.3f} "
                f"cor_med {r['correct'].get('median', float('nan')):.3f} "
                f"delta {dlt.get('median', float('nan')):+.3f} "
                f"high {r.get('high_spread_within_pair_auroc', float('nan')):.3f}"
            )


def make_selftest(path: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_problems = 36
    dim = 24
    layers = 6
    rows = []
    for p in range(n_problems):
        base_dir = rng.normal(size=dim)
        base_dir /= np.linalg.norm(base_dir)
        for s in range(5):
            T = int(rng.integers(6, 9))
            correct = s < 3
            resultants = []
            sizes = []
            cloud_rows = []
            pr = np.zeros((T, layers), dtype=np.float32)
            ae = np.zeros((T, layers), dtype=np.float32)
            out_ent = np.zeros(T, dtype=np.float32)
            out_com = np.zeros(T, dtype=np.float32)
            vec = np.zeros((T, layers, dim), dtype=np.float32)
            x = base_dir + rng.normal(scale=0.03, size=dim)
            x /= np.linalg.norm(x)
            bad_start = T // 2
            for t in range(T):
                bad = (not correct) and t >= bad_start
                n_tok = int(rng.integers(5, 10))
                sizes.append(n_tok)
                spread = 0.10 + 0.02 * rng.normal()
                entropy = 0.25 + 0.05 * rng.normal()
                moment = 2.0 + 0.2 * rng.normal()
                if bad:
                    spread += 0.22 + 0.04 * (t - bad_start)
                    entropy += 0.20
                    moment += 1.4
                    x = x + rng.normal(scale=0.25, size=dim)
                else:
                    x = x + rng.normal(scale=0.04, size=dim)
                x /= np.linalg.norm(x)
                toks = []
                for _ in range(n_tok):
                    u = x + rng.normal(scale=max(0.02, spread), size=dim)
                    u /= np.linalg.norm(u)
                    toks.append(u)
                toks = np.asarray(toks, dtype=np.float32)
                cloud_rows.append(toks[:, None, :])
                resultants.append(cloud_step_resultant(toks))
                pr[t, :] = moment + rng.normal(scale=0.05, size=layers)
                ae[t, :] = np.log(np.maximum(pr[t, :], 1e-3)) + rng.normal(scale=0.03, size=layers)
                out_ent[t] = entropy
                out_com[t] = 1.0 - entropy
                for l in range(layers):
                    vec[t, l] = x + rng.normal(scale=0.03 + 0.01 * l, size=dim)
            rows.append({
                "problem_id": p,
                "sample_idx": s,
                "is_correct": int(correct),
                "is_correct_strict": int(correct),
                "format_ok": 1,
                "n_steps": T,
                "sv_clouds": np.concatenate(cloud_rows, axis=0).astype(np.float32),
                "cloud_sizes": np.asarray(sizes, dtype=np.int32),
                "sv_pr_step_exp": pr,
                "sv_ae_step_exp": ae,
                "sv_out_entropy": out_ent,
                "sv_out_committal": out_com,
                "sv_vec_step_exp": vec,
            })
    np.savez(
        path,
        problem_ids=np.asarray([r["problem_id"] for r in rows], dtype=np.int32),
        sample_idx=np.asarray([r["sample_idx"] for r in rows], dtype=np.int32),
        is_correct=np.asarray([r["is_correct"] for r in rows], dtype=np.int32),
        is_correct_strict=np.asarray([r["is_correct_strict"] for r in rows], dtype=np.int32),
        format_ok=np.asarray([r["format_ok"] for r in rows], dtype=np.int32),
        n_steps=np.asarray([r["n_steps"] for r in rows], dtype=np.int32),
        sv_clouds=np.asarray([r["sv_clouds"] for r in rows], dtype=object),
        cloud_sizes=np.asarray([r["cloud_sizes"] for r in rows], dtype=object),
        sv_pr_step_exp=np.asarray([r["sv_pr_step_exp"] for r in rows], dtype=object),
        sv_ae_step_exp=np.asarray([r["sv_ae_step_exp"] for r in rows], dtype=object),
        sv_out_entropy=np.asarray([r["sv_out_entropy"] for r in rows], dtype=object),
        sv_out_committal=np.asarray([r["sv_out_committal"] for r in rows], dtype=object),
        sv_vectors_stored=np.asarray(True),
        sv_vec_step_exp=np.asarray([r["sv_vec_step_exp"] for r in rows], dtype=object),
        prompt_style=np.asarray("selftest"),
        step_split=np.asarray("synthetic"),
        model_name=np.asarray("synthetic"),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    rows = res["policies"]["answer_format_ok"]["results"]
    best_hmm = max(
        [float(r["within_pair_auroc_error_high"]) for r in rows if r["score"].startswith("hmm.")]
        or [float("nan")]
    )
    if not np.isfinite(best_hmm) or best_hmm < 0.75:
        raise SystemExit(f"selftest failed: best HMM within AUROC too weak ({best_hmm:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output_dir", default="outputs/latent_constraint_em")
    ap.add_argument("--policies", default="answer_format_ok")
    ap.add_argument("--band", default="mid")
    ap.add_argument("--normalize_vectors", default="l2", choices=["none", "l2", "center_chain"])
    ap.add_argument("--feature_groups", default="")
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--em_iters", type=int, default=60)
    ap.add_argument("--em_tol", type=float, default=1e-4)
    ap.add_argument("--sticky", type=float, default=0.86)
    ap.add_argument("--var_reg", type=float, default=1e-3)
    ap.add_argument("--include_deltas", action="store_true", default=True)
    ap.add_argument("--no_deltas", action="store_false", dest="include_deltas")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--min_steps", type=int, default=3)
    ap.add_argument("--min_train_sequences", type=int, default=20)
    ap.add_argument("--target_fpr", type=float, default=0.20)
    ap.add_argument("--high_spread_q", type=float, default=0.70)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "latent_constraint_selftest.npz")
            make_selftest(path, seed=args.seed)
            res = run(path, args)
            assert_selftest(res)
    else:
        if not args.input:
            raise SystemExit("pass --input or --selftest")
        res = run(args.input, args)
    jp, mp = write_outputs(res, args.output_dir, args.top)
    print_report(res, args.top)
    print(f"\nwrote {jp} and {mp}")


if __name__ == "__main__":
    main()
