#!/usr/bin/env python3
"""Same-problem latent-regime HSMM audit.

This audit is intentionally different from the cross-problem dynamic detectors.
It does not define a single "healthy trajectory".  Instead it asks whether
correct and incorrect samples of the same problem use different latent regime
grammars over local trajectory states.

Model:

    x_{i,t} | z_{i,t}=k ~ diagonal Gaussian(mu_k, var_k)

The emission parameters are shared across correct and incorrect chains.  The
class-specific parts are the initial regime distribution, transition matrix, and
explicit duration distribution.  This is a small supervised HSMM fitted by EM on
training problems and evaluated on held-out problems.

Input is the same within-problem multisample npz family produced by
`10_sample_and_extract.py` and used by the `multisample_*` audits.
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

from multisample_temporal_rupture_audit import (
    descriptive,
    finite_json,
    label_policy,
    paired_delta,
    problem_groups,
    safe_mean,
    within_pair_auroc,
)
from trajectory_difference_audit import prepare_policy_data


EPS = 1e-12
NINF = -1e100


DEFAULT_CHANNELS = "cloud_spread,out_entropy,pr_mid,ae_mid"


@dataclass
class RegimeData:
    sample_indices: np.ndarray
    problem_ids: np.ndarray
    y_err: np.ndarray
    groups: List[np.ndarray]
    channels: List[str]
    grid: np.ndarray
    tensor: np.ndarray
    residual_tensor: np.ndarray
    sequences: List[np.ndarray]
    obs_names: List[str]
    channel_coverage: Dict[str, float]


@dataclass
class HSMMParams:
    n_states: int
    max_duration: int
    obs_names: List[str]
    mu: np.ndarray
    var: np.ndarray
    pi: np.ndarray
    trans: np.ndarray
    dur: np.ndarray
    scaler_center: np.ndarray
    scaler_scale: np.ndarray


def logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    if axis is None:
        m = np.max(x)
        if not np.isfinite(m):
            return np.asarray(NINF)
        return np.asarray(m + np.log(np.sum(np.exp(x - m))))
    m = np.max(x, axis=axis, keepdims=True)
    bad = ~np.isfinite(m)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    out[bad] = NINF
    return np.squeeze(out, axis=axis)


def robust_center_scale(x: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad > EPS:
        return med, 1.4826 * mad
    sd = float(np.std(a))
    return med, sd if sd > EPS else 1.0


def robust_matrix_scaler(seqs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rows = [np.asarray(s, dtype=np.float64).reshape(-1, s.shape[-1]) for s in seqs if len(s)]
    if not rows:
        raise ValueError("no observations for scaler")
    X = np.vstack(rows)
    center = np.zeros(X.shape[1], dtype=np.float64)
    scale = np.ones(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        center[j], scale[j] = robust_center_scale(X[:, j])
    return center, scale


def apply_scaler(seqs: Sequence[np.ndarray], center: np.ndarray, scale: np.ndarray) -> List[np.ndarray]:
    out = []
    for s in seqs:
        z = (np.asarray(s, dtype=np.float64) - center) / np.maximum(scale, EPS)
        z = np.where(np.isfinite(z), z, 0.0)
        out.append(z)
    return out


def group_folds(groups: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(groups)
    if uniq.size < 2:
        return []
    rng = np.random.default_rng(seed)
    uniq = np.array(uniq, copy=True)
    rng.shuffle(uniq)
    k = int(min(max(2, k), uniq.size))
    fold_of = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[int(g)] for g in groups], dtype=int)
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def problem_center_tensor(
    tensor: np.ndarray,
    problem_ids: np.ndarray,
    *,
    center: str,
) -> np.ndarray:
    X = np.asarray(tensor, dtype=np.float64)
    if center == "none":
        return np.array(X, copy=True)
    if center != "problem_median":
        raise ValueError(f"unknown center mode {center!r}")
    Z = np.full_like(X, np.nan, dtype=np.float64)
    for p in np.unique(problem_ids):
        idx = np.where(problem_ids == p)[0]
        block = X[idx]
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(block, axis=0)
            mad = np.nanmedian(np.abs(block - med), axis=0) * 1.4826
            sd = np.nanstd(block, axis=0)
        scale = np.where(np.isfinite(mad) & (mad > EPS), mad, sd)
        scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
        med = np.where(np.isfinite(med), med, 0.0)
        Z[idx] = (block - med) / scale
    return Z


def make_observation_sequences(
    residual_tensor: np.ndarray,
    *,
    channels: Sequence[str],
    include_abs_delta: bool,
) -> Tuple[List[np.ndarray], List[str]]:
    X = np.asarray(residual_tensor, dtype=np.float64)
    seqs: List[np.ndarray] = []
    obs_names = [f"level:{c}" for c in channels]
    obs_names += [f"delta:{c}" for c in channels]
    if include_abs_delta:
        obs_names += [f"abs_delta:{c}" for c in channels]
    for i in range(X.shape[0]):
        level = X[i].T  # grid x channels
        delta = np.zeros_like(level)
        if len(level) > 1:
            delta[1:] = level[1:] - level[:-1]
        parts = [level, delta]
        if include_abs_delta:
            parts.append(np.abs(delta))
        seq = np.concatenate(parts, axis=1)
        seq = np.where(np.isfinite(seq), seq, 0.0)
        seqs.append(seq.astype(np.float64))
    return seqs, obs_names


def load_regime_data(path: str, args: argparse.Namespace) -> RegimeData:
    data = np.load(path, allow_pickle=True)
    requested = [x.strip() for x in args.channels.split(",") if x.strip()]
    bands = [x.strip() for x in args.bands.split(",") if x.strip()]
    pd = prepare_policy_data(
        data,
        policy=args.policy,
        bands=bands,
        requested_channels=requested,
        min_per_class=args.min_per_class,
        min_channel_coverage=args.min_channel_coverage,
        require_channels=args.require_channels,
        grid_size=args.grid,
        include_mahal=False,
    )
    idx = np.where(pd.contrast_mask)[0]
    if idx.size < 20:
        raise SystemExit("not enough contrastive same-problem samples")
    local_problem_ids = pd.problem_ids[idx]
    local_y = pd.y_err[idx].astype(int)
    tensor = pd.tensor[idx]
    residual = problem_center_tensor(tensor, local_problem_ids, center=args.problem_center)
    seqs, obs_names = make_observation_sequences(
        residual,
        channels=pd.channels,
        include_abs_delta=args.include_abs_delta,
    )
    local_groups = []
    for p in np.unique(local_problem_ids):
        g = np.where(local_problem_ids == p)[0]
        if np.sum(local_y[g] == 1) >= args.min_per_class and np.sum(local_y[g] == 0) >= args.min_per_class:
            local_groups.append(g)
    return RegimeData(
        sample_indices=idx,
        problem_ids=local_problem_ids,
        y_err=local_y,
        groups=local_groups,
        channels=pd.channels,
        grid=pd.grid,
        tensor=tensor,
        residual_tensor=residual,
        sequences=seqs,
        obs_names=obs_names,
        channel_coverage=pd.channel_coverage,
    )


def impute_obs(X: np.ndarray) -> np.ndarray:
    Z = np.asarray(X, dtype=np.float64).copy()
    if Z.ndim != 2:
        raise ValueError("observation matrix must be 2D")
    for j in range(Z.shape[1]):
        col = Z[:, j]
        m = np.isfinite(col)
        fill = float(np.mean(col[m])) if m.any() else 0.0
        col[~m] = fill
        Z[:, j] = col
    return Z


def kmeans_init(points: np.ndarray, k: int, seed: int, n_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    X = impute_obs(points)
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if n < k:
        raise ValueError("not enough points for states")
    centers = np.empty((k, d), dtype=np.float64)
    first = int(rng.integers(n))
    centers[0] = X[first]
    dist2 = np.sum((X - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = dist2 / max(float(dist2.sum()), EPS)
        idx = int(rng.choice(n, p=probs))
        centers[j] = X[idx]
        dist2 = np.minimum(dist2, np.sum((X - centers[j]) ** 2, axis=1))
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = X[m].mean(axis=0)
            else:
                centers[j] = X[int(rng.integers(n))]
    # Stable-ish label order for interpretability: sort by first level feature.
    order = np.argsort(centers[:, 0])
    inv = np.empty_like(order)
    inv[order] = np.arange(k)
    labels = inv[labels]
    centers = centers[order]
    var = np.zeros((k, d), dtype=np.float64)
    global_var = np.var(X, axis=0) + 1e-3
    for j in range(k):
        m = labels == j
        var[j] = np.var(X[m], axis=0) + 1e-3 if m.sum() >= 2 else global_var
    return centers, np.maximum(var, 1e-3)


def init_params(
    seqs: Sequence[np.ndarray],
    labels: np.ndarray,
    *,
    n_states: int,
    max_duration: int,
    obs_names: Sequence[str],
    seed: int,
) -> HSMMParams:
    points = np.vstack(seqs)
    mu, var = kmeans_init(points, n_states, seed=seed)
    pi = np.ones((2, n_states), dtype=np.float64) / n_states
    trans = np.ones((2, n_states, n_states), dtype=np.float64)
    for y in range(2):
        np.fill_diagonal(trans[y], 0.0)
        trans[y] /= trans[y].sum(axis=1, keepdims=True)
    d = np.arange(1, max_duration + 1, dtype=np.float64)
    base = np.exp(-d / max(2.0, max_duration / 2.0))
    base /= base.sum()
    dur = np.tile(base[None, None, :], (2, n_states, 1))
    center = np.zeros(points.shape[1], dtype=np.float64)
    scale = np.ones(points.shape[1], dtype=np.float64)
    return HSMMParams(
        n_states=n_states,
        max_duration=max_duration,
        obs_names=list(obs_names),
        mu=mu,
        var=var,
        pi=pi,
        trans=trans,
        dur=dur,
        scaler_center=center,
        scaler_scale=scale,
    )


def gaussian_log_prob(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    K, D = mu.shape
    out = np.empty((X.shape[0], K), dtype=np.float64)
    for k in range(K):
        v = np.maximum(var[k], 1e-6)
        out[:, k] = -0.5 * (
            np.sum(np.log(2.0 * np.pi * v)) + np.sum(((X - mu[k]) ** 2) / v, axis=1)
        )
    return out


def segment_sums(log_b: np.ndarray) -> np.ndarray:
    # returns K x (T+1) cumulative log emission sums
    return np.concatenate([np.zeros((log_b.shape[1], 1)), np.cumsum(log_b.T, axis=1)], axis=1)


def forward_hsmm(X: np.ndarray, params: HSMMParams, y: int) -> Tuple[float, np.ndarray, np.ndarray]:
    K = params.n_states
    Dmax = params.max_duration
    T = X.shape[0]
    log_b = gaussian_log_prob(X, params.mu, params.var)
    cum = segment_sums(log_b)
    log_pi = np.log(np.maximum(params.pi[y], EPS))
    log_A = np.full((K, K), NINF, dtype=np.float64)
    m = params.trans[y] > 0
    log_A[m] = np.log(params.trans[y][m])
    log_dur = np.log(np.maximum(params.dur[y], EPS))
    alpha = np.full((T + 1, K), NINF, dtype=np.float64)
    for t in range(1, T + 1):
        for k in range(K):
            vals = []
            for d in range(1, min(Dmax, t) + 1):
                s = t - d
                emit = cum[k, t] - cum[k, s]
                if s == 0:
                    vals.append(log_pi[k] + log_dur[k, d - 1] + emit)
                else:
                    vals.append(logsumexp(alpha[s] + log_A[:, k]) + log_dur[k, d - 1] + emit)
            alpha[t, k] = float(logsumexp(np.asarray(vals)))
    ll = float(logsumexp(alpha[T]))
    return ll, alpha, cum


def backward_hsmm(cum: np.ndarray, params: HSMMParams, y: int, T: int) -> np.ndarray:
    K = params.n_states
    Dmax = params.max_duration
    log_A = np.full((K, K), NINF, dtype=np.float64)
    m = params.trans[y] > 0
    log_A[m] = np.log(params.trans[y][m])
    log_dur = np.log(np.maximum(params.dur[y], EPS))
    beta = np.full((T + 1, K), NINF, dtype=np.float64)
    beta[T, :] = 0.0
    for t in range(T - 1, -1, -1):
        for prev in range(K):
            vals = []
            for k in range(K):
                if k == prev:
                    continue
                for d in range(1, min(Dmax, T - t) + 1):
                    end = t + d
                    emit = cum[k, end] - cum[k, t]
                    vals.append(log_A[prev, k] + log_dur[k, d - 1] + emit + beta[end, k])
            beta[t, prev] = float(logsumexp(np.asarray(vals))) if vals else NINF
    return beta


def expected_counts_one(
    X: np.ndarray,
    params: HSMMParams,
    y: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = params.n_states
    Dmax = params.max_duration
    T = X.shape[0]
    ll, alpha, cum = forward_hsmm(X, params, y)
    beta = backward_hsmm(cum, params, y, T)
    log_pi = np.log(np.maximum(params.pi[y], EPS))
    log_A = np.full((K, K), NINF, dtype=np.float64)
    m = params.trans[y] > 0
    log_A[m] = np.log(params.trans[y][m])
    log_dur = np.log(np.maximum(params.dur[y], EPS))
    gamma = np.zeros((T, K), dtype=np.float64)
    pi_counts = np.zeros(K, dtype=np.float64)
    trans_counts = np.zeros((K, K), dtype=np.float64)
    dur_counts = np.zeros((K, Dmax), dtype=np.float64)

    for t in range(1, T + 1):
        for k in range(K):
            for d in range(1, min(Dmax, t) + 1):
                s = t - d
                emit = cum[k, t] - cum[k, s]
                if s == 0:
                    lp = log_pi[k] + log_dur[k, d - 1] + emit + beta[t, k] - ll
                    w = float(np.exp(np.clip(lp, -745, 50)))
                    pi_counts[k] += w
                    dur_counts[k, d - 1] += w
                    gamma[s:t, k] += w
                else:
                    lps = alpha[s] + log_A[:, k] + log_dur[k, d - 1] + emit + beta[t, k] - ll
                    ws = np.exp(np.clip(lps, -745, 50))
                    ws[~np.isfinite(ws)] = 0.0
                    total = float(ws.sum())
                    trans_counts[:, k] += ws
                    dur_counts[k, d - 1] += total
                    gamma[s:t, k] += total
    return ll, gamma, pi_counts, trans_counts, dur_counts


def fit_hsmm(
    seqs: Sequence[np.ndarray],
    labels: np.ndarray,
    *,
    obs_names: Sequence[str],
    n_states: int,
    max_duration: int,
    n_iter: int,
    seed: int,
    smooth: float,
    min_var: float,
) -> Tuple[HSMMParams, List[float]]:
    params = init_params(
        seqs,
        labels,
        n_states=n_states,
        max_duration=max_duration,
        obs_names=obs_names,
        seed=seed,
    )
    history: List[float] = []
    X_all = [np.asarray(s, dtype=np.float64) for s in seqs]
    labels = np.asarray(labels, dtype=int)
    for _ in range(int(n_iter)):
        gamma_sum = np.zeros((n_states,), dtype=np.float64)
        x_sum = np.zeros_like(params.mu)
        x2_sum = np.zeros_like(params.mu)
        pi_counts = np.full((2, n_states), smooth, dtype=np.float64)
        trans_counts = np.full((2, n_states, n_states), smooth, dtype=np.float64)
        for y in range(2):
            np.fill_diagonal(trans_counts[y], 0.0)
        dur_counts = np.full((2, n_states, max_duration), smooth, dtype=np.float64)
        total_ll = 0.0
        for X, y in zip(X_all, labels):
            ll, gamma, pc, tc, dc = expected_counts_one(X, params, int(y))
            total_ll += ll
            g = gamma.sum(axis=0)
            gamma_sum += g
            x_sum += gamma.T @ X
            x2_sum += gamma.T @ (X ** 2)
            pi_counts[y] += pc
            trans_counts[y] += tc
            dur_counts[y] += dc
        for k in range(n_states):
            if gamma_sum[k] > EPS:
                params.mu[k] = x_sum[k] / gamma_sum[k]
                var = x2_sum[k] / gamma_sum[k] - params.mu[k] ** 2
                params.var[k] = np.maximum(var, min_var)
        params.pi = pi_counts / np.maximum(pi_counts.sum(axis=1, keepdims=True), EPS)
        for y in range(2):
            row_sum = trans_counts[y].sum(axis=1, keepdims=True)
            params.trans[y] = trans_counts[y] / np.maximum(row_sum, EPS)
            np.fill_diagonal(params.trans[y], 0.0)
            row_sum = params.trans[y].sum(axis=1, keepdims=True)
            params.trans[y] = params.trans[y] / np.maximum(row_sum, EPS)
        params.dur = dur_counts / np.maximum(dur_counts.sum(axis=2, keepdims=True), EPS)
        history.append(float(total_ll))
    return params, history


def hsmm_loglik(X: np.ndarray, params: HSMMParams, y: int, prefix_len: Optional[int] = None) -> float:
    L = X.shape[0] if prefix_len is None else max(1, min(int(prefix_len), X.shape[0]))
    ll, _, _ = forward_hsmm(X[:L], params, int(y))
    return float(ll)


def hsmm_llr(X: np.ndarray, params: HSMMParams, prefix_len: Optional[int] = None, normalize: bool = True) -> float:
    L = X.shape[0] if prefix_len is None else max(1, min(int(prefix_len), X.shape[0]))
    err = hsmm_loglik(X, params, 1, prefix_len=L)
    cor = hsmm_loglik(X, params, 0, prefix_len=L)
    val = err - cor
    return float(val / max(1, L)) if normalize else float(val)


def viterbi_hsmm(X: np.ndarray, params: HSMMParams, y: int) -> np.ndarray:
    K = params.n_states
    Dmax = params.max_duration
    T = X.shape[0]
    log_b = gaussian_log_prob(X, params.mu, params.var)
    cum = segment_sums(log_b)
    log_pi = np.log(np.maximum(params.pi[y], EPS))
    log_A = np.full((K, K), NINF, dtype=np.float64)
    m = params.trans[y] > 0
    log_A[m] = np.log(params.trans[y][m])
    log_dur = np.log(np.maximum(params.dur[y], EPS))
    delta = np.full((T + 1, K), NINF, dtype=np.float64)
    back: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for t in range(1, T + 1):
        for k in range(K):
            best = NINF
            best_prev = (-1, 0)
            for d in range(1, min(Dmax, t) + 1):
                s = t - d
                emit = cum[k, t] - cum[k, s]
                if s == 0:
                    val = log_pi[k] + log_dur[k, d - 1] + emit
                    prev = (-1, s)
                else:
                    vals = delta[s] + log_A[:, k]
                    prev_state = int(np.argmax(vals))
                    val = float(vals[prev_state] + log_dur[k, d - 1] + emit)
                    prev = (prev_state, s)
                if val > best:
                    best = val
                    best_prev = prev
            delta[t, k] = best
            back[(t, k)] = best_prev
    states = np.zeros(T, dtype=int)
    k = int(np.argmax(delta[T]))
    t = T
    while t > 0:
        prev_k, s = back[(t, k)]
        states[s:t] = k
        t = s
        k = prev_k if prev_k >= 0 else 0
    return states


def local_groups_from_problem_ids(problem_ids: np.ndarray, y: np.ndarray, min_per_class: int) -> List[np.ndarray]:
    groups = []
    for p in np.unique(problem_ids):
        idx = np.where(problem_ids == p)[0]
        if np.sum(y[idx] == 1) >= min_per_class and np.sum(y[idx] == 0) >= min_per_class:
            groups.append(idx)
    return groups


def score_permutation_pvalue(
    scores: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    obs, pairs = within_pair_auroc(groups, scores, y)
    if not np.isfinite(obs) or n_perm <= 0:
        return {"observed": obs, "pairs": pairs, "n_permutations": int(n_perm), "p_ge": None}
    rng = np.random.default_rng(seed)
    vals = []
    y_perm = np.array(y, copy=True)
    for _ in range(int(n_perm)):
        for idx in groups:
            y_perm[idx] = rng.permutation(y[idx])
        val, _ = within_pair_auroc(groups, scores, y_perm)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return {"observed": obs, "pairs": pairs, "n_permutations": int(n_perm), "p_ge": None}
    vals_arr = np.asarray(vals, dtype=np.float64)
    return {
        "observed": obs,
        "pairs": pairs,
        "n_permutations": int(len(vals_arr)),
        "p_ge": float((1.0 + np.sum(vals_arr >= obs)) / (len(vals_arr) + 1.0)),
        "null_mean": float(np.mean(vals_arr)),
        "null_q95": float(np.quantile(vals_arr, 0.95)),
    }


def static_scores(rd: RegimeData) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for ci, ch in enumerate(rd.channels):
        vals = rd.tensor[:, ci, :]
        out[f"mean:{ch}"] = np.nanmean(vals, axis=1)
        out[f"max:{ch}"] = np.nanmax(vals, axis=1)
        L = max(1, int(math.floor(0.8 * vals.shape[1])))
        out[f"mean80:{ch}"] = np.nanmean(vals[:, :L], axis=1)
    out["mean:all_channels"] = np.nanmean(rd.tensor, axis=(1, 2))
    return out


def evaluate_scores(
    rd: RegimeData,
    scores: Mapping[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    rows = {}
    for name, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        au, pairs = within_pair_auroc(rd.groups, s, rd.y_err)
        rows[name] = {
            "same_problem_paired_auroc": au,
            "n_pairs": pairs,
            "paired_delta": paired_delta(rd.groups, s, rd.y_err),
            "score_error": descriptive(s[rd.y_err == 1]),
            "score_correct": descriptive(s[rd.y_err == 0]),
            "permutation": score_permutation_pvalue(
                s,
                rd.y_err,
                rd.groups,
                n_perm=permutations,
                seed=seed + len(rows),
            ),
        }
    return rows


def crossfit_hsmm_scores(rd: RegimeData, args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    n = len(rd.sequences)
    scores: Dict[str, np.ndarray] = {
        "hsmm_llr_full": np.full(n, np.nan),
        "hsmm_llr_censor80": np.full(n, np.nan),
    }
    prefix_fracs = [float(x) for x in args.prefix_fracs.split(",") if x.strip()]
    prefix_fracs = sorted(set(max(0.05, min(1.0, x)) for x in prefix_fracs))
    for f in prefix_fracs:
        scores[f"hsmm_llr_prefix{int(round(100 * f)):02d}"] = np.full(n, np.nan)
    fold_rows: List[Dict[str, Any]] = []
    folds = group_folds(rd.problem_ids, args.folds, args.seed)
    if not folds:
        raise SystemExit("not enough problem groups for cross-fitting")
    for fold, (tr, te) in enumerate(folds):
        train_seqs = [rd.sequences[i] for i in tr]
        center, scale = robust_matrix_scaler(train_seqs)
        train_scaled = apply_scaler(train_seqs, center, scale)
        params, history = fit_hsmm(
            train_scaled,
            rd.y_err[tr],
            obs_names=rd.obs_names,
            n_states=args.states,
            max_duration=args.max_duration,
            n_iter=args.em_iters,
            seed=args.seed + 1009 * fold,
            smooth=args.smooth,
            min_var=args.min_var,
        )
        params.scaler_center = center
        params.scaler_scale = scale
        for i in te:
            seq = apply_scaler([rd.sequences[i]], center, scale)[0]
            scores["hsmm_llr_full"][i] = hsmm_llr(seq, params, normalize=True)
            scores["hsmm_llr_censor80"][i] = hsmm_llr(
                seq,
                params,
                prefix_len=max(2, int(math.floor(0.8 * len(seq)))),
                normalize=True,
            )
            for f in prefix_fracs:
                scores[f"hsmm_llr_prefix{int(round(100 * f)):02d}"][i] = hsmm_llr(
                    seq,
                    params,
                    prefix_len=max(2, int(math.floor(f * len(seq)))),
                    normalize=True,
                )
        fold_rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "train_error": int(rd.y_err[tr].sum()),
                "test_error": int(rd.y_err[te].sum()),
                "loglik_history": history,
            }
        )
    return scores, fold_rows


def transition_difference(params: HSMMParams) -> Dict[str, Any]:
    A0 = params.trans[0]
    A1 = params.trans[1]
    D0 = params.dur[0]
    D1 = params.dur[1]
    durations = np.arange(1, params.max_duration + 1, dtype=np.float64)
    mean_d0 = D0 @ durations
    mean_d1 = D1 @ durations
    return {
        "transition_l1": float(np.sum(np.abs(A1 - A0))),
        "transition_fro": float(np.sqrt(np.sum((A1 - A0) ** 2))),
        "duration_l1": float(np.sum(np.abs(D1 - D0))),
        "mean_duration_correct": mean_d0,
        "mean_duration_error": mean_d1,
        "mean_duration_error_minus_correct": mean_d1 - mean_d0,
        "pi_correct": params.pi[0],
        "pi_error": params.pi[1],
        "trans_correct": A0,
        "trans_error": A1,
        "duration_correct": D0,
        "duration_error": D1,
    }


def state_usage(rd: RegimeData, params: HSMMParams) -> Dict[str, Any]:
    scaled = apply_scaler(rd.sequences, params.scaler_center, params.scaler_scale)
    usage = {0: [], 1: []}
    durations = {0: [[] for _ in range(params.n_states)], 1: [[] for _ in range(params.n_states)]}
    for seq, y in zip(scaled, rd.y_err):
        st = viterbi_hsmm(seq, params, int(y))
        counts = np.bincount(st, minlength=params.n_states) / max(1, len(st))
        usage[int(y)].append(counts)
        start = 0
        while start < len(st):
            end = start + 1
            while end < len(st) and st[end] == st[start]:
                end += 1
            durations[int(y)][int(st[start])].append(end - start)
            start = end
    out: Dict[str, Any] = {}
    for y, label in ((0, "correct"), (1, "error")):
        mat = np.vstack(usage[y]) if usage[y] else np.empty((0, params.n_states))
        out[f"occupancy_{label}"] = np.nanmean(mat, axis=0) if len(mat) else np.full(params.n_states, np.nan)
        out[f"duration_{label}"] = [
            descriptive(np.asarray(durations[y][k], dtype=np.float64)) for k in range(params.n_states)
        ]
    if usage[0] and usage[1]:
        out["occupancy_error_minus_correct"] = out["occupancy_error"] - out["occupancy_correct"]
    return out


def fit_final_model(rd: RegimeData, args: argparse.Namespace) -> Tuple[HSMMParams, List[float], Dict[str, Any]]:
    center, scale = robust_matrix_scaler(rd.sequences)
    scaled = apply_scaler(rd.sequences, center, scale)
    params, history = fit_hsmm(
        scaled,
        rd.y_err,
        obs_names=rd.obs_names,
        n_states=args.states,
        max_duration=args.max_duration,
        n_iter=args.em_iters,
        seed=args.seed + 777,
        smooth=args.smooth,
        min_var=args.min_var,
    )
    params.scaler_center = center
    params.scaler_scale = scale
    interp = {
        "state_emission_means_scaled": params.mu,
        "state_emission_vars_scaled": params.var,
        "obs_names": rd.obs_names,
        "transition_difference": transition_difference(params),
    }
    interp.update(state_usage(rd, params))
    return params, history, interp


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    rd = load_regime_data(path, args)
    hsmm_scores, fold_rows = crossfit_hsmm_scores(rd, args)
    base_scores = static_scores(rd)
    all_scores = {**hsmm_scores, **base_scores}
    score_table = evaluate_scores(rd, all_scores, permutations=args.permutations, seed=args.seed)
    _, final_history, final_interp = fit_final_model(rd, args)
    best_static = max(
        ((name, row) for name, row in score_table.items() if not name.startswith("hsmm_")),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_paired_auroc"], nan=-1.0),
    )
    hsmm_full = score_table.get("hsmm_llr_full", {})
    hsmm_c80 = score_table.get("hsmm_llr_censor80", {})
    prefix_rows = {
        name: row
        for name, row in score_table.items()
        if name.startswith("hsmm_llr_prefix") or name in ("hsmm_llr_full", "hsmm_llr_censor80")
    }
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "policy": args.policy,
            "n_samples": int(len(rd.y_err)),
            "n_error": int(rd.y_err.sum()),
            "n_correct": int(len(rd.y_err) - rd.y_err.sum()),
            "n_contrastive_problems": int(len(rd.groups)),
            "channels": rd.channels,
            "channel_coverage": rd.channel_coverage,
            "grid": rd.grid,
            "obs_names": rd.obs_names,
            "states": int(args.states),
            "max_duration": int(args.max_duration),
            "problem_center": args.problem_center,
            "notes": {
                "no_pos_feature": "Normalized position is not used as a model input.",
                "no_healthy_template": "Problem centering uses unlabeled same-problem samples, not correct-chain templates.",
                "shared_emissions": "Correct/error classes share regime emissions; class differences are transition, duration, and initial-state grammar.",
            },
        },
        "headline": {
            "hsmm_full_same_problem_auroc": hsmm_full.get("same_problem_paired_auroc"),
            "hsmm_censor80_same_problem_auroc": hsmm_c80.get("same_problem_paired_auroc"),
            "best_static_name": best_static[0],
            "best_static_same_problem_auroc": best_static[1].get("same_problem_paired_auroc"),
            "hsmm_minus_best_static": (
                hsmm_full.get("same_problem_paired_auroc", float("nan"))
                - best_static[1].get("same_problem_paired_auroc", float("nan"))
            ),
            "censor80_minus_full": (
                hsmm_c80.get("same_problem_paired_auroc", float("nan"))
                - hsmm_full.get("same_problem_paired_auroc", float("nan"))
            ),
        },
        "scores": score_table,
        "prefix_curve": prefix_rows,
        "folds": fold_rows,
        "final_model": {
            "loglik_history": final_history,
            **final_interp,
        },
    }


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    h = res["headline"]
    lines = [
        f"# Within-Problem Regime HSMM Audit: `{res['meta']['basename']}`",
        "",
        "## Headline",
        "",
        f"- HSMM full same-problem AUROC: `{h['hsmm_full_same_problem_auroc']:.3f}`",
        f"- HSMM censor80 same-problem AUROC: `{h['hsmm_censor80_same_problem_auroc']:.3f}`",
        f"- Best static baseline: `{h['best_static_name']}` = `{h['best_static_same_problem_auroc']:.3f}`",
        f"- HSMM minus best static: `{h['hsmm_minus_best_static']:+.3f}`",
        f"- Censor80 minus full: `{h['censor80_minus_full']:+.3f}`",
        "",
        "## Prefix Curve",
        "",
        "| score | same-problem AUROC | pairs | permutation p_ge |",
        "|---|---:|---:|---:|",
    ]
    for name, row in res["prefix_curve"].items():
        perm = row.get("permutation", {})
        p = perm.get("p_ge")
        ptxt = "" if p is None else f"{p:.4f}"
        lines.append(
            f"| `{name}` | {row['same_problem_paired_auroc']:.3f} | {row['n_pairs']} | {ptxt} |"
        )
    lines += [
        "",
        "## Top Scores",
        "",
        "| score | same-problem AUROC | pairs | error median | correct median |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = sorted(
        res["scores"].items(),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_paired_auroc"], nan=-1.0),
        reverse=True,
    )
    for name, row in rows[:20]:
        err = row.get("score_error", {})
        cor = row.get("score_correct", {})
        lines.append(
            f"| `{name}` | {row['same_problem_paired_auroc']:.3f} | {row['n_pairs']} | "
            f"{err.get('median', float('nan')):.3f} | {cor.get('median', float('nan')):.3f} |"
        )
    lines += [
        "",
        "## Regime Grammar Difference",
        "",
    ]
    td = res["final_model"]["transition_difference"]
    lines.append(f"- Transition L1 difference: `{td['transition_l1']:.3f}`")
    lines.append(f"- Duration L1 difference: `{td['duration_l1']:.3f}`")
    lines.append("")
    lines.append("Anti-artifact notes: position is not an input feature; censor80 is reported to catch endpoint-only behavior.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, f"{stem}.json")
    mpath = os.path.join(output_dir, f"{stem}.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, ensure_ascii=False, indent=2)
    write_markdown(mpath, finite_json(res))
    return jpath, mpath


def print_result(res: Mapping[str, Any]) -> None:
    h = res["headline"]
    meta = res["meta"]
    print(f"\n===== within-problem regime HSMM | {meta['basename']} =====")
    print(
        f"samples {meta['n_samples']} | err {meta['n_error']} | "
        f"problems {meta['n_contrastive_problems']} | channels {meta['channels']}"
    )
    print(
        f"HSMM full {h['hsmm_full_same_problem_auroc']:.3f} | "
        f"censor80 {h['hsmm_censor80_same_problem_auroc']:.3f} | "
        f"best static {h['best_static_name']}={h['best_static_same_problem_auroc']:.3f} | "
        f"delta {h['hsmm_minus_best_static']:+.3f}"
    )
    print("\nPrefix / endpoint control:")
    for name, row in res["prefix_curve"].items():
        print(f"  {name:24s} AUROC {row['same_problem_paired_auroc']:.3f} pairs={row['n_pairs']}")
    print("\nTop scores:")
    rows = sorted(
        res["scores"].items(),
        key=lambda kv: np.nan_to_num(kv[1]["same_problem_paired_auroc"], nan=-1.0),
        reverse=True,
    )
    for name, row in rows[:12]:
        print(f"  {name:24s} AUROC {row['same_problem_paired_auroc']:.3f} pairs={row['n_pairs']}")
    td = res["final_model"]["transition_difference"]
    print("\nRegime grammar:")
    print(f"  transition L1 {td['transition_l1']:.3f} | duration L1 {td['duration_l1']:.3f}")


def _cloud_for_spread(spread: float, *, n_tok: int, dim: int) -> np.ndarray:
    r = float(np.clip(1.0 - spread, 0.02, 0.98))
    orth = math.sqrt(max(0.0, 1.0 - r * r))
    H = np.zeros((n_tok, 1, dim), dtype=np.float32)
    axes = [1, 2, 3]
    for i in range(n_tok):
        sign = 1.0 if i % 2 == 0 else -1.0
        ax = axes[(i // 2) % len(axes)]
        H[i, 0, 0] = r
        H[i, 0, ax] = sign * orth
    return H


def _object_array(xs: Sequence[object]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest(path: str, *, seed: int = 0, n_problems: int = 32, samples_per_problem: int = 4) -> None:
    """Synthetic same-problem data where order, not static level, carries signal."""
    rng = np.random.default_rng(seed)
    ids: List[str] = []
    pids: List[int] = []
    sample_idx: List[int] = []
    y_correct: List[int] = []
    n_steps: List[int] = []
    cloud_sizes: List[np.ndarray] = []
    clouds: List[np.ndarray] = []
    entropy_rows: List[np.ndarray] = []
    pr_rows: List[np.ndarray] = []
    ae_rows: List[np.ndarray] = []

    # Same approximate multiset, different grammar: correct contracts smoothly
    # then recovers; error rebounds into a detached high-risk regime mid-chain.
    correct_proto = np.array([0.75, 0.45, 0.15, -0.25, -0.55, -0.75, -0.45, -0.10, 0.20, 0.55])
    error_proto = np.array([0.75, 0.45, 0.10, -0.45, 0.35, 0.75, 0.25, -0.55, -0.70, -0.05])
    correct_proto -= correct_proto.mean()
    error_proto -= error_proto.mean()
    layers = 33
    dim = 12
    n_tok = 6
    for p in range(n_problems):
        problem_offset = rng.normal(scale=0.025)
        problem_slope = rng.normal(scale=0.015)
        for s in range(samples_per_problem):
            is_err = int(s >= samples_per_problem // 2)
            proto = error_proto if is_err else correct_proto
            jitter = rng.normal(scale=0.035, size=len(proto))
            latent = proto + jitter + problem_slope * np.linspace(-1, 1, len(proto))
            spread = np.clip(0.38 + 0.16 * latent + problem_offset, 0.08, 0.82)
            entropy = np.clip(0.52 + 0.22 * latent + problem_offset, 0.03, 1.50)
            pr = np.clip(3.2 + 0.70 * latent + 0.5 * problem_offset, 0.30, 8.0)
            ae = np.clip(0.32 + 0.20 * latent + 0.3 * problem_offset, 0.02, 1.2)
            # Remove most static mean differences; leave the regime order intact.
            entropy -= entropy.mean() - 0.52
            pr -= pr.mean() - 3.2
            ae -= ae.mean() - 0.32

            ids.append(f"p{p}_s{s}")
            pids.append(p)
            sample_idx.append(s)
            y_correct.append(0 if is_err else 1)
            n_steps.append(len(proto))
            sizes = np.full(len(proto), n_tok, dtype=np.int32)
            cloud_sizes.append(sizes)
            clouds.append(np.concatenate([_cloud_for_spread(v, n_tok=n_tok, dim=dim) for v in spread], axis=0))
            entropy_rows.append(entropy.astype(np.float32))
            pr_mat = np.tile(pr[:, None], (1, layers)).astype(np.float32)
            ae_mat = np.tile(ae[:, None], (1, layers)).astype(np.float32)
            pr_rows.append(pr_mat)
            ae_rows.append(ae_mat)

    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=object),
        problem_ids=np.asarray(pids, dtype=np.int32),
        sample_idx=np.asarray(sample_idx, dtype=np.int32),
        is_correct=np.asarray(y_correct, dtype=np.int32),
        is_correct_strict=np.asarray(y_correct, dtype=np.int32),
        format_ok=np.ones(len(ids), dtype=np.int32),
        n_steps=np.asarray(n_steps, dtype=np.int32),
        cloud_sizes=_object_array(cloud_sizes),
        sv_clouds=_object_array(clouds),
        sv_out_entropy=_object_array(entropy_rows),
        sv_pr_step_exp=_object_array(pr_rows),
        sv_ae_step_exp=_object_array(ae_rows),
        model_name=np.asarray("regime-selftest"),
        prompt_style=np.asarray("regime-selftest"),
        step_split=np.asarray("regime-selftest"),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    h = res["headline"]
    if h["hsmm_full_same_problem_auroc"] < 0.75:
        raise SystemExit("selftest failed: HSMM did not recover same-problem regime signal")
    if res["final_model"]["transition_difference"]["transition_l1"] <= 0.1:
        raise SystemExit("selftest failed: class-specific transition grammars did not diverge")


def resolve_input(args: argparse.Namespace) -> str:
    if args.input:
        return args.input
    raise SystemExit("provide --input or use --selftest")


def main() -> None:
    ap = argparse.ArgumentParser(description="Same-problem latent-regime HSMM audit")
    ap.add_argument("--input", default=None)
    ap.add_argument("--policy", default="answer_format_ok", choices=["answer", "strict", "answer_format_ok"])
    ap.add_argument("--channels", default=DEFAULT_CHANNELS)
    ap.add_argument("--bands", default="mid")
    ap.add_argument("--require_channels", action="store_true")
    ap.add_argument("--min_channel_coverage", type=float, default=0.80)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--problem_center", default="problem_median", choices=["problem_median", "none"])
    ap.add_argument("--include_abs_delta", action="store_true")
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--max_duration", type=int, default=8)
    ap.add_argument("--em_iters", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smooth", type=float, default=1e-2)
    ap.add_argument("--min_var", type=float, default=1e-3)
    ap.add_argument("--prefix_fracs", default="0.40,0.60,0.80,1.00")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/within_problem_regime_hsmm")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            npz = os.path.join(td, "regime_hsmm_selftest.npz")
            make_selftest(npz, seed=args.seed)
            args.input = npz
            res = run(npz, args)
            assert_selftest(res)
            print_result(res)
            jpath, mpath = write_outputs(res, args.output_dir, "selftest")
            print(f"\nselftest passed; saved: {jpath} | {mpath}")
        return

    path = resolve_input(args)
    res = run(path, args)
    print_result(res)
    stem = os.path.splitext(os.path.basename(path))[0]
    jpath, mpath = write_outputs(res, args.output_dir, stem)
    print(f"\nsaved: {jpath} | {mpath}")


if __name__ == "__main__":
    main()
