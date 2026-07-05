#!/usr/bin/env python3
"""Refined transition-tube audits for same-problem multisampling data.

This script runs three stricter tests of the constrained-transition-manifold
hypothesis:

1. support_holdout_oracle:
   For each problem, split correct samples into support/heldout folds.  Fit a
   same-problem correct tube on the support fold only, then score heldout
   correct samples and all error samples against the same support tube.  This
   is still diagnostic, but it avoids target leakage and avoids giving errors a
   strictly richer reference than correct heldouts.

2. tube_tail_curve:
   In addition to scalar off-tube residuals, measure the whole residual energy
   tail after projecting candidate transitions onto the correct-tube basis.
   If failures need more directions, k90/tail_auc/residual-at-k should separate
   better than one residual scalar.

3. prompt_conditioned_tube:
   Fit a local correct tube from nearest training problems under a prompt key.
   If qvec/prompt vectors are unavailable, the script falls back to a
   first-step hidden proxy and states that in the output metadata.

It also reports lightweight layer-synergy baselines when sv_vec_step_exp stores
multiple layers, because the within-problem data has a layer axis for step
vectors even when token clouds were stored at only one layer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multisample_transition_tube_audit import (
    EPS,
    TransitionTube,
    auroc_signed,
    band_cols,
    collect_score_rows,
    collect_transitions,
    descriptive,
    finite_json,
    fit_tube,
    group_folds,
    label_policy,
    make_selftest,
    normalize_steps,
    same_problem_groups,
    safe_mean,
    score_sequence,
    step_sequences,
    transitions,
    window_mask,
)


BASE_SCORE_NAMES = [
    "off_mean",
    "off_late",
    "off_max",
    "off_ratio_mean",
    "off_ratio_late",
    "off_ratio_max",
    "off_z_mean",
    "off_z_late",
    "off_z_max",
    "rank_energy",
    "transition_eff_rank",
    "off_energy_ratio",
    "tube_rank",
]

TAIL_SCORE_NAMES = [
    "tail_auc",
    "tail_slope",
    "tail_resid_k1",
    "tail_resid_k4",
    "tail_resid_k8",
    "tail_resid_k16",
    "tail_k50",
    "tail_k75",
    "tail_k90",
    "tail_k95",
]

ALL_TUBE_SCORE_NAMES = BASE_SCORE_NAMES + TAIL_SCORE_NAMES


def l2_unit_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, EPS)


def normalize_key(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all():
        return np.full(v.shape, np.nan, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / max(n, EPS)


def vector_payload_meta(data: np.lib.npyio.NpzFile, sample_limit: int = 64) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "has_sv_vec_step_exp": "sv_vec_step_exp" in data.files,
        "sv_vectors_stored": bool(data.get("sv_vectors_stored", np.array(False))),
    }
    if "layers_used" in data.files:
        meta["layers_used"] = [int(x) for x in np.asarray(data["layers_used"]).reshape(-1)]
    if "cloud_layers" in data.files:
        meta["cloud_layers"] = [int(x) for x in np.asarray(data["cloud_layers"]).reshape(-1)]
    if "sv_vec_step_exp" not in data.files:
        return meta
    shapes: List[Tuple[int, int, int]] = []
    raw = data["sv_vec_step_exp"]
    for obj in raw[: min(len(raw), sample_limit)]:
        V = np.asarray(obj)
        if V.ndim == 3:
            shapes.append((int(V.shape[0]), int(V.shape[1]), int(V.shape[2])))
    meta["example_shapes"] = shapes[:5]
    if shapes:
        meta["n_layers_min"] = int(min(s[1] for s in shapes))
        meta["n_layers_max"] = int(max(s[1] for s in shapes))
        meta["hidden_dim_min"] = int(min(s[2] for s in shapes))
        meta["hidden_dim_max"] = int(max(s[2] for s in shapes))
        meta["multilayer_step_vectors"] = bool(max(s[1] for s in shapes) > 1)
    return meta


def raw_layer_sequences(data: np.lib.npyio.NpzFile, *, normalize: str) -> List[np.ndarray]:
    if "sv_vec_step_exp" not in data.files or not bool(data.get("sv_vectors_stored", np.array(False))):
        raise SystemExit("need sv_vectors_stored=True and sv_vec_step_exp in the npz")
    out: List[np.ndarray] = []
    for obj in data["sv_vec_step_exp"]:
        V = np.asarray(obj, dtype=np.float64)
        if V.ndim != 3:
            out.append(np.empty((0, 0, 0), dtype=np.float64))
            continue
        good = np.isfinite(V).all(axis=(1, 2))
        V = V[good]
        if normalize == "l2":
            n = np.linalg.norm(V, axis=2, keepdims=True)
            V = V / np.maximum(n, EPS)
        elif normalize == "center_chain":
            V = V - np.nanmean(V, axis=0, keepdims=True)
        elif normalize != "none":
            raise ValueError(normalize)
        out.append(V)
    return out


def tube_tail_stats(
    D: np.ndarray,
    tube: TransitionTube,
    *,
    k_values: Sequence[int] = (1, 4, 8, 16),
    levels: Sequence[float] = (0.50, 0.75, 0.90, 0.95),
) -> Dict[str, float]:
    if D.size == 0 or tube.comps_max.size == 0:
        return {n: float("nan") for n in TAIL_SCORE_NAMES}
    X = D - tube.mu
    U = tube.comps_max
    C = X @ U.T
    in_e = np.sum(C ** 2, axis=0)
    P = C @ U
    off_e = float(np.sum((X - P) ** 2))
    total = float(np.sum(in_e) + off_e)
    if total <= EPS:
        out = {n: 0.0 for n in TAIL_SCORE_NAMES}
        return out
    K = int(U.shape[0])
    cum = np.cumsum(in_e) / total
    residual = np.array([(float(np.sum(in_e[k:]) + off_e) / total) for k in range(1, K + 1)], dtype=np.float64)
    out: Dict[str, float] = {
        "tail_auc": safe_mean(residual),
    }
    x = np.arange(1, K + 1, dtype=np.float64)
    y = np.log(np.maximum(residual, EPS))
    if K >= 2 and np.isfinite(y).all():
        out["tail_slope"] = float(np.polyfit(x, y, 1)[0])
    else:
        out["tail_slope"] = float("nan")
    for k in k_values:
        kk = int(min(max(1, k), K))
        out[f"tail_resid_k{k}"] = float(residual[kk - 1])
    for q in levels:
        name = f"tail_k{int(round(q * 100))}"
        if np.any(cum >= q):
            out[name] = float(np.searchsorted(cum, q) + 1)
        else:
            out[name] = float(K + 1)
    return out


def score_sequence_refined(X: np.ndarray, tube: TransitionTube, *, rank_energy: float) -> Dict[str, float]:
    out = score_sequence(X, tube, rank_energy=rank_energy)
    D = transitions(X)
    out.update(tube_tail_stats(D, tube))
    return out


def add_score(
    sums: Dict[str, np.ndarray],
    counts: Dict[str, np.ndarray],
    positions: Dict[str, np.ndarray],
    i: int,
    score: Mapping[str, float],
) -> None:
    for n in ALL_TUBE_SCORE_NAMES:
        if n in score and np.isfinite(score[n]):
            sums[n][i] += float(score[n])
            counts[n][i] += 1.0
    if "off_z_max_pos" in score and np.isfinite(score["off_z_max_pos"]):
        positions["off_z_max"][i] = float(score["off_z_max_pos"])


def finalize_scores(sums: Dict[str, np.ndarray], counts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for n, s in sums.items():
        vals = np.full_like(s, np.nan, dtype=np.float64)
        m = counts[n] > 0
        vals[m] = s[m] / counts[n][m]
        out[n] = vals
    return out


def empty_score_accumulators(n: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    sums = {name: np.zeros(n, dtype=np.float64) for name in ALL_TUBE_SCORE_NAMES}
    counts = {name: np.zeros(n, dtype=np.float64) for name in ALL_TUBE_SCORE_NAMES}
    positions = {"off_z_max": np.full(n, np.nan, dtype=np.float64)}
    return sums, counts, positions


def split_folds(items: np.ndarray, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    items = np.asarray(items, dtype=int)
    if items.size == 0:
        return []
    order = items.copy()
    rng.shuffle(order)
    k = max(1, min(int(k), int(order.size)))
    return [fold.astype(int) for fold in np.array_split(order, k) if fold.size]


def fit_refs(
    seqs: Sequence[np.ndarray],
    refs: Sequence[int],
    args: argparse.Namespace,
    *,
    min_transitions: int,
) -> Optional[TransitionTube]:
    D = collect_transitions(seqs, refs)
    return fit_tube(
        D,
        rank=args.rank,
        max_rank=args.max_rank,
        energy=args.energy,
        min_transitions=min_transitions,
    )


def support_holdout_oracle_scores(
    seqs: Sequence[np.ndarray],
    problem_ids: np.ndarray,
    y_err: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    n = len(seqs)
    sums, counts, positions = empty_score_accumulators(n)
    rng = np.random.default_rng(args.seed)
    used_problems = 0
    tubes = 0
    ranks: List[int] = []
    scored_error = set()
    scored_correct = set()
    skipped = defaultdict(int)
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        correct = idx[y_err[idx] == 0]
        errors = idx[y_err[idx] == 1]
        if correct.size < args.oracle_min_correct or errors.size == 0:
            skipped["class_count"] += 1
            continue
        folds = split_folds(correct, args.oracle_holdout_folds, rng)
        problem_used = False
        for heldout in folds:
            support = np.setdiff1d(correct, heldout, assume_unique=False)
            if support.size < args.oracle_min_support:
                skipped["support_too_small"] += 1
                continue
            tube = fit_refs(seqs, support, args, min_transitions=args.oracle_min_transitions)
            if tube is None:
                skipped["fit_failed"] += 1
                continue
            tubes += 1
            ranks.append(tube.rank)
            problem_used = True
            for i in heldout:
                s = score_sequence_refined(seqs[int(i)], tube, rank_energy=args.rank_energy)
                add_score(sums, counts, positions, int(i), s)
                scored_correct.add(int(i))
            for i in errors:
                s = score_sequence_refined(seqs[int(i)], tube, rank_energy=args.rank_energy)
                add_score(sums, counts, positions, int(i), s)
                scored_error.add(int(i))
        if problem_used:
            used_problems += 1
    meta = {
        "problems_used": int(used_problems),
        "support_tubes": int(tubes),
        "mean_rank": float(np.mean(ranks)) if ranks else None,
        "scored_error": int(len(scored_error)),
        "scored_correct_holdout": int(len(scored_correct)),
        "skipped": dict(skipped),
        "protocol": "same-problem support correct tube; heldout correct and errors scored against the same support folds",
    }
    return finalize_scores(sums, counts), positions, meta


def global_tail_crossfit_scores(
    seqs: Sequence[np.ndarray],
    problem_ids: np.ndarray,
    y_err: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    n = len(seqs)
    scores = {name: np.full(n, np.nan, dtype=np.float64) for name in ALL_TUBE_SCORE_NAMES}
    positions = {"off_z_max": np.full(n, np.nan, dtype=np.float64)}
    ranks: List[int] = []
    idx_all = np.where(mask)[0]
    for tr_rel, te_rel in group_folds(problem_ids[mask], args.folds, args.seed):
        tr_idx = idx_all[tr_rel]
        te_idx = idx_all[te_rel]
        correct = tr_idx[y_err[tr_idx] == 0]
        tube = fit_refs(seqs, correct, args, min_transitions=args.min_transitions)
        if tube is None:
            continue
        ranks.append(tube.rank)
        for i in te_idx:
            s = score_sequence_refined(seqs[int(i)], tube, rank_energy=args.rank_energy)
            for name in ALL_TUBE_SCORE_NAMES:
                if name in s:
                    scores[name][i] = s[name]
            if "off_z_max_pos" in s:
                positions["off_z_max"][i] = s["off_z_max_pos"]
    meta = {"fold_ranks": ranks, "mean_rank": float(np.mean(ranks)) if ranks else None}
    return scores, positions, meta


def extract_prompt_keys(
    data: np.lib.npyio.NpzFile,
    seqs: Sequence[np.ndarray],
    *,
    mode: str,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    n = len(seqs)
    requested = mode
    candidates = ["qvec", "question_vec", "prompt_vec", "prompt_qvec"]
    if mode in ("auto", "qvec"):
        for key in candidates:
            if key not in data.files:
                continue
            A = np.asarray(data[key], dtype=np.float64)
            if A.shape[0] == n:
                flat = A.reshape((n, -1))
                return np.asarray([normalize_key(v) for v in flat]), key, {"requested": requested, "key_shape": list(A.shape)}
            if A.ndim >= 2:
                v = normalize_key(A.reshape(-1))
                K = np.tile(v[None, :], (n, 1))
                return K, key, {"requested": requested, "key_shape": list(A.shape), "broadcast": True}
        if mode == "qvec":
            raise SystemExit("requested --retrieval_key qvec, but no qvec/prompt vector key was found")

    key_mode = "first_step_proxy" if mode == "auto" else mode
    keys: List[np.ndarray] = []
    for X in seqs:
        if X.ndim != 2 or X.shape[0] == 0:
            keys.append(np.full(0, np.nan, dtype=np.float64))
            continue
        if key_mode == "early_mean":
            m = window_mask(X.shape[0], "early")
            v = np.nanmean(X[m], axis=0)
        elif key_mode in ("first_step", "first_step_proxy"):
            v = X[0]
        else:
            raise ValueError(f"unknown retrieval key mode: {mode}")
        keys.append(normalize_key(v))
    dim = max((k.size for k in keys), default=0)
    K = np.full((n, dim), np.nan, dtype=np.float64)
    for i, k in enumerate(keys):
        if k.size == dim:
            K[i] = k
    return K, key_mode, {"requested": requested, "fallback_reason": "no qvec/prompt vector stored in npz"}


def problem_prototypes(
    keys: np.ndarray,
    problem_ids: np.ndarray,
    idxs: np.ndarray,
    y_err: np.ndarray,
    *,
    proto_policy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    protos: List[np.ndarray] = []
    pids: List[int] = []
    for p in np.unique(problem_ids[idxs]):
        pidx = idxs[problem_ids[idxs] == p]
        if proto_policy == "correct":
            pidx = pidx[y_err[pidx] == 0]
        elif proto_policy != "all":
            raise ValueError(proto_policy)
        V = keys[pidx]
        V = V[np.isfinite(V).all(axis=1)]
        if V.size == 0:
            continue
        protos.append(normalize_key(np.nanmean(V, axis=0)))
        pids.append(int(p))
    if not protos:
        return np.empty(0, dtype=int), np.empty((0, keys.shape[1]), dtype=np.float64)
    return np.asarray(pids, dtype=int), np.vstack(protos)


def conditioned_local_tube_scores(
    seqs: Sequence[np.ndarray],
    problem_ids: np.ndarray,
    y_err: np.ndarray,
    mask: np.ndarray,
    keys: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    n = len(seqs)
    scores = {name: np.full(n, np.nan, dtype=np.float64) for name in ALL_TUBE_SCORE_NAMES}
    positions = {"off_z_max": np.full(n, np.nan, dtype=np.float64)}
    idx_all = np.where(mask & np.isfinite(keys).all(axis=1))[0]
    cache: Dict[Tuple[int, ...], Optional[TransitionTube]] = {}
    local_sizes: List[int] = []
    ranks: List[int] = []
    skipped = defaultdict(int)
    folds = group_folds(problem_ids[idx_all], args.folds, args.seed)
    for tr_rel, te_rel in folds:
        tr_idx = idx_all[tr_rel]
        te_idx = idx_all[te_rel]
        proto_pids, protos = problem_prototypes(
            keys,
            problem_ids,
            tr_idx,
            y_err,
            proto_policy=args.retrieval_proto,
        )
        if proto_pids.size == 0:
            skipped["no_prototypes"] += int(te_idx.size)
            continue
        for i in te_idx:
            q = keys[int(i)]
            if not np.isfinite(q).all():
                skipped["bad_query_key"] += 1
                continue
            sims = protos @ q
            order = np.argsort(-sims)
            selected: List[int] = []
            tube: Optional[TransitionTube] = None
            for oi in order[: min(len(order), args.local_k_max)]:
                selected.append(int(proto_pids[int(oi)]))
                if len(selected) < args.local_k:
                    continue
                key = tuple(sorted(selected))
                if key in cache:
                    tube = cache[key]
                else:
                    refs = tr_idx[(y_err[tr_idx] == 0) & np.isin(problem_ids[tr_idx], np.asarray(selected, dtype=int))]
                    tube = fit_refs(seqs, refs, args, min_transitions=args.min_transitions)
                    cache[key] = tube
                if tube is not None:
                    break
            if tube is None:
                skipped["fit_failed"] += 1
                continue
            local_sizes.append(len(selected))
            ranks.append(tube.rank)
            s = score_sequence_refined(seqs[int(i)], tube, rank_energy=args.rank_energy)
            for name in ALL_TUBE_SCORE_NAMES:
                if name in s:
                    scores[name][i] = s[name]
            if "off_z_max_pos" in s:
                positions["off_z_max"][i] = s["off_z_max_pos"]
    meta = {
        "mean_local_problem_count": float(np.mean(local_sizes)) if local_sizes else None,
        "mean_rank": float(np.mean(ranks)) if ranks else None,
        "cached_tubes": int(len(cache)),
        "skipped": dict(skipped),
        "retrieval_proto": args.retrieval_proto,
    }
    return scores, positions, meta


def layer_synergy_scores(
    raw: Sequence[np.ndarray],
    mask: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    n = len(raw)
    scores = {
        "desync_mean": np.full(n, np.nan, dtype=np.float64),
        "desync_late": np.full(n, np.nan, dtype=np.float64),
        "desync_max": np.full(n, np.nan, dtype=np.float64),
        "layer_eff_rank_mean": np.full(n, np.nan, dtype=np.float64),
        "layer_eff_rank_late": np.full(n, np.nan, dtype=np.float64),
        "layer_eff_rank_max": np.full(n, np.nan, dtype=np.float64),
    }
    positions = {"desync_max": np.full(n, np.nan, dtype=np.float64)}
    skipped = defaultdict(int)
    layer_counts: List[int] = []
    for i, V in enumerate(raw):
        if not mask[i]:
            continue
        if V.ndim != 3 or V.shape[0] < 2 or V.shape[1] < 2:
            skipped["not_multilayer"] += 1
            continue
        D = V[1:] - V[:-1]
        norms = np.linalg.norm(D, axis=2, keepdims=True)
        U = D / np.maximum(norms, EPS)
        adj = np.sum(U[:, :-1, :] * U[:, 1:, :], axis=2)
        desync = 1.0 - np.nanmean(adj, axis=1)
        m_late = window_mask(desync.size, "late")
        scores["desync_mean"][i] = safe_mean(desync)
        scores["desync_late"][i] = safe_mean(desync[m_late])
        if np.isfinite(desync).any():
            k = int(np.nanargmax(desync))
            scores["desync_max"][i] = float(np.nanmax(desync))
            positions["desync_max"][i] = float((k + 1) / max(1, V.shape[0] - 1))

        effs: List[float] = []
        for t in range(U.shape[0]):
            G = U[t] @ U[t].T
            try:
                e = np.linalg.eigvalsh(G)
            except np.linalg.LinAlgError:
                continue
            e = np.maximum(e, 0.0)
            total = float(e.sum())
            if total <= EPS:
                effs.append(0.0)
            else:
                p = e[e > EPS] / total
                effs.append(float(np.exp(-np.sum(p * np.log(p)))))
        ea = np.asarray(effs, dtype=np.float64)
        if ea.size:
            ml = window_mask(ea.size, "late")
            scores["layer_eff_rank_mean"][i] = safe_mean(ea)
            scores["layer_eff_rank_late"][i] = safe_mean(ea[ml])
            scores["layer_eff_rank_max"][i] = float(np.nanmax(ea))
        layer_counts.append(int(V.shape[1]))
    meta = {
        "mean_layer_count": float(np.mean(layer_counts)) if layer_counts else None,
        "skipped": dict(skipped),
        "interpretation": "risk-high desync/eff-rank means weaker adjacent-layer transition coordination",
    }
    return scores, positions, meta


def run_policy(
    data: np.lib.npyio.NpzFile,
    *,
    policy: str,
    seqs: Sequence[np.ndarray],
    raw_layers: Optional[Sequence[np.ndarray]],
    keys: np.ndarray,
    key_meta: Mapping[str, Any],
    key_used: str,
    problem_ids: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    y_err, mask, desc = label_policy(data, policy)
    groups = same_problem_groups(problem_ids, y_err, mask, args.min_per_class)
    rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {"label_policy": desc}

    gs, gp, gm = global_tail_crossfit_scores(seqs, problem_ids, y_err, mask, args)
    diagnostics["global_tail_crossfit"] = gm
    rows.extend(collect_score_rows("global_tail", gs, gp, y_err=y_err, mask=mask, groups=groups))

    hs, hp, hm = support_holdout_oracle_scores(seqs, problem_ids, y_err, mask, args)
    diagnostics["support_holdout_oracle"] = hm
    rows.extend(collect_score_rows("support_oracle", hs, hp, y_err=y_err, mask=mask, groups=groups))

    cs, cp, cm = conditioned_local_tube_scores(seqs, problem_ids, y_err, mask, keys, args)
    diagnostics["prompt_conditioned_tube"] = {**cm, "key_used": key_used, "key_meta": finite_json(key_meta)}
    rows.extend(collect_score_rows("conditioned", cs, cp, y_err=y_err, mask=mask, groups=groups))

    if args.include_layer_sync and raw_layers is not None:
        ls, lp, lm = layer_synergy_scores(raw_layers, mask)
        diagnostics["layer_synergy"] = lm
        rows.extend(collect_score_rows("layer", ls, lp, y_err=y_err, mask=mask, groups=groups))

    rows.sort(
        key=lambda r: (
            np.nan_to_num(r["best_direction_within"], nan=-1.0),
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
        "diagnostics": diagnostics,
        "results": rows,
    }


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    problem_ids = data["problem_ids"].astype(int)
    seqs = step_sequences(data, band=args.band, normalize=args.normalize)
    raw_layers = raw_layer_sequences(data, normalize=args.normalize) if args.include_layer_sync else None
    keys, key_used, key_meta = extract_prompt_keys(data, seqs, mode=args.retrieval_key)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "prompt_style": str(data["prompt_style"]) if "prompt_style" in data.files else "unknown",
            "step_split": str(data["step_split"]) if "step_split" in data.files else "unknown",
            "model": str(data["model_name"]) if "model_name" in data.files else "unknown",
            "band": args.band,
            "normalize": args.normalize,
            "rank": int(args.rank),
            "max_rank": int(args.max_rank),
            "energy": float(args.energy),
            "rank_energy": float(args.rank_energy),
            "vector_payload": vector_payload_meta(data),
            "retrieval_key_used": key_used,
            "retrieval_key_meta": finite_json(key_meta),
            "notes": {
                "support_holdout_oracle": "same-problem correct support tube; heldout correct and errors scored against same support folds",
                "tube_tail_curve": "tail_auc/k90/residual-at-k measure how many correct-tube directions are needed",
                "prompt_conditioned_tube": "local tube from nearest training problems under qvec/prompt key, or first-step proxy if qvec is absent",
                "layer_synergy": "available only when sv_vec_step_exp has multiple layers",
            },
        },
        "policies": {
            pol: run_policy(
                data,
                policy=pol,
                seqs=seqs,
                raw_layers=raw_layers,
                keys=keys,
                key_meta=key_meta,
                key_used=key_used,
                problem_ids=problem_ids,
                args=args,
            )
            for pol in policies
        },
    }


def write_outputs(res: Mapping[str, Any], output_dir: str, top: int) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = (
        f"multisample_tube_refinement_"
        f"{os.path.splitext(str(res['meta']['basename']))[0]}_"
        f"{res['meta']['band']}_{res['meta']['normalize']}"
    )
    jp = os.path.join(output_dir, stem + ".json")
    mp = os.path.join(output_dir, stem + ".md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, indent=2, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(f"# Multisample Tube Refinement Audit: {res['meta']['basename']}\n\n")
        f.write("## Result Analysis\n\n")
        f.write("- `support_oracle.*` is the stricter same-problem diagnostic: correct heldouts and errors use the same support-correct tube folds.\n")
        f.write("- `global_tail.*` is the deployable-style cross-problem tube with spectral-tail features added.\n")
        f.write("- `conditioned.*` is a local tube from nearest training problems under the reported retrieval key.\n")
        f.write("- `layer.*` tests adjacent-layer transition coordination when multiple step-vector layers are stored.\n")
        f.write("- Same-problem paired AUROC is the headline metric; cross-problem AUROC is context.\n\n")
        f.write("Metadata:\n\n")
        f.write(f"```json\n{json.dumps(finite_json(res['meta']), indent=2, ensure_ascii=False)}\n```\n\n")
        for pol, sec in res["policies"].items():
            f.write(f"### {pol}\n\n")
            f.write(
                f"{sec['n_error']} error / {sec['n_correct']} correct samples; "
                f"{sec['n_contrastive_problems']} contrastive problems.\n\n"
            )
            f.write(f"Diagnostics: `{json.dumps(finite_json(sec['diagnostics']), ensure_ascii=False)}`\n\n")
            f.write("| score | within | cross | err med | cor med | delta med | err argpos | cor argpos |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in sec["results"][:top]:
                dlt = r["paired_delta_error_minus_correct"]
                ae = r.get("argpos_error", {})
                ac = r.get("argpos_correct", {})
                f.write(
                    f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
                    f"{r['cross_auroc_error_high']:.3f} | "
                    f"{r['error'].get('median', float('nan')):.3f} | "
                    f"{r['correct'].get('median', float('nan')):.3f} | "
                    f"{dlt.get('median', float('nan')):.3f} | "
                    f"{ae.get('median', float('nan')):.3f} | {ac.get('median', float('nan')):.3f} |\n"
                )
            f.write("\n")
        f.write("## Follow-Up Research Direction\n\n")
        f.write("- If `support_oracle.*` remains much stronger than `global_tail.*`, the correct transition manifold is problem-conditioned.\n")
        f.write("- If `conditioned.*` approaches `support_oracle.*`, local prompt/prefix conditioning is the right path toward a deployable tube monitor.\n")
        f.write("- If tail features beat off-residual features, write the mechanism as rank/tail inflation rather than plain distance from a tube.\n")
        f.write("- If `layer.*` scores work, add a layer-coordination term to the monitor instead of only averaging layers into one vector.\n\n")
        f.write("## Optimization Suggestions\n\n")
        f.write("- Prefer `answer_format_ok` for headline same-problem tests to avoid format failures dominating geometry.\n")
        f.write("- Inspect argpos for max-style features before claiming online detection.\n")
        f.write("- Re-extract qvec/prompt anchors for multisample data; first-step proxy is only a fallback, not the final paper setting.\n")
    return jp, mp


def print_report(res: Mapping[str, Any], top: int) -> None:
    meta = res["meta"]
    print(f"\n===== multisample tube refinement | {meta['basename']} | {meta['band']} {meta['normalize']} =====")
    print(f"prompt={meta['prompt_style']} step_split={meta['step_split']} model={meta['model']}")
    vp = meta.get("vector_payload", {})
    print(
        "vectors "
        f"layers={vp.get('n_layers_min')}..{vp.get('n_layers_max')} "
        f"hidden={vp.get('hidden_dim_min')}..{vp.get('hidden_dim_max')} "
        f"retrieval_key={meta['retrieval_key_used']}"
    )
    for pol, sec in res["policies"].items():
        print(f"\n[{pol}] err={sec['n_error']} cor={sec['n_correct']} contrastive={sec['n_contrastive_problems']}")
        print(f"diagnostics {finite_json(sec['diagnostics'])}")
        for r in sec["results"][:top]:
            dlt = r["paired_delta_error_minus_correct"]
            ae = r.get("argpos_error", {})
            ac = r.get("argpos_correct", {})
            print(
                f"  {r['score']:36s} within {r['within_pair_auroc_error_high']:.3f} "
                f"cross {r['cross_auroc_error_high']:.3f} "
                f"err_med {r['error'].get('median', float('nan')):.3f} "
                f"cor_med {r['correct'].get('median', float('nan')):.3f} "
                f"delta {dlt.get('median', float('nan')):+.3f} "
                f"argpos_e {ae.get('median', float('nan')):.2f} "
                f"argpos_c {ac.get('median', float('nan')):.2f}"
            )


def assert_selftest(res: Mapping[str, Any]) -> None:
    rows = res["policies"]["answer_format_ok"]["results"]
    support = [
        float(r["within_pair_auroc_error_high"])
        for r in rows
        if r["score"].startswith("support_oracle.") and np.isfinite(r["within_pair_auroc_error_high"])
    ]
    if not support or max(support) < 0.70:
        raise SystemExit(f"selftest failed: support-oracle signal too weak ({max(support) if support else float('nan'):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output_dir", default="outputs/multisample_tube_refinement")
    ap.add_argument("--policies", default="answer_format_ok")
    ap.add_argument("--band", default="mid")
    ap.add_argument("--normalize", default="l2", choices=["none", "l2", "center_chain"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--energy", type=float, default=0.90)
    ap.add_argument("--rank_energy", type=float, default=0.90)
    ap.add_argument("--max_rank", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--min_transitions", type=int, default=40)
    ap.add_argument("--oracle_min_correct", type=int, default=3)
    ap.add_argument("--oracle_min_support", type=int, default=2)
    ap.add_argument("--oracle_min_transitions", type=int, default=8)
    ap.add_argument("--oracle_holdout_folds", type=int, default=5)
    ap.add_argument("--retrieval_key", default="auto", choices=["auto", "qvec", "first_step", "early_mean"])
    ap.add_argument("--retrieval_proto", default="all", choices=["all", "correct"])
    ap.add_argument("--local_k", type=int, default=12)
    ap.add_argument("--local_k_max", type=int, default=48)
    ap.add_argument("--include_layer_sync", action="store_true", default=True)
    ap.add_argument("--no_layer_sync", action="store_false", dest="include_layer_sync")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tube_refinement_selftest.npz")
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
