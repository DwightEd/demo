#!/usr/bin/env python3
"""Constraint-anchor flow audit for stepwise reasoning.

This script tests a more specific version of the current project hypothesis:

    A reasoning step is not only "geometrically coherent" or "fragmented".
    It is also anchored to a source.  Errors can occur when the step remains
    internally coherent but its hidden computation shifts away from the problem
    constraints and toward the recent/self-generated prefix.

The audit therefore builds a per-step anchor posterior over typed sources:

    question, earlier_prefix, recent_prev, other

The posterior is computed from hidden-state subspace projection density, not a
single cosine similarity.  Surface text provides a second posterior from number
support.  Their KL mismatch and the step-to-step posterior transport are then
evaluated as first-error detectors under same-problem paired ranking.

This is intentionally a falsification script.  If these anchor-flow quantities
do not beat spread/kappa controls, the "lost anchor" story is not yet supported
by the current representation data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multisample_temporal_rupture_audit import descriptive, finite_json, within_pair_auroc
from premise_constraint_audit import bootstrap_within_increment, pair_rescue_report
from second_moment_dynamics_audit import oof_scores
from token_stream_geometry_audit import chain_lengths, load_token_matrix, source_info


EPS = 1e-12
ANCHORS = ("question", "earlier_prefix", "recent_prev", "other")
NUMBER_RE = re.compile(r"(?<![\w/])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?")


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class AnchorStepRow:
    chain_idx: int
    problem_id: int
    step_idx: int
    gold_error_step: int
    phase: str
    y_first_error: int
    text: str
    features: Dict[str, float]


def scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(x)


def auroc(score: Iterable[float], y: Iterable[int]) -> float:
    s = np.asarray(score, dtype=np.float64)
    yy = np.asarray(y, dtype=int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    p = int((yy == 1).sum())
    n = int((yy == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[yy == 1].sum() - p * (p + 1) / 2.0) / (p * n))


def bdir(x: float) -> float:
    return float(max(x, 1.0 - x)) if np.isfinite(x) else float("nan")


def safe_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def safe_std(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if a.size > 1 else (0.0 if a.size else float("nan"))


def close_num(a: float, b: float, *, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b), 1.0)


def parse_number(raw: str) -> Optional[float]:
    s = raw.strip().replace(",", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            val = float(a) / float(b)
        else:
            val = float(s)
    except Exception:
        return None
    if pct:
        val /= 100.0
    return float(val) if math.isfinite(val) else None


def extract_numbers(text: str) -> List[float]:
    vals: List[float] = []
    for m in NUMBER_RE.finditer(str(text)):
        prefix = str(text)[max(0, m.start() - 12) : m.start()].lower()
        if re.search(r"(step|part|case|line|#)\s*$", prefix):
            continue
        v = parse_number(m.group(0))
        if v is not None:
            vals.append(v)
    return vals


def number_in(value: float, bank: Sequence[float]) -> bool:
    return any(close_num(value, b) for b in bank)


def normalize_prob(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=np.float64)
    x = np.where(np.isfinite(x) & (x > 0), x, 0.0)
    s = float(x.sum())
    if s <= EPS:
        return np.ones_like(x, dtype=np.float64) / max(1, x.size)
    return x / s


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    pp = normalize_prob(np.asarray(p, dtype=np.float64) + EPS)
    qq = normalize_prob(np.asarray(q, dtype=np.float64) + EPS)
    return float(np.sum(pp * (np.log(pp + EPS) - np.log(qq + EPS))))


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    pp = normalize_prob(p)
    qq = normalize_prob(q)
    m = 0.5 * (pp + qq)
    return float(0.5 * kl_div(pp, m) + 0.5 * kl_div(qq, m))


def entropy(p: np.ndarray) -> float:
    pp = normalize_prob(p)
    return float(-np.sum(pp * np.log(pp + EPS)))


def exp_weights(n: int, beta: float) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    if n == 1 or abs(beta) <= EPS:
        return np.ones(n, dtype=np.float64) / n
    pos = np.linspace(0.0, 1.0, n)
    z = float(beta) * pos
    z -= float(z.max())
    w = np.exp(z)
    return w / max(float(w.sum()), EPS)


def unit_rows(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    if X.size == 0:
        return np.empty((0, H.shape[1] if np.asarray(H).ndim == 2 else 0), dtype=np.float64), np.empty(0)
    norms = np.linalg.norm(X, axis=1)
    ok = norms > EPS
    X = X[ok]
    norms = norms[ok]
    if X.shape[0] == 0:
        return np.empty((0, X.shape[1]), dtype=np.float64), np.empty(0, dtype=np.float64)
    return X / np.maximum(norms[:, None], EPS), norms


def preprocess_rows(H: np.ndarray, *, center: Optional[np.ndarray], unitize: bool) -> np.ndarray:
    X = np.asarray(H, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    if X.shape[0] == 0:
        return np.empty((0, H.shape[1] if np.asarray(H).ndim == 2 else 0), dtype=np.float64)
    if center is not None and np.asarray(center).shape[-1] == X.shape[1]:
        X = X - np.asarray(center, dtype=np.float64)[None, :]
    if unitize:
        norms = np.linalg.norm(X, axis=1)
        ok = norms > EPS
        X = X[ok] / np.maximum(norms[ok, None], EPS)
    return X


def basis_from_rows(A: np.ndarray, *, rank: int) -> np.ndarray:
    X = np.asarray(A, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if X.shape[0] == 1:
        v = X[0]
        n = float(np.linalg.norm(v))
        return (v / n)[:, None] if n > EPS else np.empty((X.shape[1], 0), dtype=np.float64)
    X = X - np.mean(X, axis=0, keepdims=True) if X.shape[0] > 2 else X
    try:
        _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.empty((X.shape[1], 0), dtype=np.float64)
    k = int(min(rank, vt.shape[0]))
    if k <= 0:
        return np.empty((X.shape[1], 0), dtype=np.float64)
    V = vt[:k].T
    n = np.linalg.norm(V, axis=0)
    ok = n > EPS
    return V[:, ok] / np.maximum(n[ok][None, :], EPS)


def projection_density(X: np.ndarray, V: np.ndarray) -> Tuple[float, float, int]:
    """Return projection fraction, rank-normalized density, and basis rank."""
    if X.ndim != 2 or V.ndim != 2 or X.shape[0] == 0 or V.shape[1] == 0:
        return float("nan"), float("nan"), 0
    if X.shape[1] != V.shape[0]:
        return float("nan"), float("nan"), 0
    denom = float(np.sum(X * X))
    if denom <= EPS:
        return float("nan"), float("nan"), int(V.shape[1])
    proj = X @ V
    frac = float(np.sum(proj * proj) / denom)
    density = frac / max(1, int(V.shape[1]))
    return frac, density, int(V.shape[1])


def anchor_posterior(
    step_H: np.ndarray,
    anchors: Mapping[str, Optional[np.ndarray]],
    *,
    chain_center: Optional[np.ndarray],
    unitize: bool,
    rank: int,
    temp: float,
    other_score: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    X = preprocess_rows(step_H, center=chain_center, unitize=unitize)
    raw_scores: List[float] = []
    feats: Dict[str, float] = {}
    for name in ANCHORS:
        if name == "other":
            frac = float("nan")
            density = float(other_score)
            brank = 1
        else:
            A0 = anchors.get(name)
            if A0 is None:
                frac = float("nan")
                density = float("nan")
                brank = 0
            else:
                A = preprocess_rows(A0, center=chain_center, unitize=unitize)
                V = basis_from_rows(A, rank=rank)
                frac, density, brank = projection_density(X, V)
        feats[f"anchor_{name}_proj"] = float(frac)
        feats[f"anchor_{name}_density"] = float(density)
        feats[f"anchor_{name}_rank"] = float(brank)
        raw_scores.append(float(density) if np.isfinite(density) else -np.inf)
    scores = np.asarray(raw_scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        p = np.ones(len(ANCHORS), dtype=np.float64) / len(ANCHORS)
    else:
        z = scores.copy()
        floor = float(np.nanmin(scores[finite])) - 5.0
        z[~finite] = floor
        z = z / max(float(temp), EPS)
        z -= float(np.max(z))
        p = np.exp(z)
        p = normalize_prob(p)
    for name, val in zip(ANCHORS, p):
        feats[f"p_hidden_{name}"] = float(val)
    feats["p_hidden_valid"] = float(p[0] + p[1] + p[2])
    feats["p_hidden_prefix"] = float(p[1] + p[2])
    feats["hidden_anchor_entropy"] = entropy(p)
    feats["hidden_anchor_eff"] = float(math.exp(feats["hidden_anchor_entropy"]))
    feats["hidden_self_hijack"] = float((p[2] + p[3]) - (p[0] + p[1]))
    feats["hidden_recent_over_question"] = float(p[2] - p[0])
    feats["hidden_prefix_over_question"] = float((p[1] + p[2]) - p[0])
    return p, feats


def text_anchor_posterior(
    step_text: str,
    *,
    question_numbers: Sequence[float],
    earlier_numbers: Sequence[float],
    recent_numbers: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    nums = extract_numbers(step_text)
    counts = np.ones(len(ANCHORS), dtype=np.float64) * 0.25
    unsupported = 0
    for v in nums:
        if number_in(v, question_numbers):
            counts[0] += 1.0
        elif number_in(v, recent_numbers):
            counts[2] += 1.0
        elif number_in(v, earlier_numbers):
            counts[1] += 1.0
        else:
            counts[3] += 1.0
            unsupported += 1
    p = normalize_prob(counts)
    feats = {f"p_text_{name}": float(val) for name, val in zip(ANCHORS, p)}
    feats["text_anchor_entropy"] = entropy(p)
    feats["text_n_numbers"] = float(len(nums))
    feats["text_unsupported_numbers"] = float(unsupported)
    feats["text_unsupported_frac"] = float(unsupported / max(1, len(nums)))
    return p, feats


def slice_steps(H: np.ndarray, lengths: np.ndarray) -> List[np.ndarray]:
    steps: List[np.ndarray] = []
    cur = 0
    n = int(H.shape[0])
    for ln in np.asarray(lengths, dtype=int):
        hi = min(n, cur + max(0, int(ln)))
        if hi > cur:
            steps.append(np.asarray(H[cur:hi], dtype=np.float64))
        else:
            steps.append(np.empty((0, H.shape[1]), dtype=np.float64))
        cur = hi
    return steps


def get_steps_text(data: np.lib.npyio.NpzFile, idx: int, T: int) -> List[str]:
    if "steps_text" not in data.files:
        return ["" for _ in range(T)]
    try:
        arr = data["steps_text"][idx]
    except Exception:
        return ["" for _ in range(T)]
    out = [str(x) for x in list(arr)[:T]]
    if len(out) < T:
        out.extend([""] * (T - len(out)))
    return out


def get_question_text(data: np.lib.npyio.NpzFile, idx: int) -> str:
    for key in ("questions", "question", "prompts", "prompt"):
        if key in data.files:
            try:
                return scalar_str(data[key][idx])
            except Exception:
                return scalar_str(data[key])
    return ""


def select_q_anchor(data: np.lib.npyio.NpzFile, idx: int, layer_used: int, dim: int) -> Optional[np.ndarray]:
    if "qvec" not in data.files:
        return None
    q = data["qvec"]
    try:
        qi = np.asarray(q[idx], dtype=np.float64)
    except Exception:
        qi = np.asarray(q, dtype=np.float64)
    if qi.ndim == 1 and qi.size == dim:
        return qi[None, :]
    if qi.ndim == 2 and qi.shape[1] == dim:
        layers = [int(x) for x in data["sv_layers"]] if "sv_layers" in data.files else []
        if layer_used in layers:
            return qi[layers.index(layer_used)][None, :]
        return qi[min(qi.shape[0] - 1, qi.shape[0] // 2)][None, :]
    return None


def step_entropy_trace(data: np.lib.npyio.NpzFile, idx: int, lengths: np.ndarray) -> np.ndarray:
    T = len(lengths)
    out = np.full(T, np.nan, dtype=np.float64)
    if "sv_out_entropy" in data.files:
        try:
            v = np.asarray(data["sv_out_entropy"][idx], dtype=np.float64).reshape(-1)
            out[: min(T, v.size)] = v[:T]
            return out
        except Exception:
            pass
    if "tok_U_D" in data.files:
        try:
            tok = np.asarray(data["tok_U_D"][idx], dtype=np.float64).reshape(-1)
            cur = 0
            for t, ln in enumerate(lengths):
                hi = min(tok.size, cur + int(ln))
                if hi > cur:
                    out[t] = float(np.nanmean(tok[cur:hi]))
                cur = hi
        except Exception:
            pass
    return out


def phase_for(gold: int, t: int) -> str:
    if gold < 0:
        return "correct_chain"
    if t < gold:
        return "pre_error"
    if t == gold:
        return "first_error"
    return "post_error"


def y_for_phase(phase: str, control_pool: str) -> int:
    if phase == "first_error":
        return 1
    if phase == "post_error":
        return -1
    if control_pool == "correct_chain" and phase != "correct_chain":
        return -1
    if control_pool == "pre_error" and phase != "pre_error":
        return -1
    return 0


def build_rows(path: str, args: argparse.Namespace) -> Tuple[List[AnchorStepRow], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "gold_error_step" not in data.files:
        raise SystemExit("constraint_anchor_flow_audit requires gold_error_step for step-level first-error evaluation")
    source, layer_i, layer_used = source_info(data, path, args)
    gold = data["gold_error_step"].astype(int)
    problem_ids = data["problem_ids"].astype(int) if "problem_ids" in data.files else np.arange(len(gold))
    N = len(gold) if args.max_chains <= 0 else min(len(gold), int(args.max_chains))
    rows: List[AnchorStepRow] = []
    skipped = {"missing_hidden": 0, "too_few_steps": 0, "too_few_tokens": 0}

    iterator = range(N)
    if not args.no_progress:
        iterator = tqdm(iterator, desc="anchor-flow rows", unit="chain")
    for idx in iterator:
        H = load_token_matrix(data, path, args, idx=idx, source=source, layer_i=layer_i)
        if H is None or np.asarray(H).ndim != 2 or np.asarray(H).shape[0] == 0:
            skipped["missing_hidden"] += 1
            continue
        H = np.asarray(H, dtype=np.float64)
        lengths, _ranges = chain_lengths(data, idx, H.shape[0], source)
        if lengths.size < 2:
            skipped["too_few_steps"] += 1
            continue
        step_mats = slice_steps(H, lengths)
        T = len(step_mats)
        if T < 2:
            skipped["too_few_steps"] += 1
            continue
        if sum(m.shape[0] for m in step_mats) < args.min_tokens:
            skipped["too_few_tokens"] += 1
            continue

        chain_center = np.mean(H, axis=0) if args.center_chain else None
        q_anchor = select_q_anchor(data, idx, layer_used, H.shape[1])
        texts = get_steps_text(data, idx, T)
        question_text = get_question_text(data, idx)
        question_nums = extract_numbers(question_text)
        entropy_trace = step_entropy_trace(data, idx, lengths)

        hidden_posts: List[np.ndarray] = []
        step_nums: List[List[float]] = []
        for t in range(T):
            step_nums.append(extract_numbers(texts[t]))
            prev = step_mats[t - 1] if t > 0 else None
            earlier = np.concatenate(step_mats[: max(0, t - 1)], axis=0) if t > 1 else None
            anchors: Dict[str, Optional[np.ndarray]] = {
                "question": q_anchor,
                "earlier_prefix": earlier,
                "recent_prev": prev,
                "other": None,
            }
            p_hidden, feats = anchor_posterior(
                step_mats[t],
                anchors,
                chain_center=chain_center,
                unitize=not args.raw_projection,
                rank=args.anchor_rank,
                temp=args.posterior_temp,
                other_score=args.other_score,
            )
            hidden_posts.append(p_hidden)

            U, norms = unit_rows(step_mats[t])
            if U.shape[0] >= 1:
                w = exp_weights(U.shape[0], args.kappa_beta)
                mu = w @ U
                kappa = float(np.linalg.norm(mu))
                feats["kappa"] = kappa
                feats["spread"] = float(1.0 - kappa)
                feats["tok_norm_mean"] = float(np.mean(norms)) if norms.size else float("nan")
            else:
                feats["kappa"] = float("nan")
                feats["spread"] = float("nan")
                feats["tok_norm_mean"] = float("nan")
            feats["n_tok"] = float(step_mats[t].shape[0])
            feats["logN"] = float(math.log1p(max(0, step_mats[t].shape[0])))
            feats["pos"] = float(t / max(1, T - 1))
            feats["entropy"] = float(entropy_trace[t]) if t < entropy_trace.size else float("nan")

            earlier_nums = [x for xs in step_nums[: max(0, t - 1)] for x in xs]
            recent_nums = step_nums[t - 1] if t > 0 else []
            p_text, text_feats = text_anchor_posterior(
                texts[t],
                question_numbers=question_nums,
                earlier_numbers=earlier_nums,
                recent_numbers=recent_nums,
            )
            feats.update(text_feats)
            feats["text_hidden_kl"] = kl_div(p_text, p_hidden)
            feats["text_hidden_js"] = js_div(p_text, p_hidden)
            feats["hidden_text_kl"] = kl_div(p_hidden, p_text)
            feats["text_q_minus_hidden_q"] = float(p_text[0] - p_hidden[0])
            feats["text_valid_minus_hidden_valid"] = float((p_text[0] + p_text[1] + p_text[2]) - (p_hidden[0] + p_hidden[1] + p_hidden[2]))

            if t > 0:
                prev_p = hidden_posts[t - 1]
                feats["anchor_jump_l1"] = float(np.sum(np.abs(p_hidden - prev_p)))
                feats["anchor_transition_js"] = js_div(prev_p, p_hidden)
                feats["anchor_transition_kl"] = kl_div(prev_p, p_hidden)
                feats["delta_p_question"] = float(p_hidden[0] - prev_p[0])
                feats["drop_p_question"] = float(prev_p[0] - p_hidden[0])
                feats["rise_p_recent"] = float(p_hidden[2] - prev_p[2])
            else:
                feats["anchor_jump_l1"] = float("nan")
                feats["anchor_transition_js"] = float("nan")
                feats["anchor_transition_kl"] = float("nan")
                feats["delta_p_question"] = float("nan")
                feats["drop_p_question"] = float("nan")
                feats["rise_p_recent"] = float("nan")

            feats["risk_low_question"] = float(-p_hidden[0])
            feats["risk_low_valid"] = float(-(p_hidden[0] + p_hidden[1] + p_hidden[2]))
            feats["risk_recent_over_question"] = float(p_hidden[2] - p_hidden[0])
            feats["risk_other_over_question"] = float(p_hidden[3] - p_hidden[0])
            feats["risk_self_hijack"] = feats["hidden_self_hijack"]
            feats["risk_text_hidden_kl"] = feats["text_hidden_kl"]
            feats["risk_transition"] = feats["anchor_transition_js"]
            feats["risk_drop_question"] = feats["drop_p_question"]
            feats["risk_anchor_fragment"] = float(feats["spread"] * feats["hidden_anchor_entropy"]) if np.isfinite(feats["spread"]) else float("nan")
            feats["risk_coherent_hijack"] = float(feats["kappa"] * max(0.0, feats["hidden_self_hijack"])) if np.isfinite(feats["kappa"]) else float("nan")
            if np.isfinite(feats["entropy"]):
                feats["risk_confident_hijack"] = float(max(0.0, feats["hidden_self_hijack"]) * (1.0 / (1.0 + max(0.0, feats["entropy"]))))
            else:
                feats["risk_confident_hijack"] = float("nan")

            phase = phase_for(int(gold[idx]), t)
            rows.append(
                AnchorStepRow(
                    chain_idx=int(idx),
                    problem_id=int(problem_ids[idx]),
                    step_idx=int(t),
                    gold_error_step=int(gold[idx]),
                    phase=phase,
                    y_first_error=y_for_phase(phase, args.control_pool),
                    text=texts[t],
                    features=feats,
                )
            )

    meta = {
        "npz": path,
        "source": source,
        "layer": int(layer_used),
        "layer_index": int(layer_i),
        "n_chains_seen": int(N),
        "n_rows": int(len(rows)),
        "skipped": skipped,
        "anchor_rank": int(args.anchor_rank),
        "posterior_temp": float(args.posterior_temp),
        "other_score": float(args.other_score),
        "center_chain": bool(args.center_chain),
        "projection": "raw_rows" if args.raw_projection else "unit_rows",
        "control_pool": args.control_pool,
        "has_qvec": bool("qvec" in data.files),
        "has_questions": bool(any(k in data.files for k in ("questions", "question", "prompts", "prompt"))),
    }
    return rows, meta


def feature_array(rows: Sequence[AnchorStepRow], name: str) -> np.ndarray:
    return np.asarray([r.features.get(name, float("nan")) for r in rows], dtype=np.float64)


def row_arrays(rows: Sequence[AnchorStepRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray([r.y_first_error for r in rows], dtype=int)
    problem_ids = np.asarray([r.problem_id for r in rows], dtype=int)
    chain_ids = np.asarray([r.chain_idx for r in rows], dtype=int)
    phases = np.asarray([r.phase for r in rows], dtype=object)
    return y, problem_ids, chain_ids, phases


def problem_row_groups(problem_ids: np.ndarray, y: np.ndarray, mask: np.ndarray, *, min_per_class: int) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for p in np.unique(problem_ids[mask]):
        idx = np.where(mask & (problem_ids == p))[0]
        if np.sum(y[idx] == 1) >= min_per_class and np.sum(y[idx] == 0) >= min_per_class:
            groups.append(idx)
    return groups


def position_matched_within_pair_auroc(
    rows: Sequence[AnchorStepRow],
    vals: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    step_gap: int,
) -> Tuple[float, int]:
    """Same-problem paired AUROC with exact/near step-position matching.

    Ordinary first-error localization can be dominated by the fact that errors
    occur late.  This stricter metric only compares a first-error row with
    control rows from the same problem whose step index is within `step_gap`.
    """
    v = np.asarray(vals, dtype=np.float64)
    conc = 0.0
    pairs = 0
    problems = sorted({r.problem_id for i, r in enumerate(rows) if mask[i]})
    for p in problems:
        idx = [i for i, r in enumerate(rows) if mask[i] and r.problem_id == p and np.isfinite(v[i])]
        err = [i for i in idx if y[i] == 1]
        ctrl = [i for i in idx if y[i] == 0]
        for e in err:
            matched = [c for c in ctrl if abs(rows[c].step_idx - rows[e].step_idx) <= int(step_gap)]
            for c in matched:
                conc += 1.0 if v[e] > v[c] else (0.5 if v[e] == v[c] else 0.0)
            pairs += len(matched)
    return (conc / pairs if pairs else float("nan")), int(pairs)


def eval_score(
    name: str,
    score: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    mask: np.ndarray,
    *,
    rows: Optional[Sequence[AnchorStepRow]] = None,
    pos_match_step_gap: int = 0,
) -> Dict[str, Any]:
    s = np.asarray(score, dtype=np.float64)
    mm = mask & np.isfinite(s)
    raw_auc = auroc(s[mm], y[mm]) if mm.any() else float("nan")
    sign = 1.0 if (not np.isfinite(raw_auc) or raw_auc >= 0.5) else -1.0
    ss = sign * s
    within, pairs = within_pair_auroc(groups, ss, y)
    pm_within, pm_pairs = (
        position_matched_within_pair_auroc(rows, ss, y, mask, step_gap=pos_match_step_gap)
        if rows is not None
        else (float("nan"), 0)
    )
    return {
        "score": name,
        "n": int(mm.sum()),
        "cross_auroc_error_high": float(auroc(ss[mm], y[mm])) if mm.any() else float("nan"),
        "raw_cross_auroc": float(raw_auc),
        "sign": float(sign),
        "within_pair_auroc_error_high": float(within),
        "within_pairs": int(pairs),
        "pos_matched_within_auroc_error_high": float(pm_within),
        "pos_matched_pairs": int(pm_pairs),
        "err_median": float(np.nanmedian(ss[mm & (y == 1)])) if np.any(mm & (y == 1)) else float("nan"),
        "ctrl_median": float(np.nanmedian(ss[mm & (y == 0)])) if np.any(mm & (y == 0)) else float("nan"),
    }


def design(rows: Sequence[AnchorStepRow], names: Sequence[str]) -> np.ndarray:
    X = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for j, name in enumerate(names):
        X[:, j] = feature_array(rows, name)
    return X


def finite_feature_names(rows: Sequence[AnchorStepRow], candidates: Sequence[str], *, min_coverage: float, mask: np.ndarray) -> List[str]:
    out: List[str] = []
    for name in candidates:
        x = feature_array(rows, name)
        cov = float(np.mean(np.isfinite(x[mask]))) if mask.any() else 0.0
        if cov >= min_coverage:
            out.append(name)
    return out


def summarize_coherent_slice(
    rows: Sequence[AnchorStepRow],
    scores: Mapping[str, np.ndarray],
    *,
    y: np.ndarray,
    problem_ids: np.ndarray,
    base_mask: np.ndarray,
    spread_q: float,
    min_per_class: int,
) -> Dict[str, Any]:
    spread = feature_array(rows, "spread")
    if not np.isfinite(spread[base_mask]).any():
        return {"ok": False, "reason": "spread unavailable"}
    thresh = float(np.nanquantile(spread[base_mask], spread_q))
    coh = base_mask & np.isfinite(spread) & (spread <= thresh)
    groups = problem_row_groups(problem_ids, y, coh, min_per_class=min_per_class)
    out = {
        "ok": bool(len(groups) > 0),
        "spread_quantile": float(spread_q),
        "spread_threshold": thresh,
        "n": int(coh.sum()),
        "err": int(np.sum(coh & (y == 1))),
        "ctrl": int(np.sum(coh & (y == 0))),
        "groups": int(len(groups)),
        "scores": [],
    }
    if not groups:
        return out
    for name, vals in scores.items():
        out["scores"].append(eval_score(name, vals, y, groups, coh, rows=rows, pos_match_step_gap=0))
    out["scores"].sort(key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0), reverse=True)
    return out


def trajectory_transition_summary(rows: Sequence[AnchorStepRow], names: Sequence[str]) -> Dict[str, Any]:
    by_chain: Dict[int, List[AnchorStepRow]] = {}
    for r in rows:
        by_chain.setdefault(r.chain_idx, []).append(r)
    out: Dict[str, Any] = {}
    for name in names:
        pre_to_first: List[float] = []
        first_to_post: List[float] = []
        ctrl_delta: List[float] = []
        for _cid, rs in by_chain.items():
            rs = sorted(rs, key=lambda r: r.step_idx)
            gold = rs[0].gold_error_step if rs else -1
            vals = {r.step_idx: r.features.get(name, float("nan")) for r in rs}
            if gold >= 1 and gold in vals and (gold - 1) in vals:
                a = vals[gold - 1]
                b = vals[gold]
                if np.isfinite(a) and np.isfinite(b):
                    pre_to_first.append(float(b - a))
            if gold >= 0 and (gold + 1) in vals and gold in vals:
                a = vals[gold]
                b = vals[gold + 1]
                if np.isfinite(a) and np.isfinite(b):
                    first_to_post.append(float(b - a))
            if gold < 0:
                for r0, r1 in zip(rs[:-1], rs[1:]):
                    a = r0.features.get(name, float("nan"))
                    b = r1.features.get(name, float("nan"))
                    if np.isfinite(a) and np.isfinite(b):
                        ctrl_delta.append(float(b - a))
        out[name] = {
            "pre_to_first_error": descriptive(pre_to_first),
            "first_to_post_error": descriptive(first_to_post),
            "correct_chain_step_delta": descriptive(ctrl_delta),
        }
    return out


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    rows, meta = build_rows(path, args)
    y, problem_ids, chain_ids, phases = row_arrays(rows)
    eval_mask = y >= 0
    groups = problem_row_groups(problem_ids, y, eval_mask, min_per_class=args.min_per_class)

    base_candidates = ["logN", "pos", "spread", "entropy", "tok_norm_mean"]
    anchor_candidates = [
        "risk_low_question",
        "risk_recent_over_question",
        "risk_other_over_question",
        "risk_self_hijack",
        "risk_text_hidden_kl",
        "risk_transition",
        "risk_drop_question",
        "hidden_anchor_entropy",
        "anchor_jump_l1",
        "anchor_transition_js",
        "text_hidden_kl",
        "hidden_recent_over_question",
        "hidden_prefix_over_question",
        "text_q_minus_hidden_q",
        "text_unsupported_frac",
        "risk_anchor_fragment",
        "risk_coherent_hijack",
        "risk_confident_hijack",
    ]
    baseline_names = finite_feature_names(rows, base_candidates, min_coverage=args.min_feature_coverage, mask=eval_mask)
    anchor_names = finite_feature_names(rows, anchor_candidates, min_coverage=args.min_feature_coverage, mask=eval_mask)

    single_rows: List[Dict[str, Any]] = []
    single_scores: Dict[str, np.ndarray] = {}
    for name in baseline_names + anchor_names:
        vals = feature_array(rows, name)
        ev = eval_score(
            name,
            vals,
            y,
            groups,
            eval_mask,
            rows=rows,
            pos_match_step_gap=args.pos_match_step_gap,
        )
        single_rows.append(ev)
        sign = ev["sign"]
        single_scores[name] = sign * vals
    single_rows.sort(key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0), reverse=True)

    model_scores: Dict[str, np.ndarray] = {}
    model_rows: List[Dict[str, Any]] = []
    if eval_mask.sum() >= 20 and len(np.unique(y[eval_mask])) == 2:
        yy = y[eval_mask]
        gg = problem_ids[eval_mask]
        rr = [rows[i] for i in np.where(eval_mask)[0]]
        baseline_no_pos = [n for n in baseline_names if n != "pos"]
        if baseline_names:
            sb = oof_scores(design(rr, baseline_names), yy, gg, folds=args.folds, seed=args.seed)
            full = np.full(len(rows), np.nan)
            full[np.where(eval_mask)[0]] = sb
            model_scores["OOF:baseline"] = full
            model_rows.append(
                eval_score(
                    "OOF:baseline",
                    full,
                    y,
                    groups,
                    eval_mask,
                    rows=rows,
                    pos_match_step_gap=args.pos_match_step_gap,
                )
            )
        if baseline_no_pos:
            sb_np = oof_scores(design(rr, baseline_no_pos), yy, gg, folds=args.folds, seed=args.seed)
            full = np.full(len(rows), np.nan)
            full[np.where(eval_mask)[0]] = sb_np
            model_scores["OOF:baseline_no_pos"] = full
            model_rows.append(
                eval_score(
                    "OOF:baseline_no_pos",
                    full,
                    y,
                    groups,
                    eval_mask,
                    rows=rows,
                    pos_match_step_gap=args.pos_match_step_gap,
                )
            )
        if anchor_names:
            sa = oof_scores(design(rr, anchor_names), yy, gg, folds=args.folds, seed=args.seed)
            full = np.full(len(rows), np.nan)
            full[np.where(eval_mask)[0]] = sa
            model_scores["OOF:anchor"] = full
            model_rows.append(
                eval_score(
                    "OOF:anchor",
                    full,
                    y,
                    groups,
                    eval_mask,
                    rows=rows,
                    pos_match_step_gap=args.pos_match_step_gap,
                )
            )
        if baseline_names and anchor_names:
            sj = oof_scores(design(rr, baseline_names + anchor_names), yy, gg, folds=args.folds, seed=args.seed)
            full = np.full(len(rows), np.nan)
            full[np.where(eval_mask)[0]] = sj
            model_scores["OOF:baseline+anchor"] = full
            model_rows.append(
                eval_score(
                    "OOF:baseline+anchor",
                    full,
                    y,
                    groups,
                    eval_mask,
                    rows=rows,
                    pos_match_step_gap=args.pos_match_step_gap,
                )
            )
        if baseline_no_pos and anchor_names:
            sj_np = oof_scores(design(rr, baseline_no_pos + anchor_names), yy, gg, folds=args.folds, seed=args.seed)
            full = np.full(len(rows), np.nan)
            full[np.where(eval_mask)[0]] = sj_np
            model_scores["OOF:baseline_no_pos+anchor"] = full
            model_rows.append(
                eval_score(
                    "OOF:baseline_no_pos+anchor",
                    full,
                    y,
                    groups,
                    eval_mask,
                    rows=rows,
                    pos_match_step_gap=args.pos_match_step_gap,
                )
            )
    model_rows.sort(key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0), reverse=True)

    best_baseline = max(
        [r for r in single_rows if r["score"] in baseline_names]
        + [r for r in model_rows if r["score"] in {"OOF:baseline", "OOF:baseline_no_pos"}],
        key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0),
        default=None,
    )
    best_anchor = max(
        [r for r in single_rows if r["score"] in anchor_names]
        + [
            r
            for r in model_rows
            if r["score"] in {"OOF:anchor", "OOF:baseline+anchor", "OOF:baseline_no_pos+anchor"}
        ],
        key=lambda r: np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0),
        default=None,
    )

    score_bank = {**single_scores, **model_scores}
    increment = {}
    rescue = {}
    if best_baseline and best_anchor and best_baseline["score"] in score_bank and best_anchor["score"] in score_bank:
        increment = bootstrap_within_increment(
            score_bank[best_anchor["score"]],
            score_bank[best_baseline["score"]],
            groups=groups,
            y_err=y,
            n_boot=args.bootstrap,
            seed=args.seed + 11,
        )
        rescue = pair_rescue_report(
            score_bank[best_anchor["score"]],
            score_bank[best_baseline["score"]],
            groups=groups,
            y_err=y,
        )

    coherent_scores = {}
    for name in ("risk_self_hijack", "risk_text_hidden_kl", "risk_coherent_hijack", "risk_transition"):
        if name in single_scores:
            coherent_scores[name] = single_scores[name]
    if "OOF:anchor" in model_scores:
        coherent_scores["OOF:anchor"] = model_scores["OOF:anchor"]
    if "OOF:baseline+anchor" in model_scores:
        coherent_scores["OOF:baseline+anchor"] = model_scores["OOF:baseline+anchor"]
    coherent = summarize_coherent_slice(
        rows,
        coherent_scores,
        y=y,
        problem_ids=problem_ids,
        base_mask=eval_mask,
        spread_q=args.coherent_spread_q,
        min_per_class=args.min_per_class,
    )

    transitions = trajectory_transition_summary(
        rows,
        [
            "p_hidden_question",
            "p_hidden_recent_prev",
            "p_hidden_other",
            "hidden_self_hijack",
            "text_hidden_kl",
            "anchor_transition_js",
            "spread",
            "risk_coherent_hijack",
        ],
    )

    headline = {
        "best_baseline": best_baseline or {},
        "best_anchor": best_anchor or {},
        "increment_over_best_baseline": increment,
        "baseline_miss_rescue": rescue,
        "coherent_slice": coherent,
    }

    res = {
        "meta": {
            **meta,
            "eval_rows": int(eval_mask.sum()),
            "first_error_rows": int(np.sum(eval_mask & (y == 1))),
            "control_rows": int(np.sum(eval_mask & (y == 0))),
            "post_error_rows": int(np.sum(phases == "post_error")),
            "problem_groups": int(len(groups)),
            "within_pairs": int(sum(int(np.sum(y[g] == 1)) * int(np.sum(y[g] == 0)) for g in groups)),
            "pos_match_step_gap": int(args.pos_match_step_gap),
            "baseline_features": baseline_names,
            "anchor_features": anchor_names,
        },
        "headline": headline,
        "single_scores": single_rows,
        "model_scores": model_rows,
        "trajectory_transitions": transitions,
    }
    if args.examples_per_class > 0:
        res["examples"] = collect_examples(rows, score_bank, best_anchor, args.examples_per_class)
    return res


def collect_examples(
    rows: Sequence[AnchorStepRow],
    scores: Mapping[str, np.ndarray],
    best_anchor: Optional[Mapping[str, Any]],
    n: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if not best_anchor or best_anchor.get("score") not in scores:
        return {}
    vals = scores[str(best_anchor["score"])]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for phase in ("first_error", "pre_error", "correct_chain"):
        idx = [i for i, r in enumerate(rows) if r.phase == phase and np.isfinite(vals[i])]
        idx = sorted(idx, key=lambda i: vals[i], reverse=True)[:n]
        out[phase] = [
            {
                "chain_idx": rows[i].chain_idx,
                "problem_id": rows[i].problem_id,
                "step_idx": rows[i].step_idx,
                "score": float(vals[i]),
                "text": rows[i].text[:300],
                "features": {k: rows[i].features.get(k) for k in (
                    "p_hidden_question",
                    "p_hidden_earlier_prefix",
                    "p_hidden_recent_prev",
                    "p_hidden_other",
                    "hidden_self_hijack",
                    "text_hidden_kl",
                    "anchor_transition_js",
                    "spread",
                    "kappa",
                )},
            }
            for i in idx
        ]
    return out


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]
    lines: List[str] = []
    lines.append(f"# Constraint Anchor Flow Audit: {os.path.basename(str(meta['npz']))}\n")
    lines.append("## Summary\n")
    lines.append(
        f"- Rows {meta['eval_rows']} | first-error {meta['first_error_rows']} | controls {meta['control_rows']} | "
        f"problems {meta['problem_groups']} | source {meta['source']} L{meta['layer']}."
    )
    bb = head.get("best_baseline") or {}
    ba = head.get("best_anchor") or {}
    if bb:
        lines.append(
            f"- Best baseline `{bb['score']}` within {bb['within_pair_auroc_error_high']:.3f} "
            f"pos-matched {bb.get('pos_matched_within_auroc_error_high')} "
            f"cross {bb['cross_auroc_error_high']:.3f}."
        )
    if ba:
        lines.append(
            f"- Best anchor `{ba['score']}` within {ba['within_pair_auroc_error_high']:.3f} "
            f"pos-matched {ba.get('pos_matched_within_auroc_error_high')} "
            f"cross {ba['cross_auroc_error_high']:.3f}."
        )
    inc = head.get("increment_over_best_baseline") or {}
    if inc:
        lines.append(f"- Increment over baseline: {inc.get('point')} CI [{inc.get('lo')}, {inc.get('hi')}] sig={inc.get('sig')}.")
    coh = head.get("coherent_slice") or {}
    if coh:
        lines.append(
            f"- Coherent low-spread slice: ok={coh.get('ok')} n={coh.get('n')} err={coh.get('err')} "
            f"groups={coh.get('groups')} spread_thr={coh.get('spread_threshold')}."
        )
    lines.append("")
    lines.append("## Model Scores\n")
    lines.append("| score | within | pos-matched | cross | pairs | pos pairs |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in res.get("model_scores", []):
        lines.append(
            f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
            f"{r.get('pos_matched_within_auroc_error_high')} | {r['cross_auroc_error_high']:.3f} | "
            f"{r['within_pairs']} | {r.get('pos_matched_pairs')} |"
        )
    lines.append("")
    lines.append("## Top Single Scores\n")
    lines.append("| score | within | pos-matched | cross | sign | err med | ctrl med |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in res.get("single_scores", [])[:16]:
        lines.append(
            f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
            f"{r.get('pos_matched_within_auroc_error_high')} | {r['cross_auroc_error_high']:.3f} | "
            f"{r['sign']:.0f} | {r['err_median']:.3f} | {r['ctrl_median']:.3f} |"
        )
    if coh and coh.get("scores"):
        lines.append("")
        lines.append("## Coherent Low-Spread Slice\n")
        lines.append("| score | within | pos-matched | cross | pairs | pos pairs |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in coh["scores"][:10]:
            lines.append(
                f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
                f"{r.get('pos_matched_within_auroc_error_high')} | {r['cross_auroc_error_high']:.3f} | "
                f"{r['within_pairs']} | {r.get('pos_matched_pairs')} |"
            )
    lines.append("")
    lines.append("## Trajectory Transitions\n")
    lines.append("| feature | pre->first mean | correct delta mean | first->post mean |")
    lines.append("|---|---:|---:|---:|")
    for name, st in res.get("trajectory_transitions", {}).items():
        a = st.get("pre_to_first_error", {}).get("mean")
        b = st.get("correct_chain_step_delta", {}).get("mean")
        c = st.get("first_to_post_error", {}).get("mean")
        lines.append(f"| {name} | {a} | {b} | {c} |")
    lines.append("")
    lines.append("## Interpretation Guardrails\n")
    lines.append("- Hidden anchor posterior is a subspace-projection diagnostic, not causal proof.")
    lines.append("- Text-hidden KL uses numeric support only; low text coverage means insufficient evidence.")
    lines.append("- Same-problem paired AUROC is the headline ranking metric.")
    lines.append("- The coherent slice tests whether anchor-flow can help when spread/kappa is not already high-risk.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    clean = finite_json(res)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    write_markdown(mpath, clean)
    return jpath, mpath


def _object_array(xs: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest(path: str, *, seed: int = 0, n_problems: int = 18, samples_per_problem: int = 5, dim: int = 48) -> None:
    rng = np.random.default_rng(seed)
    problem_ids: List[int] = []
    gold_steps: List[int] = []
    questions: List[str] = []
    steps_text: List[np.ndarray] = []
    step_ranges: List[np.ndarray] = []
    clouds: List[np.ndarray] = []
    sizes: List[np.ndarray] = []
    qvecs: List[np.ndarray] = []
    ent: List[np.ndarray] = []
    layer = 16
    for p in range(n_problems):
        q = rng.normal(size=dim)
        q /= max(float(np.linalg.norm(q)), EPS)
        r = rng.normal(size=dim)
        r = r - np.dot(r, q) * q
        r /= max(float(np.linalg.norm(r)), EPS)
        a = int(rng.integers(3, 12))
        b = int(rng.integers(2, 9))
        c = int(rng.integers(2, 6))
        for s in range(samples_per_problem):
            err = s >= samples_per_problem // 2
            gold = 2 if err else -1
            problem_ids.append(p)
            gold_steps.append(gold)
            questions.append(f"Tom has {a} apples, buys {b} more, then makes {c} equal bags. How many apples are assigned?")
            steps = np.array(
                [
                    f"Step 1: Read the question numbers {a}, {b}, and {c}.",
                    f"Step 2: Combine apples {a} + {b} = {a + b}.",
                    f"Step 3: Use the question total and bags: {a + b} * {c} = {(a + b) * c if not err else (a + b + 1) * c}.",
                    f"#### {(a + b) * c if not err else (a + b + 1) * c}",
                ],
                dtype=object,
            )
            steps_text.append(steps)
            lens = np.array([6, 7, 7, 4], dtype=np.int32)
            lo = np.cumsum(np.r_[0, lens[:-1]])
            hi = lo + lens - 1
            step_ranges.append(np.stack([lo, hi], axis=1).astype(np.int32))
            centers = [
                q,
                0.65 * q + 0.35 * r,
                (0.70 * q + 0.30 * r) if not err else r,
                (0.70 * q + 0.30 * r) if not err else r,
            ]
            chunks = []
            for t, center in enumerate(centers):
                cc = center / max(float(np.linalg.norm(center)), EPS)
                chunks.append(cc[None, :] + 0.035 * rng.normal(size=(int(lens[t]), dim)))
            clouds.append(np.concatenate(chunks, axis=0)[:, None, :].astype(np.float32))
            sizes.append(lens)
            qvecs.append(q[None, :].astype(np.float32))
            # Entropy is intentionally weak; the selftest should rely on anchor flow.
            ent.append((0.25 + 0.02 * rng.normal(size=4)).astype(np.float32))
    np.savez_compressed(
        path,
        problem_ids=np.asarray(problem_ids, dtype=np.int32),
        gold_error_step=np.asarray(gold_steps, dtype=np.int32),
        questions=np.asarray(questions, dtype=object),
        steps_text=_object_array(steps_text),
        step_token_ranges=_object_array(step_ranges),
        sv_clouds=_object_array(clouds),
        cloud_sizes=_object_array(sizes),
        cloud_layers=np.asarray([layer], dtype=np.int32),
        qvec=_object_array(qvecs),
        sv_layers=np.asarray([layer], dtype=np.int32),
        sv_out_entropy=_object_array(ent),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    head = res["headline"]
    best_anchor = head.get("best_anchor") or {}
    if not best_anchor:
        raise AssertionError("selftest produced no anchor score")
    within = float(best_anchor.get("within_pair_auroc_error_high", float("nan")))
    if not np.isfinite(within) or within < 0.80:
        raise AssertionError(f"anchor flow selftest too weak: {within}")
    coh = head.get("coherent_slice") or {}
    if coh.get("ok") and coh.get("scores"):
        top = float(coh["scores"][0]["within_pair_auroc_error_high"])
        if np.isfinite(top) and top < 0.70:
            raise AssertionError(f"coherent-slice anchor flow too weak: {top}")


def print_result(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    print(f"===== constraint anchor flow | {os.path.basename(str(meta['npz']))} =====")
    print(
        f"rows {meta['eval_rows']} | first-error {meta['first_error_rows']} | controls {meta['control_rows']} | "
        f"problems {meta['problem_groups']} | source {meta['source']} L{meta['layer']}"
    )
    bb = res["headline"].get("best_baseline") or {}
    ba = res["headline"].get("best_anchor") or {}
    if bb:
        print(
            f"best baseline {bb['score']} within={bb['within_pair_auroc_error_high']:.3f} "
            f"posmatch={bb.get('pos_matched_within_auroc_error_high')} cross={bb['cross_auroc_error_high']:.3f}"
        )
    if ba:
        print(
            f"best anchor   {ba['score']} within={ba['within_pair_auroc_error_high']:.3f} "
            f"posmatch={ba.get('pos_matched_within_auroc_error_high')} cross={ba['cross_auroc_error_high']:.3f}"
        )
    inc = res["headline"].get("increment_over_best_baseline") or {}
    if inc:
        print(f"increment over baseline: {inc.get('point')} CI=[{inc.get('lo')}, {inc.get('hi')}] sig={inc.get('sig')}")
    coh = res["headline"].get("coherent_slice") or {}
    if coh:
        print(
            f"coherent low-spread slice: ok={coh.get('ok')} n={coh.get('n')} err={coh.get('err')} "
            f"groups={coh.get('groups')}"
        )
    print("\nModel scores:")
    for r in res.get("model_scores", [])[:8]:
        print(
            f"  {r['score']:<30} within {r['within_pair_auroc_error_high']:.3f} "
            f"posmatch {r.get('pos_matched_within_auroc_error_high')} cross {r['cross_auroc_error_high']:.3f}"
        )
    print("\nTop single scores:")
    for r in res.get("single_scores", [])[:12]:
        print(
            f"  {r['score']:<32} within {r['within_pair_auroc_error_high']:.3f} "
            f"posmatch {r.get('pos_matched_within_auroc_error_high')} cross {r['cross_auroc_error_high']:.3f}"
        )
    print("\nTrajectory transitions:")
    for name, st in list(res.get("trajectory_transitions", {}).items())[:8]:
        a = st.get("pre_to_first_error", {}).get("mean")
        b = st.get("correct_chain_step_delta", {}).get("mean")
        c = st.get("first_to_post_error", {}).get("mean")
        print(f"  {name:<28} pre->first {a} ctrl {b} first->post {c}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=False, default="", help="input npz")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--hidden_dir", default="", help="override hidden shard directory for full_hidden npz files")
    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--control_pool", choices=["pre_and_correct", "pre_error", "correct_chain"], default="pre_and_correct")
    ap.add_argument("--anchor_rank", type=int, default=4)
    ap.add_argument("--posterior_temp", type=float, default=0.08)
    ap.add_argument("--other_score", type=float, default=0.02)
    ap.add_argument("--center_chain", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--raw_projection", action="store_true", help="use raw hidden rows instead of unit directions for anchor posterior")
    ap.add_argument("--kappa_beta", type=float, default=1.0)
    ap.add_argument("--min_tokens", type=int, default=4)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--min_feature_coverage", type=float, default=0.55)
    ap.add_argument("--coherent_spread_q", type=float, default=0.50)
    ap.add_argument(
        "--pos_match_step_gap",
        type=int,
        default=0,
        help="extra strict paired AUROC: compare only controls within this absolute step-index gap",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--examples_per_class", type=int, default=3)
    ap.add_argument("--output_dir", default="outputs/constraint_anchor_flow")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "constraint_anchor_flow_selftest.npz")
            make_selftest(path, seed=args.seed)
            args.input = path
            args.no_progress = True
            res = run(path, args)
            assert_selftest(res)
            print_result(res)
            print("selftest passed")
            return
    if not args.input:
        raise SystemExit("--input is required unless --selftest is used")
    res = run(args.input, args)
    print_result(res)
    stem = os.path.splitext(os.path.basename(args.input))[0] + f"_L{res['meta']['layer']}_constraint_anchor_flow"
    jpath, mpath = write_outputs(res, args.output_dir, stem)
    print(f"\nsaved: {jpath}")
    print(f"saved: {mpath}")


if __name__ == "__main__":
    main()
