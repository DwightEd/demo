from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def finite_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def auroc(y: Iterable[int], score: Iterable[float]) -> float:
    y = np.asarray(list(y), dtype=np.int32)
    s = np.asarray(list(score), dtype=np.float64)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks for ties.
    vals, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k in np.where(counts > 1)[0]:
            idx = inv == k
            ranks[idx] = np.mean(ranks[idx])
    pos = y == 1
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(~pos))
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def auprc(y: Iterable[int], score: Iterable[float]) -> float:
    y = np.asarray(list(y), dtype=np.int32)
    s = np.asarray(list(score), dtype=np.float64)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if y.size == 0 or len(np.unique(y)) < 2 or np.sum(y == 1) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    tp_all = np.cumsum(y == 1)
    fp_all = np.cumsum(y == 0)
    # Evaluate only at score thresholds.  Treating tied observations one by
    # one makes average precision depend on input row order and can give a
    # constant score an AP different from class prevalence.
    threshold_ends = np.r_[np.flatnonzero(np.diff(s) != 0), y.size - 1]
    tp = tp_all[threshold_ends]
    fp = fp_all[threshold_ends]
    recall = tp / max(tp[-1], 1)
    precision = tp / np.maximum(tp + fp, 1)
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def binary_metric_stats(y: Iterable[int], score: Iterable[float]) -> Dict[str, Any]:
    y_arr = np.asarray(list(y), dtype=np.int32)
    score_arr = np.asarray(list(score), dtype=np.float64)
    if y_arr.shape != score_arr.shape:
        raise ValueError("labels and scores must have the same shape")
    finite = np.isfinite(score_arr)
    y_valid = y_arr[finite]
    n = int(y_valid.size)
    pos = int(np.sum(y_valid == 1))
    neg = int(np.sum(y_valid == 0))
    return {
        "n": n,
        "pos": pos,
        "neg": neg,
        "coverage": float(n / y_arr.size) if y_arr.size else float("nan"),
        "prevalence": float(pos / n) if n else float("nan"),
        "auroc": auroc(y_arr, score_arr),
        "auprc": auprc(y_arr, score_arr),
    }


def load_metric_npz(path: str | Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def evaluate_first_error(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    step_scores = np.asarray(metrics["step_scores"], dtype=np.float64)
    names = [str(x) for x in metrics["step_score_names"].tolist()]
    gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
    n_steps = np.asarray(metrics["n_steps"], dtype=np.int64)
    rows = []
    y = []
    by_name = {name: [] for name in names}
    for i in range(step_scores.shape[0]):
        if gold[i] < 0:
            continue
        for j in range(int(n_steps[i])):
            y.append(1 if j == int(gold[i]) else 0)
            for k, name in enumerate(names):
                by_name[name].append(float(step_scores[i, j, k]))
            rows.append((i, j))
    metric_stats = {name: binary_metric_stats(y, vals) for name, vals in by_name.items()}
    single = {name: stats["auroc"] for name, stats in metric_stats.items()}
    return {
        "rows": len(y),
        "pos": int(np.sum(y)),
        "single": single,
        "metric_stats": metric_stats,
    }


def rank_first_errors(metrics: Mapping[str, Any], score_name: str) -> Dict[str, Any]:
    step_scores = np.asarray(metrics["step_scores"], dtype=np.float64)
    names = [str(x) for x in metrics["step_score_names"].tolist()]
    if score_name not in names:
        return {"n": 0, "top1": float("nan"), "mean_rank": float("nan")}
    k = names.index(score_name)
    gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
    n_steps = np.asarray(metrics["n_steps"], dtype=np.int64)
    ranks = []
    top1 = []
    for i in range(step_scores.shape[0]):
        g = int(gold[i])
        if g < 0 or g >= int(n_steps[i]):
            continue
        s = step_scores[i, : int(n_steps[i]), k]
        if not np.isfinite(s[g]):
            continue
        order = np.argsort(-np.nan_to_num(s, nan=-np.inf))
        rank = int(np.where(order == g)[0][0]) + 1
        ranks.append(rank)
        top1.append(1 if rank == 1 else 0)
    return {
        "n": int(len(ranks)),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
    }


def evaluate_response(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    chain_scores = np.asarray(metrics["chain_scores"], dtype=np.float64)
    names = [str(x) for x in metrics["chain_score_names"].tolist()]
    if "is_correct" in metrics:
        correct = np.asarray(metrics["is_correct"], dtype=np.float64)
        if np.isfinite(correct).any() and np.any(correct >= 0):
            y = (correct == 0).astype(np.int32)
        else:
            gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
            y = (gold >= 0).astype(np.int32)
    else:
        gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
        y = (gold >= 0).astype(np.int32)
    metric_stats = {
        name: binary_metric_stats(y, chain_scores[:, k])
        for k, name in enumerate(names)
    }
    single = {name: stats["auroc"] for name, stats in metric_stats.items()}
    pr = {name: stats["auprc"] for name, stats in metric_stats.items()}
    return {
        "n": int(y.size),
        "pos": int(np.sum(y == 1)),
        "single": single,
        "auprc": pr,
        "metric_stats": metric_stats,
        "ablation_best": _ablation_best(names, chain_scores, y),
    }


def evaluate_all(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    summary = {
        "n_chains": int(len(metrics["chain_idx"])),
        "first_error": evaluate_first_error(metrics),
        "response": evaluate_response(metrics),
        "rank": {},
    }
    for name in [str(x) for x in metrics["step_score_names"].tolist()]:
        summary["rank"][name] = rank_first_errors(metrics, name)
    return summary


def _ablation_best(
    names: Sequence[str],
    chain_scores: np.ndarray,
    y: np.ndarray,
    min_coverage: float = 0.80,
) -> Dict[str, Dict[str, Any]]:
    groups = {
        "prompt_flow": ("prompt", "prefix", "random", "off_prompt"),
        "uncertainty": ("entropy", "nll"),
        "icr": ("icr",),
        "representation_geometry": ("geom_",),
        "spectral_chain_dynamics": ("sd_",),
        "layer_time_geometry": ("ltg_",),
        "controls": ("step_len", "rel_pos"),
        "combined": tuple(str(x) for x in names),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for group, keys in groups.items():
        best_name = None
        best_auc = float("nan")
        best_pr = float("nan")
        best_coverage = float("nan")
        best_n = 0
        for k, name in enumerate(names):
            if group != "combined" and not any(key in name for key in keys):
                continue
            stats = binary_metric_stats(y, chain_scores[:, k])
            if not np.isfinite(stats["coverage"]) or stats["coverage"] < float(min_coverage):
                continue
            auc = stats["auroc"]
            if np.isfinite(auc) and (best_name is None or auc > best_auc):
                best_name = name
                best_auc = auc
                best_pr = stats["auprc"]
                best_coverage = stats["coverage"]
                best_n = stats["n"]
        out[group] = {
            "best_metric": best_name,
            "auroc": best_auc,
            "auprc": best_pr,
            "coverage": best_coverage,
            "n": int(best_n),
            "min_coverage": float(min_coverage),
        }
    return out


def save_json(obj: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(finite_json(obj), f, indent=2, ensure_ascii=False)
