#!/usr/bin/env python3
"""Report token-trajectory signal distributions and shape changes.

This is a lightweight reporting entry point.  It deliberately avoids the
classification, cross-fitting, bootstrap, and alarm logic in
`token_stream_geometry_audit.py`.

Two modes are supported:

1. Read an existing `*.profiles.jsonl` written by
   `token_stream_geometry_audit.py --save_profiles`.
2. Read an `.npz` dataset and compute the same causal token-stream traces
   directly.

The output is descriptive: per-chain signal distributions, normalized
trajectory curves, and rise-then-fall shape summaries.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multisample_temporal_rupture_audit import descriptive
from token_stream_geometry_audit import (
    auroc,
    build_token_stream_features,
    chain_lengths,
    error_token_from_gold,
    finite_json,
    finite_slope,
    hump_metrics,
    load_token_matrix,
    parse_ints,
    resolve_labels,
    resolve_stream_backend,
    safe_mean,
    safe_std,
    source_info,
)


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - progress bars are optional
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class SignalRow:
    idx: int
    problem_id: int
    y_err: int
    n_tokens: int
    n_steps: int
    error_token: float
    traces: Dict[str, np.ndarray]


def qstats(values: Iterable[float]) -> Dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": safe_std(x),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
    }


def cohen_d(error_vals: np.ndarray, correct_vals: np.ndarray) -> float:
    a = np.asarray(error_vals, dtype=np.float64)
    b = np.asarray(correct_vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(max(((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1), EPS))
    return float((np.mean(a) - np.mean(b)) / pooled)


def as_trace(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    out: List[float] = []
    for v in values:
        out.append(float(v) if v is not None else float("nan"))
    return np.asarray(out, dtype=np.float64)


def finite_first_last(x: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(x)
    if not m.any():
        return float("nan"), float("nan")
    vals = x[m]
    return float(vals[0]), float(vals[-1])


def position_mask(n: int, lo: float, hi: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=bool)
    pos = np.linspace(0.0, 1.0, n)
    return (pos >= lo) & (pos <= hi)


def chain_trace_stats(trace: np.ndarray) -> Dict[str, float]:
    x = np.asarray(trace, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    first, last = finite_first_last(x)
    if finite.size == 0:
        base = {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "first": float("nan"),
            "last": float("nan"),
            "early_mean": float("nan"),
            "mid_mean": float("nan"),
            "late_mean": float("nan"),
            "late_minus_early": float("nan"),
            "amplitude": float("nan"),
            "volatility": float("nan"),
            "slope": float("nan"),
            "argmax_pos": float("nan"),
            "argmin_pos": float("nan"),
        }
    else:
        d = np.abs(np.diff(x))
        base = {
            "mean": float(np.mean(finite)),
            "std": safe_std(finite),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "first": first,
            "last": last,
            "early_mean": safe_mean(x[position_mask(x.size, 0.00, 0.25)]),
            "mid_mean": safe_mean(x[position_mask(x.size, 0.375, 0.625)]),
            "late_mean": safe_mean(x[position_mask(x.size, 0.75, 1.00)]),
            "late_minus_early": float("nan"),
            "amplitude": float(np.max(finite) - np.min(finite)),
            "volatility": safe_mean(d),
            "slope": finite_slope(x),
            "argmax_pos": float(int(np.nanargmax(x)) / max(1, x.size - 1)),
            "argmin_pos": float(int(np.nanargmin(x)) / max(1, x.size - 1)),
        }
        if np.isfinite(base["early_mean"]) and np.isfinite(base["late_mean"]):
            base["late_minus_early"] = float(base["late_mean"] - base["early_mean"])
    for k, v in hump_metrics(x).items():
        base[k] = float(v)
    return base


def trace_names(rows: Sequence[SignalRow]) -> List[str]:
    names = sorted({name for row in rows for name in row.traces})
    preferred_prefixes = (
        "resultant_w",
        "spread_w",
        "eff_rank_raw_w",
        "eff_rank_unit_w",
        "alpha_raw_w",
        "alpha_unit_w",
        "lam1_raw_w",
        "lam1_unit_w",
    )
    preferred = [n for n in names if n.startswith(preferred_prefixes)]
    rest = [n for n in names if n not in preferred]
    return preferred + rest


def select_trace_names(rows: Sequence[SignalRow], patterns: str, max_signals: int) -> List[str]:
    names = trace_names(rows)
    if patterns:
        pats = [p.strip() for p in patterns.replace(";", ",").split(",") if p.strip()]
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
    if max_signals > 0:
        names = names[: int(max_signals)]
    return names


def load_profiles(path: str, *, max_samples: int = 0) -> List[SignalRow]:
    rows: List[SignalRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            traces = {str(k): as_trace(v) for k, v in (obj.get("traces") or {}).items()}
            rows.append(
                SignalRow(
                    idx=int(obj.get("idx", len(rows))),
                    problem_id=int(obj.get("problem_id", obj.get("idx", len(rows)))),
                    y_err=int(obj.get("y_err", 0)),
                    n_tokens=int(obj.get("n_tokens", 0)),
                    n_steps=int(obj.get("n_steps", 0)),
                    error_token=float(obj["error_token"]) if obj.get("error_token") is not None else float("nan"),
                    traces=traces,
                )
            )
            if max_samples and len(rows) >= int(max_samples):
                break
    return rows


def load_from_npz(path: str, args: argparse.Namespace) -> Tuple[List[SignalRow], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "problem_ids" in data.files:
        pids_all = data["problem_ids"].astype(int)
    else:
        n0 = len(data["gold_error_step"]) if "gold_error_step" in data.files else len(data["is_correct"])
        pids_all = np.arange(n0, dtype=int)
    y_err_all, mask_all, policy_desc = resolve_labels(data, args.policy)
    source, layer_i, layer_used = source_info(data, path, args)
    backend_kind, torch_mod, stream_device = resolve_stream_backend(args)

    if args.max_problems:
        keep = []
        seen = set()
        for pid in pids_all:
            p = int(pid)
            if p not in seen:
                seen.add(p)
                keep.append(p)
            if len(keep) >= int(args.max_problems):
                break
        mask_all = mask_all & np.isin(pids_all, np.asarray(keep, dtype=int))

    windows = parse_ints(args.windows)
    alpha_windows = parse_ints(args.alpha_windows)
    rows: List[SignalRow] = []
    iterator = range(len(pids_all))
    if not args.no_progress:
        iterator = tqdm(iterator, desc="trajectory-report", unit="chain", dynamic_ncols=True)
    for idx in iterator:
        if not mask_all[idx]:
            continue
        H = load_token_matrix(data, path, args, idx=idx, source=source, layer_i=layer_i)
        if H is None or H.ndim != 2 or H.shape[0] < int(args.min_tokens):
            continue
        if args.max_tokens and H.shape[0] > int(args.max_tokens):
            H = H[: int(args.max_tokens)]
        n_tokens = int(H.shape[0])
        lengths, ranges = chain_lengths(data, idx, n_tokens, source)
        lengths = lengths[lengths > 0]
        if lengths.size == 0:
            lengths = np.asarray([n_tokens], dtype=int)
        gold = int(data["gold_error_step"][idx]) if "gold_error_step" in data.files else -1
        err_tok = error_token_from_gold(gold, ranges, n_tokens) if ranges is not None else float("nan")

        _feats, _risk, profiles = build_token_stream_features(
            H,
            windows=windows,
            alpha_windows=alpha_windows,
            decay=float(args.decay),
            min_window=int(args.min_window),
            alpha_k=int(args.alpha_k),
            alpha_stride=int(args.alpha_stride),
            no_alpha=bool(args.no_alpha),
            stream_backend_kind=backend_kind,
            torch_mod=torch_mod,
            stream_device=stream_device,
        )
        rows.append(
            SignalRow(
                idx=int(idx),
                problem_id=int(pids_all[idx]),
                y_err=int(y_err_all[idx]),
                n_tokens=n_tokens,
                n_steps=int(len(lengths)),
                error_token=err_tok,
                traces=profiles,
            )
        )
        if args.max_samples and len(rows) >= int(args.max_samples):
            break
    data.close()
    meta = {
        "source": source,
        "layer_used": int(layer_used),
        "policy": policy_desc,
        "stream_backend": backend_kind,
        "stream_device": str(stream_device) if stream_device is not None else "",
    }
    return rows, meta


def build_chain_summary(rows: Sequence[SignalRow], names: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        for name in names:
            if name not in row.traces:
                continue
            stats = chain_trace_stats(row.traces[name])
            item: Dict[str, Any] = {
                "idx": row.idx,
                "problem_id": row.problem_id,
                "y_err": row.y_err,
                "n_tokens": row.n_tokens,
                "n_steps": row.n_steps,
                "error_token": row.error_token if np.isfinite(row.error_token) else "",
                "signal": name,
            }
            item.update(stats)
            out.append(item)
    return out


def distribution_summary(chain_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    signals = sorted({str(r["signal"]) for r in chain_rows})
    stat_names = [
        "mean",
        "std",
        "min",
        "max",
        "first",
        "last",
        "early_mean",
        "mid_mean",
        "late_mean",
        "late_minus_early",
        "amplitude",
        "volatility",
        "slope",
        "hump_score",
        "hump_present",
        "hump_peak_pos",
        "hump_rise",
        "hump_fall",
    ]
    out: List[Dict[str, Any]] = []
    for signal in signals:
        sub = [r for r in chain_rows if r["signal"] == signal]
        y = np.asarray([int(r["y_err"]) for r in sub], dtype=int)
        for stat in stat_names:
            vals = np.asarray([float(r.get(stat, float("nan"))) for r in sub], dtype=np.float64)
            err = vals[y == 1]
            cor = vals[y == 0]
            q_err = qstats(err)
            q_cor = qstats(cor)
            row: Dict[str, Any] = {
                "signal": signal,
                "stat": stat,
                "n": int(np.isfinite(vals).sum()),
                "n_error": int(np.isfinite(err).sum()),
                "n_correct": int(np.isfinite(cor).sum()),
                "error_mean": q_err["mean"],
                "correct_mean": q_cor["mean"],
                "error_median": q_err["median"],
                "correct_median": q_cor["median"],
                "error_q25": q_err["q25"],
                "error_q75": q_err["q75"],
                "correct_q25": q_cor["q25"],
                "correct_q75": q_cor["q75"],
                "diff_mean_error_minus_correct": (
                    float(q_err["mean"] - q_cor["mean"])
                    if np.isfinite(q_err["mean"]) and np.isfinite(q_cor["mean"])
                    else float("nan")
                ),
                "diff_median_error_minus_correct": (
                    float(q_err["median"] - q_cor["median"])
                    if np.isfinite(q_err["median"]) and np.isfinite(q_cor["median"])
                    else float("nan")
                ),
                "cohen_d_error_minus_correct": cohen_d(err, cor),
                "cross_auroc_error_high": auroc(vals, y),
            }
            auc = row["cross_auroc_error_high"]
            row["cross_best_direction"] = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
            out.append(row)
    return out


def binned_trace(trace: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64).reshape(-1)
    out = np.full(int(bins), np.nan, dtype=np.float64)
    if x.size == 0:
        return out
    pos = np.linspace(0.0, 1.0, x.size)
    bid = np.minimum((pos * bins).astype(int), bins - 1)
    for b in range(bins):
        vals = x[bid == b]
        out[b] = safe_mean(vals)
    return out


def trajectory_bins(rows: Sequence[SignalRow], names: Sequence[str], bins: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in names:
        by_group: Dict[str, List[np.ndarray]] = {"all": [], "correct": [], "error": []}
        for row in rows:
            if name not in row.traces:
                continue
            curve = binned_trace(row.traces[name], bins)
            by_group["all"].append(curve)
            by_group["error" if row.y_err else "correct"].append(curve)
        for group, curves in by_group.items():
            if not curves:
                continue
            A = np.asarray(curves, dtype=np.float64)
            for b in range(bins):
                vals = A[:, b]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    row = {
                        "signal": name,
                        "group": group,
                        "bin": b,
                        "pos_lo": float(b / bins),
                        "pos_hi": float((b + 1) / bins),
                        "n": 0,
                        "mean": float("nan"),
                        "se": float("nan"),
                        "q25": float("nan"),
                        "median": float("nan"),
                        "q75": float("nan"),
                    }
                else:
                    row = {
                        "signal": name,
                        "group": group,
                        "bin": b,
                        "pos_lo": float(b / bins),
                        "pos_hi": float((b + 1) / bins),
                        "n": int(vals.size),
                        "mean": float(np.mean(vals)),
                        "se": float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0,
                        "q25": float(np.quantile(vals, 0.25)),
                        "median": float(np.quantile(vals, 0.50)),
                        "q75": float(np.quantile(vals, 0.75)),
                    }
                out.append(row)
    return out


def bin_shape_rows(bin_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    signals = sorted({str(r["signal"]) for r in bin_rows})
    for signal in signals:
        for group in ("correct", "error", "all"):
            sub = [r for r in bin_rows if r["signal"] == signal and r["group"] == group and int(r["n"]) > 0]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: int(r["bin"]))
            means = np.asarray([float(r["mean"]) for r in sub], dtype=np.float64)
            bins = np.asarray([int(r["bin"]) for r in sub], dtype=int)
            if not np.isfinite(means).any():
                continue
            imax = int(np.nanargmax(means))
            out.append(
                {
                    "signal": signal,
                    "group": group,
                    "start": float(means[0]),
                    "middle": float(means[len(means) // 2]),
                    "end": float(means[-1]),
                    "end_minus_start": float(means[-1] - means[0]),
                    "peak": float(means[imax]),
                    "peak_bin": int(bins[imax]),
                    "peak_pos": float((bins[imax] + 0.5) / max(len(means), 1)),
                    "rise_to_peak": float(means[imax] - means[0]),
                    "fall_from_peak": float(means[imax] - means[-1]),
                }
            )
    return out


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, float) and not np.isfinite(v):
                    clean[k] = ""
                else:
                    clean[k] = v
            writer.writerow(clean)


def fmt(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def sort_value(x: Any, default: float = -1.0) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return float(default)
    return val if math.isfinite(val) else float(default)


def find_dist(dist_rows: Sequence[Mapping[str, Any]], signal: str, stat: str) -> Optional[Mapping[str, Any]]:
    for row in dist_rows:
        if row["signal"] == signal and row["stat"] == stat:
            return row
    return None


def find_shape(shape_rows: Sequence[Mapping[str, Any]], signal: str, group: str) -> Optional[Mapping[str, Any]]:
    for row in shape_rows:
        if row["signal"] == signal and row["group"] == group:
            return row
    return None


def write_markdown(
    path: str,
    *,
    result: Mapping[str, Any],
    dist_rows: Sequence[Mapping[str, Any]],
    shape_rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# Trajectory Signal Report")
    lines.append("")
    lines.append("This report is descriptive.  It does not fit probes, select thresholds, or run bootstrap tests.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Input: `{result['input']}`")
    lines.append(f"- Mode: `{result['mode']}`")
    lines.append(f"- Samples: {result['n_samples']} ({result['n_error']} error, {result['n_correct']} correct)")
    lines.append(f"- Signals: {len(names)}")
    if result.get("meta"):
        for k, v in result["meta"].items():
            lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Signal Distributions")
    lines.append("")
    lines.append(
        "| signal | mean err/cor | late-early err/cor | slope err/cor | hump rate err/cor | peak pos err/cor | mean AUROC |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for signal in names:
        mean_r = find_dist(dist_rows, signal, "mean") or {}
        le_r = find_dist(dist_rows, signal, "late_minus_early") or {}
        slope_r = find_dist(dist_rows, signal, "slope") or {}
        hump_r = find_dist(dist_rows, signal, "hump_present") or {}
        peak_r = find_dist(dist_rows, signal, "hump_peak_pos") or {}
        lines.append(
            "| `{}` | {}/{} | {}/{} | {}/{} | {}/{} | {}/{} | {} |".format(
                signal,
                fmt(mean_r.get("error_median")),
                fmt(mean_r.get("correct_median")),
                fmt(le_r.get("error_median")),
                fmt(le_r.get("correct_median")),
                fmt(slope_r.get("error_median")),
                fmt(slope_r.get("correct_median")),
                fmt(hump_r.get("error_mean")),
                fmt(hump_r.get("correct_mean")),
                fmt(peak_r.get("error_median")),
                fmt(peak_r.get("correct_median")),
                fmt(mean_r.get("cross_auroc_error_high")),
            )
        )
    lines.append("")
    lines.append("## Normalized Trajectory Shape")
    lines.append("")
    lines.append("| signal | group | start | middle | end | end-start | peak pos | rise | fall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for signal in names:
        for group in ("correct", "error"):
            row = find_shape(shape_rows, signal, group)
            if not row:
                continue
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    signal,
                    group,
                    fmt(row.get("start")),
                    fmt(row.get("middle")),
                    fmt(row.get("end")),
                    fmt(row.get("end_minus_start")),
                    fmt(row.get("peak_pos")),
                    fmt(row.get("rise_to_peak")),
                    fmt(row.get("fall_from_peak")),
                )
            )
    lines.append("")
    lines.append("## How To Read")
    lines.append("")
    lines.append("- `resultant_w*` is directional concentration. Higher means recent token directions are more aligned.")
    lines.append("- `spread_w* = 1 - resultant_w*` is directional breadth. Higher means directions are more dispersed.")
    lines.append("- `eff_rank_*` reports the effective rank of the local token matrix spectrum.")
    lines.append("- A real rise-then-fall pattern should show positive `rise`, positive `fall`, and an interior `peak pos`.")
    lines.append("- Large error/correct differences in this report are descriptive leads; they still need same-problem controls before becoming claims.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for k, v in result.get("files", {}).items():
        lines.append(f"- {k}: `{v}`")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_result(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(args.profiles) == bool(args.input):
        raise SystemExit("pass exactly one of --profiles or --input")
    if args.profiles:
        rows = load_profiles(args.profiles, max_samples=args.max_samples)
        source_path = args.profiles
        mode = "profiles"
        meta: Dict[str, Any] = {}
    else:
        rows, meta = load_from_npz(args.input, args)
        source_path = args.input
        mode = "npz"
    if not rows:
        raise SystemExit("no usable rows")
    names = select_trace_names(rows, args.signals, args.max_signals)
    if not names:
        raise SystemExit("no matching traces; adjust --signals or recompute profiles with more traces")

    chain_rows = build_chain_summary(rows, names)
    dist_rows = distribution_summary(chain_rows)
    bin_rows = trajectory_bins(rows, names, bins=args.bins)
    shape_rows = bin_shape_rows(bin_rows)

    stem = os.path.splitext(os.path.basename(source_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, f"{stem}_trajectory_signal_report")
    chain_csv = prefix + ".chains.csv"
    dist_csv = prefix + ".distribution.csv"
    bins_csv = prefix + ".bins.csv"
    shape_csv = prefix + ".shape.csv"
    json_path = prefix + ".json"
    md_path = prefix + ".md"

    write_csv(chain_csv, chain_rows)
    write_csv(dist_csv, dist_rows)
    write_csv(bins_csv, bin_rows)
    write_csv(shape_csv, shape_rows)

    result: Dict[str, Any] = {
        "input": source_path,
        "mode": mode,
        "meta": meta,
        "n_samples": int(len(rows)),
        "n_error": int(sum(r.y_err == 1 for r in rows)),
        "n_correct": int(sum(r.y_err == 0 for r in rows)),
        "signals": list(names),
        "config": {
            "bins": int(args.bins),
            "windows": str(args.windows),
            "alpha_windows": str(args.alpha_windows),
            "no_alpha": bool(args.no_alpha),
            "decay": float(args.decay),
            "min_window": int(args.min_window),
            "alpha_stride": int(args.alpha_stride),
        },
        "distribution": finite_json(dist_rows),
        "trajectory_shape": finite_json(shape_rows),
        "files": {
            "chain_csv": chain_csv,
            "distribution_csv": dist_csv,
            "bins_csv": bins_csv,
            "shape_csv": shape_csv,
            "json": json_path,
            "markdown": md_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(finite_json(result), f, indent=2, ensure_ascii=False)
    write_markdown(md_path, result=result, dist_rows=dist_rows, shape_rows=shape_rows, names=names)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--profiles", default="", help="existing *.profiles.jsonl from token_stream_geometry_audit.py")
    src.add_argument("--input", default="", help="npz dataset; traces are computed directly")
    ap.add_argument("--policy", default="answer_format_ok", choices=["answer", "strict", "answer_format_ok", "gold_error_step"])
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--hidden_dir", default="", help="override hidden shard directory for full_hidden npz files")
    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--stream_backend", default="auto", choices=["auto", "cpu", "torch", "cuda"])
    ap.add_argument("--stream_device", default="", help="torch device for stream resultant, e.g. cuda, cuda:0, or cpu")
    ap.add_argument("--windows", default="8,16,32,64")
    ap.add_argument("--alpha_windows", default="16,32,64")
    ap.add_argument("--decay", type=float, default=0.08)
    ap.add_argument("--min_window", type=int, default=6)
    ap.add_argument("--min_tokens", type=int, default=12)
    ap.add_argument("--alpha_k", type=int, default=12)
    ap.add_argument("--alpha_stride", type=int, default=4)
    ap.add_argument("--no_alpha", action="store_true", help="skip alpha/effective-rank computation when reading npz")
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_problems", type=int, default=0)
    ap.add_argument("--signals", default="", help="comma-separated glob filters, e.g. 'spread_w*,eff_rank_raw_w*'")
    ap.add_argument("--max_signals", type=int, default=24)
    ap.add_argument("--bins", type=int, default=20, help="normalized token-position bins for trajectory curves")
    ap.add_argument("--output_dir", default="outputs/trajectory_signal_report")
    ap.add_argument("--no_progress", action="store_true")
    return ap


def print_result(res: Mapping[str, Any]) -> None:
    print(
        "===== trajectory signal report | {} | {} =====".format(
            os.path.basename(str(res["input"])),
            res["mode"],
        )
    )
    print(
        "samples {} | err {} | correct {} | signals {}".format(
            res["n_samples"],
            res["n_error"],
            res["n_correct"],
            len(res["signals"]),
        )
    )
    print("\nTop signal/stat AUROCs:")
    rows = list(res.get("distribution") or [])
    rows.sort(key=lambda r: sort_value(r.get("cross_best_direction")), reverse=True)
    for row in rows[:12]:
        print(
            "  {}/{} auc {} best {} err_med {} cor_med {}".format(
                row.get("signal"),
                row.get("stat"),
                fmt(row.get("cross_auroc_error_high")),
                fmt(row.get("cross_best_direction")),
                fmt(row.get("error_median")),
                fmt(row.get("correct_median")),
            )
        )
    print("\nSaved:")
    for v in (res.get("files") or {}).values():
        print(f"  {v}")


def main() -> None:
    args = build_arg_parser().parse_args()
    res = make_result(args)
    print_result(res)


if __name__ == "__main__":
    main()
