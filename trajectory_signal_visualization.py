#!/usr/bin/env python3
"""Visualize step-signal trajectories without hiding sample-level structure.

This script is the visual companion to `trajectory_phase_transition_audit.py`.
It keeps every reasoning chain as a row, then adds a small set of chain-level
dynamic scores that preserve local spikes instead of averaging them away.

Outputs:
  - all-chain raw and normalized heatmaps for each requested signal;
  - all-chain heatmaps for prefix-relative event metrics such as break_z;
  - first-error aligned heatmaps for wrong chains;
  - response-level dynamic-score comparison;
  - per-chain case cards for selected or all chains;
  - CSV/JSON/HTML index files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import trajectory_phase_transition_audit as tpt


EPS = 1e-12


def finite_json(x: Any) -> Any:
    return tpt.finite_json(x)


def finite_quantile(vals: Iterable[float], q: float) -> float:
    return tpt.finite_quantile(vals, q)


def finite_values(vals: Iterable[float]) -> np.ndarray:
    x = np.asarray([float(v) for v in vals], dtype=np.float64)
    return x[np.isfinite(x)]


def auroc(score: Iterable[float], y: Iterable[int]) -> float:
    return tpt.auroc(score, y)


def clean_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Mapping[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as f:
        json.dump(finite_json(obj), f, indent=2, ensure_ascii=False)
        tmp = f.name
    os.replace(tmp, path)


def event_value(row: tpt.EventRow, name: str) -> float:
    return float(row.features.get(name, float("nan")))


def logmeanexp(vals: Sequence[float], tau: float) -> float:
    x = finite_values(vals)
    if x.size == 0:
        return float("nan")
    tau = max(float(tau), EPS)
    m = float(np.max(x))
    return float(m + tau * np.log(np.mean(np.exp((x - m) / tau))))


def topk_mean(vals: Sequence[float], k: int) -> float:
    x = finite_values(vals)
    if x.size == 0:
        return float("nan")
    kk = min(max(int(k), 1), int(x.size))
    return float(np.mean(np.sort(x)[-kk:]))


def empirical_hazard(vals: np.ndarray, threshold: float, high: float) -> np.ndarray:
    x = np.asarray(vals, dtype=np.float64)
    den = max(float(high - threshold), EPS)
    p = (x - threshold) / den
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, 1.0)


def assign_first_error_modes(rows: Sequence[tpt.EventRow], args: argparse.Namespace) -> Dict[int, str]:
    first_rows = [r for r in rows if r.phase == "first_error"]
    control_rows = [r for r in rows if r.phase in {"correct_chain", "pre_error"}]
    level_thr = finite_quantile([event_value(r, "level_z") for r in control_rows], args.event_q)
    jump_thr = finite_quantile([event_value(r, "jump_z") for r in control_rows], args.event_q)
    break_thr = finite_quantile([event_value(r, "break_z") for r in control_rows], args.event_q)
    pre_std_thr = finite_quantile([event_value(r, "prefix_std") for r in control_rows], args.stable_q)
    pre_mean_thr = finite_quantile([event_value(r, "prefix_mean") for r in control_rows], args.event_q)
    pre_slope_thr = finite_quantile([event_value(r, "prefix_slope") for r in control_rows], args.event_q)
    out: Dict[int, str] = {}
    for r in first_rows:
        level = event_value(r, "level_z")
        jump = event_value(r, "jump_z")
        br = event_value(r, "break_z")
        pstd = event_value(r, "prefix_std")
        pmean = event_value(r, "prefix_mean")
        pslope = event_value(r, "prefix_slope")
        if np.isfinite(pstd) and np.isfinite(br) and pstd <= pre_std_thr and br >= break_thr:
            mode = "stable_prefix_break"
        elif np.isfinite(pslope) and np.isfinite(jump) and pslope >= pre_slope_thr and jump < jump_thr:
            mode = "gradual_drift"
        elif np.isfinite(pmean) and pmean >= pre_mean_thr:
            mode = "persistently_unstable_prefix"
        elif (np.isfinite(level) and level >= level_thr) or (np.isfinite(jump) and jump >= jump_thr):
            mode = "isolated_jump"
        else:
            mode = "no_clear_geometry_event"
        out[int(r.chain_idx)] = mode
    return out


def augment_chain_rows(
    chain_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[tpt.EventRow],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_chain: Dict[int, List[tpt.EventRow]] = {}
    for r in event_rows:
        by_chain.setdefault(int(r.chain_idx), []).append(r)
    control_breaks = [
        event_value(r, "break_z")
        for r in event_rows
        if r.phase in {"correct_chain", "pre_error"}
    ]
    break_thr = finite_quantile(control_breaks, args.event_q)
    break_high = finite_quantile(control_breaks, min(0.995, max(args.event_q + 0.05, 0.95)))
    if not np.isfinite(break_high) or break_high <= break_thr:
        break_high = break_thr + 1.0
    mode_by_chain = assign_first_error_modes(event_rows, args)
    out: List[Dict[str, Any]] = []
    for base in chain_rows:
        row = dict(base)
        vals = sorted(by_chain.get(int(row["chain_idx"]), []), key=lambda r: r.step_idx)
        br = np.asarray([event_value(r, "break_z") for r in vals], dtype=np.float64)
        finite = br[np.isfinite(br)]
        p = empirical_hazard(br, break_thr, break_high)
        noisy_or = 1.0 - float(np.prod(1.0 - p[np.isfinite(p)])) if p.size else float("nan")
        row["dynamic_lme_break_z"] = logmeanexp(br, args.lse_tau)
        row["dynamic_topk_break_z"] = topk_mean(br, args.topk)
        row["dynamic_burst_area"] = float(np.sum(np.maximum(0.0, finite - break_thr))) if finite.size else float("nan")
        row["dynamic_burst_count"] = int(np.sum(finite >= break_thr)) if finite.size else 0
        row["dynamic_noisy_or_break"] = noisy_or
        row["dynamic_first_cross_step"] = float("nan")
        row["dynamic_first_cross_pos"] = float("nan")
        for rr, pp in zip(vals, p):
            if np.isfinite(pp) and pp > 0:
                row["dynamic_first_cross_step"] = int(rr.step_idx)
                row["dynamic_first_cross_pos"] = float(rr.step_idx / max(float(row.get("n_events", 1)), 1.0))
                break
        gold = int(row.get("gold_error_step", -1))
        row["first_error_mode"] = mode_by_chain.get(int(row["chain_idx"]), "correct_chain" if gold < 0 else "unclassified")
        out.append(row)
    meta = {
        "break_threshold": float(break_thr),
        "break_high": float(break_high),
        "event_q": float(args.event_q),
        "topk": int(args.topk),
        "lse_tau": float(args.lse_tau),
    }
    return out, meta


def response_score_table(chain_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    names = [
        "max_signal_value",
        "mean_signal_value",
        "max_level_z",
        "max_jump_z",
        "max_break_z",
        "max_shock_z",
        "dynamic_lme_break_z",
        "dynamic_topk_break_z",
        "dynamic_burst_area",
        "dynamic_burst_count",
        "dynamic_noisy_or_break",
    ]
    y = [int(r.get("y_chain_error", 0)) for r in chain_rows]
    rows = []
    for name in names:
        auc = auroc([float(r.get(name, float("nan"))) for r in chain_rows], y)
        rows.append({"score": name, "chain_auroc": auc})
    rows.sort(key=lambda r: np.nan_to_num(r["chain_auroc"], nan=-1.0), reverse=True)
    return {"scores": rows}


def sort_chain_ids(chain_rows: Sequence[Mapping[str, Any]], sort_by: str) -> List[int]:
    rows = list(chain_rows)
    if sort_by == "break":
        rows.sort(
            key=lambda r: (
                -int(r.get("y_chain_error", 0)),
                -np.nan_to_num(float(r.get("max_break_z", float("nan"))), nan=-1e9),
                int(r.get("gold_error_step", 10**9)) if int(r.get("gold_error_step", -1)) >= 0 else 10**9,
            )
        )
    elif sort_by == "gold":
        rows.sort(
            key=lambda r: (
                -int(r.get("y_chain_error", 0)),
                int(r.get("gold_error_step", 10**9)) if int(r.get("gold_error_step", -1)) >= 0 else 10**9,
                -np.nan_to_num(float(r.get("max_break_z", float("nan"))), nan=-1e9),
            )
        )
    else:
        rows.sort(key=lambda r: int(r.get("chain_idx", 0)))
    return [int(r["chain_idx"]) for r in rows]


def build_signal_matrix(
    seqs: Sequence[Mapping[str, Any]], signal: str, order: Sequence[int]
) -> Tuple[np.ndarray, Dict[int, Mapping[str, Any]]]:
    by_chain = {int(item["chain_idx"]): item for item in seqs}
    max_t = 0
    for cid in order:
        item = by_chain.get(int(cid))
        if item and signal in item["signals"]:
            max_t = max(max_t, int(len(item["signals"][signal])))
    mat = np.full((len(order), max_t), np.nan, dtype=np.float64)
    for r, cid in enumerate(order):
        item = by_chain.get(int(cid))
        if not item or signal not in item["signals"]:
            continue
        x = np.asarray(item["signals"][signal], dtype=np.float64)
        mat[r, : x.size] = x
    return mat, by_chain


def build_event_matrix(
    event_rows: Sequence[tpt.EventRow], metric: str, order: Sequence[int]
) -> np.ndarray:
    max_t = 0
    for r in event_rows:
        max_t = max(max_t, int(r.step_idx) + 1)
    mat = np.full((len(order), max_t), np.nan, dtype=np.float64)
    pos = {int(cid): i for i, cid in enumerate(order)}
    for row in event_rows:
        rr = pos.get(int(row.chain_idx))
        if rr is None:
            continue
        mat[rr, int(row.step_idx)] = event_value(row, metric)
    return mat


def normalized_matrix(mat: np.ndarray, bins: int) -> np.ndarray:
    out = np.full((mat.shape[0], int(bins)), np.nan, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, int(bins))
    for i in range(mat.shape[0]):
        x = np.asarray(mat[i], dtype=np.float64)
        finite = np.where(np.isfinite(x))[0]
        if finite.size == 0:
            continue
        x = x[: finite[-1] + 1]
        m = np.isfinite(x)
        if np.sum(m) == 1:
            out[i, :] = x[m][0]
            continue
        pos = np.linspace(0.0, 1.0, x.size)
        out[i, :] = np.interp(grid, pos[m], x[m])
    return out


def aligned_event_matrix(
    event_rows: Sequence[tpt.EventRow],
    metric: str,
    order: Sequence[int],
    win: int,
) -> Tuple[np.ndarray, List[int]]:
    cols = list(range(-int(win), int(win) + 1))
    mat = np.full((len(order), len(cols)), np.nan, dtype=np.float64)
    pos = {int(cid): i for i, cid in enumerate(order)}
    colpos = {rel: i for i, rel in enumerate(cols)}
    for row in event_rows:
        if row.gold_error_step < 0:
            continue
        rr = pos.get(int(row.chain_idx))
        if rr is None:
            continue
        rel = int(row.step_idx) - int(row.gold_error_step)
        cc = colpos.get(rel)
        if cc is not None:
            mat[rr, cc] = event_value(row, metric)
    return mat, cols


def figure_size(n_rows: int, n_cols: int, args: argparse.Namespace) -> Tuple[float, float]:
    width = min(max(8.0, 0.20 * max(n_cols, 20)), float(args.max_fig_width))
    height = min(max(4.0, 2.0 + 0.018 * max(n_rows, 1)), float(args.max_fig_height))
    return width, height


def plot_heatmap(
    mat: np.ndarray,
    path: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    args: argparse.Namespace,
    gold_steps: Optional[Sequence[float]] = None,
    boundary: Optional[int] = None,
    xticks: Optional[Sequence[int]] = None,
    xticklabels: Optional[Sequence[str]] = None,
) -> str:
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=figure_size(mat.shape[0], mat.shape[1], args))
    masked = np.ma.masked_invalid(mat)
    finite = mat[np.isfinite(mat)]
    if finite.size:
        vmin = float(np.quantile(finite, args.vmin_q))
        vmax = float(np.quantile(finite, args.vmax_q))
        if vmax <= vmin:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    if gold_steps is not None:
        yy = []
        xx = []
        for i, g in enumerate(gold_steps):
            if np.isfinite(g) and 0 <= g < mat.shape[1]:
                yy.append(i)
                xx.append(float(g))
        if xx:
            ax.scatter(xx, yy, marker="|", s=12, c="white", linewidths=0.9, label="gold first error")
            ax.scatter(xx, yy, marker="|", s=6, c="black", linewidths=0.6)
    if boundary is not None and 0 < boundary < mat.shape[0]:
        ax.axhline(boundary - 0.5, color="white", lw=0.8, ls="--")
        ax.axhline(boundary - 0.5, color="black", lw=0.4, ls="--")
    if xticks is not None and xticklabels is not None:
        ax.set_xticks(list(xticks))
        ax.set_xticklabels(list(xticklabels), rotation=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=int(args.dpi))
    plt.close(fig)
    return path


def plot_response_score_bar(scores: Sequence[Mapping[str, Any]], path: str, args: argparse.Namespace) -> str:
    ensure_dir(os.path.dirname(path))
    rows = list(scores)[: min(len(scores), int(args.max_score_bars))]
    labels = [str(r["score"]) for r in rows][::-1]
    vals = [float(r["chain_auroc"]) for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.34 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, vals, color="#4677c8")
    ax.axvline(0.5, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("response AUROC")
    ax.set_title("Response-level dynamic scores")
    fig.tight_layout()
    fig.savefig(path, dpi=int(args.dpi))
    plt.close(fig)
    return path


def selected_case_ids(chain_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[int]:
    rows = list(chain_rows)
    if args.plot_all_cases:
        return [int(r["chain_idx"]) for r in rows]
    n = max(int(args.case_count), 0)
    if n == 0:
        return []
    err = [r for r in rows if int(r.get("y_chain_error", 0)) == 1]
    cor = [r for r in rows if int(r.get("y_chain_error", 0)) == 0]
    err_hi = sorted(err, key=lambda r: np.nan_to_num(float(r.get("max_break_z", np.nan)), nan=-1e9), reverse=True)
    err_lo = sorted(err, key=lambda r: np.nan_to_num(float(r.get("max_break_z", np.nan)), nan=1e9))
    cor_hi = sorted(cor, key=lambda r: np.nan_to_num(float(r.get("max_break_z", np.nan)), nan=-1e9), reverse=True)
    buckets = [err_hi, err_lo, cor_hi]
    ids: List[int] = []
    per = max(1, int(math.ceil(n / len(buckets))))
    for bucket in buckets:
        for r in bucket[:per]:
            cid = int(r["chain_idx"])
            if cid not in ids:
                ids.append(cid)
            if len(ids) >= n:
                return ids
    return ids[:n]


def plot_case_card(
    chain_id: int,
    seq_item: Mapping[str, Any],
    chain_row: Mapping[str, Any],
    event_rows: Sequence[tpt.EventRow],
    signal: str,
    path: str,
    args: argparse.Namespace,
) -> str:
    ensure_dir(os.path.dirname(path))
    sig = np.asarray(seq_item["signals"][signal], dtype=np.float64)
    rows = sorted([r for r in event_rows if int(r.chain_idx) == int(chain_id)], key=lambda r: r.step_idx)
    steps = np.arange(sig.size)
    br = np.full(sig.size, np.nan, dtype=np.float64)
    level = np.full(sig.size, np.nan, dtype=np.float64)
    jump = np.full(sig.size, np.nan, dtype=np.float64)
    shock = np.full(sig.size, np.nan, dtype=np.float64)
    for r in rows:
        if 0 <= int(r.step_idx) < sig.size:
            br[int(r.step_idx)] = event_value(r, "break_z")
            level[int(r.step_idx)] = event_value(r, "level_z")
            jump[int(r.step_idx)] = event_value(r, "jump_z")
            shock[int(r.step_idx)] = event_value(r, "shock_z")
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    ax[0].plot(steps, sig, "-o", ms=3, lw=1.2, color="#2b6cb0", label=signal)
    ax[0].set_ylabel(signal)
    ax[0].grid(alpha=0.25)
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(steps, br, "-o", ms=3, lw=1.2, color="#c2410c", label="break_z")
    ax[1].plot(steps, level, "--", lw=0.9, color="#64748b", label="level_z")
    ax[1].plot(steps, jump, ":", lw=1.1, color="#0f766e", label="jump_z")
    ax[1].plot(steps, shock, "-.", lw=0.9, color="#7c3aed", label="shock_z")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("prefix-relative event score")
    ax[1].grid(alpha=0.25)
    ax[1].legend(loc="upper left", fontsize=8, ncol=4)
    gold = int(chain_row.get("gold_error_step", -1))
    if gold >= 0:
        for a in ax:
            a.axvline(gold, color="red", lw=1.0, alpha=0.8)
    if np.isfinite(br).any():
        peak = int(np.nanargmax(br))
        for a in ax:
            a.axvline(peak, color="black", lw=0.8, ls="--", alpha=0.7)
    title = (
        f"chain {chain_id} | problem {int(chain_row.get('problem_id', -1))} | "
        f"gold {gold} | error {int(chain_row.get('y_chain_error', 0))} | "
        f"mode {chain_row.get('first_error_mode', '')} | "
        f"max_break {float(chain_row.get('max_break_z', float('nan'))):.2f}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=int(args.dpi))
    plt.close(fig)
    return path


def write_chain_csv(rows: Sequence[Mapping[str, Any]], path: str) -> str:
    ensure_dir(os.path.dirname(path))
    cols = [
        "signal",
        "chain_idx",
        "problem_id",
        "gold_error_step",
        "y_chain_error",
        "first_error_mode",
        "n_events",
        "max_signal_value",
        "mean_signal_value",
        "max_level_z",
        "max_jump_z",
        "max_break_z",
        "max_shock_z",
        "dynamic_lme_break_z",
        "dynamic_topk_break_z",
        "dynamic_burst_area",
        "dynamic_burst_count",
        "dynamic_noisy_or_break",
        "dynamic_first_cross_step",
        "dynamic_first_cross_pos",
        "gold_break_z_rank",
        "gold_break_z_percentile",
        "gold_break_z_top1",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def html_index(path: str, title: str, figures: Sequence[str], case_paths: Sequence[str], summary: Mapping[str, Any]) -> str:
    ensure_dir(os.path.dirname(path))
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.35}"
        "img{max-width:100%;border:1px solid #ddd;margin:8px 0}"
        "code{background:#f5f5f5;padding:2px 4px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}"
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<h2>Summary</h2>",
        "<pre>" + html.escape(json.dumps(finite_json(summary), indent=2, ensure_ascii=False)) + "</pre>",
        "<h2>All-Chain Views</h2>",
        "<div class='grid'>",
    ]
    root = os.path.dirname(path)
    for fig in figures:
        rel = os.path.relpath(fig, root).replace("\\", "/")
        lines.append(f"<div><h3>{html.escape(os.path.basename(fig))}</h3><a href='{rel}'><img src='{rel}'></a></div>")
    lines.extend(["</div>", "<h2>Case Cards</h2>", "<div class='grid'>"])
    for fig in case_paths:
        rel = os.path.relpath(fig, root).replace("\\", "/")
        lines.append(f"<div><h3>{html.escape(os.path.basename(fig))}</h3><a href='{rel}'><img src='{rel}'></a></div>")
    lines.extend(["</div>", "</body></html>"])
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=root) as f:
        f.write("\n".join(lines))
        tmp = f.name
    os.replace(tmp, path)
    return path


def run_visualization(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    seqs, meta = tpt.get_signal_sequences(data, args)
    signal_names = [s.strip() for s in args.signals.split(",") if s.strip()]
    signal_names = [s for s in signal_names if s in meta["available_signals"]]
    if not signal_names:
        raise SystemExit(f"no requested signals are available; requested {args.signals}, available {meta['available_signals']}")

    stem = clean_name(os.path.splitext(os.path.basename(path))[0])
    outdir = args.output_dir
    figdir = os.path.join(outdir, "figures")
    casedir = os.path.join(outdir, "cases")
    ensure_dir(figdir)
    ensure_dir(casedir)
    outputs: Dict[str, Any] = {
        "meta": dict(meta, input=path, stem=stem, signals=signal_names),
        "signals": {},
        "figures": [],
        "case_figures": [],
    }
    for sig in signal_names:
        event_rows = tpt.build_event_rows(seqs, sig, args)
        chain_rows0 = tpt.build_chain_summaries(event_rows, sig, args)
        chain_rows, dyn_meta = augment_chain_rows(chain_rows0, event_rows, args)
        score_table = response_score_table(chain_rows)
        order = sort_chain_ids(chain_rows, args.sort_by)
        chain_by_id = {int(r["chain_idx"]): r for r in chain_rows}
        error_boundary = sum(int(chain_by_id[c].get("y_chain_error", 0)) == 1 for c in order)
        gold_steps = [
            float(chain_by_id[c].get("gold_error_step", float("nan"))) if int(chain_by_id[c].get("gold_error_step", -1)) >= 0 else float("nan")
            for c in order
        ]

        sig_mat, seq_by_id = build_signal_matrix(seqs, sig, order)
        raw_path = os.path.join(figdir, f"{stem}_{sig}_raw_step_heatmap.png")
        plot_heatmap(
            sig_mat,
            raw_path,
            title=f"{stem} | {sig} by raw step | rows sorted by {args.sort_by}",
            xlabel="raw step index",
            ylabel="reasoning chains",
            cmap=args.signal_cmap,
            args=args,
            gold_steps=gold_steps,
            boundary=error_boundary,
        )
        norm_mat = normalized_matrix(sig_mat, args.norm_bins)
        norm_gold = []
        for cid in order:
            row = chain_by_id[cid]
            g = int(row.get("gold_error_step", -1))
            n_events = int(row.get("n_events", 0)) + int(args.min_prefix)
            norm_gold.append((g / max(n_events - 1, 1)) * (args.norm_bins - 1) if g >= 0 else float("nan"))
        norm_path = os.path.join(figdir, f"{stem}_{sig}_normalized_step_heatmap.png")
        plot_heatmap(
            norm_mat,
            norm_path,
            title=f"{stem} | {sig} by normalized step | rows sorted by {args.sort_by}",
            xlabel="normalized step position",
            ylabel="reasoning chains",
            cmap=args.signal_cmap,
            args=args,
            gold_steps=norm_gold,
            boundary=error_boundary,
        )
        outputs["figures"].extend([raw_path, norm_path])

        for metric in [m.strip() for m in args.event_metrics.split(",") if m.strip()]:
            ev_mat = build_event_matrix(event_rows, metric, order)
            ev_path = os.path.join(figdir, f"{stem}_{sig}_{metric}_event_heatmap.png")
            plot_heatmap(
                ev_mat,
                ev_path,
                title=f"{stem} | {sig}/{metric} by raw step",
                xlabel="raw step index",
                ylabel="reasoning chains",
                cmap=args.event_cmap,
                args=args,
                gold_steps=gold_steps,
                boundary=error_boundary,
            )
            outputs["figures"].append(ev_path)
            err_order = [cid for cid in order if int(chain_by_id[cid].get("y_chain_error", 0)) == 1]
            aligned, rels = aligned_event_matrix(event_rows, metric, err_order, args.align_window)
            al_path = os.path.join(figdir, f"{stem}_{sig}_{metric}_first_error_aligned_heatmap.png")
            tick_step = max(1, int(math.ceil(len(rels) / 9)))
            ticks = list(range(0, len(rels), tick_step))
            labels = [f"{rels[i]:+d}" for i in ticks]
            plot_heatmap(
                aligned,
                al_path,
                title=f"{stem} | {sig}/{metric} aligned to first error",
                xlabel="step offset from gold first error",
                ylabel="wrong chains",
                cmap=args.event_cmap,
                args=args,
                xticks=ticks,
                xticklabels=labels,
            )
            outputs["figures"].append(al_path)

        score_path = os.path.join(figdir, f"{stem}_{sig}_response_dynamic_scores.png")
        plot_response_score_bar(score_table["scores"], score_path, args)
        outputs["figures"].append(score_path)
        chain_csv = os.path.join(outdir, f"{stem}_{sig}_chain_dynamic_scores.csv")
        write_chain_csv(chain_rows, chain_csv)

        case_ids = selected_case_ids(chain_rows, args)
        case_paths = []
        for cid in case_ids:
            seq_item = seq_by_id.get(int(cid))
            if not seq_item:
                continue
            cpath = os.path.join(casedir, f"{stem}_{sig}_chain_{int(cid):05d}.png")
            plot_case_card(int(cid), seq_item, chain_by_id[int(cid)], event_rows, sig, cpath, args)
            case_paths.append(cpath)
        outputs["case_figures"].extend(case_paths)

        outputs["signals"][sig] = {
            "dynamic_meta": dyn_meta,
            "response_scores": score_table["scores"],
            "n_chain_rows": int(len(chain_rows)),
            "chain_csv": chain_csv,
            "case_count": int(len(case_paths)),
        }

    json_path = os.path.join(outdir, f"{stem}_trajectory_signal_visualization.json")
    save_json(outputs, json_path)
    index_path = os.path.join(outdir, f"{stem}_trajectory_signal_visualization.html")
    html_index(index_path, f"Trajectory Signal Visualization: {stem}", outputs["figures"], outputs["case_figures"], outputs)
    outputs["json"] = json_path
    outputs["html"] = index_path
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", help="input .npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--layer_is_id", action="store_true")
    ap.add_argument("--signals", default="spread,entropy_mean")
    ap.add_argument("--event_metrics", default="break_z,level_z,jump_z,shock_z")
    ap.add_argument("--min_prefix", type=int, default=2)
    ap.add_argument("--event_q", type=float, default=0.90)
    ap.add_argument("--stable_q", type=float, default=0.50)
    ap.add_argument("--scale_floor", type=float, default=1e-3)
    ap.add_argument("--std_floor_frac", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--lse_tau", type=float, default=1.0)
    ap.add_argument("--sort_by", choices=["break", "gold", "chain"], default="break")
    ap.add_argument("--norm_bins", type=int, default=64)
    ap.add_argument("--align_window", type=int, default=5)
    ap.add_argument("--case_count", type=int, default=36)
    ap.add_argument("--plot_all_cases", action="store_true")
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/trajectory_signal_visualization")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--max_fig_width", type=float, default=18.0)
    ap.add_argument("--max_fig_height", type=float, default=70.0)
    ap.add_argument("--vmin_q", type=float, default=0.02)
    ap.add_argument("--vmax_q", type=float, default=0.98)
    ap.add_argument("--signal_cmap", default="viridis")
    ap.add_argument("--event_cmap", default="magma")
    ap.add_argument("--max_score_bars", type=int, default=14)
    ap.add_argument("--selftest", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.selftest:
        ensure_dir(args.output_dir)
        path = os.path.join(args.output_dir, "trajectory_signal_visualization_selftest.npz")
        tpt.make_selftest(path, seed=args.seed)
        args.input = path
        args.layer = 16
        args.nearest_layer = True
        args.case_count = min(args.case_count, 12)
    if not args.input:
        raise SystemExit("pass input .npz or --selftest")
    out = run_visualization(args.input, args)
    print(f"===== trajectory signal visualization | {os.path.basename(args.input)} =====")
    print(f"signals: {', '.join(out['meta']['signals'])} | chains {out['meta']['n_chains_loaded']}")
    for sig, block in out["signals"].items():
        best = block["response_scores"][0] if block["response_scores"] else {"score": "", "chain_auroc": float("nan")}
        print(f"  {sig}: best response score {best['score']} AUROC {best['chain_auroc']:.3f}")
        print(f"    chain csv: {block['chain_csv']}")
        print(f"    case cards: {block['case_count']}")
    print(f"html: {out['html']}")
    print(f"json: {out['json']}")


if __name__ == "__main__":
    main()
