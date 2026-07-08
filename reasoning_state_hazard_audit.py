#!/usr/bin/env python3
"""Length-controlled latent reasoning-state hazard audit.

This script tests the stronger version of the hidden-geometry hypothesis:

    reasoning errors occur when the emitted step diverges from a latent
    constraint state, especially at high-load state-transition points.

It deliberately does not reduce a chain to one mean/max scalar.  It builds
step-level feature sequences, separates nuisance controls from geometry, uses
out-of-fold grouped hazard models, and aggregates step hazards into response
scores with noisy-OR.

Main questions:

1. First-error localization:
   does geometry improve detection of the first wrong step beyond length,
   position, token count, text complexity, and entropy controls?

2. Pre-error awareness:
   before the first wrong step is emitted, do hidden-state geometry features
   already contain a decodable future-error signal?

3. Response detection:
   does an online hazard aggregation improve response-level wrong-answer AUC
   without washing out local step evidence?

Optional prefix-flow features can be merged from `prefix_innovation_audit.py`.
Those require raw hidden states and can use that script's CUDA path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False


@dataclass
class StepRow:
    chain_idx: int
    problem_id: int
    step_idx: int
    gold_error_step: int
    phase: str
    y_first_error: int
    y_future_error: int
    y_chain_error: int
    features: Dict[str, float] = field(default_factory=dict)


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


def safe_name(text: Any, max_len: int = 90) -> str:
    s = str(text)
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    v = "".join(out).strip("_")
    if not v:
        v = "x"
    return v[:max_len]


def safe_mean(vals: Iterable[float]) -> float:
    x = np.asarray([float(v) for v in vals], dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def descriptive(vals: Iterable[float]) -> Dict[str, Any]:
    x = np.asarray([float(v) for v in vals], dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
    }


def auroc(score: Iterable[float], y: Iterable[int]) -> float:
    s = np.asarray(list(score), dtype=np.float64)
    yy = np.asarray(list(y), dtype=int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    p = int(np.sum(yy == 1))
    n = int(np.sum(yy == 0))
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
    return float((np.sum(ranks[yy == 1]) - p * (p + 1) / 2.0) / (p * n))


def finite_slope(vals: np.ndarray, pos: Optional[np.ndarray] = None) -> float:
    y = np.asarray(vals, dtype=np.float64)
    if pos is None:
        x = np.arange(y.size, dtype=np.float64)
        x = x / max(1.0, float(y.size - 1))
    else:
        x = np.asarray(pos, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    xx = x[m] - float(np.mean(x[m]))
    yy = y[m] - float(np.mean(y[m]))
    den = float(np.dot(xx, xx))
    return float(np.dot(xx, yy) / den) if den > EPS else float("nan")


def robust_center_scale(vals: np.ndarray, min_scale: float = 1e-4) -> Tuple[float, float]:
    x = np.asarray(vals, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > EPS:
        return med, max(1.4826 * mad, min_scale)
    sd = float(np.std(x))
    return med, max(sd, min_scale)


def phase_for(gold: int, step_idx: int) -> str:
    if gold < 0:
        return "correct_chain"
    if step_idx < gold:
        return "pre_error"
    if step_idx == gold:
        return "first_error"
    return "post_error"


def first_error_label(phase: str, control_pool: str) -> int:
    if phase == "first_error":
        return 1
    if phase == "post_error":
        return -1
    if control_pool == "pre_and_correct":
        return 0
    if control_pool == "pre_error":
        return 0 if phase == "pre_error" else -1
    if control_pool == "correct_chain":
        return 0 if phase == "correct_chain" else -1
    raise ValueError(control_pool)


def future_error_label(phase: str) -> int:
    if phase == "pre_error":
        return 1
    if phase == "correct_chain":
        return 0
    return -1


def select_layer_index(data: Mapping[str, Any], n_layers: int, args: argparse.Namespace) -> Tuple[int, int]:
    layers = None
    for key in ("layers_used", "cloud_store_layers"):
        if key in data:
            layers = [int(x) for x in data[key]]
            break
    if layers and len(layers) == n_layers:
        arr = np.asarray(layers, dtype=int)
        if args.nearest_layer:
            idx = int(np.argmin(np.abs(arr - int(args.layer))))
            return idx, int(arr[idx])
        if int(args.layer) not in set(layers):
            raise SystemExit(f"layer {args.layer} not present; available {layers}. Use --nearest_layer.")
        return layers.index(int(args.layer)), int(args.layer)
    if n_layers == 1:
        return 0, int(args.layer)
    if 0 <= int(args.layer) < n_layers:
        return int(args.layer), int(args.layer)
    if args.nearest_layer:
        idx = max(0, min(n_layers - 1, int(args.layer)))
        return idx, idx
    raise SystemExit("cannot map requested layer to stepcloud layer axis")


def profile_stats(vals: np.ndarray, layer_pos: Optional[np.ndarray] = None) -> Dict[str, float]:
    x = np.asarray(vals, dtype=np.float64).reshape(-1)
    m = np.isfinite(x)
    if int(m.sum()) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "first": float("nan"),
            "last": float("nan"),
            "slope": float("nan"),
            "early_mean": float("nan"),
            "late_mean": float("nan"),
            "late_minus_early": float("nan"),
            "argmax_pos": float("nan"),
            "argmin_pos": float("nan"),
        }
    n = x.size
    pos = np.linspace(0.0, 1.0, n) if layer_pos is None else np.asarray(layer_pos, dtype=np.float64)
    early = pos <= 0.33
    late = pos >= 0.67
    early_mean = safe_mean(x[early]) if early.any() else safe_mean(x[: max(1, n // 3)])
    late_mean = safe_mean(x[late]) if late.any() else safe_mean(x[max(0, n - max(1, n // 3)) :])
    finite_idx = np.where(m)[0]
    imax = int(finite_idx[np.argmax(x[m])])
    imin = int(finite_idx[np.argmin(x[m])])
    return {
        "mean": float(np.mean(x[m])),
        "std": float(np.std(x[m], ddof=1)) if int(m.sum()) > 1 else 0.0,
        "min": float(np.min(x[m])),
        "max": float(np.max(x[m])),
        "first": float(x[finite_idx[0]]),
        "last": float(x[finite_idx[-1]]),
        "slope": finite_slope(x, pos),
        "early_mean": float(early_mean),
        "late_mean": float(late_mean),
        "late_minus_early": float(late_mean - early_mean) if np.isfinite(late_mean) and np.isfinite(early_mean) else float("nan"),
        "argmax_pos": float(pos[imax]) if imax < pos.size else float("nan"),
        "argmin_pos": float(pos[imin]) if imin < pos.size else float("nan"),
    }


def prefix_event(seq: np.ndarray, step_idx: int, *, scale_floor: float, std_floor_frac: float) -> Dict[str, float]:
    x = np.asarray(seq, dtype=np.float64)
    cur = float(x[step_idx]) if 0 <= step_idx < x.size else float("nan")
    prev = float(x[step_idx - 1]) if step_idx > 0 and np.isfinite(x[step_idx - 1]) else float("nan")
    prefix = x[:step_idx]
    prefix = prefix[np.isfinite(prefix)]
    min_scale = max(float(scale_floor), float(std_floor_frac) * float(np.nanstd(x)))
    center, scale = robust_center_scale(prefix, min_scale=min_scale)
    jump = cur - prev if np.isfinite(cur) and np.isfinite(prev) else float("nan")
    level = cur - center if np.isfinite(cur) else float("nan")
    level_z = level / max(scale, EPS) if np.isfinite(level) else float("nan")
    jump_z = jump / max(scale, EPS) if np.isfinite(jump) else float("nan")
    return {
        "level_z": level_z,
        "jump_z": jump_z,
        "break_z": max(0.0, level_z if np.isfinite(level_z) else 0.0)
        + max(0.0, jump_z if np.isfinite(jump_z) else 0.0),
        "shock_z": max(level_z if np.isfinite(level_z) else float("-inf"), jump_z if np.isfinite(jump_z) else float("-inf")),
        "prefix_mean": float(np.mean(prefix)) if prefix.size else float("nan"),
        "prefix_std": float(np.std(prefix, ddof=1)) if prefix.size > 1 else 0.0,
    }


def edis_trace(vals: np.ndarray, window: int = 8, burst_threshold: float = 1.36, rebound_threshold: float = 1.33) -> float:
    x = np.asarray(vals, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    w = min(int(window), max(2, int(x.size // 2)))
    burst = 0
    if x.size > w:
        for t in range(0, x.size - w):
            if x[t + w] - x[t] > burst_threshold:
                burst += 1
    rebound = 0
    running_min = float(x[0])
    for t in range(1, x.size):
        if float(x[t]) - running_min > rebound_threshold:
            rebound += 1
        running_min = min(running_min, float(x[t]))
    return float(0.5 * (burst + rebound) * (1.0 + float(np.var(x))))


def step_token_counts(data: Mapping[str, Any], idx: int, T: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if "step_token_ranges" in data:
        try:
            rng = np.asarray(data["step_token_ranges"][idx], dtype=int)
            if rng.ndim == 2 and rng.shape[0] >= T and rng.shape[1] >= 2:
                counts = np.maximum(1, rng[:T, 1] - rng[:T, 0] + 1).astype(np.float64)
                return counts, rng[:T]
        except Exception:
            pass
    if "cloud_sizes" in data:
        try:
            sizes = np.asarray(data["cloud_sizes"][idx], dtype=np.float64).reshape(-1)
            if sizes.size >= T:
                return np.maximum(1.0, sizes[:T]), None
        except Exception:
            pass
    return np.ones(T, dtype=np.float64), None


def entropy_step_features(data: Mapping[str, Any], idx: int, T: int, ranges: Optional[np.ndarray]) -> List[Dict[str, float]]:
    rows = [{"ctrl_entropy_mean": float("nan"), "ctrl_entropy_std": float("nan"), "ctrl_entropy_edis": float("nan")} for _ in range(T)]
    if "tok_U_D" not in data or ranges is None:
        return rows
    try:
        ent = np.asarray(data["tok_U_D"][idx], dtype=np.float64)
    except Exception:
        return rows
    if ent.size == 0:
        return rows
    base = int(ranges[0, 0])
    for j in range(T):
        lo = max(0, int(ranges[j, 0]) - base)
        hi = min(ent.size, int(ranges[j, 1]) - base + 1)
        seg = ent[lo:hi]
        finite = seg[np.isfinite(seg)]
        if finite.size:
            rows[j]["ctrl_entropy_mean"] = float(np.mean(finite))
            rows[j]["ctrl_entropy_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            rows[j]["ctrl_entropy_edis"] = edis_trace(finite)
    return rows


def maybe_step_texts(data: Mapping[str, Any], idx: int, T: int) -> List[str]:
    for key in ("step_texts", "steps", "reasoning_steps", "chain_steps"):
        if key not in data:
            continue
        try:
            obj = data[key][idx]
            if isinstance(obj, str):
                continue
            vals = [str(x) for x in list(obj)]
            if len(vals) >= T:
                return vals[:T]
        except Exception:
            continue
    return ["" for _ in range(T)]


def text_complexity_features(text: str) -> Dict[str, float]:
    s = str(text or "")
    n = max(1, len(s))
    equation_chars = sum(1 for ch in s if ch in "=+-*/^_()[]{}<>\\")
    digit_chars = sum(1 for ch in s if ch.isdigit())
    number_count = len(re.findall(r"[-+]?\d+(?:\.\d+)?", s))
    frac_count = s.count("\\frac") + s.count("/")
    return {
        "ctrl_text_chars": float(len(s)),
        "ctrl_text_log_chars": float(math.log1p(len(s))),
        "ctrl_text_digit_chars": float(digit_chars),
        "ctrl_text_number_count": float(number_count),
        "ctrl_text_equation_frac": float(equation_chars / n),
        "ctrl_text_frac_count": float(frac_count),
    }


def add_profile_family(
    rows: List[StepRow],
    matrices: Dict[str, np.ndarray],
    *,
    layer_idx: int,
    layer_used: int,
    layers: np.ndarray,
    args: argparse.Namespace,
) -> List[str]:
    event_sources: List[str] = []
    layer_pos = np.linspace(0.0, 1.0, int(layers.size)) if layers.size else None
    for source_name, M in matrices.items():
        A = np.asarray(M, dtype=np.float64)
        if A.ndim != 3:
            continue
        T, L, F = A.shape
        if T != len(rows) or L == 0 or F == 0:
            continue
        for f in range(F):
            base = f"{source_name}_f{f}"
            if source_name.endswith("_resultant") and f == 0:
                base = source_name
            for t in range(T):
                vals = A[t, :, f]
                prof = profile_stats(vals, layer_pos=layer_pos)
                if 0 <= layer_idx < vals.size:
                    rows[t].features[f"geo_{base}_L{layer_used}"] = float(vals[layer_idx])
                for stat, val in prof.items():
                    rows[t].features[f"geo_{base}_{stat}"] = float(val)
            event_sources.append(f"geo_{base}_mean")
            if 0 <= layer_idx < L:
                event_sources.append(f"geo_{base}_L{layer_used}")
    return event_sources


def feature_names_from_data(data: Mapping[str, Any], key: str, fallback_n: int) -> List[str]:
    name_key = {
        "stepcloud": "cloud_feature_names",
        "stepgeom": "geom_feature_names",
        "stepattn": "attn_names",
    }.get(key, "")
    if name_key and name_key in data:
        names = [safe_name(x) for x in data[name_key]]
        if len(names) >= fallback_n:
            return names[:fallback_n]
    return [f"f{i}" for i in range(fallback_n)]


def add_step_tensor_features(
    rows: List[StepRow],
    tensor: np.ndarray,
    feature_names: Sequence[str],
    *,
    prefix: str,
    layer_idx: int,
    layer_used: int,
    layers: np.ndarray,
) -> List[str]:
    A = np.asarray(tensor, dtype=np.float64)
    event_sources: List[str] = []
    if A.ndim == 2:
        A = A[:, None, :]
    if A.ndim != 3 or A.shape[0] != len(rows):
        return event_sources
    layer_pos = np.linspace(0.0, 1.0, A.shape[1]) if A.shape[1] else None
    for f in range(A.shape[2]):
        name = safe_name(feature_names[f] if f < len(feature_names) else f"f{f}")
        src = f"{prefix}_{name}"
        for t in range(A.shape[0]):
            vals = A[t, :, f]
            prof = profile_stats(vals, layer_pos=layer_pos)
            if 0 <= layer_idx < vals.size:
                rows[t].features[f"geo_{src}_L{layer_used}"] = float(vals[layer_idx])
            for stat, val in prof.items():
                rows[t].features[f"geo_{src}_{stat}"] = float(val)
        event_sources.append(f"geo_{src}_mean")
        if 0 <= layer_idx < A.shape[1]:
            event_sources.append(f"geo_{src}_L{layer_used}")
    return event_sources


def add_prefix_events(rows: List[StepRow], source_names: Sequence[str], args: argparse.Namespace) -> None:
    for name in sorted(set(source_names)):
        seq = np.asarray([r.features.get(name, float("nan")) for r in rows], dtype=np.float64)
        if not np.isfinite(seq).any():
            continue
        compact = safe_name(name.replace("geo_", ""), max_len=70)
        for t, row in enumerate(rows):
            if t < int(args.min_prefix):
                continue
            ev = prefix_event(seq, t, scale_floor=args.scale_floor, std_floor_frac=args.std_floor_frac)
            for k, v in ev.items():
                row.features[f"geoevt_{compact}_{k}"] = float(v)


def build_rows(path: str, args: argparse.Namespace) -> Tuple[List[StepRow], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "stepcloud" not in data.files or "gold_error_step" not in data.files:
        raise SystemExit("reasoning_state_hazard_audit requires stepcloud and gold_error_step")
    gold = np.asarray(data["gold_error_step"], dtype=int)
    problem_ids = np.asarray(data["problem_ids"], dtype=int) if "problem_ids" in data.files else np.arange(len(gold))
    N = len(gold) if args.max_chains <= 0 else min(len(gold), int(args.max_chains))
    rows: List[StepRow] = []
    skipped = {"bad_stepcloud": 0, "too_few_steps": 0}
    layer_used = int(args.layer)
    layer_idx_seen = -1
    iterator = range(N)
    if not args.no_progress:
        iterator = tqdm(iterator, desc="reasoning-state rows", unit="chain")
    for idx in iterator:
        try:
            sc = np.asarray(data["stepcloud"][idx], dtype=np.float64)
        except Exception:
            skipped["bad_stepcloud"] += 1
            continue
        if sc.ndim != 3 or sc.shape[0] < 1 or sc.shape[1] < 1:
            skipped["bad_stepcloud"] += 1
            continue
        T, n_layers, n_cloud = sc.shape
        if T < 2:
            skipped["too_few_steps"] += 1
            continue
        layer_idx, layer_used = select_layer_index(data, n_layers, args)
        layer_idx_seen = layer_idx
        layers = (
            np.asarray(data["layers_used"], dtype=int)
            if "layers_used" in data.files and len(data["layers_used"]) == n_layers
            else np.arange(n_layers, dtype=int)
        )
        counts, ranges = step_token_counts(data, idx, T)
        total_tokens = float(np.sum(counts[np.isfinite(counts)])) if np.isfinite(counts).any() else float(T)
        ent_rows = entropy_step_features(data, idx, T, ranges)
        texts = maybe_step_texts(data, idx, T) if args.include_text_complexity else ["" for _ in range(T)]
        chain_rows: List[StepRow] = []
        for t in range(T):
            phase = phase_for(int(gold[idx]), t)
            step_count = float(counts[t]) if t < counts.size and np.isfinite(counts[t]) else float("nan")
            prefix_tokens = float(np.sum(counts[:t])) if t > 0 else 0.0
            feats = {
                "ctrl_step_idx": float(t),
                "ctrl_pos": float(t / max(T - 1, 1)),
                "ctrl_n_steps": float(T),
                "ctrl_log_n_steps": float(math.log1p(T)),
                "ctrl_step_tokens": step_count,
                "ctrl_log_step_tokens": float(math.log1p(step_count)) if np.isfinite(step_count) else float("nan"),
                "ctrl_prefix_tokens": prefix_tokens,
                "ctrl_log_prefix_tokens": float(math.log1p(prefix_tokens)),
                "ctrl_response_tokens": total_tokens,
                "ctrl_log_response_tokens": float(math.log1p(total_tokens)),
                "ctrl_step_token_frac": float(step_count / max(total_tokens, EPS)) if np.isfinite(step_count) else float("nan"),
            }
            feats.update(ent_rows[t])
            if args.include_text_complexity:
                feats.update(text_complexity_features(texts[t]))
            chain_rows.append(
                StepRow(
                    chain_idx=int(idx),
                    problem_id=int(problem_ids[idx]),
                    step_idx=int(t),
                    gold_error_step=int(gold[idx]),
                    phase=phase,
                    y_first_error=first_error_label(phase, args.control_pool),
                    y_future_error=future_error_label(phase),
                    y_chain_error=int(gold[idx] >= 0),
                    features=feats,
                )
            )

        cloud_names = feature_names_from_data(data, "stepcloud", n_cloud)
        event_sources: List[str] = []
        event_sources.extend(add_step_tensor_features(chain_rows, sc, cloud_names, prefix="cloud", layer_idx=layer_idx, layer_used=layer_used, layers=layers))
        cloud_raw_names = [str(x) for x in data["cloud_feature_names"]] if "cloud_feature_names" in data.files else []
        if "resultant" in cloud_raw_names:
            result_idx = cloud_raw_names.index("resultant")
            spread = np.array(sc, copy=True)
            spread[:, :, result_idx] = 1.0 - spread[:, :, result_idx]
            event_sources.extend(
                add_step_tensor_features(
                    chain_rows,
                    spread[:, :, result_idx : result_idx + 1],
                    ["spread"],
                    prefix="cloud",
                    layer_idx=layer_idx,
                    layer_used=layer_used,
                    layers=layers,
                )
            )
        for key, prefix in (("stepgeom", "geom"), ("stepattn", "attn")):
            if key not in data.files:
                continue
            try:
                arr = np.asarray(data[key][idx], dtype=np.float64)
            except Exception:
                continue
            names = feature_names_from_data(data, key, arr.shape[-1] if arr.ndim >= 2 else 0)
            event_sources.extend(add_step_tensor_features(chain_rows, arr, names, prefix=prefix, layer_idx=layer_idx, layer_used=layer_used, layers=layers))
        if args.entropy_as_geometry:
            for t, row in enumerate(chain_rows):
                for nm in ("ctrl_entropy_mean", "ctrl_entropy_std", "ctrl_entropy_edis"):
                    if nm in row.features:
                        row.features["geo_" + nm.removeprefix("ctrl_")] = row.features[nm]
            event_sources.extend(["geo_entropy_mean", "geo_entropy_std", "geo_entropy_edis"])
        add_prefix_events(chain_rows, event_sources, args)
        rows.extend(chain_rows)
    meta = {
        "input": os.path.abspath(path),
        "basename": os.path.basename(path),
        "n_chains_seen": int(N),
        "n_rows": int(len(rows)),
        "n_error_chains": int(np.sum(gold[:N] >= 0)),
        "layer_requested": int(args.layer),
        "layer_used": int(layer_used),
        "layer_axis_index": int(layer_idx_seen),
        "skipped": skipped,
    }
    if args.include_prefix_flow:
        merge_prefix_flow(path, rows, args, meta)
    return rows, meta


def merge_prefix_flow(path: str, rows: List[StepRow], args: argparse.Namespace, meta: Dict[str, Any]) -> None:
    try:
        import prefix_innovation_audit as pia
    except Exception as exc:
        if args.require_prefix_flow:
            raise SystemExit(f"--include_prefix_flow requires prefix_innovation_audit import: {exc}") from exc
        meta["prefix_flow"] = {"available": False, "reason": f"import failed: {exc}"}
        return
    ns = argparse.Namespace(
        input=path,
        layer=args.layer,
        nearest_layer=args.nearest_layer,
        hidden_dir=args.hidden_dir,
        no_mmap=args.no_mmap,
        control_pool=args.control_pool,
        rank=args.prefix_rank,
        beta=args.prefix_beta,
        raw_hidden=args.prefix_raw_hidden,
        center_chain=args.prefix_center_chain,
        z_top_k=args.prefix_z_top_k,
        z_thresh=args.prefix_z_thresh,
        std_floor_frac=args.prefix_std_floor_frac,
        dim_top_k=8,
        prefix_backend=args.prefix_backend,
        prefix_device=args.prefix_device,
        prefix_dtype=args.prefix_dtype,
        min_per_class=1,
        min_feature_coverage=0.0,
        folds=args.folds,
        bootstrap=0,
        seed=args.seed,
        max_chains=args.max_chains,
        output_dir=args.output_dir,
        no_progress=args.no_progress,
        selftest=False,
    )
    try:
        prows, pmeta = pia.build_rows(path, ns)
    except Exception as exc:
        if args.require_prefix_flow:
            raise
        meta["prefix_flow"] = {"available": False, "reason": str(exc)}
        return
    keep = [
        "off_prefix_subspace",
        "off_prev_subspace",
        "innovation_off_prefix",
        "innovation_off_prev",
        "innovation_norm",
        "mean_step_cos_prev",
        "q_align_current",
        "q_align_drop",
        "innovation_q_cos",
        "prefix_z_abs_max",
        "prefix_z_top_abs_mean",
        "prefix_z_l2",
        "prefix_z_n_gt",
        "risk_combined_off_z",
    ]
    by_key = {(int(r.chain_idx), int(r.step_idx)): r for r in prows}
    merged = 0
    for row in rows:
        src = by_key.get((row.chain_idx, row.step_idx))
        if src is None:
            continue
        for name in keep:
            row.features[f"pf_{name}"] = float(src.features.get(name, float("nan")))
        merged += 1
    meta["prefix_flow"] = {
        "available": True,
        "merged_rows": int(merged),
        "source_meta": finite_json(pmeta),
    }


def feature_matrix(rows: Sequence[StepRow], names: Sequence[str]) -> np.ndarray:
    X = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for j, name in enumerate(names):
        X[:, j] = [r.features.get(name, float("nan")) for r in rows]
    return X


def select_features(rows: Sequence[StepRow], prefix: str, mask: np.ndarray, min_coverage: float, min_unique: int) -> List[str]:
    names = sorted({k for r in rows for k in r.features if k.startswith(prefix)})
    out: List[str] = []
    for name in names:
        x = np.asarray([r.features.get(name, float("nan")) for r in rows], dtype=np.float64)
        cov = float(np.mean(np.isfinite(x[mask]))) if mask.any() else 0.0
        vals = np.unique(np.round(x[mask & np.isfinite(x)], 10))
        if cov >= min_coverage and vals.size >= min_unique:
            out.append(name)
    return out


def group_folds(group_ids: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(group_ids, dtype=int)
    uniq = np.unique(groups)
    if uniq.size < 2:
        return []
    rng = np.random.default_rng(seed)
    uniq = np.array(uniq, copy=True)
    rng.shuffle(uniq)
    k = int(min(max(2, k), uniq.size))
    fold_of = {int(g): i % k for i, g in enumerate(uniq)}
    fold = np.asarray([fold_of[int(g)] for g in groups], dtype=int)
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


def fit_linear_witness(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X0 = np.where(np.isfinite(X), X, med)
    mu = np.mean(X0, axis=0)
    sd = np.std(X0, axis=0)
    sd = np.where(sd > EPS, sd, 1.0)
    Z = (X0 - mu) / sd
    err = Z[y == 1]
    cor = Z[y == 0]
    w = np.zeros(Z.shape[1], dtype=np.float64) if err.size == 0 or cor.size == 0 else np.mean(err, axis=0) - np.mean(cor, axis=0)
    return med, sd, w


def apply_linear_witness(X: np.ndarray, med: np.ndarray, sd: np.ndarray, w: np.ndarray) -> np.ndarray:
    X0 = np.where(np.isfinite(X), X, med)
    return ((X0 - med) / np.maximum(sd, EPS)) @ w


def oof_predict(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    *,
    folds: int,
    seed: int,
    C: float,
) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=np.float64)
    for tr, te in group_folds(group_ids, folds, seed):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr = X[tr]
        Xte = X[te]
        if HAVE_SKLEARN:
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=4000, class_weight="balanced", C=float(C), solver="lbfgs"),
            )
            clf.fit(Xtr, y[tr])
            scores[te] = clf.predict_proba(Xte)[:, 1]
        else:
            med, sd, w = fit_linear_witness(Xtr, y[tr])
            raw = apply_linear_witness(Xte, med, sd, w)
            scores[te] = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
    return scores


def residualize_oof(
    X_geo: np.ndarray,
    X_ctrl: np.ndarray,
    group_ids: np.ndarray,
    *,
    folds: int,
    seed: int,
    ridge: float,
) -> np.ndarray:
    if X_geo.size == 0:
        return X_geo.copy()
    if X_ctrl.size == 0:
        return X_geo.copy()
    out = np.full_like(X_geo, np.nan, dtype=np.float64)
    for tr, te in group_folds(group_ids, folds, seed):
        Ctr = X_ctrl[tr]
        Cte = X_ctrl[te]
        Gtr = X_geo[tr]
        Gte = X_geo[te]
        c_med = np.nanmedian(Ctr, axis=0)
        c_med = np.where(np.isfinite(c_med), c_med, 0.0)
        Ctr0 = np.where(np.isfinite(Ctr), Ctr, c_med)
        Cte0 = np.where(np.isfinite(Cte), Cte, c_med)
        c_mu = np.mean(Ctr0, axis=0)
        c_sd = np.std(Ctr0, axis=0)
        c_sd = np.where(c_sd > EPS, c_sd, 1.0)
        Ztr = (Ctr0 - c_mu) / c_sd
        Zte = (Cte0 - c_mu) / c_sd
        g_med = np.nanmedian(Gtr, axis=0)
        g_med = np.where(np.isfinite(g_med), g_med, 0.0)
        Gtr0 = np.where(np.isfinite(Gtr), Gtr, g_med)
        Gte0 = np.where(np.isfinite(Gte), Gte, g_med)
        A = np.column_stack([np.ones(Ztr.shape[0]), Ztr])
        B = np.column_stack([np.ones(Zte.shape[0]), Zte])
        reg = np.eye(A.shape[1], dtype=np.float64) * float(ridge)
        reg[0, 0] = 0.0
        coef = np.linalg.pinv(A.T @ A + reg) @ A.T @ Gtr0
        out[te] = Gte0 - B @ coef
    return out


def within_group_auroc(groups: Sequence[np.ndarray], score: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    conc = 0.0
    pairs = 0
    s = np.asarray(score, dtype=np.float64)
    yy = np.asarray(y, dtype=int)
    for group in groups:
        idx = np.asarray(group, dtype=int)
        pos = [float(s[i]) for i in idx if yy[i] == 1 and np.isfinite(s[i])]
        neg = [float(s[i]) for i in idx if yy[i] == 0 and np.isfinite(s[i])]
        for a in pos:
            for b in neg:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        pairs += len(pos) * len(neg)
    return (float(conc / pairs) if pairs else float("nan")), int(pairs)


def grouped_eval_sets(row_subset: Sequence[StepRow], y: np.ndarray, min_per_class: int) -> Dict[str, List[np.ndarray]]:
    chain_ids = np.asarray([r.chain_idx for r in row_subset], dtype=int)
    problem_ids = np.asarray([r.problem_id for r in row_subset], dtype=int)
    out: Dict[str, List[np.ndarray]] = {}
    for label, ids in (("within_chain", chain_ids), ("within_problem", problem_ids)):
        groups: List[np.ndarray] = []
        for gid in np.unique(ids):
            idx = np.where(ids == gid)[0]
            if np.sum(y[idx] == 1) >= min_per_class and np.sum(y[idx] == 0) >= min_per_class:
                groups.append(idx)
        out[label] = groups
    return out


def bootstrap_auc_delta(
    score_a: np.ndarray,
    score_b: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    yy = np.asarray(y, dtype=int)
    gids = np.asarray(group_ids, dtype=int)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 4:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    base_a = auroc(a[m], yy[m])
    base_b = auroc(b[m], yy[m])
    rng = np.random.default_rng(seed)
    uniq = np.unique(gids[m])
    deltas: List[float] = []
    for _ in range(int(n_boot)):
        sample_groups = rng.choice(uniq, size=uniq.size, replace=True)
        idxs: List[int] = []
        valid_idx = np.where(m)[0]
        for g in sample_groups:
            idxs.extend(valid_idx[gids[valid_idx] == g].tolist())
        if not idxs:
            continue
        ii = np.asarray(idxs, dtype=int)
        da = auroc(a[ii], yy[ii])
        db = auroc(b[ii], yy[ii])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(float(da - db))
    d = np.asarray(deltas, dtype=np.float64)
    return {
        "auc_a": float(base_a),
        "auc_b": float(base_b),
        "delta": float(base_a - base_b) if np.isfinite(base_a) and np.isfinite(base_b) else float("nan"),
        "ci_low": float(np.quantile(d, 0.025)) if d.size else float("nan"),
        "ci_high": float(np.quantile(d, 0.975)) if d.size else float("nan"),
        "n_boot": int(d.size),
    }


def model_eval_row(name: str, score: np.ndarray, y: np.ndarray, row_subset: Sequence[StepRow], min_per_class: int) -> Dict[str, Any]:
    m = np.isfinite(score)
    groups = grouped_eval_sets(row_subset, y, min_per_class)
    wc, wc_pairs = within_group_auroc(groups["within_chain"], score, y)
    wp, wp_pairs = within_group_auroc(groups["within_problem"], score, y)
    return {
        "model": name,
        "n_scored": int(np.sum(m)),
        "pooled_auroc": float(auroc(score[m], y[m])) if m.any() else float("nan"),
        "within_chain_auroc": float(wc),
        "within_chain_pairs": int(wc_pairs),
        "within_problem_auroc": float(wp),
        "within_problem_pairs": int(wp_pairs),
        "positive_mean": safe_mean(score[(y == 1) & m]),
        "negative_mean": safe_mean(score[(y == 0) & m]),
    }


def evaluate_task(
    rows: Sequence[StepRow],
    *,
    task_name: str,
    y_all: np.ndarray,
    mask: np.ndarray,
    control_names: Sequence[str],
    geometry_names: Sequence[str],
    prefix_names: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    idx = np.where(mask)[0]
    if idx.size < 10 or len(np.unique(y_all[idx])) < 2:
        return {"available": False, "reason": "not enough labeled rows for both classes"}
    row_subset = [rows[int(i)] for i in idx]
    y = y_all[idx].astype(int)
    group_ids_all = np.asarray(
        [r.chain_idx if args.cv_group == "chain" else r.problem_id for r in row_subset],
        dtype=int,
    )
    Xc = feature_matrix(row_subset, control_names)
    Xg = feature_matrix(row_subset, list(geometry_names) + list(prefix_names))
    Xpf = feature_matrix(row_subset, prefix_names)
    Xgres = residualize_oof(Xg, Xc, group_ids_all, folds=args.folds, seed=args.seed + 101, ridge=args.residual_ridge)
    model_specs: List[Tuple[str, np.ndarray]] = []
    if Xc.shape[1] > 0:
        model_specs.append(("controls", Xc))
    if Xg.shape[1] > 0:
        model_specs.append(("geometry", Xg))
    if Xgres.shape[1] > 0:
        model_specs.append(("geometry_residualized", Xgres))
    if Xc.shape[1] > 0 and Xg.shape[1] > 0:
        model_specs.append(("controls+geometry", np.column_stack([Xc, Xg])))
        model_specs.append(("controls+geometry_residualized", np.column_stack([Xc, Xgres])))
    if Xpf.shape[1] > 0:
        model_specs.append(("prefix_flow", Xpf))
        if Xc.shape[1] > 0:
            model_specs.append(("controls+prefix_flow", np.column_stack([Xc, Xpf])))
    scores: Dict[str, np.ndarray] = {}
    eval_rows: List[Dict[str, Any]] = []
    for name, X in model_specs:
        if X.shape[1] == 0:
            continue
        s = oof_predict(X, y, group_ids_all, folds=args.folds, seed=args.seed + 7, C=args.C)
        scores[name] = s
        eval_rows.append(model_eval_row(name, s, y, row_subset, args.min_per_class))
    eval_rows.sort(key=lambda r: np.nan_to_num(r["pooled_auroc"], nan=-1.0), reverse=True)
    increments: Dict[str, Any] = {}
    for name in ("controls+geometry", "controls+geometry_residualized", "controls+prefix_flow"):
        if "controls" in scores and name in scores:
            increments[f"{name}_vs_controls"] = bootstrap_auc_delta(
                scores[name],
                scores["controls"],
                y,
                group_ids_all,
                n_boot=args.bootstrap,
                seed=args.seed + 211,
            )
    response = response_detection(row_subset, y, scores, args)
    return {
        "available": True,
        "task": task_name,
        "n_rows": int(idx.size),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "cv_group": args.cv_group,
        "models": eval_rows,
        "increments": increments,
        "response_detection": response,
    }


def response_detection(row_subset: Sequence[StepRow], y: np.ndarray, scores: Mapping[str, np.ndarray], args: argparse.Namespace) -> Dict[str, Any]:
    chain_ids = np.asarray([r.chain_idx for r in row_subset], dtype=int)
    y_chain_by_id: Dict[int, int] = {}
    for r in row_subset:
        y_chain_by_id[int(r.chain_idx)] = int(r.y_chain_error)
    chain_order = sorted(y_chain_by_id)
    yy = np.asarray([y_chain_by_id[c] for c in chain_order], dtype=int)
    out_rows: List[Dict[str, Any]] = []
    score_bank: Dict[str, np.ndarray] = {}
    for model_name, s in scores.items():
        noisy: List[float] = []
        maxs: List[float] = []
        topk: List[float] = []
        for c in chain_order:
            vals = np.asarray([s[i] for i, r in enumerate(row_subset) if int(r.chain_idx) == c], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                noisy.append(float("nan"))
                maxs.append(float("nan"))
                topk.append(float("nan"))
                continue
            p = np.clip(vals, 0.0, 1.0)
            noisy.append(float(1.0 - np.prod(1.0 - p)))
            maxs.append(float(np.max(p)))
            kk = min(max(1, int(args.response_topk)), p.size)
            topk.append(float(np.mean(np.sort(p)[-kk:])))
        for agg_name, arr in (("noisy_or", noisy), ("max", maxs), ("topk_mean", topk)):
            aa = np.asarray(arr, dtype=np.float64)
            score_bank[f"{model_name}:{agg_name}"] = aa
            out_rows.append(
                {
                    "model": model_name,
                    "aggregation": agg_name,
                    "chain_auroc": float(auroc(aa, yy)),
                    "n_chains": int(np.sum(np.isfinite(aa))),
                }
            )
    out_rows.sort(key=lambda r: np.nan_to_num(r["chain_auroc"], nan=-1.0), reverse=True)
    increments: Dict[str, Any] = {}
    for geom_name in ("controls+geometry", "controls+geometry_residualized", "controls+prefix_flow"):
        for agg_name in ("noisy_or", "max", "topk_mean"):
            a = f"{geom_name}:{agg_name}"
            b = f"controls:{agg_name}"
            if a in score_bank and b in score_bank:
                increments[f"{a}_vs_{b}"] = bootstrap_auc_delta(
                    score_bank[a],
                    score_bank[b],
                    yy,
                    np.asarray(chain_order, dtype=int),
                    n_boot=args.bootstrap,
                    seed=args.seed + 307,
                )
    return {"scores": out_rows, "increments": increments}


def feature_coverage(rows: Sequence[StepRow], names: Sequence[str], mask: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in names:
        x = np.asarray([r.features.get(name, float("nan")) for r in rows], dtype=np.float64)
        out[name] = float(np.mean(np.isfinite(x[mask]))) if mask.any() else 0.0
    return out


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    rows, meta = build_rows(path, args)
    if not rows:
        raise SystemExit("no step rows were built")
    y_first = np.asarray([r.y_first_error for r in rows], dtype=int)
    y_future = np.asarray([r.y_future_error for r in rows], dtype=int)
    first_mask = y_first >= 0
    future_mask = y_future >= 0
    control_names = select_features(rows, "ctrl_", first_mask | future_mask, args.min_feature_coverage, args.min_unique)
    geometry_names = select_features(rows, "geo_", first_mask | future_mask, args.min_feature_coverage, args.min_unique)
    geometry_event_names = select_features(rows, "geoevt_", first_mask | future_mask, args.min_feature_coverage, args.min_unique)
    prefix_names = select_features(rows, "pf_", first_mask | future_mask, args.min_feature_coverage, args.min_unique)
    all_geometry = geometry_names + geometry_event_names
    if args.require_geometry and not all_geometry and not prefix_names:
        raise SystemExit("no geometry features passed coverage; lower --min_feature_coverage or inspect input keys")
    first = evaluate_task(
        rows,
        task_name="first_error",
        y_all=y_first,
        mask=first_mask,
        control_names=control_names,
        geometry_names=all_geometry,
        prefix_names=prefix_names,
        args=args,
    )
    future = evaluate_task(
        rows,
        task_name="pre_error_future",
        y_all=y_future,
        mask=future_mask,
        control_names=control_names,
        geometry_names=all_geometry,
        prefix_names=prefix_names,
        args=args,
    )
    meta.update(
        {
            "control_features": control_names,
            "geometry_features": geometry_names,
            "geometry_event_features": geometry_event_names,
            "prefix_flow_features": prefix_names,
            "feature_counts": {
                "control": len(control_names),
                "geometry_current": len(geometry_names),
                "geometry_event": len(geometry_event_names),
                "prefix_flow": len(prefix_names),
            },
            "coverage_summary": {
                "control": descriptive(feature_coverage(rows, control_names, first_mask | future_mask).values()),
                "geometry_current": descriptive(feature_coverage(rows, geometry_names, first_mask | future_mask).values()),
                "geometry_event": descriptive(feature_coverage(rows, geometry_event_names, first_mask | future_mask).values()),
                "prefix_flow": descriptive(feature_coverage(rows, prefix_names, first_mask | future_mask).values()),
            },
            "method": {
                "controls": "length/position/token/text/entropy nuisance stream",
                "geometry": "step x layer hidden geometry profiles plus prefix-relative geometric events",
                "residualization": "out-of-fold ridge residuals of geometry on controls",
                "hazard": "grouped out-of-fold logistic step hazard",
                "response": "online noisy-OR, max, and top-k mean aggregation over step hazards",
            },
        }
    )
    return {
        "meta": meta,
        "tasks": {
            "first_error": first,
            "pre_error_future": future,
        },
        "rows": rows,
    }


def write_json(obj: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as f:
        json.dump(finite_json(obj), f, ensure_ascii=False, indent=2)
        tmp = f.name
    os.replace(tmp, path)


def write_rows_csv(rows: Sequence[StepRow], path: str, feature_names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "chain_idx",
        "problem_id",
        "step_idx",
        "gold_error_step",
        "phase",
        "y_first_error",
        "y_future_error",
        "y_chain_error",
    ] + list(feature_names)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {
                "chain_idx": r.chain_idx,
                "problem_id": r.problem_id,
                "step_idx": r.step_idx,
                "gold_error_step": r.gold_error_step,
                "phase": r.phase,
                "y_first_error": r.y_first_error,
                "y_future_error": r.y_future_error,
                "y_chain_error": r.y_chain_error,
            }
            for name in feature_names:
                row[name] = r.features.get(name, float("nan"))
            w.writerow(row)


def markdown_report(res: Mapping[str, Any]) -> str:
    meta = res["meta"]
    lines = [
        f"# Reasoning State Hazard Audit: `{meta['basename']}`",
        "",
        "## Material Passport",
        "",
        "- Verification Status: ANALYZED",
        "- Artifact Type: code experiment output",
        "- Core question: whether hidden geometry adds length-controlled hazard signal.",
        "",
        "## Setup",
        "",
        f"- Chains seen: {meta['n_chains_seen']} | rows: {meta['n_rows']} | error chains: {meta['n_error_chains']}",
        f"- Layer used: {meta['layer_used']} | CV group: grouped out-of-fold",
        f"- Feature counts: controls {meta['feature_counts']['control']}, geometry-current {meta['feature_counts']['geometry_current']}, "
        f"geometry-event {meta['feature_counts']['geometry_event']}, prefix-flow {meta['feature_counts']['prefix_flow']}",
        "",
        "## Interpretation Guardrail",
        "",
        "The headline is not raw geometry AUC.  The key quantity is `controls+geometry` or "
        "`controls+geometry_residualized` minus `controls` under grouped out-of-fold evaluation.",
        "",
    ]
    for task_name, task in res["tasks"].items():
        lines.extend([f"## Task: `{task_name}`", ""])
        if not task.get("available"):
            lines.append(f"- Not available: {task.get('reason')}")
            lines.append("")
            continue
        lines.append(f"- Rows: {task['n_rows']} | positives: {task['positives']} | negatives: {task['negatives']}")
        lines.extend(["", "### Step Hazard Models", ""])
        lines.append("| model | pooled AUC | within-chain AUC | chain pairs | within-problem AUC | problem pairs |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in task.get("models", []):
            lines.append(
                f"| `{row['model']}` | {row['pooled_auroc']:.3f} | {row['within_chain_auroc']:.3f} | "
                f"{row['within_chain_pairs']} | {row['within_problem_auroc']:.3f} | {row['within_problem_pairs']} |"
            )
        lines.extend(["", "### Step-Level Increments", ""])
        lines.append("| comparison | delta pooled AUC | 95% CI |")
        lines.append("|---|---:|---|")
        for name, inc in task.get("increments", {}).items():
            lines.append(f"| `{name}` | {inc['delta']:+.3f} | [{inc['ci_low']:+.3f}, {inc['ci_high']:+.3f}] |")
        lines.extend(["", "### Response Hazard Aggregation", ""])
        lines.append("| model | aggregation | response AUC | chains |")
        lines.append("|---|---|---:|---:|")
        for row in task.get("response_detection", {}).get("scores", []):
            lines.append(f"| `{row['model']}` | `{row['aggregation']}` | {row['chain_auroc']:.3f} | {row['n_chains']} |")
        lines.extend(["", "### Response-Level Increments", ""])
        lines.append("| comparison | delta response AUC | 95% CI |")
        lines.append("|---|---:|---|")
        for name, inc in task.get("response_detection", {}).get("increments", {}).items():
            lines.append(f"| `{name}` | {inc['delta']:+.3f} | [{inc['ci_low']:+.3f}, {inc['ci_high']:+.3f}] |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    compact = {k: v for k, v in res.items() if k != "rows"}
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    cpath = os.path.join(output_dir, stem + ".rows.csv")
    write_json(compact, jpath)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_dir) as f:
        f.write(markdown_report(compact))
        tmp = f.name
    os.replace(tmp, mpath)
    feature_names = (
        res["meta"].get("control_features", [])
        + res["meta"].get("geometry_features", [])
        + res["meta"].get("geometry_event_features", [])
        + res["meta"].get("prefix_flow_features", [])
    )
    write_rows_csv(res["rows"], cpath, feature_names)
    return jpath, mpath, cpath


def object_array(items: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def make_selftest(path: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 180
    layers = np.arange(8, 16, dtype=np.int32)
    names = np.asarray(["resultant", "lam1", "eff_rank"], dtype=object)
    clouds = []
    ranges = []
    ents = []
    gold = []
    pids = []
    cursor_problem = 0
    for i in range(n):
        err = i % 2 == 0
        T = int(rng.integers(5, 11) + (1 if err else 0))
        g = int(rng.integers(2, T - 1)) if err else -1
        counts = rng.integers(4, 16, size=T)
        rr = []
        pos0 = 0
        ent = []
        for t, c in enumerate(counts):
            rr.append([pos0, pos0 + int(c) - 1])
            base_ent = rng.normal(2.0 + 0.03 * T + 0.02 * t, 0.2, size=int(c))
            ent.extend(base_ent.tolist())
            pos0 += int(c)
        A = np.zeros((T, layers.size, names.size), dtype=np.float64)
        for t in range(T):
            pos = t / max(T - 1, 1)
            for li, lp in enumerate(np.linspace(0.0, 1.0, layers.size)):
                load = 0.02 * T + 0.06 * pos + 0.03 * math.sin(2 * math.pi * lp)
                pre_tension = 0.05 if err and t == g - 1 else 0.0
                first_error_shock = 0.16 if err and t == g else 0.0
                A[t, li, 0] = 0.78 - load - pre_tension - first_error_shock + rng.normal(0, 0.015)
                A[t, li, 1] = 0.45 + load + 0.08 * first_error_shock + rng.normal(0, 0.015)
                A[t, li, 2] = 9.0 + 4.0 * load + 3.0 * first_error_shock + rng.normal(0, 0.15)
        clouds.append(A)
        ranges.append(np.asarray(rr, dtype=np.int32))
        ents.append(np.asarray(ent, dtype=np.float64))
        gold.append(g)
        pids.append(cursor_problem)
        cursor_problem += 1
    np.savez(
        path,
        stepcloud=object_array(clouds),
        cloud_feature_names=names,
        layers_used=layers,
        step_token_ranges=object_array(ranges),
        tok_U_D=object_array(ents),
        gold_error_step=np.asarray(gold, dtype=np.int32),
        problem_ids=np.asarray(pids, dtype=np.int32),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    first = res["tasks"]["first_error"]
    if not first.get("available"):
        raise AssertionError("first-error selftest task unavailable")
    aucs = {r["model"]: r["pooled_auroc"] for r in first["models"]}
    ctrl = aucs.get("controls", float("nan"))
    geom = aucs.get("controls+geometry", float("nan"))
    gres = aucs.get("controls+geometry_residualized", float("nan"))
    if not np.isfinite(geom) or geom < 0.85:
        raise AssertionError(f"selftest failed: controls+geometry pooled AUC too low: {geom}")
    if np.isfinite(ctrl) and np.isfinite(geom) and geom - ctrl < 0.08:
        raise AssertionError(f"selftest failed: geometry increment too small: geom={geom}, controls={ctrl}")
    if not np.isfinite(gres) or gres < 0.80:
        raise AssertionError(f"selftest failed: residualized geometry AUC too low: {gres}")
    resp = first["response_detection"]["scores"]
    best_resp = max([r["chain_auroc"] for r in resp if r["model"] in {"controls+geometry", "controls+geometry_residualized"}], default=float("nan"))
    if not np.isfinite(best_resp) or best_resp < 0.85:
        raise AssertionError(f"selftest failed: response hazard AUC too low: {best_resp}")


def print_result(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    print(f"\n===== reasoning state hazard | {meta['basename']} =====")
    print(
        f"rows {meta['n_rows']} | chains {meta['n_chains_seen']} | "
        f"features ctrl {meta['feature_counts']['control']} geo {meta['feature_counts']['geometry_current']} "
        f"geoevt {meta['feature_counts']['geometry_event']} pf {meta['feature_counts']['prefix_flow']}"
    )
    for task_name, task in res["tasks"].items():
        print(f"\nTask {task_name}:")
        if not task.get("available"):
            print(f"  unavailable: {task.get('reason')}")
            continue
        print(f"  rows {task['n_rows']} pos {task['positives']} neg {task['negatives']}")
        for row in task.get("models", [])[:8]:
            print(
                f"  {row['model']:<32} pooled {row['pooled_auroc']:.3f} "
                f"within-chain {row['within_chain_auroc']:.3f}"
            )
        for name, inc in task.get("increments", {}).items():
            print(f"  inc {name:<48} {inc['delta']:+.3f} [{inc['ci_low']:+.3f},{inc['ci_high']:+.3f}]")
        best_resp = task.get("response_detection", {}).get("scores", [])[:5]
        if best_resp:
            print("  response:")
            for row in best_resp:
                print(f"    {row['model']}/{row['aggregation']:<12} AUC {row['chain_auroc']:.3f}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="", help="input .npz with stepcloud and gold_error_step")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--nearest_layer", action="store_true")
    ap.add_argument("--min_prefix", type=int, default=1)
    ap.add_argument("--control_pool", choices=["pre_and_correct", "pre_error", "correct_chain"], default="pre_and_correct")
    ap.add_argument("--cv_group", choices=["chain", "problem"], default="chain")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--residual_ridge", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--min_feature_coverage", type=float, default=0.70)
    ap.add_argument("--min_unique", type=int, default=3)
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--scale_floor", type=float, default=1e-4)
    ap.add_argument("--std_floor_frac", type=float, default=0.10)
    ap.add_argument("--response_topk", type=int, default=2)
    ap.add_argument("--include_text_complexity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--entropy_as_geometry", action="store_true", help="also treat entropy dynamics as geometry features")
    ap.add_argument("--require_geometry", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include_prefix_flow", action="store_true")
    ap.add_argument("--require_prefix_flow", action="store_true")
    ap.add_argument("--hidden_dir", default="")
    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--prefix_rank", type=int, default=8)
    ap.add_argument("--prefix_beta", type=float, default=1.0)
    ap.add_argument("--prefix_raw_hidden", action="store_true")
    ap.add_argument("--prefix_center_chain", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefix_z_top_k", type=int, default=16)
    ap.add_argument("--prefix_z_thresh", type=float, default=3.0)
    ap.add_argument("--prefix_std_floor_frac", type=float, default=0.10)
    ap.add_argument("--prefix_backend", default="auto", choices=["auto", "cpu", "torch", "cuda"])
    ap.add_argument("--prefix_device", default="")
    ap.add_argument("--prefix_dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--output_dir", default="outputs/reasoning_state_hazard")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "reasoning_state_hazard_selftest.npz")
            make_selftest(path, args.seed)
            args.input = path
            args.no_progress = True
            args.bootstrap = min(args.bootstrap, 100)
            res = run(path, args)
            assert_selftest(res)
            print_result(res)
            print("selftest passed")
            return
    if not args.input:
        raise SystemExit("--input is required unless --selftest is used")
    res = run(args.input, args)
    print_result(res)
    stem = os.path.splitext(os.path.basename(args.input))[0] + f"_L{res['meta']['layer_used']}_reasoning_state_hazard"
    jpath, mpath, cpath = write_outputs(res, args.output_dir, stem)
    print(f"\nsaved: {jpath}")
    print(f"saved: {mpath}")
    print(f"saved: {cpath}")


if __name__ == "__main__":
    main()
