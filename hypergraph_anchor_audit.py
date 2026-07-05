"""Prompt-anchor geometry / hypergraph audit over token hidden matrices.

This script is deliberately non-parametric.  It does not assume a single
"correct reasoning" trajectory and it does not train a hypergraph model.  It
asks a smaller question:

  Does the response token stream locally disconnect from the prompt/question
  anchor, and can that disconnection recover step boundaries or first-error
  locations?

Construction for one chain/layer:
  nodes                 sliding response-token windows + one virtual prompt node
  temporal edges         adjacent windows
  hidden-neighbor edges  kNN in hidden-centroid cosine space
  prompt-anchor edges    window centroid to qvec similarity

Existing step labels are used only for evaluation, not for graph construction.

The optional hypergraph export follows the local `hypergraph-hallucination`
project's object schema:

  x, he_incidence_index, he_attr, he_mark, he_member_counts, y_token, response_idx

but replaces attention-derived hyperedges with hidden-geometry hyperedges.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hidden_io import _fn, load_chain
from audit_utils import auroc, bdir, finite_json, safe_mean, safe_std

try:  # Optional: remote GPU env usually has torch; local Windows env may not.
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None


EPS = 1e-9


@dataclass
class ChainResult:
    idx: int
    group: int
    gold: int
    correct: bool
    n_steps: int
    n_tokens: int
    step_features: Dict[str, np.ndarray]
    boundary_features: Dict[str, np.ndarray]
    boundary_pos: np.ndarray
    step_starts: np.ndarray
    hypergraph_stats: Dict[str, object]


def unit(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, float)
    return v / max(float(np.linalg.norm(v)), EPS)


def unit_rows(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, EPS)


def layer_index(have: Sequence[int], layer: int, *, nearest: bool = False) -> Optional[int]:
    vals = [int(x) for x in have]
    if layer in vals:
        return vals.index(layer)
    if nearest and vals:
        return int(np.argmin([abs(x - layer) for x in vals]))
    return None


def object_array(xs: Sequence[object]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.npz:
        npz = args.npz
        stem = os.path.splitext(os.path.basename(npz))[0].replace("full_", "")
    else:
        if not args.dataset:
            raise SystemExit("provide npz path or --dataset")
        stem = args.dataset
        npz = os.path.join(args.data_dir, "features", f"full_{stem}.npz")
    if args.hidden_dir:
        hidden = args.hidden_dir
    else:
        hidden = os.path.join(args.data_dir, "hidden", stem)
    return npz, hidden, stem


def hidden_candidates(hidden_dir: str, cid: object, idx: int, dataset: str) -> List[str]:
    vals = [
        os.path.join(hidden_dir, _fn(cid)),
        os.path.join(hidden_dir, f"{dataset}-{idx}.npy"),
        os.path.join(hidden_dir, f"{idx}.npy"),
    ]
    out = []
    for p in vals:
        if p not in out:
            out.append(p)
    return out


def load_hidden_any(hidden_dir: str, cid: object, idx: int, dataset: str) -> Optional[np.ndarray]:
    for p in hidden_candidates(hidden_dir, cid, idx, dataset):
        if os.path.exists(p):
            return np.load(p, mmap_mode="r")
    return None


def rel_ranges(abs_ranges: np.ndarray, n_tokens: int) -> np.ndarray:
    rr = np.asarray(abs_ranges, int)
    if rr.ndim != 2 or rr.shape[1] != 2:
        return np.zeros((0, 2), int)
    a0 = int(rr[0, 0])
    rel = np.zeros_like(rr)
    rel[:, 0] = np.clip(rr[:, 0] - a0, 0, n_tokens)
    rel[:, 1] = np.clip(rr[:, 1] - a0 + 1, 0, n_tokens)  # half-open
    rel[:, 1] = np.maximum(rel[:, 1], rel[:, 0])
    return rel


def make_windows(n_tokens: int, window: int, stride: int) -> List[Tuple[int, int]]:
    if n_tokens <= 0:
        return []
    if n_tokens <= window:
        return [(0, n_tokens)]
    starts = list(range(0, max(1, n_tokens - window + 1), stride))
    if starts[-1] + window < n_tokens:
        starts.append(n_tokens - window)
    return [(s, min(n_tokens, s + window)) for s in starts]


def window_cloud(H: np.ndarray, windows: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    U = unit_rows(H)
    centroids, spread = [], []
    for lo, hi in windows:
        seg = U[int(lo) : int(hi)]
        if len(seg) == 0:
            centroids.append(np.full(H.shape[1], np.nan))
            spread.append(np.nan)
            continue
        m = seg.mean(axis=0)
        r = float(np.linalg.norm(m))
        centroids.append(m / max(r, EPS))
        spread.append(1.0 - r)
    return np.asarray(centroids, float), np.asarray(spread, float)


def prompt_edge_strength(anchor_cos: np.ndarray, gamma: float) -> np.ndarray:
    # Map cosine to a non-negative edge weight.  This keeps the graph diagnostic
    # stable even when raw qvec cosine has model/layer-specific scale.
    a = np.clip((np.asarray(anchor_cos, float) + 1.0) / 2.0, 0.0, 1.0)
    return np.power(a, gamma)


def build_graph(
    C: np.ndarray,
    prompt_weight: np.ndarray,
    *,
    top_k: int,
    temporal_weight: float,
    nn_gamma: float,
) -> np.ndarray:
    n = len(C)
    W = np.zeros((n + 1, n + 1), float)
    if n == 0:
        return W
    sim = C @ C.T
    sim[~np.isfinite(sim)] = -1.0
    for i in range(n - 1):
        w = temporal_weight * max(0.0, float((sim[i, i + 1] + 1.0) / 2.0))
        W[i, i + 1] = W[i + 1, i] = max(W[i, i + 1], w)
    k = min(max(0, int(top_k)), max(0, n - 1))
    if k:
        for i in range(n):
            order = np.argsort(sim[i])[::-1]
            added = 0
            for j in order:
                if j == i or sim[i, j] <= 0:
                    continue
                w = float(sim[i, j] ** nn_gamma)
                W[i, j] = W[j, i] = max(W[i, j], w)
                added += 1
                if added >= k:
                    break
    prompt = n
    for i, w in enumerate(prompt_weight):
        if np.isfinite(w) and w > 0:
            W[i, prompt] = W[prompt, i] = float(w)
    return W


def _as_numpy(x):
    if torch is not None and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_tensor(x, *, dtype=None):
    if torch is None:
        arr = np.asarray(x)
        return arr.astype(dtype) if dtype is not None else arr
    if dtype in (np.int64, int, "long"):
        return torch.as_tensor(x, dtype=torch.long)
    return torch.as_tensor(x, dtype=torch.float32)


def add_hyperedge(
    node_lists: List[np.ndarray],
    attrs: List[List[float]],
    marks: List[List[float]],
    members: Sequence[int],
    weights: Sequence[float],
    *,
    kind: float,
    prompt_edge: bool,
) -> None:
    uniq = np.asarray(sorted(set(int(x) for x in members)), int)
    if len(uniq) < 2:
        return
    w = np.asarray(weights, float)
    w = w[np.isfinite(w)]
    mean_w = float(w.mean()) if len(w) else 0.0
    max_w = float(w.max()) if len(w) else mean_w
    node_lists.append(uniq)
    attrs.append([max(0.0, min(1.0, mean_w)), max(0.0, min(1.0, max_w)), max(0.0, min(1.0, kind))])
    marks.append([1.0, 0.0] if prompt_edge else [0.0, 1.0])


def project_hypergraph_object(
    C: np.ndarray,
    spread: np.ndarray,
    anchor: np.ndarray,
    prompt_ratio: np.ndarray,
    W: np.ndarray,
    windows: Sequence[Tuple[int, int]],
    rel: np.ndarray,
    boundary_features: Dict[str, np.ndarray],
    *,
    gold: int,
    correct: bool,
    top_k: int,
    min_prompt_edge: float,
    source_id: int,
) -> Dict[str, object]:
    """Build a hypergraph-hallucination-compatible object from hidden geometry.

    The local project constructs one hyperedge per response-token attention row.
    Here we use hidden geometry instead:
      - prompt-anchor hyperedges: virtual prompt node + one response window
      - temporal hyperedges: adjacent response windows
      - neighbor hyperedges: one window plus its strongest hidden-neighbor windows
    """
    n = len(C)
    prompt_node = 0
    response_idx = 1
    if n == 0:
        x = np.zeros((1, 8), np.float32)
        return {
            "source_id": int(source_id),
            "response_idx": int(response_idx),
            "token_ids": np.arange(1, dtype=np.int64),
            "x": _maybe_tensor(x),
            "he_incidence_index": _maybe_tensor(np.zeros((2, 0), np.int64), dtype=np.int64),
            "he_attr": _maybe_tensor(np.zeros((0, 3), np.float32)),
            "he_mark": _maybe_tensor(np.zeros((0, 2), np.float32)),
            "he_member_counts": _maybe_tensor(np.zeros((0,), np.float32)),
            "y_token": _maybe_tensor(np.zeros((1,), np.float32)),
        }

    deg = W[:n, :].sum(axis=1)
    mids = np.asarray([(a + b) / 2.0 for a, b in windows], float)
    denom = max(1.0, float(max(b for _, b in windows)))
    prev_jump = np.zeros(n, float)
    prev_break = np.zeros(n, float)
    if n > 1:
        hj = np.asarray(boundary_features.get("hidden_jump", np.zeros(n - 1)), float)
        bb = np.asarray(boundary_features.get("boundary_break", np.zeros(n - 1)), float)
        prev_jump[1:] = np.nan_to_num(hj, nan=0.0)
        prev_break[1:] = np.nan_to_num(bb, nan=0.0)

    prompt_x = np.array([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]], np.float32)
    node_x = np.column_stack(
        [
            np.zeros(n, float),  # prompt flag
            np.nan_to_num(anchor, nan=0.0),
            np.nan_to_num(prompt_ratio, nan=0.0),
            np.nan_to_num(spread, nan=0.0),
            deg / max(float(np.nanmax(deg)) if np.isfinite(deg).any() else 1.0, EPS),
            mids / denom,
            prev_jump,
            prev_break,
        ]
    ).astype(np.float32)
    x = np.vstack([prompt_x, node_x])

    node_lists: List[np.ndarray] = []
    attrs: List[List[float]] = []
    marks: List[List[float]] = []

    # Prompt-anchor hyperedges.  These are the hidden-geometry analogue of the
    # project's prompt->response attention hyperedges.
    for i in range(n):
        w = float(W[i, n])
        if np.isfinite(w) and w >= min_prompt_edge:
            add_hyperedge(
                node_lists,
                attrs,
                marks,
                [prompt_node, response_idx + i],
                [w],
                kind=0.0,
                prompt_edge=True,
            )

    # Temporal response hyperedges.
    for i in range(n - 1):
        w = float(W[i, i + 1])
        if np.isfinite(w) and w > 0:
            add_hyperedge(
                node_lists,
                attrs,
                marks,
                [response_idx + i, response_idx + i + 1],
                [w],
                kind=0.5,
                prompt_edge=False,
            )

    # Hidden-neighbor row hyperedges, mirroring attention-row hyperedges.
    k = min(max(1, int(top_k)), max(1, n - 1))
    simW = W[:n, :n].copy()
    np.fill_diagonal(simW, 0.0)
    for i in range(n):
        order = np.argsort(simW[i])[::-1]
        nbrs = [j for j in order if simW[i, j] > 0][:k]
        if not nbrs:
            continue
        members = [response_idx + i] + [response_idx + int(j) for j in nbrs]
        weights = [float(simW[i, j]) for j in nbrs]
        add_hyperedge(node_lists, attrs, marks, members, weights, kind=1.0, prompt_edge=False)

    if node_lists:
        he_nodes, he_ids = [], []
        for he_id, members in enumerate(node_lists):
            he_nodes.extend(members.tolist())
            he_ids.extend([he_id] * len(members))
        he_incidence_index = np.vstack([np.asarray(he_nodes, np.int64), np.asarray(he_ids, np.int64)])
        he_attr = np.asarray(attrs, np.float32)
        he_mark = np.asarray(marks, np.float32)
        he_member_counts = np.asarray([len(m) for m in node_lists], np.float32)
    else:
        he_incidence_index = np.zeros((2, 0), np.int64)
        he_attr = np.zeros((0, 3), np.float32)
        he_mark = np.zeros((0, 2), np.float32)
        he_member_counts = np.zeros((0,), np.float32)

    y_token = np.zeros(n + 1, np.float32)
    if (not correct) and 0 <= gold < len(rel):
        lo, hi = int(rel[gold, 0]), int(rel[gold, 1])
        for i, (a, b) in enumerate(windows):
            if min(hi, b) - max(lo, a) > 0:
                y_token[response_idx + i] = 1.0

    return {
        "source_id": int(source_id),
        "response_idx": int(response_idx),
        "token_ids": _maybe_tensor(np.arange(n + 1, dtype=np.int64), dtype=np.int64),
        "x": _maybe_tensor(x),
        "he_incidence_index": _maybe_tensor(he_incidence_index, dtype=np.int64),
        "he_attr": _maybe_tensor(he_attr),
        "he_mark": _maybe_tensor(he_mark),
        "he_member_counts": _maybe_tensor(he_member_counts),
        "y_token": _maybe_tensor(y_token),
        "window_ranges": np.asarray(windows, np.int64),
        "step_ranges_rel": np.asarray(rel, np.int64),
        "gold_step": int(gold),
        "schema_source": "adapted_from_hypergraph_hallucination",
    }


def hypergraph_schema_stats(obj: Dict[str, object]) -> Dict[str, object]:
    x = _as_numpy(obj["x"])
    he_index = _as_numpy(obj["he_incidence_index"])
    he_attr = _as_numpy(obj["he_attr"])
    he_mark = _as_numpy(obj["he_mark"])
    he_count = _as_numpy(obj["he_member_counts"])
    y = _as_numpy(obj["y_token"])
    valid = (
        x.ndim == 2
        and he_index.ndim == 2
        and he_index.shape[0] == 2
        and he_attr.ndim == 2
        and he_mark.ndim == 2
        and len(he_attr) == len(he_mark) == len(he_count)
        and len(y) == len(x)
        and (he_index.shape[1] == 0 or (he_index[0].min() >= 0 and he_index[0].max() < len(x)))
        and (he_index.shape[1] == 0 or (he_index[1].min() >= 0 and he_index[1].max() < len(he_attr)))
    )
    return {
        "valid": bool(valid),
        "n_nodes": int(x.shape[0]) if x.ndim == 2 else 0,
        "node_dim": int(x.shape[1]) if x.ndim == 2 else 0,
        "n_hyperedges": int(he_attr.shape[0]) if he_attr.ndim == 2 else 0,
        "he_attr_dim": int(he_attr.shape[1]) if he_attr.ndim == 2 and he_attr.size else (int(he_attr.shape[1]) if he_attr.ndim == 2 else 0),
        "n_incidence": int(he_index.shape[1]) if he_index.ndim == 2 else 0,
        "prompt_edges": int((he_mark[:, 0] > he_mark[:, 1]).sum()) if he_mark.ndim == 2 and len(he_mark) else 0,
        "response_edges": int((he_mark[:, 1] >= he_mark[:, 0]).sum()) if he_mark.ndim == 2 and len(he_mark) else 0,
        "positive_nodes": int(np.nan_to_num(y).sum()),
        "response_idx": int(obj.get("response_idx", 0)),
    }


def numpy_message_passing_smoke(obj: Dict[str, object]) -> Dict[str, object]:
    """Numpy equivalent of the project's node->hyperedge->node sanity path."""
    x = np.asarray(_as_numpy(obj["x"]), np.float32)
    he_index = np.asarray(_as_numpy(obj["he_incidence_index"]), np.int64)
    he_attr = np.asarray(_as_numpy(obj["he_attr"]), np.float32)
    he_count = np.asarray(_as_numpy(obj["he_member_counts"]), np.float32)
    if x.ndim != 2 or he_index.shape[1] == 0 or len(he_attr) == 0:
        return {"ok": False, "reason": "empty graph"}
    node_ids = he_index[0]
    he_ids = he_index[1]
    agg = np.zeros((len(he_attr), x.shape[1]), np.float32)
    np.add.at(agg, he_ids, x[node_ids])
    agg = agg / np.maximum(he_count[:, None], EPS)
    out = np.zeros_like(x)
    np.add.at(out, node_ids, agg[he_ids])
    deg = np.bincount(node_ids, minlength=len(x)).astype(np.float32)[:, None]
    out = out / np.maximum(deg, EPS)
    return {
        "ok": bool(np.isfinite(out).all()),
        "mean_update_norm": float(np.linalg.norm(out, axis=1).mean()),
        "max_update_norm": float(np.linalg.norm(out, axis=1).max()),
    }


def save_project_hypergraph(obj: Dict[str, object], path_no_ext: str) -> str:
    if torch is not None:
        path = f"{path_no_ext}.pt"
        torch.save(obj, path)
        return path
    path = f"{path_no_ext}.npz"
    payload = {}
    for k, v in obj.items():
        if isinstance(v, (str, int, float, bool)):
            payload[k] = np.asarray(v)
        else:
            payload[k] = _as_numpy(v)
    np.savez_compressed(path, **payload)
    return path


def local_boundary_scores(
    C: np.ndarray,
    anchor: np.ndarray,
    prompt_ratio: np.ndarray,
    W: np.ndarray,
    windows: Sequence[Tuple[int, int]],
    *,
    span: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n = len(C)
    if n < 2:
        return np.zeros(0, float), {
            "hidden_jump": np.zeros(0, float),
            "anchor_drop": np.zeros(0, float),
            "prompt_ratio_drop": np.zeros(0, float),
            "boundary_break": np.zeros(0, float),
        }
    pos = np.asarray([(windows[i][1] + windows[i + 1][0]) / 2.0 for i in range(n - 1)], float)
    hidden_jump = np.full(n - 1, np.nan)
    anchor_drop = np.full(n - 1, np.nan)
    prompt_drop = np.full(n - 1, np.nan)
    boundary_break = np.full(n - 1, np.nan)
    for i in range(n - 1):
        hidden_jump[i] = 1.0 - float(C[i] @ C[i + 1])
        anchor_drop[i] = float(anchor[i] - anchor[i + 1])
        prompt_drop[i] = float(prompt_ratio[i] - prompt_ratio[i + 1])
        left = np.arange(max(0, i - span + 1), i + 1)
        right = np.arange(i + 1, min(n, i + 1 + span))
        if len(left) == 0 or len(right) == 0:
            continue
        cross = float(W[np.ix_(left, right)].sum())
        vol_l = float(W[left, :].sum())
        vol_r = float(W[right, :].sum())
        bridge = cross / max(EPS, math.sqrt(vol_l * vol_r))
        boundary_break[i] = 1.0 - min(1.0, bridge)
    return pos, {
        "hidden_jump": hidden_jump,
        "anchor_drop": anchor_drop,
        "prompt_ratio_drop": prompt_drop,
        "boundary_break": boundary_break,
    }


def overlap_windows(step: Tuple[int, int], windows: Sequence[Tuple[int, int]]) -> np.ndarray:
    lo, hi = int(step[0]), int(step[1])
    idx = []
    for i, (a, b) in enumerate(windows):
        if min(hi, b) - max(lo, a) > 0:
            idx.append(i)
    return np.asarray(idx, int)


def nearest_boundary(boundary_pos: np.ndarray, token_pos: float) -> int:
    if len(boundary_pos) == 0:
        return -1
    return int(np.argmin(np.abs(boundary_pos - float(token_pos))))


def score_steps(
    rel: np.ndarray,
    windows: Sequence[Tuple[int, int]],
    anchor: np.ndarray,
    spread: np.ndarray,
    prompt_ratio: np.ndarray,
    boundary_pos: np.ndarray,
    boundary_features: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    T = len(rel)
    feats = {
        "geom_anchor_mean": np.full(T, np.nan),
        "geom_anchor_min": np.full(T, np.nan),
        "geom_anchor_loss": np.full(T, np.nan),
        "prompt_degree_ratio": np.full(T, np.nan),
        "prompt_degree_min": np.full(T, np.nan),
        "window_spread": np.full(T, np.nan),
        "entry_hidden_jump": np.full(T, np.nan),
        "entry_anchor_drop": np.full(T, np.nan),
        "entry_prompt_ratio_drop": np.full(T, np.nan),
        "entry_boundary_break": np.full(T, np.nan),
        "inside_boundary_break": np.full(T, np.nan),
    }
    for t in range(T):
        idx = overlap_windows((int(rel[t, 0]), int(rel[t, 1])), windows)
        if len(idx):
            feats["geom_anchor_mean"][t] = safe_mean(anchor[idx])
            feats["geom_anchor_min"][t] = float(np.nanmin(anchor[idx]))
            feats["geom_anchor_loss"][t] = 1.0 - feats["geom_anchor_mean"][t]
            feats["prompt_degree_ratio"][t] = safe_mean(prompt_ratio[idx])
            feats["prompt_degree_min"][t] = float(np.nanmin(prompt_ratio[idx]))
            feats["window_spread"][t] = safe_mean(spread[idx])
        if t > 0:
            bi = nearest_boundary(boundary_pos, float(rel[t, 0]))
            if bi >= 0:
                feats["entry_hidden_jump"][t] = boundary_features["hidden_jump"][bi]
                feats["entry_anchor_drop"][t] = boundary_features["anchor_drop"][bi]
                feats["entry_prompt_ratio_drop"][t] = boundary_features["prompt_ratio_drop"][bi]
                feats["entry_boundary_break"][t] = boundary_features["boundary_break"][bi]
        if len(boundary_pos):
            lo, hi = int(rel[t, 0]), int(rel[t, 1])
            m = (boundary_pos >= lo) & (boundary_pos <= hi)
            vals = boundary_features["boundary_break"][m]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                feats["inside_boundary_break"][t] = float(vals.max())
    return feats


def analyze_chain(
    H_all: np.ndarray,
    rel: np.ndarray,
    qv: np.ndarray,
    *,
    source_id: int,
    gold: int,
    correct: bool,
    layer_col: int,
    window: int,
    stride: int,
    top_k: int,
    temporal_weight: float,
    prompt_gamma: float,
    nn_gamma: float,
    boundary_span: int,
    min_prompt_edge: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, object], Dict[str, object]]:
    H = np.asarray(H_all[:, layer_col, :], np.float32)
    windows = make_windows(len(H), window, stride)
    C, spread = window_cloud(H, windows)
    q = unit(qv)
    anchor = C @ q
    pweight = prompt_edge_strength(anchor, prompt_gamma)
    W = build_graph(C, pweight, top_k=top_k, temporal_weight=temporal_weight, nn_gamma=nn_gamma)
    prompt_node = len(C)
    deg = W.sum(axis=1)
    prompt_ratio = W[:prompt_node, prompt_node] / np.maximum(deg[:prompt_node], EPS)
    bpos, bfeats = local_boundary_scores(
        C,
        anchor,
        prompt_ratio,
        W,
        windows,
        span=boundary_span,
    )
    sfeats = score_steps(rel, windows, anchor, spread, prompt_ratio, bpos, bfeats)
    hg = project_hypergraph_object(
        C,
        spread,
        anchor,
        prompt_ratio,
        W,
        windows,
        rel,
        bfeats,
        gold=gold,
        correct=correct,
        top_k=top_k,
        min_prompt_edge=min_prompt_edge,
        source_id=source_id,
    )
    stats = hypergraph_schema_stats(hg)
    smoke = numpy_message_passing_smoke(hg)
    stats["message_passing_smoke"] = smoke
    return sfeats, bfeats, bpos, np.asarray([r[0] for r in rel], float), hg, stats


def qvec_for_chain(z, i: int, layer: int) -> Optional[np.ndarray]:
    if "qvec" not in z.files:
        return None
    qv = np.asarray(z["qvec"], float)
    sv_layers = [int(x) for x in z["sv_layers"]] if "sv_layers" in z.files else []
    qi = layer_index(sv_layers, layer, nearest=True) if sv_layers else 0
    if qi is None:
        return None
    if qv.ndim == 3:
        return np.asarray(qv[i, qi], float)
    if qv.ndim == 2:
        return np.asarray(qv[qi], float)
    return None


def load_results(npz_path: str, hidden_dir: str, dataset: str, args: argparse.Namespace) -> Tuple[List[ChainResult], Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=True)
    if "hidden_layers" in z.files:
        hidden_layers = [int(x) for x in z["hidden_layers"]]
    else:
        hidden_layers = [10, 14, 18, 22]
    lc = layer_index(hidden_layers, args.layer)
    if lc is None:
        raise SystemExit(f"layer {args.layer} not found in hidden layers {hidden_layers}")
    ids = z["ids"] if "ids" in z.files else np.array([f"{dataset}-{i}" for i in range(len(z["gold_error_step"]))], dtype=object)
    groups = z["problem_ids"] if "problem_ids" in z.files else np.arange(len(ids))
    ges = np.asarray(z["gold_error_step"], int)
    ranges = z["step_token_ranges"]
    N = len(ges) if not args.max_chains else min(args.max_chains, len(ges))
    out: List[ChainResult] = []
    missing_hidden = 0
    missing_qvec = 0
    exported = 0
    export_dir = ""
    if args.export_hypergraphs:
        export_dir = args.export_dir or os.path.join(
            args.output_dir,
            f"project_hypergraphs_{dataset}_L{args.layer}",
        )
        if os.path.exists(export_dir) and args.overwrite_export:
            shutil.rmtree(export_dir)
        os.makedirs(export_dir, exist_ok=True)
    for i in range(N):
        H = load_hidden_any(hidden_dir, ids[i], i, dataset)
        if H is None:
            missing_hidden += 1
            continue
        qv = qvec_for_chain(z, i, args.layer)
        if qv is None:
            missing_qvec += 1
            continue
        rr = rel_ranges(np.asarray(ranges[i], int), int(H.shape[0]))
        if len(rr) == 0:
            continue
        correct = bool(ges[i] < 0)
        sfeats, bfeats, bpos, starts, hg, hg_stats = analyze_chain(
            H,
            rr,
            qv,
            source_id=i,
            gold=int(ges[i]),
            correct=correct,
            layer_col=lc,
            window=args.window,
            stride=args.stride,
            top_k=args.top_k,
            temporal_weight=args.temporal_weight,
            prompt_gamma=args.prompt_gamma,
            nn_gamma=args.nn_gamma,
            boundary_span=args.boundary_span,
            min_prompt_edge=args.min_prompt_edge,
        )
        if args.export_hypergraphs and (args.export_limit <= 0 or exported < args.export_limit):
            saved = save_project_hypergraph(hg, os.path.join(export_dir, f"hypergraph_{dataset}_{i:06d}"))
            hg_stats["saved_path"] = saved
            exported += 1
        out.append(
            ChainResult(
                idx=i,
                group=int(groups[i]),
                gold=int(ges[i]),
                correct=correct,
                n_steps=len(rr),
                n_tokens=int(H.shape[0]),
                step_features=sfeats,
                boundary_features=bfeats,
                boundary_pos=bpos,
                step_starts=starts,
                hypergraph_stats=hg_stats,
            )
        )
    meta = {
        "npz": npz_path,
        "hidden_dir": hidden_dir,
        "dataset": dataset,
        "layer": args.layer,
        "hidden_layers": hidden_layers,
        "n_seen": int(N),
        "n_loaded": int(len(out)),
        "missing_hidden": int(missing_hidden),
        "missing_qvec": int(missing_qvec),
        "export_hypergraphs": bool(args.export_hypergraphs),
        "export_dir": export_dir,
        "exported_hypergraphs": int(exported),
        "export_format": "pt" if torch is not None else "npz",
        "torch_available": bool(torch is not None),
        "window": args.window,
        "stride": args.stride,
        "top_k": args.top_k,
    }
    return out, meta


def flatten_labeled(chains: Sequence[ChainResult], names: Sequence[str], *, high_feature: Optional[str] = None, q: float = 0.70):
    vals = []
    if high_feature:
        for c in chains:
            v = c.step_features.get(high_feature)
            if v is None:
                continue
            for t in range(c.n_steps):
                if c.correct or t <= c.gold:
                    if np.isfinite(v[t]):
                        vals.append(v[t])
        threshold = float(np.quantile(vals, q)) if vals else float("inf")
    else:
        threshold = -float("inf")
    X, y, g, keys = [], [], [], []
    for c in chains:
        gate = c.step_features.get(high_feature, np.full(c.n_steps, np.inf)) if high_feature else np.full(c.n_steps, np.inf)
        for t in range(c.n_steps):
            if high_feature and (not np.isfinite(gate[t]) or gate[t] < threshold):
                continue
            if c.correct or t < c.gold:
                yy = 0
            elif t == c.gold:
                yy = 1
            else:
                continue
            X.append([c.step_features.get(nm, np.full(c.n_steps, np.nan))[t] for nm in names])
            y.append(yy)
            g.append(c.group)
            keys.append((c.idx, t))
    return np.asarray(X, float), np.asarray(y, int), np.asarray(g), keys


def feature_table(chains: Sequence[ChainResult], names: Sequence[str], *, high_feature: Optional[str] = None, top: int = 20):
    rows = []
    for nm in names:
        X, y, _, _ = flatten_labeled(chains, [nm], high_feature=high_feature)
        if X.size == 0:
            continue
        s = X[:, 0]
        m = np.isfinite(s)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        raw = auroc(s[m], y[m])
        rows.append(
            {
                "feature": nm,
                "auroc_bestdir": bdir(raw),
                "raw_auroc_high_is_error": raw,
                "mean_non_error": safe_mean(s[(y == 0) & m]),
                "mean_gold_error": safe_mean(s[(y == 1) & m]),
                "n": int(m.sum()),
                "err": int(y[m].sum()),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["auroc_bestdir"], nan=-1), reverse=True)
    return rows[:top]


def within_chain_rank(chains: Sequence[ChainResult], feature: str, sign: float):
    top1, exp1, pct = [], [], []
    for c in chains:
        if c.correct or c.gold < 0 or c.gold >= c.n_steps:
            continue
        v = c.step_features.get(feature)
        if v is None:
            continue
        s = sign * np.asarray(v, float)
        m = np.isfinite(s)
        m[np.arange(c.n_steps) > c.gold] = False
        if not m[c.gold] or m.sum() < 2:
            continue
        cand = s[m]
        better = int((cand > s[c.gold]).sum())
        top1.append(float(better == 0))
        exp1.append(1.0 / m.sum())
        pct.append(better / max(1, m.sum() - 1))
    return {
        "top1": safe_mean(top1),
        "expected_top1": safe_mean(exp1),
        "mean_pct": safe_mean(pct),
        "n": int(len(top1)),
        "sign": float(sign),
    }


def localization_table(chains: Sequence[ChainResult], names: Sequence[str], *, top: int):
    det = {r["feature"]: r for r in feature_table(chains, names, top=len(names))}
    rows = []
    for nm, r in det.items():
        sign = 1.0 if r["raw_auroc_high_is_error"] >= 0.5 else -1.0
        loc = within_chain_rank(chains, nm, sign)
        if loc["n"] > 0:
            rows.append({"feature": nm, **loc})
    rows.sort(key=lambda r: np.nan_to_num(r["top1"], nan=-1) - np.nan_to_num(r["expected_top1"], nan=0), reverse=True)
    return rows[:top]


def boundary_recovery(chains: Sequence[ChainResult], names: Sequence[str], *, tolerance: float):
    rows = []
    for nm in names:
        s_all, y_all = [], []
        for c in chains:
            scores = c.boundary_features.get(nm)
            if scores is None or len(scores) == 0:
                continue
            true = c.step_starts[1:]
            for pos, score in zip(c.boundary_pos, scores):
                if not np.isfinite(score):
                    continue
                lab = int(len(true) > 0 and np.min(np.abs(true - pos)) <= tolerance)
                s_all.append(score)
                y_all.append(lab)
        s = np.asarray(s_all, float)
        y = np.asarray(y_all, int)
        if len(s) < 30 or len(np.unique(y)) < 2:
            continue
        raw = auroc(s, y)
        rows.append(
            {
                "feature": nm,
                "auroc_bestdir": bdir(raw),
                "raw_auroc_high_is_boundary": raw,
                "mean_non_boundary": safe_mean(s[y == 0]),
                "mean_boundary": safe_mean(s[y == 1]),
                "n": int(len(s)),
                "boundary": int(y.sum()),
            }
        )
    rows.sort(key=lambda r: np.nan_to_num(r["auroc_bestdir"], nan=-1), reverse=True)
    return rows


def event_study(chains: Sequence[ChainResult], names: Sequence[str], *, window: int):
    out = {}
    for nm in names:
        rows = []
        for d in range(-window, window + 1):
            vals = []
            for c in chains:
                if c.correct or c.gold < 0:
                    continue
                t = c.gold + d
                if 0 <= t < c.n_steps and nm in c.step_features:
                    vals.append(c.step_features[nm][t])
            rows.append({"delta": d, "mean": safe_mean(vals), "std": safe_std(vals), "n": int(np.isfinite(vals).sum())})
        out[nm] = rows
    return out


def summarize_hypergraph_stats(chains: Sequence[ChainResult]) -> Dict[str, object]:
    stats = [c.hypergraph_stats for c in chains if c.hypergraph_stats]
    if not stats:
        return {}
    smoke_ok = [
        bool(s.get("message_passing_smoke", {}).get("ok", False))
        for s in stats
    ]
    return {
        "schema": "hypergraph-hallucination-compatible",
        "n_graphs": int(len(stats)),
        "valid_graphs": int(sum(bool(s.get("valid", False)) for s in stats)),
        "message_passing_smoke_ok": int(sum(smoke_ok)),
        "mean_nodes": safe_mean([s.get("n_nodes", np.nan) for s in stats]),
        "mean_hyperedges": safe_mean([s.get("n_hyperedges", np.nan) for s in stats]),
        "mean_incidence": safe_mean([s.get("n_incidence", np.nan) for s in stats]),
        "mean_prompt_edges": safe_mean([s.get("prompt_edges", np.nan) for s in stats]),
        "mean_response_edges": safe_mean([s.get("response_edges", np.nan) for s in stats]),
        "mean_positive_nodes": safe_mean([s.get("positive_nodes", np.nan) for s in stats]),
        "first_saved_path": next((s.get("saved_path") for s in stats if s.get("saved_path")), ""),
    }


def run(npz: str, hidden_dir: str, dataset: str, args: argparse.Namespace) -> Dict[str, object]:
    chains, meta = load_results(npz, hidden_dir, dataset, args)
    names = [
        "geom_anchor_mean",
        "geom_anchor_min",
        "geom_anchor_loss",
        "prompt_degree_ratio",
        "prompt_degree_min",
        "window_spread",
        "entry_hidden_jump",
        "entry_anchor_drop",
        "entry_prompt_ratio_drop",
        "entry_boundary_break",
        "inside_boundary_break",
    ]
    bnames = ["hidden_jump", "anchor_drop", "prompt_ratio_drop", "boundary_break"]
    res = {
        "meta": meta,
        "n_chains": len(chains),
        "n_error_chains": int(sum(not c.correct for c in chains)),
        "hypotheses": {
            "hidden_geometry_lookback": "response windows should remain geometrically connected to qvec if grounded",
            "hypergraph_boundary": "step/free boundaries should appear as local graph cuts or prompt-anchor drops",
            "small_data_policy": "no hypergraph neural model yet; use non-parametric graph diagnostics first",
        },
        "overall_features": feature_table(chains, names, top=args.top),
        "high_break_features": feature_table(chains, names, high_feature="entry_boundary_break", top=args.top),
        "localization": localization_table(chains, names, top=args.top),
        "boundary_recovery": boundary_recovery(chains, bnames, tolerance=args.tolerance),
        "event_study": event_study(chains, names[:8], window=args.event_window),
        "project_hypergraph": summarize_hypergraph_stats(chains),
    }
    return res


def print_rows(rows, *, label: str, key: str = "feature", n: int = 12) -> None:
    print(f"\n{label}:")
    for r in rows[:n]:
        if "err" in r:
            print(
                f"  {r[key]:24s} AUROC {r['auroc_bestdir']:.3f} "
                f"nonerr {r['mean_non_error']:+.3f} err {r['mean_gold_error']:+.3f} "
                f"n={r['n']} err={r['err']}"
            )
        else:
            print(
                f"  {r[key]:24s} AUROC {r['auroc_bestdir']:.3f} "
                f"nonbd {r['mean_non_boundary']:+.3f} bd {r['mean_boundary']:+.3f} "
                f"n={r['n']} bd={r['boundary']}"
            )


def print_result(res: Dict[str, object]) -> None:
    meta = res["meta"]
    print(f"\n===== hypergraph anchor audit | {os.path.basename(meta['npz'])} | L{meta['layer']} =====")
    print(
        f"chains {res['n_chains']} | error chains {res['n_error_chains']} | "
        f"window {meta['window']} stride {meta['stride']} top_k {meta['top_k']}"
    )
    if meta.get("missing_hidden") or meta.get("missing_qvec"):
        print(f"missing hidden {meta.get('missing_hidden')} | missing qvec {meta.get('missing_qvec')}")
    hg = res.get("project_hypergraph", {})
    if hg:
        print(
            "project hypergraph: "
            f"valid {hg.get('valid_graphs')}/{hg.get('n_graphs')} | "
            f"mp-smoke {hg.get('message_passing_smoke_ok')}/{hg.get('n_graphs')} | "
            f"mean nodes {hg.get('mean_nodes'):.1f} | mean hyperedges {hg.get('mean_hyperedges'):.1f} | "
            f"format {meta.get('export_format')}"
        )
        if meta.get("export_hypergraphs"):
            print(f"exported {meta.get('exported_hypergraphs')} hypergraphs to {meta.get('export_dir')}")
            if hg.get("first_saved_path"):
                print(f"first saved: {hg.get('first_saved_path')}")
    print_rows(res["overall_features"], label="Step/gold-error prompt-anchor scores")
    print_rows(res["high_break_features"], label="High-boundary-break subset scores")
    print_rows(res["boundary_recovery"], label="Boundary-free step-boundary recovery")
    print("\nWithin-chain first-error localization:")
    for r in res["localization"][:12]:
        gain = r["top1"] - r["expected_top1"]
        print(f"  {r['feature']:24s} top1 {r['top1']:.3f} exp {r['expected_top1']:.3f} gain {gain:+.3f} n={r['n']}")


def make_selftest(root: str, *, layer: int, seed: int = 5) -> Tuple[str, str]:
    rng = np.random.default_rng(seed)
    hidden = os.path.join(root, "hidden", "selftest")
    os.makedirs(hidden, exist_ok=True)
    layers = np.asarray([10, 14, 18, 22], int)
    lc = layer_index(layers, layer, nearest=True)
    d = 64
    ids, groups, gold, ranges, qvecs = [], [], [], [], []
    for i in range(80):
        q = unit(rng.normal(size=d))
        off = rng.normal(size=d)
        off = unit(off - np.dot(off, q) * q)
        T = int(rng.integers(5, 8))
        lens = rng.integers(18, 32, size=T)
        is_err = i % 3 == 0
        g = int(rng.integers(2, T - 1)) if is_err else -1
        dirs = []
        for t in range(T):
            if is_err and t >= g:
                mix = 0.35 if t == g else 0.15
            elif (not is_err) and t == 2 and i % 5 == 0:
                mix = 0.55  # healthy exploration, then recovery
            else:
                mix = 0.82
            direction = unit(mix * q + (1.0 - mix) * off + 0.05 * rng.normal(size=d))
            dirs.append(direction)
        toks = []
        for t, n in enumerate(lens):
            for _ in range(int(n)):
                toks.append(dirs[t] + 0.12 * rng.normal(size=d))
        H = np.zeros((len(toks), len(layers), d), np.float32)
        base = np.asarray(toks, np.float32)
        for j in range(len(layers)):
            noise = 0.03 * rng.normal(size=base.shape)
            H[:, j, :] = base + noise
        cid = f"selftest-{i}"
        np.save(os.path.join(hidden, _fn(cid)), H)
        lo = np.cumsum(np.r_[0, lens[:-1]]) + 100
        rr = np.stack([lo, lo + lens - 1], axis=1).astype(int)
        ids.append(cid)
        groups.append(i)
        gold.append(g)
        ranges.append(rr)
        q_layer = np.zeros((len(layers), d), np.float32)
        for j in range(len(layers)):
            q_layer[j] = q + 0.01 * rng.normal(size=d)
        qvecs.append(q_layer)
    npz = os.path.join(root, "hypergraph_anchor_selftest.npz")
    np.savez_compressed(
        npz,
        ids=np.asarray(ids, dtype=object),
        problem_ids=np.asarray(groups, int),
        gold_error_step=np.asarray(gold, int),
        step_token_ranges=object_array(ranges),
        hidden_stored=np.array(True),
        hidden_dir=np.asarray(hidden),
        hidden_layers=layers,
        sv_layers=layers,
        qvec=np.asarray(qvecs, np.float32),
    )
    return npz, hidden


def assert_selftest(res: Dict[str, object]) -> None:
    rows = {r["feature"]: r for r in res["overall_features"]}
    if rows.get("geom_anchor_loss", {}).get("auroc_bestdir", 0.0) < 0.75:
        raise SystemExit("selftest failed: geom_anchor_loss did not recover injected prompt drift")
    loc = {r["feature"]: r for r in res["localization"]}
    if loc.get("geom_anchor_loss", {}).get("top1", 0.0) < 0.60:
        raise SystemExit("selftest failed: geom_anchor_loss did not localize injected first errors")
    brec = {r["feature"]: r for r in res["boundary_recovery"]}
    if brec.get("hidden_jump", {}).get("auroc_bestdir", 0.0) < 0.60:
        raise SystemExit("selftest failed: hidden_jump did not recover synthetic step boundaries")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt-anchor geometry / hypergraph audit")
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--dataset", choices=["gsm8k", "math", "omnimath"], default=None)
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--hidden_dir", default=None)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--temporal_weight", type=float, default=0.75)
    ap.add_argument("--prompt_gamma", type=float, default=2.0)
    ap.add_argument("--nn_gamma", type=float, default=2.0)
    ap.add_argument("--min_prompt_edge", type=float, default=0.05)
    ap.add_argument("--boundary_span", type=int, default=2)
    ap.add_argument("--tolerance", type=float, default=10.0)
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--output_dir", default="outputs/hypergraph_anchor")
    ap.add_argument("--export_hypergraphs", action="store_true")
    ap.add_argument("--export_dir", default="")
    ap.add_argument("--export_limit", type=int, default=0, help="0 means export all loaded chains")
    ap.add_argument("--overwrite_export", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            npz, hidden = make_selftest(td, layer=args.layer)
            res = run(npz, hidden, "selftest", args)
            assert_selftest(res)
            print_result(res)
            os.makedirs(args.output_dir, exist_ok=True)
            out_file = os.path.join(args.output_dir, f"selftest_L{args.layer}.json")
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(finite_json(res), fh, indent=2, ensure_ascii=False)
            print(f"\nselftest passed; saved: {out_file}")
        return

    npz, hidden, stem = resolve_paths(args)
    res = run(npz, hidden, stem, args)
    print_result(res)
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = stem
    if args.max_chains:
        out_name += f"_n{args.max_chains}"
    out_file = os.path.join(args.output_dir, f"{out_name}_L{args.layer}.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(finite_json(res), fh, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_file}")


if __name__ == "__main__":
    main()
