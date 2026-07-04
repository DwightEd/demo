#!/usr/bin/env python3
"""Full token-level hypergraph training over multi-layer reasoning hidden states.

This is the "complete" counterpart to hypergraph_anchor_audit.py.  The audit
script builds non-parametric graph diagnostics; this script trains a real
HyperCHARM-style hypergraph neural model.

Relation to the local hypergraph-hallucination project:
  - processed_hypergraph.py builds token nodes and attention-row hyperedges.
  - train_hypergraph.py trains a token/node classifier with HyperCHARM:
      node -> hyperedge aggregation -> hyperedge-conditioned node update,
      BCEWithLogitsLoss on response nodes, pos_weight for class imbalance.

Here we keep that training objective and message passing mechanism, but replace
attention-derived hyperedges with hidden-geometry hyperedges over our reasoning
hidden shards:
  nodes      = one virtual prompt node + every response token
  x          = selected multi-layer token hidden states, optionally plus
               interpretable diagnostic channels
  hyperedges = prompt-anchor, causal temporal, and hidden-neighbor hyperedges
  y_token    = tokens inside the gold first-error step; post-error tokens are
               masked out of loss/evaluation

The model predicts token/window risk online from prefix-causal graph structure;
metrics are reported at token level and after aggregation back to reasoning
steps.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
except ImportError:  # pragma: no cover - local Windows may not have torch
    torch = None
    nn = None
    F = None
    AdamW = None

try:
    if torch is not None:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    else:  # pragma: no cover
        Data = None
        DataLoader = None
except ImportError:  # pragma: no cover - remote GPU env should have PyG
    Data = None
    DataLoader = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
except ImportError as exc:  # pragma: no cover
    raise SystemExit("hypergraph_token_hgn.py needs scikit-learn") from exc

from hidden_io import _fn


EPS = 1e-9


def parse_layers(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def finite_json(obj):
    if isinstance(obj, dict):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def require_torch() -> None:
    if torch is None:
        raise SystemExit("This script needs torch for training. Run it on the GPU server.")
    if Data is None or DataLoader is None:
        raise SystemExit("This script needs torch_geometric for HyperGraphData batching.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unit_rows(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def layer_indices(have: Sequence[int], want: Sequence[int]) -> List[int]:
    have = [int(x) for x in have]
    out = []
    for w in want:
        if int(w) not in have:
            raise SystemExit(f"layer {w} not found in hidden layers {have}")
        out.append(have.index(int(w)))
    return out


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
    rel[:, 1] = np.clip(rr[:, 1] - a0 + 1, 0, n_tokens)
    rel[:, 1] = np.maximum(rel[:, 1], rel[:, 0])
    return rel


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.npz:
        npz = args.npz
        stem = os.path.splitext(os.path.basename(npz))[0].replace("full_", "")
    else:
        if not args.dataset:
            raise SystemExit("provide npz path or --dataset")
        stem = args.dataset
        npz = os.path.join(args.data_dir, "features", f"full_{stem}.npz")
    hidden = args.hidden_dir or os.path.join(args.data_dir, "hidden", stem)
    return npz, hidden, stem


def nearest_layer_index(have: Sequence[int], want: int) -> int:
    vals = [int(x) for x in have]
    if int(want) in vals:
        return vals.index(int(want))
    return int(np.argmin([abs(x - int(want)) for x in vals]))


def qvec_for_chain(z, i: int, layers: Sequence[int], *, allow_missing: bool) -> Optional[np.ndarray]:
    if "qvec" not in z.files:
        if allow_missing:
            return None
        raise SystemExit("qvec missing; pass --allow_missing_qvec to train without prompt-anchor edges")
    qv = np.asarray(z["qvec"], np.float32)
    sv_layers = [int(x) for x in z["sv_layers"]] if "sv_layers" in z.files else [int(x) for x in layers]
    q_cols = [nearest_layer_index(sv_layers, int(layer)) for layer in layers]
    if qv.ndim == 3:
        return np.asarray(qv[i, q_cols, :], np.float32)
    if qv.ndim == 2:
        return np.asarray(qv[q_cols, :], np.float32)
    if allow_missing:
        return None
    raise SystemExit(f"unsupported qvec shape {qv.shape}")


def step_token_counts(ranges: np.ndarray, gold: int) -> Tuple[int, int]:
    rel = np.asarray(ranges, int)
    if len(rel) == 0:
        return 0, 0
    if gold < 0:
        return 0, int(rel[-1, 1])
    if gold >= len(rel):
        return 0, int(rel[-1, 1])
    pos = int(rel[gold, 1] - rel[gold, 0])
    total = int(rel[gold, 1])
    return max(0, pos), max(0, total)


def local_spread(U: np.ndarray, radius: int) -> np.ndarray:
    # U: (R, L, D), unit-normalized.  Return mean local 1-resultant across layers.
    R, L, _ = U.shape
    out = np.zeros((R, L), np.float32)
    for t in range(R):
        lo = max(0, t - radius)
        hi = min(R, t + radius + 1)
        m = U[lo:hi].mean(axis=0)
        out[t] = 1.0 - np.linalg.norm(m, axis=-1)
    return out


def hidden_jump(Uflat: np.ndarray) -> np.ndarray:
    R = len(Uflat)
    out = np.zeros(R, np.float32)
    if R > 1:
        out[1:] = 1.0 - np.sum(Uflat[1:] * Uflat[:-1], axis=1)
    return out


def prompt_edge_strength(anchor: np.ndarray, gamma: float) -> np.ndarray:
    a = np.clip((anchor + 1.0) / 2.0, 0.0, 1.0)
    return np.power(a, gamma)


def add_hyperedge(
    node_lists: List[np.ndarray],
    attrs: List[List[float]],
    marks: List[List[float]],
    members: Sequence[int],
    weights: Sequence[float],
    *,
    kind: float,
    prompt_edge: bool,
    age: float = 0.0,
) -> None:
    uniq = np.asarray(sorted(set(int(x) for x in members)), dtype=np.int64)
    if len(uniq) < 2:
        return
    w = np.asarray(weights, np.float32)
    w = w[np.isfinite(w)]
    mean_w = float(w.mean()) if len(w) else 0.0
    max_w = float(w.max()) if len(w) else mean_w
    node_lists.append(uniq)
    attrs.append(
        [
            max(0.0, min(1.0, mean_w)),
            max(0.0, min(1.0, max_w)),
            max(0.0, min(1.0, float(kind))),
            max(0.0, min(1.0, float(age))),
        ]
    )
    marks.append([1.0, 0.0] if prompt_edge else [0.0, 1.0])


def build_node_features(
    H: np.ndarray,
    q: Optional[np.ndarray],
    *,
    x_mode: str,
    hidden_form: str,
    local_radius: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    H = np.asarray(H, np.float32)
    R, L, D = H.shape
    U = unit_rows(H)
    Uflat = U.reshape(R, L * D) / math.sqrt(max(1, L))
    if q is None:
        qU = np.zeros((L, D), np.float32)
        q_raw = np.zeros((L, D), np.float32)
    else:
        q_raw = np.asarray(q, np.float32)
        qU = unit_rows(q_raw)
    qflat = qU.reshape(L * D) / math.sqrt(max(1, L))
    raw_flat = H.reshape(R, L * D) / math.sqrt(max(1, L))
    q_raw_flat = q_raw.reshape(L * D) / math.sqrt(max(1, L))

    anchor_by_layer = np.einsum("rld,ld->rl", U, qU)
    spread_by_layer = local_spread(U, local_radius)
    norms = np.log(np.maximum(np.linalg.norm(H, axis=-1), EPS))
    jump = hidden_jump(Uflat)
    pos = np.linspace(0.0, 1.0, R, dtype=np.float32) if R > 1 else np.zeros(R, np.float32)

    diag = np.column_stack(
        [
            np.zeros(R, np.float32),  # prompt flag
            anchor_by_layer.mean(axis=1),
            anchor_by_layer.min(axis=1),
            anchor_by_layer.std(axis=1),
            spread_by_layer.mean(axis=1),
            spread_by_layer.max(axis=1),
            norms.mean(axis=1),
            norms.std(axis=1),
            jump,
            pos,
        ]
    ).astype(np.float32)
    prompt_diag = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], np.float32)

    if hidden_form == "unit":
        hidden_resp = Uflat
        hidden_prompt = qflat[None, :]
    elif hidden_form == "raw":
        hidden_resp = raw_flat
        hidden_prompt = q_raw_flat[None, :]
    elif hidden_form == "both":
        hidden_resp = np.column_stack([raw_flat, Uflat])
        hidden_prompt = np.column_stack([q_raw_flat[None, :], qflat[None, :]])
    else:
        raise SystemExit(f"unknown --hidden_form {hidden_form}")

    if x_mode == "hidden":
        x_resp = hidden_resp.astype(np.float32)
        x_prompt = hidden_prompt.astype(np.float32)
    elif x_mode == "diag":
        x_resp = diag
        x_prompt = prompt_diag
    elif x_mode == "hidden_diag":
        x_resp = np.column_stack([hidden_resp, diag]).astype(np.float32)
        x_prompt = np.column_stack([hidden_prompt, prompt_diag]).astype(np.float32)
    else:
        raise SystemExit(f"unknown --x_mode {x_mode}")

    stats = {
        "anchor_mean": anchor_by_layer.mean(axis=1).astype(np.float32),
        "anchor_min": anchor_by_layer.min(axis=1).astype(np.float32),
        "spread": spread_by_layer.mean(axis=1).astype(np.float32),
        "jump": jump.astype(np.float32),
    }
    return np.vstack([x_prompt, x_resp]).astype(np.float32), stats, Uflat.astype(np.float32)


def build_hyperedges(
    sim_repr: np.ndarray,
    token_stats: Dict[str, np.ndarray],
    *,
    top_k: int,
    temporal_radius: int,
    causal: bool,
    prompt_gamma: float,
    min_prompt_edge: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = len(sim_repr)
    node_lists: List[np.ndarray] = []
    attrs: List[List[float]] = []
    marks: List[List[float]] = []
    prompt_node = 0
    response_offset = 1

    pweight = prompt_edge_strength(token_stats["anchor_mean"], prompt_gamma)
    for i, w in enumerate(pweight):
        if np.isfinite(w) and float(w) >= min_prompt_edge:
            add_hyperedge(
                node_lists,
                attrs,
                marks,
                [prompt_node, response_offset + i],
                [float(w)],
                kind=0.0,
                prompt_edge=True,
                age=0.0,
            )

    for i in range(R):
        lo = max(0, i - temporal_radius)
        hi = i + 1 if causal else min(R, i + temporal_radius + 1)
        members = [response_offset + j for j in range(lo, hi)]
        if len(members) >= 2:
            weights = [1.0 / max(1, i - j + 1) for j in range(lo, hi)]
            add_hyperedge(
                node_lists,
                attrs,
                marks,
                members,
                weights,
                kind=0.33,
                prompt_edge=False,
                age=min(1.0, (hi - lo) / max(1, R)),
            )

    if R > 1 and top_k > 0:
        sim = sim_repr @ sim_repr.T
        sim[~np.isfinite(sim)] = -1.0
        for i in range(R):
            if causal:
                candidates = np.arange(0, i, dtype=int)
            else:
                candidates = np.asarray([j for j in range(R) if j != i], dtype=int)
            if len(candidates) == 0:
                continue
            vals = sim[i, candidates]
            order = candidates[np.argsort(vals)[::-1]]
            nbrs = [int(j) for j in order if sim[i, j] > 0][:top_k]
            if not nbrs:
                continue
            members = [response_offset + i] + [response_offset + j for j in nbrs]
            weights = [float(max(0.0, sim[i, j])) for j in nbrs]
            max_age = max((i - j for j in nbrs), default=0) if causal else 0
            add_hyperedge(
                node_lists,
                attrs,
                marks,
                members,
                weights,
                kind=1.0,
                prompt_edge=False,
                age=min(1.0, max_age / max(1, R)),
            )

    if node_lists:
        nodes, he_ids = [], []
        for hid, members in enumerate(node_lists):
            nodes.extend(members.tolist())
            he_ids.extend([hid] * len(members))
        he_index = np.vstack([np.asarray(nodes, np.int64), np.asarray(he_ids, np.int64)])
        he_attr = np.asarray(attrs, np.float32)
        he_mark = np.asarray(marks, np.float32)
        he_count = np.asarray([len(m) for m in node_lists], np.float32)
    else:
        he_index = np.zeros((2, 0), np.int64)
        he_attr = np.zeros((0, 4), np.float32)
        he_mark = np.zeros((0, 2), np.float32)
        he_count = np.zeros((0,), np.float32)
    return he_index, he_attr, he_mark, he_count


def build_labels(rel: np.ndarray, gold: int, n_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.zeros(n_tokens + 1, np.float32)
    mask = np.zeros(n_tokens + 1, np.bool_)
    if n_tokens > 0:
        mask[1:] = True
    if gold >= 0 and gold < len(rel):
        lo, hi = int(rel[gold, 0]), int(rel[gold, 1])
        y[1 + lo : 1 + hi] = 1.0
        # For first-error localization, post-error tokens are not clean negatives.
        mask[1 + hi :] = False
    return y, mask


if torch is not None and Data is not None:

    class HyperGraphData(Data):
        @property
        def num_hedges(self) -> int:
            return int(self.he_attr.size(0)) if hasattr(self, "he_attr") and self.he_attr is not None else 0

        def __inc__(self, key, value, *args, **kwargs):
            if key == "he_index":
                return torch.tensor([[self.num_nodes], [self.num_hedges]], dtype=torch.long)
            return super().__inc__(key, value, *args, **kwargs)

        def __cat_dim__(self, key, value, *args, **kwargs):
            if key == "he_index":
                return 1
            return super().__cat_dim__(key, value, *args, **kwargs)


    def to_data(obj: Dict[str, np.ndarray], *, normalize_x: bool) -> HyperGraphData:
        x = torch.as_tensor(obj["x"], dtype=torch.float32)
        if normalize_x and x.numel() > 0:
            prompt = x[:1].clone()
            resp = x[1:]
            if len(resp):
                resp = torch.clamp(resp, -5.0, 5.0)
                resp = (resp - resp.mean(dim=0, keepdim=True)) / (resp.std(dim=0, keepdim=True) + 1e-6)
            x = torch.cat([prompt, resp], dim=0)
        he_attr = torch.as_tensor(obj["he_attr"], dtype=torch.float32)
        if he_attr.numel() > 0:
            he_attr = torch.clamp(he_attr, 0.0, 1.0)
        return HyperGraphData(
            x=x,
            he_index=torch.as_tensor(obj["he_index"], dtype=torch.long),
            he_attr=he_attr,
            he_mark=torch.as_tensor(obj["he_mark"], dtype=torch.float32),
            he_count=torch.as_tensor(obj["he_count"], dtype=torch.float32),
            y=torch.as_tensor(obj["y"], dtype=torch.float32),
            loss_mask=torch.as_tensor(obj["loss_mask"], dtype=torch.bool),
            node_pos=torch.arange(len(obj["y"]), dtype=torch.long),
            response_idx=torch.tensor(1, dtype=torch.long),
            step_ranges=torch.as_tensor(obj["step_ranges"], dtype=torch.long),
            gold_step=torch.tensor(int(obj["gold_step"]), dtype=torch.long),
            chain_idx=torch.tensor(int(obj["chain_idx"]), dtype=torch.long),
            group=torch.tensor(int(obj["group"]), dtype=torch.long),
        )


    def make_mlp(in_dim, hidden_dims, out_dim, activation=nn.ReLU, dropout=0.0):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)


    class HyperCharmLayer(nn.Module):
        """Same node->hyperedge->node mechanism as train_hypergraph.py."""

        def __init__(self, node_dim: int, hedge_dim: int, hidden_dim: int, residual: bool = True):
            super().__init__()
            self.residual = residual
            self.node2edge = make_mlp(node_dim + 2, [hidden_dim], hidden_dim)
            self.edge2node = make_mlp(hedge_dim + hidden_dim, [hidden_dim], node_dim)
            self.ln_out = nn.LayerNorm(node_dim)

        def forward(self, x, he_index, he_attr, he_mark, he_count):
            node_ids = he_index[0]
            he_ids = he_index[1]
            msg_ne = self.node2edge(torch.cat([x[node_ids], he_mark[he_ids]], dim=-1))
            agg_e = torch.zeros((he_attr.size(0), msg_ne.size(-1)), device=x.device)
            agg_e.index_add_(0, he_ids, msg_ne)
            agg_e = agg_e / (he_count.unsqueeze(-1).to(x.device) + 1e-6)

            inc_msg = self.edge2node(torch.cat([he_attr[he_ids], agg_e[he_ids]], dim=-1))
            inc_msg = F.relu(inc_msg)

            out = torch.zeros_like(x)
            out.index_add_(0, node_ids, inc_msg)
            node_deg = torch.bincount(node_ids, minlength=x.size(0)).float().unsqueeze(-1).to(x.device)
            out = out / (node_deg + 1e-6)
            out = self.ln_out(out)
            return x + out if self.residual else out


    class HyperCHARMNode(nn.Module):
        def __init__(self, node_dim: int, hedge_dim: int, hidden_dim: int, gnn_layers: int, dropout: float, residual_mp: bool):
            super().__init__()
            self.in_proj = nn.Linear(node_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [
                    HyperCharmLayer(hidden_dim, hedge_dim, hidden_dim, residual=residual_mp)
                    for _ in range(gnn_layers)
                ]
            )
            self.pred = nn.Sequential(
                nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                nn.LayerNorm(max(8, hidden_dim // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(8, hidden_dim // 2), 1),
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, data):
            h = F.relu(self.in_proj(data.x.float()))
            for layer in self.layers:
                h = layer(h, data.he_index.long(), data.he_attr.float(), data.he_mark.float(), data.he_count.float())
            return self.pred(h).view(-1)


else:
    HyperGraphData = object
    HyperCHARMNode = object


@dataclass
class GraphMeta:
    chain_idx: int
    group: int
    gold_step: int
    n_tokens: int
    n_steps: int


class ReasoningHypergraphDataset:
    def __init__(
        self,
        npz_path: str,
        hidden_dir: str,
        dataset: str,
        indices: Sequence[int],
        *,
        layers: Sequence[int],
        x_mode: str,
        hidden_form: str,
        local_radius: int,
        top_k: int,
        temporal_radius: int,
        causal: bool,
        prompt_gamma: float,
        min_prompt_edge: float,
        allow_missing_qvec: bool,
        normalize_x: bool,
    ):
        self.z = np.load(npz_path, allow_pickle=True)
        self.hidden_dir = hidden_dir
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        hidden_layers = [int(x) for x in self.z["hidden_layers"]] if "hidden_layers" in self.z.files else [10, 14, 18, 22]
        self.layer_cols = layer_indices(hidden_layers, layers)
        self.layers = [hidden_layers[i] for i in self.layer_cols]
        self.x_mode = x_mode
        self.hidden_form = hidden_form
        self.local_radius = int(local_radius)
        self.top_k = int(top_k)
        self.temporal_radius = int(temporal_radius)
        self.causal = bool(causal)
        self.prompt_gamma = float(prompt_gamma)
        self.min_prompt_edge = float(min_prompt_edge)
        self.allow_missing_qvec = bool(allow_missing_qvec)
        self.normalize_x = bool(normalize_x)
        self.ids = self.z["ids"] if "ids" in self.z.files else np.array([f"{dataset}-{i}" for i in range(len(self.z["gold_error_step"]))], dtype=object)
        self.groups = self.z["problem_ids"] if "problem_ids" in self.z.files else np.arange(len(self.ids))
        self.gold = np.asarray(self.z["gold_error_step"], int)
        self.ranges = self.z["step_token_ranges"]

    def __len__(self) -> int:
        return len(self.indices)

    def graph_label(self, local_idx: int) -> int:
        i = self.indices[local_idx]
        return int(self.gold[i] >= 0)

    def meta(self, local_idx: int) -> GraphMeta:
        i = self.indices[local_idx]
        H = load_hidden_any(self.hidden_dir, self.ids[i], i, self.dataset)
        n_tokens = int(H.shape[0]) if H is not None else 0
        rr = rel_ranges(np.asarray(self.ranges[i], int), n_tokens)
        return GraphMeta(int(i), int(self.groups[i]), int(self.gold[i]), n_tokens, len(rr))

    def label_counts(self) -> Tuple[int, int]:
        pos = 0
        total = 0
        for i in self.indices:
            # Count labels from ranges only; no need to materialize hidden.
            rr_abs = np.asarray(self.ranges[i], int)
            if rr_abs.ndim != 2 or len(rr_abs) == 0:
                continue
            n_tokens = int(rr_abs[-1, 1] - rr_abs[0, 0] + 1)
            rr = rel_ranges(rr_abs, n_tokens)
            p, t = step_token_counts(rr, int(self.gold[i]))
            pos += p
            total += t
        return int(pos), int(max(0, total - pos))

    def __getitem__(self, local_idx: int):
        require_torch()
        i = self.indices[local_idx]
        H_all = load_hidden_any(self.hidden_dir, self.ids[i], i, self.dataset)
        if H_all is None:
            raise FileNotFoundError(f"missing hidden shard for {self.ids[i]} index {i}")
        H = np.asarray(H_all[:, self.layer_cols, :], np.float32)
        q = qvec_for_chain(self.z, i, self.layers, allow_missing=self.allow_missing_qvec)
        rr = rel_ranges(np.asarray(self.ranges[i], int), int(H.shape[0]))
        x, token_stats, sim_repr = build_node_features(
            H,
            q,
            x_mode=self.x_mode,
            hidden_form=self.hidden_form,
            local_radius=self.local_radius,
        )
        he_index, he_attr, he_mark, he_count = build_hyperedges(
            sim_repr,
            token_stats,
            top_k=self.top_k,
            temporal_radius=self.temporal_radius,
            causal=self.causal,
            prompt_gamma=self.prompt_gamma,
            min_prompt_edge=self.min_prompt_edge,
        )
        y, loss_mask = build_labels(rr, int(self.gold[i]), int(H.shape[0]))
        obj = {
            "x": x,
            "he_index": he_index,
            "he_attr": he_attr,
            "he_mark": he_mark,
            "he_count": he_count,
            "y": y,
            "loss_mask": loss_mask,
            "step_ranges": rr,
            "gold_step": int(self.gold[i]),
            "chain_idx": int(i),
            "group": int(self.groups[i]),
        }
        return to_data(obj, normalize_x=self.normalize_x)


def auroc_safe(y, p) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    m = np.isfinite(p)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], p[m]))


def aupr_safe(y, p) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    m = np.isfinite(p)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(average_precision_score(y[m], p[m]))


def bdir(a: float) -> float:
    return float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan")


def evaluate_loader(model, loader, device: str) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prob = torch.sigmoid(model(batch))
            mask = batch.loss_mask.bool()
            ys.append(batch.y[mask].detach().cpu().numpy())
            ps.append(prob[mask].detach().cpu().numpy())
    if not ys:
        return {"node_auroc": float("nan"), "node_aupr": float("nan")}
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {"node_auroc": auroc_safe(y, p), "node_aupr": aupr_safe(y, p)}


def step_scores_for_graph(data, prob: np.ndarray, *, pool: str) -> Tuple[np.ndarray, np.ndarray]:
    rr = data.step_ranges.detach().cpu().numpy().astype(int)
    gold = int(data.gold_step.detach().cpu().item())
    scores, labels = [], []
    for t, (lo, hi) in enumerate(rr):
        if hi <= lo:
            continue
        if gold >= 0 and t > gold:
            continue
        vals = prob[1 + int(lo) : 1 + int(hi)]
        if len(vals) == 0:
            continue
        score = float(np.max(vals)) if pool == "max" else float(np.mean(vals))
        lab = int(gold >= 0 and t == gold)
        scores.append(score)
        labels.append(lab)
    return np.asarray(scores, float), np.asarray(labels, int)


def predict_dataset(model, ds: ReasoningHypergraphDataset, device: str, *, step_pool: str) -> Dict[str, object]:
    model.eval()
    node_y, node_p = [], []
    step_y, step_p, step_groups = [], [], []
    graph_y, graph_p, graph_groups = [], [], []
    loc_top1, loc_exp = [], []
    with torch.no_grad():
        for k in range(len(ds)):
            data = ds[k].to(device)
            prob = torch.sigmoid(model(data)).detach().cpu().numpy()
            mask = data.loss_mask.detach().cpu().numpy().astype(bool)
            node_y.append(data.y.detach().cpu().numpy()[mask])
            node_p.append(prob[mask])

            ss, yy = step_scores_for_graph(data.cpu(), prob, pool=step_pool)
            if len(ss):
                step_y.append(yy)
                step_p.append(ss)
                step_groups.append(np.full(len(ss), int(data.group.cpu().item()), int))
                graph_y.append(int(data.gold_step.cpu().item() >= 0))
                graph_p.append(float(np.max(ss)))
                graph_groups.append(int(data.group.cpu().item()))
            gold = int(data.gold_step.cpu().item())
            if gold >= 0 and len(ss) and yy.sum() == 1:
                gi = int(np.where(yy == 1)[0][0])
                better = int((ss > ss[gi]).sum())
                loc_top1.append(float(better == 0))
                loc_exp.append(1.0 / len(ss))
    out = {}
    if node_y:
        ny = np.concatenate(node_y)
        npred = np.concatenate(node_p)
        out["node_y"] = ny
        out["node_p"] = npred
        out["node_auroc"] = auroc_safe(ny, npred)
        out["node_aupr"] = aupr_safe(ny, npred)
    if step_y:
        sy = np.concatenate(step_y)
        sp = np.concatenate(step_p)
        sg = np.concatenate(step_groups)
        out["step_y"] = sy
        out["step_p"] = sp
        out["step_groups"] = sg
        out["step_auroc"] = auroc_safe(sy, sp)
        out["step_aupr"] = aupr_safe(sy, sp)
    if graph_y:
        gy = np.asarray(graph_y, int)
        gp = np.asarray(graph_p, float)
        out["graph_y"] = gy
        out["graph_p"] = gp
        out["graph_groups"] = np.asarray(graph_groups, int)
        out["graph_auroc"] = auroc_safe(gy, gp)
        out["graph_aupr"] = aupr_safe(gy, gp)
    out["loc_top1"] = float(np.mean(loc_top1)) if loc_top1 else float("nan")
    out["loc_expected_top1"] = float(np.mean(loc_exp)) if loc_exp else float("nan")
    out["loc_n"] = int(len(loc_top1))
    return out


def make_val_indices(train_idx: Sequence[int], groups: np.ndarray, gold: np.ndarray, *, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx, int)
    ug = np.unique(groups[train_idx])
    rng.shuffle(ug)
    target = max(1, int(round(len(ug) * val_frac)))
    val_groups = set(int(x) for x in ug[:target])
    val = [int(i) for i in train_idx if int(groups[i]) in val_groups]
    tr = [int(i) for i in train_idx if int(groups[i]) not in val_groups]
    # Keep both classes in train if possible.
    if len(np.unique((gold[tr] >= 0).astype(int))) < 2 and len(ug) > target:
        return make_val_indices(train_idx, groups, gold, val_frac=max(val_frac / 2, 0.05), seed=seed + 1)
    return tr, val


def train_fold(
    fold: int,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    common: Dict[str, object],
    args: argparse.Namespace,
    device: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    train_ds = ReasoningHypergraphDataset(indices=train_idx, **common)
    val_ds = ReasoningHypergraphDataset(indices=val_idx, **common)
    test_ds = ReasoningHypergraphDataset(indices=test_idx, **common)
    sample = train_ds[0]
    node_dim = int(sample.x.shape[1])
    hedge_dim = int(sample.he_attr.shape[1]) if sample.he_attr.numel() else 4
    pos, neg = train_ds.label_counts()
    pos_weight = neg / max(pos, 1)
    if args.pos_weight_cap > 0:
        pos_weight = min(pos_weight, args.pos_weight_cap)
    print(f"\n[fold {fold}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} node_dim={node_dim} he_dim={hedge_dim} pos_weight={pos_weight:.3f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.num_workers)
    model = HyperCHARMNode(
        node_dim=node_dim,
        hedge_dim=hedge_dim,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        residual_mp=not args.no_residual_mp,
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * max(1, len(train_loader))))

    best_state = None
    best_val = -float("inf")
    patience = int(args.patience)
    bad = 0
    history = []
    for ep in range(int(args.epochs)):
        model.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            mask = batch.loss_mask.bool()
            if mask.sum() == 0:
                continue
            loss = loss_fn(logits[mask], batch.y[mask].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            scheduler.step()
            total += float(loss.detach().cpu().item())
            batches += 1
        val_m = evaluate_loader(model, val_loader, device)
        val_score = val_m.get("node_aupr", float("nan"))
        if not np.isfinite(val_score):
            val_score = val_m.get("node_auroc", -float("inf"))
        avg_loss = total / max(1, batches)
        history.append({"epoch": ep + 1, "loss": avg_loss, **val_m})
        print(
            f"[fold {fold} epoch {ep+1:02d}] loss={avg_loss:.4f} "
            f"val_node_AUROC={val_m['node_auroc']:.4f} val_node_AUPR={val_m['node_aupr']:.4f}"
        )
        if val_score > best_val:
            best_val = float(val_score)
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[fold {fold}] early stopping at epoch {ep+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = predict_dataset(model, test_ds, device, step_pool=args.step_pool)
    summary = {
        "fold": int(fold),
        "train_graphs": int(len(train_ds)),
        "val_graphs": int(len(val_ds)),
        "test_graphs": int(len(test_ds)),
        "node_dim": int(node_dim),
        "he_attr_dim": int(hedge_dim),
        "train_pos_tokens": int(pos),
        "train_neg_tokens": int(neg),
        "pos_weight": float(pos_weight),
        "history": history,
        "test_node_auroc": float(pred.get("node_auroc", float("nan"))),
        "test_node_aupr": float(pred.get("node_aupr", float("nan"))),
        "test_step_auroc": float(pred.get("step_auroc", float("nan"))),
        "test_step_aupr": float(pred.get("step_aupr", float("nan"))),
        "test_graph_auroc": float(pred.get("graph_auroc", float("nan"))),
        "test_graph_aupr": float(pred.get("graph_aupr", float("nan"))),
        "loc_top1": float(pred.get("loc_top1", float("nan"))),
        "loc_expected_top1": float(pred.get("loc_expected_top1", float("nan"))),
        "loc_n": int(pred.get("loc_n", 0)),
    }
    return summary, pred


def concatenate_preds(preds: Sequence[Dict[str, object]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for prefix in ["node", "step", "graph"]:
        ys = [p.get(f"{prefix}_y") for p in preds if f"{prefix}_y" in p]
        ps = [p.get(f"{prefix}_p") for p in preds if f"{prefix}_p" in p]
        if ys and ps:
            y = np.concatenate([np.asarray(v) for v in ys])
            p = np.concatenate([np.asarray(v) for v in ps])
            out[f"oof_{prefix}_auroc"] = auroc_safe(y, p)
            out[f"oof_{prefix}_aupr"] = aupr_safe(y, p)
            out[f"oof_{prefix}_n"] = int(len(y))
            out[f"oof_{prefix}_pos"] = int(np.asarray(y, int).sum())
    loc_top = [p.get("loc_top1") for p in preds if np.isfinite(p.get("loc_top1", float("nan")))]
    loc_exp = [p.get("loc_expected_top1") for p in preds if np.isfinite(p.get("loc_expected_top1", float("nan")))]
    loc_n = [p.get("loc_n", 0) for p in preds]
    out["oof_loc_top1"] = float(np.mean(loc_top)) if loc_top else float("nan")
    out["oof_loc_expected_top1"] = float(np.mean(loc_exp)) if loc_exp else float("nan")
    out["oof_loc_n"] = int(np.sum(loc_n))
    return out


def inspect_original_implementation() -> Dict[str, object]:
    return {
        "original_processed_hypergraph": {
            "nodes": "tokens",
            "x": "self-attention diagonal per flattened layer/head",
            "hyperedges": "one attention-row hyperedge per response token/head above tau",
            "he_attr": "mean attention, max attention, normalized head id",
            "he_mark": "[prompt-cross, response-only]",
            "label": "y_token hallucination labels",
        },
        "original_train_hypergraph": {
            "message_passing": "node2edge MLP([x_node, he_mark]) -> mean per hyperedge -> edge2node MLP([he_attr, agg_e]) -> mean per node -> LayerNorm -> residual",
            "prediction": "node-level logits",
            "loss": "BCEWithLogitsLoss(pos_weight) on response nodes",
            "selection": "best validation AUPR with early stopping",
        },
        "this_script_changes": {
            "nodes": "virtual prompt node + response tokens",
            "x": "selected multi-layer raw hidden and/or unit directions, optionally concatenated with anchor/spread/jump diagnostics",
            "hyperedges": "prompt-anchor, causal temporal, and hidden-neighbor hyperedges",
            "label": "tokens in gold first-error step; post-error tokens masked out",
            "evaluation": "token OOF + step aggregation + within-chain first-error localization",
        },
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    require_torch()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    npz, hidden_dir, stem = resolve_paths(args)
    z = np.load(npz, allow_pickle=True)
    groups = z["problem_ids"] if "problem_ids" in z.files else np.arange(len(z["gold_error_step"]))
    gold = np.asarray(z["gold_error_step"], int)
    n = len(gold) if args.max_chains <= 0 else min(args.max_chains, len(gold))
    indices = np.arange(n)
    labels = (gold[:n] >= 0).astype(int)
    n_splits = min(args.folds, len(np.unique(groups[:n])))
    if n_splits < 2:
        raise SystemExit("need at least two groups for GroupKFold")

    common = {
        "npz_path": npz,
        "hidden_dir": hidden_dir,
        "dataset": stem,
        "layers": parse_layers(args.layers),
        "x_mode": args.x_mode,
        "hidden_form": args.hidden_form,
        "local_radius": args.local_radius,
        "top_k": args.top_k,
        "temporal_radius": args.temporal_radius,
        "causal": bool(args.causal),
        "prompt_gamma": args.prompt_gamma,
        "min_prompt_edge": args.min_prompt_edge,
        "allow_missing_qvec": bool(args.allow_missing_qvec),
        "normalize_x": bool(args.normalize_x),
    }

    folds = []
    preds = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (trval, te) in enumerate(splitter.split(indices, labels, groups[:n]), start=1):
        tr, val = make_val_indices(indices[trval], groups[:n], gold[:n], val_frac=args.val_frac, seed=args.seed + fold)
        summary, pred = train_fold(fold, tr, val, indices[te], common, args, device)
        folds.append(summary)
        preds.append(pred)

    aggregate = concatenate_preds(preds)
    res = {
        "meta": {
            "npz": npz,
            "hidden_dir": hidden_dir,
            "dataset": stem,
            "n_seen": int(n),
            "error_chains": int(labels.sum()),
            "layers": parse_layers(args.layers),
            "x_mode": args.x_mode,
            "hidden_form": args.hidden_form,
            "causal": bool(args.causal),
            "top_k": int(args.top_k),
            "temporal_radius": int(args.temporal_radius),
            "local_radius": int(args.local_radius),
            "device": str(device),
            "original_implementation": inspect_original_implementation(),
        },
        "folds": folds,
        "aggregate": aggregate,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    causal_tag = "causal" if args.causal else "noncausal"
    out_name = f"{stem}_layers-{args.layers.replace(',', '-')}_{args.x_mode}_{args.hidden_form}_{causal_tag}"
    if args.max_chains:
        out_name += f"_n{args.max_chains}"
    out_path = os.path.join(args.output_dir, f"{out_name}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(finite_json(res), fh, indent=2, ensure_ascii=False)
    res["saved"] = out_path
    return res


def make_selftest(root: str, *, seed: int) -> Tuple[str, str]:
    rng = np.random.default_rng(seed)
    hidden_dir = os.path.join(root, "hidden", "selftest")
    os.makedirs(hidden_dir, exist_ok=True)
    layers = np.asarray([10, 14, 18, 22], int)
    d = 48
    ids, groups, gold, ranges, qvec = [], [], [], [], []
    for i in range(60):
        q = unit_rows(rng.normal(size=(len(layers), d)).astype(np.float32))
        off = unit_rows(rng.normal(size=(len(layers), d)).astype(np.float32))
        T = int(rng.integers(4, 7))
        lens = rng.integers(14, 22, size=T)
        is_err = i % 3 == 0
        g = int(rng.integers(1, T - 1)) if is_err else -1
        toks = []
        for t, ln in enumerate(lens):
            if is_err and t == g:
                mix = 0.15
            elif is_err and t > g:
                mix = 0.35
            else:
                mix = 0.85
            center = unit_rows(mix * q + (1 - mix) * off + 0.03 * rng.normal(size=(len(layers), d)))
            for _ in range(int(ln)):
                toks.append(center + 0.10 * rng.normal(size=(len(layers), d)))
        H = np.asarray(toks, np.float32)
        cid = f"selftest-{i}"
        np.save(os.path.join(hidden_dir, _fn(cid)), H.astype(np.float16))
        lo = np.cumsum(np.r_[0, lens[:-1]]) + 100
        rr = np.stack([lo, lo + lens - 1], axis=1).astype(int)
        ids.append(cid)
        groups.append(i)
        gold.append(g)
        ranges.append(rr)
        qvec.append(q.astype(np.float32))
    obj = np.empty(len(ranges), dtype=object)
    obj[:] = ranges
    npz = os.path.join(root, "full_selftest.npz")
    np.savez_compressed(
        npz,
        ids=np.asarray(ids, dtype=object),
        problem_ids=np.asarray(groups, int),
        gold_error_step=np.asarray(gold, int),
        step_token_ranges=obj,
        hidden_layers=layers,
        sv_layers=layers,
        qvec=np.asarray(qvec, np.float32),
    )
    return npz, hidden_dir


def print_result(res: Dict[str, object]) -> None:
    meta = res["meta"]
    agg = res["aggregate"]
    print(f"\n===== hypergraph token HGN | {os.path.basename(meta['npz'])} =====")
    print(
        f"chains {meta['n_seen']} | error chains {meta['error_chains']} | "
        f"layers {meta['layers']} | x_mode {meta['x_mode']} | hidden_form {meta['hidden_form']} | causal {meta['causal']}"
    )
    for k in [
        "oof_node_auroc",
        "oof_node_aupr",
        "oof_step_auroc",
        "oof_step_aupr",
        "oof_graph_auroc",
        "oof_graph_aupr",
        "oof_loc_top1",
        "oof_loc_expected_top1",
    ]:
        if k in agg:
            print(f"  {k:24s} {agg[k]:.4f}")
    print(f"\nsaved: {res['saved']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train full token-level HyperCHARM on reasoning hidden hypergraphs")
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--dataset", choices=["gsm8k", "math", "omnimath"], default=None)
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--hidden_dir", default=None)
    ap.add_argument("--layers", default="10,14,18,22")
    ap.add_argument("--x_mode", choices=["hidden", "diag", "hidden_diag"], default="hidden_diag")
    ap.add_argument("--hidden_form", choices=["raw", "unit", "both"], default="both")
    ap.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--temporal_radius", type=int, default=8)
    ap.add_argument("--local_radius", type=int, default=6)
    ap.add_argument("--prompt_gamma", type=float, default=2.0)
    ap.add_argument("--min_prompt_edge", type=float, default=0.05)
    ap.add_argument("--allow_missing_qvec", action="store_true")
    ap.add_argument("--normalize_x", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--gnn_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--pos_weight_cap", type=float, default=20.0)
    ap.add_argument("--step_pool", choices=["max", "mean"], default="max")
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/hypergraph_hgn")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        require_torch()
        with tempfile.TemporaryDirectory() as td:
            npz, hidden_dir = make_selftest(td, seed=args.seed)
            args.npz = npz
            args.hidden_dir = hidden_dir
            args.dataset = None
            args.data_dir = td
            args.max_chains = 0
            args.folds = min(args.folds, 3)
            args.epochs = min(args.epochs, 8)
            args.output_dir = os.path.join(td, "out")
            res = run(args)
            print_result(res)
            if res["aggregate"].get("oof_step_auroc", 0.0) < 0.70:
                raise SystemExit("selftest failed: HGN did not recover synthetic first-error steps")
        return

    res = run(args)
    print_result(res)


if __name__ == "__main__":
    main()
