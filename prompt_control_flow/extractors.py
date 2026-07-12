from __future__ import annotations

from typing import Dict, Protocol

import numpy as np

from .config import ExtractionConfig, MetricNames
from .data import ChainRecord
from .metrics import compute_step_prompt_flow_metrics
from .teacher_forcing import ForwardCache, prompt_token_indices


EPS = 1e-12


class MechanismExtractor(Protocol):
    name: str
    requires_hidden: bool
    requires_attention: bool
    requires_logits: bool

    def compute(
        self,
        cache: ForwardCache,
        record: ChainRecord,
        cfg: ExtractionConfig,
    ) -> Dict[str, np.ndarray]:
        ...


class PromptResidualFlowExtractor:
    name = "prompt_flow"
    requires_hidden = True
    requires_attention = False
    requires_logits = False

    def compute(self, cache: ForwardCache, record: ChainRecord, cfg: ExtractionConfig) -> Dict[str, np.ndarray]:
        if cache.hidden_states is None:
            return {}
        hidden = [np.asarray(h, dtype=np.float64) for h in cache.hidden_states]
        logits = np.asarray(cache.logits, dtype=np.float64) if (cfg.include_entropy and cache.logits is not None) else None
        return compute_step_prompt_flow_metrics(
            hidden,
            logits,
            prompt_token_indices=prompt_token_indices(cache.offset_mapping, len(_prompt_text(record))),
            response_token_start=int(cache.response_start_token),
            step_ranges=cache.step_token_ranges,
            layers=cfg.layers,
            subspace_k=cfg.subspace_k,
            prefix_k=cfg.prefix_k,
            rng=np.random.default_rng(cfg.random_seed + int(record.chain_idx)),
            center_subspaces=cfg.center_subspaces,
        )


class UncertaintyExtractor:
    name = "uncertainty"
    requires_hidden = False
    requires_attention = False
    requires_logits = True

    def compute(self, cache: ForwardCache, record: ChainRecord, cfg: ExtractionConfig) -> Dict[str, np.ndarray]:
        n_steps = len(cache.step_token_ranges)
        out = {
            MetricNames.TOKEN_ENTROPY: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_NLL: np.full(n_steps, np.nan, dtype=np.float64),
        }
        if cache.logits is None or cache.input_ids is None:
            return out
        logits = np.asarray(cache.logits, dtype=np.float64)
        input_ids = _to_numpy(cache.input_ids).astype(np.int64, copy=False)
        for j, (a, b) in enumerate(cache.step_token_ranges):
            target = np.arange(max(int(a), 1), int(b) + 1, dtype=np.int64)
            pred = target - 1
            ok = (pred >= 0) & (pred < logits.shape[0]) & (target < input_ids.shape[0])
            if not np.any(ok):
                continue
            lp = logits[pred[ok]]
            m = np.max(lp, axis=1, keepdims=True)
            exp = np.exp(lp - m)
            prob = exp / np.maximum(np.sum(exp, axis=1, keepdims=True), EPS)
            ent = -np.sum(prob * np.log(np.maximum(prob, EPS)), axis=1)
            tgt = input_ids[target[ok]]
            p_tgt = prob[np.arange(tgt.size), tgt]
            nll = -np.log(np.maximum(p_tgt, EPS))
            out[MetricNames.TOKEN_ENTROPY][j] = _safe_mean(ent)
            out[MetricNames.TOKEN_NLL][j] = _safe_mean(nll)
        return out


class ICRResidualMismatchExtractor:
    name = "icr"
    requires_hidden = True
    requires_attention = True
    requires_logits = False

    def compute(self, cache: ForwardCache, record: ChainRecord, cfg: ExtractionConfig) -> Dict[str, np.ndarray]:
        n_steps = len(cache.step_token_ranges)
        out = {
            MetricNames.ICR_MEAN: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.ICR_MAX: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.ICR_TOP20_MEAN: np.full(n_steps, np.nan, dtype=np.float64),
        }
        for layer in cfg.layers:
            out[f"icr_layer_mean_{int(layer)}"] = np.full(n_steps, np.nan, dtype=np.float64)
        if cache.hidden_states is None or cache.attentions is None:
            return out

        hs = cache.hidden_states
        attn = cache.attentions
        for j, (a, b) in enumerate(cache.step_token_ranges):
            all_vals = []
            for layer in cfg.layers:
                l = int(layer)
                if l < 0 or l + 1 >= len(hs) or l >= len(attn):
                    continue
                layer_vals = _icr_for_step_layer(
                    np.asarray(hs[l], dtype=np.float64),
                    np.asarray(hs[l + 1], dtype=np.float64),
                    np.asarray(attn[l], dtype=np.float64),
                    int(a),
                    int(b),
                    top_k=int(cfg.icr_top_k),
                    top_p=cfg.icr_top_p,
                )
                if layer_vals.size:
                    out[f"icr_layer_mean_{l}"][j] = _safe_mean(layer_vals)
                    all_vals.extend(layer_vals.tolist())
            vals = np.asarray(all_vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[MetricNames.ICR_MEAN][j] = float(np.mean(vals))
                out[MetricNames.ICR_MAX][j] = float(np.max(vals))
                k = max(1, int(np.ceil(0.2 * vals.size)))
                out[MetricNames.ICR_TOP20_MEAN][j] = float(np.mean(np.sort(vals)[-k:]))
        return out


def build_extractors(*, prompt_flow: bool, uncertainty: bool, icr: bool) -> list[MechanismExtractor]:
    extractors: list[MechanismExtractor] = []
    if prompt_flow:
        extractors.append(PromptResidualFlowExtractor())
    if uncertainty:
        extractors.append(UncertaintyExtractor())
    if icr:
        extractors.append(ICRResidualMismatchExtractor())
    return extractors


def _icr_for_step_layer(
    h_l: np.ndarray,
    h_next: np.ndarray,
    attn_layer: np.ndarray,
    a: int,
    b: int,
    *,
    top_k: int,
    top_p: float | None,
) -> np.ndarray:
    vals = []
    seq_len = min(h_l.shape[0], h_next.shape[0], attn_layer.shape[-1])
    for pos in range(max(a, 0), min(b + 1, seq_len)):
        row = attn_layer[:, pos, : pos + 1]
        if row.size == 0:
            continue
        att = np.mean(row, axis=0)
        k = len(att)
        if top_p is not None:
            k = max(1, int(float(top_p) * len(att)))
        elif top_k is not None and top_k > 0:
            k = min(int(top_k), len(att))
        if k <= 0:
            continue
        idx = np.argpartition(att, -k)[-k:]
        a_top = att[idx]
        delta = h_next[pos] - h_l[pos]
        selected = h_l[idx]
        denom = np.linalg.norm(selected, axis=1) + EPS
        contrib = np.sum(delta[None, :] * selected, axis=1) / denom
        vals.append(_js_from_vectors(contrib, a_top))
    return np.asarray(vals, dtype=np.float64)


def _js_from_vectors(x: np.ndarray, y: np.ndarray) -> float:
    px = _softmax(_zscore(np.asarray(x, dtype=np.float64)))
    py = _softmax(_zscore(np.asarray(y, dtype=np.float64)))
    m = 0.5 * (px + py)
    return float(0.5 * _kl(px, m) + 0.5 * _kl(py, m))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.maximum(np.sum(e), EPS)


def _zscore(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return (x - np.mean(x)) / (s + EPS)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(p, EPS)
    q = np.maximum(q, EPS)
    return float(np.sum(p * np.log(p / q)))


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _prompt_text(record: ChainRecord) -> str:
    return f"Problem: {record.problem}\n\nSolution:\n\n"
