from __future__ import annotations

from typing import Dict, Protocol

import numpy as np

from .config import ExtractionConfig, MetricNames
from .data import ChainRecord
from .metrics import compute_step_prompt_flow_metrics
from .teacher_forcing import ForwardCache


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
        return compute_step_prompt_flow_metrics(
            cache.hidden_states,
            None,
            prompt_token_indices=cache.prompt_content_token_indices,
            question_token_indices=cache.question_token_indices,
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
            MetricNames.TOKEN_ENTROPY_MAX: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_ENTROPY_FIRST: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_ENTROPY_LAST: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_NLL: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_NLL_MAX: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_NLL_FIRST: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_NLL_LAST: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_CHOSEN_LOGPROB: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_CHOSEN_LOGPROB_MIN: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_MARGIN: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_MARGIN_MIN: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_TOPK_MASS: np.full(n_steps, np.nan, dtype=np.float64),
            MetricNames.TOKEN_TOPK_MASS_MIN: np.full(n_steps, np.nan, dtype=np.float64),
        }
        compact = {
            name: _to_numpy(values).astype(np.float64, copy=False)
            for name, values in cache.token_output_summaries.items()
        }
        if compact:
            for j, (a, b) in enumerate(cache.step_token_ranges):
                target = np.arange(int(a), int(b) + 1, dtype=np.int64)
                target = target[(target >= 0) & (target < cache.seq_len)]
                if target.size == 0:
                    continue
                ent = compact["entropy"][target]
                nll = compact["nll"][target]
                chosen = compact["chosen_logprob"][target]
                margin = compact["top1_top2_margin"][target]
                top_mass = compact["topk_mass"][target]
                _fill_uncertainty_aggregates(out, j, ent, nll, chosen, margin, top_mass)
            return out
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
            chosen = -nll
            top_values = np.partition(lp, -2, axis=1)[:, -2:]
            margin = np.max(top_values, axis=1) - np.min(top_values, axis=1)
            top_mass = np.sum(
                np.partition(prob, -min(10, prob.shape[1]), axis=1)[
                    :, -min(10, prob.shape[1]) :
                ],
                axis=1,
            )
            _fill_uncertainty_aggregates(out, j, ent, nll, chosen, margin, top_mass)
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
                depth = int(layer)
                block = depth - 1
                if depth <= 0 or depth >= len(hs) or block >= len(attn):
                    continue
                layer_vals = _icr_for_step_layer(
                    hs[depth - 1],
                    hs[depth],
                    attn[block],
                    int(a),
                    int(b),
                    top_k=int(cfg.icr_top_k),
                    top_p=cfg.icr_top_p,
                )
                if layer_vals.size:
                    out[f"icr_layer_mean_{depth}"][j] = _safe_mean(layer_vals)
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
    h_l,
    h_next,
    attn_layer,
    a: int,
    b: int,
    *,
    top_k: int,
    top_p: float | None,
) -> np.ndarray:
    if hasattr(h_l, "detach"):
        return _icr_for_step_layer_torch(
            h_l,
            h_next,
            attn_layer,
            a,
            b,
            top_k=top_k,
            top_p=top_p,
        )
    h_l = np.asarray(h_l, dtype=np.float64)
    h_next = np.asarray(h_next, dtype=np.float64)
    attn_layer = np.asarray(attn_layer, dtype=np.float64)
    vals = []
    seq_len = min(h_l.shape[0], h_next.shape[0], attn_layer.shape[-1])
    target_positions = range(max(int(a), 1), min(int(b) + 1, seq_len))
    for target_pos in target_positions:
        pos = target_pos - 1
        row = attn_layer[:, pos, : pos + 1]
        if row.size == 0:
            continue
        att = np.mean(row, axis=0)
        k = len(att)
        if top_p is not None:
            order = np.argsort(att)[::-1]
            mass = np.cumsum(att[order]) / max(float(np.sum(att)), EPS)
            k = max(1, int(np.searchsorted(mass, float(top_p), side="left") + 1))
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


def _icr_for_step_layer_torch(
    h_l,
    h_next,
    attn_layer,
    a: int,
    b: int,
    *,
    top_k: int,
    top_p: float | None,
) -> np.ndarray:
    import torch

    values = []
    seq_len = min(h_l.shape[0], h_next.shape[0], attn_layer.shape[-1])
    with torch.inference_mode():
        for target_pos in range(max(int(a), 1), min(int(b) + 1, seq_len)):
            prediction_pos = target_pos - 1
            attention = attn_layer[:, prediction_pos, : prediction_pos + 1].float()
            if attention.numel() == 0:
                continue
            attention = attention.mean(dim=0)
            if top_p is not None:
                sorted_attention, sorted_idx = torch.sort(
                    attention, descending=True
                )
                cumulative = torch.cumsum(sorted_attention, dim=0) / sorted_attention.sum().clamp_min(EPS)
                crossing = torch.nonzero(cumulative >= float(top_p), as_tuple=False)
                count = (
                    int(crossing[0, 0].item()) + 1
                    if crossing.numel()
                    else int(sorted_idx.numel())
                )
                idx = sorted_idx[: max(1, count)]
            else:
                count = min(max(int(top_k), 1), int(attention.numel()))
                idx = torch.topk(attention, k=count).indices
            selected_attention = attention.index_select(0, idx)
            delta = h_next[prediction_pos].float() - h_l[prediction_pos].float()
            selected_states = h_l.index_select(0, idx).float()
            contribution = torch.sum(
                selected_states * delta[None, :], dim=1
            ) / torch.linalg.vector_norm(selected_states, dim=1).clamp_min(EPS)
            values.append(_js_from_torch_vectors(contribution, selected_attention))
    if not values:
        return np.zeros(0, dtype=np.float64)
    return torch.stack(values).detach().float().cpu().numpy().astype(np.float64)


def _js_from_torch_vectors(x, y):
    import torch

    x = (x - x.mean()) / x.std(unbiased=False).clamp_min(EPS)
    y = (y - y.mean()) / y.std(unbiased=False).clamp_min(EPS)
    px = torch.softmax(x, dim=0).clamp_min(EPS)
    py = torch.softmax(y, dim=0).clamp_min(EPS)
    midpoint = 0.5 * (px + py)
    return 0.5 * torch.sum(px * torch.log(px / midpoint)) + 0.5 * torch.sum(
        py * torch.log(py / midpoint)
    )


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


def _fill_uncertainty_aggregates(
    out: Dict[str, np.ndarray],
    step_pos: int,
    entropy: np.ndarray,
    nll: np.ndarray,
    chosen_logprob: np.ndarray,
    margin: np.ndarray,
    topk_mass: np.ndarray,
) -> None:
    values = [
        np.asarray(x, dtype=np.float64)
        for x in (entropy, nll, chosen_logprob, margin, topk_mass)
    ]
    if any(x.size == 0 or not np.isfinite(x).all() for x in values):
        return
    ent, nll_values, chosen, margin_values, mass = values
    out[MetricNames.TOKEN_ENTROPY][step_pos] = float(np.mean(ent))
    out[MetricNames.TOKEN_ENTROPY_MAX][step_pos] = float(np.max(ent))
    out[MetricNames.TOKEN_ENTROPY_FIRST][step_pos] = float(ent[0])
    out[MetricNames.TOKEN_ENTROPY_LAST][step_pos] = float(ent[-1])
    out[MetricNames.TOKEN_NLL][step_pos] = float(np.mean(nll_values))
    out[MetricNames.TOKEN_NLL_MAX][step_pos] = float(np.max(nll_values))
    out[MetricNames.TOKEN_NLL_FIRST][step_pos] = float(nll_values[0])
    out[MetricNames.TOKEN_NLL_LAST][step_pos] = float(nll_values[-1])
    out[MetricNames.TOKEN_CHOSEN_LOGPROB][step_pos] = float(np.mean(chosen))
    out[MetricNames.TOKEN_CHOSEN_LOGPROB_MIN][step_pos] = float(np.min(chosen))
    out[MetricNames.TOKEN_MARGIN][step_pos] = float(np.mean(margin_values))
    out[MetricNames.TOKEN_MARGIN_MIN][step_pos] = float(np.min(margin_values))
    out[MetricNames.TOKEN_TOPK_MASS][step_pos] = float(np.mean(mass))
    out[MetricNames.TOKEN_TOPK_MASS_MIN][step_pos] = float(np.min(mass))


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)
