from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .config import MetricNames
from .geometry import orthonormal_basis, projection_energy_fraction, random_basis


EPS = 1e-12


def _safe_nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _safe_nanmax(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")


def compute_step_prompt_flow_metrics(
    hidden_states: Sequence[np.ndarray],
    logits: np.ndarray | None,
    *,
    prompt_token_indices: np.ndarray,
    response_token_start: int,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
    subspace_k: int,
    prefix_k: int,
    rng: np.random.Generator,
    center_subspaces: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute per-step prompt/prefix/random residual-flow metrics.

    Hidden states follow HuggingFace convention: element 0 is embeddings and
    element l+1 is the output after transformer block l.  For a token at
    position p, logits[p-1] predicts input_ids[p], so the residual update used
    to predict generated tokens in a step is measured at positions p-1.
    """

    if len(step_ranges) == 0:
        raise ValueError("empty step_ranges")
    n_steps = len(step_ranges)
    metric_names = [
        MetricNames.PROMPT_FRAC,
        MetricNames.PREFIX_FRAC,
        MetricNames.RANDOM_FRAC,
        MetricNames.OFF_PROMPT,
        MetricNames.PROMPT_CONTROL_RATIO,
        MetricNames.PREFIX_LOCK_RATIO,
        MetricNames.TOKEN_ENTROPY,
        MetricNames.TOKEN_NLL,
        MetricNames.STEP_LEN,
        MetricNames.REL_POS,
    ]
    out = {name: np.full(n_steps, np.nan, dtype=np.float64) for name in metric_names}

    for j, (a, b) in enumerate(step_ranges):
        target_positions = np.arange(max(int(a), 1), int(b) + 1, dtype=np.int64)
        pred_positions = target_positions - 1
        if target_positions.size == 0:
            continue
        out[MetricNames.STEP_LEN][j] = float(target_positions.size)
        out[MetricNames.REL_POS][j] = float(j / max(n_steps - 1, 1))

        prompt_fracs: List[np.ndarray] = []
        prefix_fracs: List[np.ndarray] = []
        random_fracs: List[np.ndarray] = []
        ent_vals: List[np.ndarray] = []
        nll_vals: List[np.ndarray] = []

        for layer in layers:
            l = int(layer)
            if l < 0 or l + 1 >= len(hidden_states):
                continue
            h_l = np.asarray(hidden_states[l], dtype=np.float64)
            h_next = np.asarray(hidden_states[l + 1], dtype=np.float64)
            valid_pred = pred_positions[pred_positions < h_l.shape[0]]
            if valid_pred.size == 0:
                continue
            delta = h_next[valid_pred] - h_l[valid_pred]
            if delta.ndim != 2 or delta.shape[0] == 0:
                continue

            prompt_idx = prompt_token_indices[prompt_token_indices < h_l.shape[0]]
            prompt_idx = prompt_idx[prompt_idx >= 0]
            if prompt_idx.size >= 1:
                try:
                    q_basis = orthonormal_basis(
                        h_l[prompt_idx],
                        subspace_k,
                        center=center_subspaces,
                    ).basis
                    prompt_fracs.append(projection_energy_fraction(delta, q_basis))
                    random_fracs.append(projection_energy_fraction(delta, random_basis(delta.shape[1], q_basis.shape[1], rng)))
                except ValueError:
                    pass

            prefix_end = max(int(a) - 1, int(response_token_start))
            prefix_idx = np.arange(int(response_token_start), prefix_end, dtype=np.int64)
            prefix_idx = prefix_idx[(prefix_idx >= 0) & (prefix_idx < h_l.shape[0])]
            if prefix_idx.size >= 2:
                try:
                    p_basis = orthonormal_basis(
                        h_l[prefix_idx],
                        prefix_k,
                        center=center_subspaces,
                    ).basis
                    prefix_fracs.append(projection_energy_fraction(delta, p_basis))
                except ValueError:
                    pass

        pf = np.concatenate(prompt_fracs) if prompt_fracs else np.asarray([], dtype=np.float64)
        prf = np.concatenate(prefix_fracs) if prefix_fracs else np.asarray([], dtype=np.float64)
        rf = np.concatenate(random_fracs) if random_fracs else np.asarray([], dtype=np.float64)

        out[MetricNames.PROMPT_FRAC][j] = _safe_nanmean(pf)
        out[MetricNames.PREFIX_FRAC][j] = _safe_nanmean(prf)
        out[MetricNames.RANDOM_FRAC][j] = _safe_nanmean(rf)
        out[MetricNames.OFF_PROMPT][j] = 1.0 - out[MetricNames.PROMPT_FRAC][j] if np.isfinite(out[MetricNames.PROMPT_FRAC][j]) else float("nan")
        den = out[MetricNames.PREFIX_FRAC][j] + out[MetricNames.OFF_PROMPT][j] + EPS
        out[MetricNames.PROMPT_CONTROL_RATIO][j] = out[MetricNames.PROMPT_FRAC][j] / den if np.isfinite(den) else float("nan")
        out[MetricNames.PREFIX_LOCK_RATIO][j] = (out[MetricNames.PREFIX_FRAC][j] + out[MetricNames.OFF_PROMPT][j]) / (out[MetricNames.PROMPT_FRAC][j] + EPS)

        if logits is not None:
            # logits[p-1] predicts token p; caller fills nll later if token ids are available.
            lp = np.asarray(logits[pred_positions[pred_positions < logits.shape[0]]], dtype=np.float64)
            if lp.size:
                m = np.max(lp, axis=1, keepdims=True)
                prob = np.exp(lp - m)
                prob /= np.maximum(np.sum(prob, axis=1, keepdims=True), EPS)
                ent = -np.sum(prob * np.log(np.maximum(prob, EPS)), axis=1)
                out[MetricNames.TOKEN_ENTROPY][j] = _safe_nanmean(ent)

    return out


def compute_step_residual_vectors(
    hidden_states: Sequence[np.ndarray],
    *,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
) -> np.ndarray:
    """Return one residual-flow vector per reasoning step.

    For each step and selected layer, this takes the mean residual update over
    the prediction positions that generated the step, then concatenates layers.
    The output is the common input for learned latent charts such as PCA,
    diffusion maps, or a VAE.  It is intentionally not a classifier.
    """

    if len(step_ranges) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    hidden_dim = None
    for h in hidden_states:
        arr = np.asarray(h)
        if arr.ndim == 2:
            hidden_dim = int(arr.shape[1])
            break
    if hidden_dim is None:
        raise ValueError("hidden_states must contain rank-2 arrays")

    out = np.zeros((len(step_ranges), len(layers) * hidden_dim), dtype=np.float32)
    for j, (a, b) in enumerate(step_ranges):
        target_positions = np.arange(max(int(a), 1), int(b) + 1, dtype=np.int64)
        pred_positions = target_positions - 1
        chunks: List[np.ndarray] = []
        for layer in layers:
            l = int(layer)
            if l < 0 or l + 1 >= len(hidden_states):
                chunks.append(np.zeros(hidden_dim, dtype=np.float32))
                continue
            h_l = np.asarray(hidden_states[l], dtype=np.float32)
            h_next = np.asarray(hidden_states[l + 1], dtype=np.float32)
            valid_pred = pred_positions[pred_positions < h_l.shape[0]]
            if valid_pred.size == 0:
                chunks.append(np.zeros(hidden_dim, dtype=np.float32))
                continue
            delta = h_next[valid_pred] - h_l[valid_pred]
            chunks.append(np.mean(delta, axis=0).astype(np.float32, copy=False))
        out[j] = np.concatenate(chunks, axis=0)
    return out


def compute_step_state_vectors(
    hidden_states: Sequence[np.ndarray],
    *,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
) -> np.ndarray:
    """Return the backward-compatible flattened state vector for each step.

    New geometry code should prefer :func:`compute_step_layer_state_vectors`,
    which preserves the layer axis.  This wrapper intentionally keeps the old
    ``[step, layer * hidden]`` schema for existing audits.
    """

    tensor = compute_step_layer_state_vectors(
        hidden_states,
        step_ranges=step_ranges,
        layers=layers,
    )
    if tensor.size == 0:
        return np.zeros((tensor.shape[0], 0), dtype=np.float32)
    return tensor.reshape(tensor.shape[0], -1)


def compute_step_layer_state_vectors(
    hidden_states: Sequence[np.ndarray],
    *,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
) -> np.ndarray:
    """Mean-pool step states while preserving network depth.

    The returned tensor has shape ``[n_steps, n_layers, hidden_dim]``.  Its
    entry ``out[t, l]`` is the mean hidden state over tokens in reasoning step
    ``t`` at the explicitly requested hidden-state depth ``layers[l]``.  No
    layer concatenation, supervision, or learned readout is applied here.
    """

    if len(step_ranges) == 0:
        return np.zeros((0, len(layers), 0), dtype=np.float32)

    hidden_dim = None
    for h in hidden_states:
        arr = np.asarray(h)
        if arr.ndim == 2:
            hidden_dim = int(arr.shape[1])
            break
    if hidden_dim is None:
        raise ValueError("hidden_states must contain rank-2 arrays")

    out = np.zeros((len(step_ranges), len(layers), hidden_dim), dtype=np.float32)
    for j, (a, b) in enumerate(step_ranges):
        target_positions = np.arange(int(a), int(b) + 1, dtype=np.int64)
        for layer_pos, layer in enumerate(layers):
            l = int(layer)
            if l < 0 or l >= len(hidden_states):
                continue
            h_l = np.asarray(hidden_states[l], dtype=np.float32)
            valid = target_positions[(target_positions >= 0) & (target_positions < h_l.shape[0])]
            if valid.size == 0:
                continue
            out[j, layer_pos] = np.mean(h_l[valid], axis=0).astype(np.float32, copy=False)
    return out


def summarize_step_metrics(metric_series: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """Chain-level summaries used by response diagnosis."""

    summaries: Dict[str, float] = {}
    for name, vals in metric_series.items():
        x = np.asarray(vals, dtype=np.float64)
        if not np.isfinite(x).any():
            summaries[f"mean_{name}"] = float("nan")
            summaries[f"max_{name}"] = float("nan")
            continue
        summaries[f"mean_{name}"] = _safe_nanmean(x)
        summaries[f"max_{name}"] = _safe_nanmax(x)
    # A rough survival-style score from off-prompt/prefix lock.
    risk = np.asarray(metric_series.get(MetricNames.PREFIX_LOCK_RATIO, []), dtype=np.float64)
    risk = risk[np.isfinite(risk)]
    if risk.size:
        z = np.clip((risk - np.nanmedian(risk)) / (np.nanstd(risk) + EPS), -20, 20)
        p = 1.0 / (1.0 + np.exp(-z))
        summaries["survival_prefix_lock"] = float(1.0 - np.prod(1.0 - np.clip(p, 1e-6, 1 - 1e-6)))
    else:
        summaries["survival_prefix_lock"] = float("nan")
    return summaries
