from __future__ import annotations

from typing import Sequence

import numpy as np


def source_head_pre_output(
    attention,
    values,
    *,
    target_indices,
    source_mask,
):
    """Decompose each head's pre-``W_O`` vector over selected source tokens.

    ``attention`` has shape ``[batch, heads, query, key]`` and ``values`` has
    shape ``[batch, key, kv_heads, head_dim]``. Grouped-query value heads are
    repeated exactly as in decoder-only attention before the weighted sum.
    """

    import torch

    if attention.ndim != 4 or values.ndim != 4:
        raise ValueError("attention and values must be rank-four tensors")
    batch, heads, _queries, keys = attention.shape
    if values.shape[0] != batch or values.shape[1] != keys:
        raise ValueError("attention keys and value positions are misaligned")
    if source_mask.shape != (batch, keys):
        raise ValueError("source_mask must have shape [batch, key]")
    if target_indices.shape != (batch,):
        raise ValueError("target_indices must have shape [batch]")
    kv_heads = int(values.shape[2])
    if heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if kv_heads != heads:
        values = values.repeat_interleave(heads // kv_heads, dim=2)
    batch_index = torch.arange(batch, device=attention.device)
    weights = attention[batch_index, :, target_indices, :]
    selected = weights * source_mask[:, None, :].to(weights.dtype)
    contribution = torch.einsum(
        "bhk,bkhd->bhd", selected, values.to(dtype=selected.dtype)
    )
    return contribution, selected.sum(dim=-1)


def length_matched_control_mask(
    visible_lengths: Sequence[int] | np.ndarray,
    evidence_ranges: np.ndarray,
    padded_length: int,
) -> np.ndarray:
    """Choose a deterministic non-overlapping circular control window.

    The control has exactly the same number of visible tokens as the evidence
    span. It is placed half a visible sequence away and searched cyclically if
    that proposal overlaps the evidence span.
    """

    lengths = np.asarray(visible_lengths, dtype=np.int64)
    ranges = np.asarray(evidence_ranges, dtype=np.int64)
    if ranges.shape != (len(lengths), 2):
        raise ValueError("evidence ranges must have shape [batch, 2]")
    result = np.zeros((len(lengths), int(padded_length)), dtype=bool)
    for row, (visible, bounds) in enumerate(zip(lengths, ranges, strict=True)):
        start, stop = (int(bounds[0]), int(bounds[1]))
        width = stop - start
        if not (0 <= start < stop <= int(visible) <= int(padded_length)):
            raise ValueError("evidence range is outside the visible sequence")
        candidates = max(int(visible) - width + 1, 0)
        if candidates < 1 or width * 2 > int(visible):
            raise ValueError("sequence has no length-matched non-evidence control")
        proposal = (start + int(visible) // 2) % candidates
        selected = None
        for offset in range(candidates):
            candidate = (proposal + offset) % candidates
            candidate_stop = candidate + width
            if candidate_stop <= start or candidate >= stop:
                selected = candidate
                break
        if selected is None:
            raise ValueError("could not place a non-overlapping control window")
        result[row, selected : selected + width] = True
    return result


def head_residual_writes(pre_output, output_projection):
    """Apply each head's corresponding ``W_O`` slice independently."""

    import torch

    if pre_output.ndim != 3 or output_projection.ndim != 2:
        raise ValueError("head pre-output and W_O must be rank three/two")
    batch, heads, head_dim = pre_output.shape
    hidden_out, hidden_in = output_projection.shape
    if hidden_in != heads * head_dim:
        raise ValueError("W_O input width does not match concatenated attention heads")
    weights = output_projection.to(dtype=pre_output.dtype).reshape(
        hidden_out, heads, head_dim
    ).permute(1, 0, 2)
    return torch.einsum("bhd,hod->bho", pre_output, weights)


def cosine_alignment(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    values = np.asarray(left, dtype=np.float64)
    targets = np.asarray(right, dtype=np.float64)
    if values.shape[-1] != targets.shape[-1]:
        raise ValueError("alignment dimensions differ")
    while targets.ndim < values.ndim:
        targets = np.expand_dims(targets, axis=-2)
    numerator = np.sum(values * targets, axis=-1)
    denominator = np.linalg.norm(values, axis=-1) * np.linalg.norm(targets, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )
