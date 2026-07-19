"""Granularity-safe pooling and objectives for hallucination detection.

Token, step, and response supervision are intentionally separate.  In
particular, ``token_bce`` requires an explicit exact-label mask so that a
first-error *step* label cannot silently be expanded into fabricated token
labels.  Step supervision uses an explicit risk set, excluding post-error
steps.  Pooling defaults to a mean of logits; a length-normalized
log-mean-exp alternative is available, while unnormalized max pooling is not.
"""

from __future__ import annotations

import math
from typing import Any, Optional

try:  # pragma: no cover - availability depends on execution environment
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("attention HyperCHARM objectives require PyTorch")


_POOLING_MODES = frozenset({"mean", "logmeanexp"})


def _as_vector(values: Any, *, name: str, dtype=None, device=None):
    require_torch()
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    return tensor


def _pool_vector(logits, *, pooling: str, temperature: float):
    if pooling not in _POOLING_MODES:
        raise ValueError(
            f"pooling must be one of {sorted(_POOLING_MODES)}; "
            "max pooling is intentionally unsupported because it is length-biased"
        )
    if logits.numel() == 0:
        raise ValueError("cannot pool an empty token range")
    if pooling == "mean":
        return logits.mean()
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scale = float(temperature)
    return scale * (
        torch.logsumexp(logits / scale, dim=0) - math.log(int(logits.numel()))
    )


def pool_token_logits_to_steps(
    token_logits: Any,
    step_ranges: Any,
    *,
    pooling: str = "mean",
    temperature: float = 1.0,
):
    """Pool token logits over absolute half-open ``[start, end)`` step ranges.

    The returned tensor has one logit per step.  Ranges must be ordered,
    non-overlapping, non-empty, and lie on the same full prompt+response token
    axis as ``token_logits``.
    """

    logits = _as_vector(token_logits, name="token_logits")
    ranges = torch.as_tensor(step_ranges, dtype=torch.long, device=logits.device)
    if ranges.ndim != 2 or ranges.size(-1) != 2:
        raise ValueError("step_ranges must have shape [num_steps, 2]")
    if ranges.size(0) == 0:
        return logits.new_empty((0,))

    starts, ends = ranges[:, 0], ranges[:, 1]
    if bool((starts < 0).any()) or bool((ends > logits.numel()).any()):
        raise ValueError("step_ranges contain offsets outside token_logits")
    if bool((ends <= starts).any()):
        raise ValueError("every step range must be non-empty and half-open")
    if ranges.size(0) > 1 and bool((starts[1:] < ends[:-1]).any()):
        raise ValueError("step_ranges must be ordered and non-overlapping")

    return torch.stack(
        [
            _pool_vector(
                logits[int(start) : int(end)],
                pooling=pooling,
                temperature=temperature,
            )
            for start, end in ranges.tolist()
        ]
    )


def pool_token_logits_to_response(
    token_logits: Any,
    response_start: Any,
    *,
    response_end: Optional[Any] = None,
    pooling: str = "mean",
    temperature: float = 1.0,
):
    """Pool response-token logits to one length-normalized response logit."""

    logits = _as_vector(token_logits, name="token_logits")
    start_tensor = torch.as_tensor(response_start)
    if start_tensor.numel() != 1:
        raise ValueError("response_start must be a scalar for a single graph")
    start = int(start_tensor.item())
    if response_end is None:
        end = int(logits.numel())
    else:
        end_tensor = torch.as_tensor(response_end)
        if end_tensor.numel() != 1:
            raise ValueError("response_end must be a scalar for a single graph")
        end = int(end_tensor.item())
    if start < 0 or end > logits.numel() or end <= start:
        raise ValueError("response range must be a non-empty slice of token_logits")
    return _pool_vector(
        logits[start:end], pooling=pooling, temperature=temperature
    )


def make_first_error_targets(
    num_steps: int,
    gold_step: int,
    *,
    device: Optional[Any] = None,
):
    """Return binary step targets and the observable first-error risk set.

    ``gold_step == -1`` denotes a fully correct response: every step is a
    negative at-risk observation.  For an erroneous response, only steps up to
    and including the first erroneous step are in the risk set; later steps are
    consequences and are excluded rather than relabelled as negatives.
    """

    require_torch()
    num_steps = int(num_steps)
    gold_step = int(gold_step)
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if gold_step < -1 or gold_step >= num_steps:
        raise ValueError("gold_step must be -1 or a valid zero-based step index")

    targets = torch.zeros(num_steps, dtype=torch.float32, device=device)
    risk_mask = torch.ones(num_steps, dtype=torch.bool, device=device)
    if gold_step >= 0:
        targets[gold_step] = 1.0
        risk_mask[gold_step + 1 :] = False
    return targets, risk_mask


def _masked_bce(
    logits: Any,
    targets: Any,
    mask: Any,
    *,
    logits_name: str,
    mask_name: str,
    pos_weight: Optional[Any],
    reduction: str,
):
    require_torch()
    score = _as_vector(logits, name=logits_name)
    target = _as_vector(
        targets, name="targets", dtype=score.dtype, device=score.device
    )
    selected_mask = _as_vector(
        mask, name=mask_name, dtype=torch.bool, device=score.device
    )
    if score.shape != target.shape or score.shape != selected_mask.shape:
        raise ValueError(f"{logits_name}, targets, and {mask_name} must share shape")
    if not bool(selected_mask.any()):
        raise ValueError(f"{mask_name} selects no supervised observations")
    selected_logits = score[selected_mask]
    selected_targets = target[selected_mask]
    if not torch.isfinite(selected_logits).all() or not torch.isfinite(selected_targets).all():
        raise ValueError("selected logits and targets must be finite")
    if bool(((selected_targets < 0) | (selected_targets > 1)).any()):
        raise ValueError("binary targets must lie in [0, 1]")

    weight_tensor = None
    if pos_weight is not None:
        weight_tensor = torch.as_tensor(
            pos_weight, dtype=score.dtype, device=score.device
        )
        if weight_tensor.numel() != 1 or not bool(weight_tensor > 0):
            raise ValueError("pos_weight must be one positive scalar")
    return F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        pos_weight=weight_tensor,
        reduction=reduction,
    )


def token_bce(
    token_logits: Any,
    token_targets: Any,
    *,
    exact_label_mask: Any,
    pos_weight: Optional[Any] = None,
    reduction: str = "mean",
):
    """BCE over tokens carrying genuine token-level annotations only.

    ``exact_label_mask`` is mandatory by design.  A dataset that only provides
    a first-error step must call ``step_bce`` instead of expanding that label to
    every token in the step.
    """

    return _masked_bce(
        token_logits,
        token_targets,
        exact_label_mask,
        logits_name="token_logits",
        mask_name="exact_label_mask",
        pos_weight=pos_weight,
        reduction=reduction,
    )


def step_bce(
    step_scores: Any,
    step_targets: Any,
    *,
    risk_mask: Any,
    pos_weight: Optional[Any] = None,
    reduction: str = "mean",
):
    """BCE over step scores restricted to the observable first-error risk set."""

    return _masked_bce(
        step_scores,
        step_targets,
        risk_mask,
        logits_name="step_scores",
        mask_name="risk_mask",
        pos_weight=pos_weight,
        reduction=reduction,
    )


def response_bce(
    response_scores: Any,
    response_targets: Any,
    *,
    pos_weight: Optional[Any] = None,
    reduction: str = "mean",
):
    """Binary response-level hallucination loss without token pseudo-labels."""

    scores = _as_vector(response_scores, name="response_scores")
    targets = _as_vector(
        response_targets,
        name="response_targets",
        dtype=scores.dtype,
        device=scores.device,
    )
    if scores.shape != targets.shape:
        raise ValueError("response_scores and response_targets must share shape")
    mask = torch.ones_like(scores, dtype=torch.bool)
    return _masked_bce(
        scores,
        targets,
        mask,
        logits_name="response_scores",
        mask_name="response_mask",
        pos_weight=pos_weight,
        reduction=reduction,
    )


__all__ = [
    "make_first_error_targets",
    "pool_token_logits_to_response",
    "pool_token_logits_to_steps",
    "response_bce",
    "step_bce",
    "token_bce",
]
