from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FlowSignatureConfig:
    """Configuration for a sketched order-two log-signature.

    The representation is translation invariant because it uses hidden-state
    increments. Hidden increments are normalized by prefix total variation;
    the extra progress channel preserves when a direction change happens
    without exposing the raw number of reasoning steps.
    """

    projection_dim: int = 8
    phase_points: int = 16
    progress_weight: float = 1.0
    state_normalization: str = "none"
    seed: int = 0
    eps: float = 1e-8

    def validate(self) -> None:
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.phase_points < 2:
            raise ValueError("phase_points must be at least 2")
        if self.progress_weight <= 0:
            raise ValueError("progress_weight must be positive")
        if self.state_normalization not in {"none", "l2"}:
            raise ValueError("state_normalization must be 'none' or 'l2'")


@dataclass
class FlowSignatureEncoding:
    order1_prefix: np.ndarray
    order2_prefix: np.ndarray
    total_variation: np.ndarray
    phase_grid: np.ndarray
    projection: np.ndarray
    feature_metadata: dict[str, int | float | str]


def make_orthogonal_projection(
    input_dim: int,
    output_dim: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a deterministic data-independent orthogonal JL sketch."""

    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    out = min(int(input_dim), int(output_dim))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(input_dim, out, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.to(device=device, dtype=dtype)


def _normalize_states(states: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    if mode == "none":
        return states
    if mode == "l2":
        return states / states.norm(dim=-1, keepdim=True).clamp_min(eps)
    raise ValueError(f"unknown state normalization {mode!r}")


def _shuffle_increments(
    increments: torch.Tensor,
    lengths: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Permute hidden increments while keeping endpoints and step counts fixed."""

    shuffled = increments.clone()
    rng = np.random.default_rng(int(seed))
    for i, length in enumerate(lengths.detach().cpu().tolist()):
        k = int(length)
        if k > 1:
            perm = torch.as_tensor(rng.permutation(k), device=increments.device)
            shuffled[i, :k] = increments[i, perm]
    return shuffled


def _gather_boundaries(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather B x boundary x ... tensors at B x phase boundary indices."""

    batch = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[batch, indices]


def _prefix_logsignatures(
    hidden_increments: torch.Tensor,
    increment_counts: torch.Tensor,
    *,
    phase_points: int,
    progress_weight: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact order-1 and order-2 prefixes of piecewise-linear paths.

    Parameters
    ----------
    hidden_increments:
        Tensor shaped ``[B, K_max, L, P]``.
    increment_counts:
        Number of valid increments for every trajectory.

    Returns
    -------
    order1, order2, total_variation
        Prefix features on a common normalized phase grid. The endpoint is
        exact; intermediate grid points split an original segment exactly,
        rather than resampling and shortcutting path corners.
    """

    if hidden_increments.ndim != 4:
        raise ValueError("hidden_increments must have shape [batch, time, layer, dim]")
    batch_size, max_increments, n_layers, hidden_dim = hidden_increments.shape
    if torch.any(increment_counts < 1):
        raise ValueError("every trajectory must contain at least one increment")

    device = hidden_increments.device
    dtype = hidden_increments.dtype
    active = (
        torch.arange(max_increments, device=device)[None, :]
        < increment_counts[:, None]
    )
    hidden_increments = hidden_increments * active[:, :, None, None]
    hidden_norms = hidden_increments.norm(dim=-1)
    shared_segment_length = hidden_norms.mean(dim=-1)
    shared_total_length = shared_segment_length.sum(dim=1, keepdim=True)
    uniform_delta = active.to(dtype=dtype) / increment_counts.to(dtype=dtype)[:, None]
    arc_delta = torch.where(
        shared_total_length > float(eps),
        shared_segment_length / shared_total_length.clamp_min(float(eps)),
        uniform_delta,
    )
    time_delta = (
        arc_delta[:, :, None, None]
        .expand(batch_size, max_increments, n_layers, 1)
        * float(progress_weight)
    )
    increments = torch.cat([hidden_increments, time_delta], dim=-1)
    coord_dim = hidden_dim + 1

    prefix = torch.zeros(batch_size, n_layers, coord_dim, device=device, dtype=dtype)
    area = torch.zeros(
        batch_size,
        n_layers,
        coord_dim,
        coord_dim,
        device=device,
        dtype=dtype,
    )
    variation = torch.zeros(batch_size, n_layers, device=device, dtype=dtype)
    prefix_boundaries = [prefix.clone()]
    area_boundaries = [area.clone()]
    variation_boundaries = [variation.clone()]
    for step in range(max_increments):
        delta = increments[:, step]
        area = area + 0.5 * (
            prefix.unsqueeze(-1) * delta.unsqueeze(-2)
            - delta.unsqueeze(-1) * prefix.unsqueeze(-2)
        )
        prefix = prefix + delta
        variation = variation + hidden_norms[:, step]
        prefix_boundaries.append(prefix.clone())
        area_boundaries.append(area.clone())
        variation_boundaries.append(variation.clone())

    prefix_at_boundary = torch.stack(prefix_boundaries, dim=1)
    area_at_boundary = torch.stack(area_boundaries, dim=1)
    variation_at_boundary = torch.stack(variation_boundaries, dim=1)

    phase = torch.linspace(
        1.0 / phase_points,
        1.0,
        phase_points,
        device=device,
        dtype=dtype,
    )
    arc_boundaries = torch.cat(
        [
            torch.zeros(batch_size, 1, device=device, dtype=dtype),
            torch.cumsum(arc_delta, dim=1),
        ],
        dim=1,
    )
    targets = phase[None, :].expand(batch_size, phase_points).contiguous()
    lower = torch.searchsorted(arc_boundaries.contiguous(), targets, right=True) - 1
    endpoint = targets >= 1.0 - 10.0 * float(eps)
    lower = torch.where(endpoint, increment_counts[:, None], lower)
    lower = torch.maximum(lower, torch.zeros_like(lower))
    lower = torch.minimum(lower, increment_counts[:, None])

    prefix0 = _gather_boundaries(prefix_at_boundary, lower)
    area0 = _gather_boundaries(area_at_boundary, lower)
    variation0 = _gather_boundaries(variation_at_boundary, lower)
    next_index = torch.minimum(lower, (increment_counts - 1)[:, None])
    next_delta = _gather_boundaries(increments, next_index)
    next_hidden_norm = _gather_boundaries(hidden_norms, next_index)
    arc_start = _gather_boundaries(arc_boundaries, lower)
    next_arc = _gather_boundaries(arc_delta, next_index)
    alpha = (targets - arc_start) / next_arc.clamp_min(float(eps))
    alpha = torch.where(endpoint, torch.zeros_like(alpha), alpha).clamp(0.0, 1.0)
    partial = next_delta * alpha[:, :, None, None]
    prefix_phase = prefix0 + partial
    area_phase = area0 + 0.5 * (
        prefix0.unsqueeze(-1) * partial.unsqueeze(-2)
        - partial.unsqueeze(-1) * prefix0.unsqueeze(-2)
    )
    variation_phase = variation0 + next_hidden_norm * alpha[:, :, None]
    safe_variation = variation_phase.clamp_min(float(eps))

    order1 = prefix_phase[..., :hidden_dim] / safe_variation.unsqueeze(-1)
    upper_i, upper_j = torch.triu_indices(coord_dim, coord_dim, offset=1, device=device)
    area_upper = area_phase[..., upper_i, upper_j]
    pair_scale = torch.empty_like(area_upper)
    hidden_hidden = upper_j < hidden_dim
    pair_scale[..., hidden_hidden] = safe_variation.unsqueeze(-1).square()
    pair_scale[..., ~hidden_hidden] = safe_variation.unsqueeze(-1)
    area_upper = area_upper / pair_scale.clamp_min(float(eps))

    order1 = order1.reshape(batch_size, phase_points, n_layers * hidden_dim)
    order2 = torch.cat(
        [
            order1.reshape(batch_size, phase_points, n_layers, hidden_dim),
            area_upper,
        ],
        dim=-1,
    ).reshape(batch_size, phase_points, -1)
    return order1, order2, variation_phase[:, -1]


def encode_reasoning_flows(
    trajectories: Sequence[np.ndarray],
    cfg: FlowSignatureConfig,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    include_shuffled: bool = True,
) -> tuple[FlowSignatureEncoding, FlowSignatureEncoding | None]:
    """Encode ragged ``[step, layer, hidden]`` trajectories in GPU batches."""

    cfg.validate()
    if not trajectories:
        raise ValueError("no trajectories to encode")
    arrays = [np.asarray(x) for x in trajectories]
    if any(x.ndim != 3 for x in arrays):
        raise ValueError("all trajectories must have shape [step, layer, hidden]")
    if any(x.shape[0] < 2 for x in arrays):
        raise ValueError("all trajectories must have at least two states")
    n_layers = int(arrays[0].shape[1])
    hidden_dim = int(arrays[0].shape[2])
    if any(x.shape[1:] != (n_layers, hidden_dim) for x in arrays):
        raise ValueError("all trajectories must share layer and hidden dimensions")

    device = torch.device(device)
    projection = make_orthogonal_projection(
        hidden_dim,
        cfg.projection_dim,
        seed=cfg.seed,
        device=device,
    )
    projected_dim = int(projection.shape[1])
    chronological_o1: list[np.ndarray] = []
    chronological_o2: list[np.ndarray] = []
    chronological_tv: list[np.ndarray] = []
    shuffled_o1: list[np.ndarray] = []
    shuffled_o2: list[np.ndarray] = []
    shuffled_tv: list[np.ndarray] = []

    for start in range(0, len(arrays), max(1, int(batch_size))):
        block = arrays[start : start + max(1, int(batch_size))]
        lengths = torch.as_tensor([x.shape[0] - 1 for x in block], device=device)
        max_states = max(x.shape[0] for x in block)
        padded = torch.zeros(
            len(block),
            max_states,
            n_layers,
            hidden_dim,
            dtype=torch.float32,
            device=device,
        )
        for row, array in enumerate(block):
            value = torch.as_tensor(array, dtype=torch.float32, device=device)
            padded[row, : value.shape[0]] = value
        padded = _normalize_states(padded, cfg.state_normalization, cfg.eps)
        projected = torch.einsum("btld,dp->btlp", padded, projection)
        increments = projected[:, 1:] - projected[:, :-1]
        o1, o2, tv = _prefix_logsignatures(
            increments,
            lengths,
            phase_points=cfg.phase_points,
            progress_weight=cfg.progress_weight,
            eps=cfg.eps,
        )
        chronological_o1.append(o1.detach().cpu().numpy().astype(np.float32, copy=False))
        chronological_o2.append(o2.detach().cpu().numpy().astype(np.float32, copy=False))
        chronological_tv.append(tv.detach().cpu().numpy().astype(np.float32, copy=False))

        if include_shuffled:
            shuffled_increments = _shuffle_increments(
                increments,
                lengths,
                seed=cfg.seed + 104729 + start,
            )
            so1, so2, stv = _prefix_logsignatures(
                shuffled_increments,
                lengths,
                phase_points=cfg.phase_points,
                progress_weight=cfg.progress_weight,
                eps=cfg.eps,
            )
            shuffled_o1.append(so1.detach().cpu().numpy().astype(np.float32, copy=False))
            shuffled_o2.append(so2.detach().cpu().numpy().astype(np.float32, copy=False))
            shuffled_tv.append(stv.detach().cpu().numpy().astype(np.float32, copy=False))

    pair_count = projected_dim * (projected_dim + 1) // 2
    metadata: dict[str, int | float | str] = {
        "hidden_dim": hidden_dim,
        "projected_dim": projected_dim,
        "num_layers": n_layers,
        "phase_points": cfg.phase_points,
        "order1_features_per_layer": projected_dim,
        "order2_features_per_layer": projected_dim + pair_count,
        "progress_weight": cfg.progress_weight,
        "state_normalization": cfg.state_normalization,
        "normalization": "prefix_total_variation",
        "second_level": "antisymmetric_levy_area",
    }
    phase_grid = np.linspace(1.0 / cfg.phase_points, 1.0, cfg.phase_points, dtype=np.float32)
    chronological = FlowSignatureEncoding(
        order1_prefix=np.concatenate(chronological_o1, axis=0),
        order2_prefix=np.concatenate(chronological_o2, axis=0),
        total_variation=np.concatenate(chronological_tv, axis=0),
        phase_grid=phase_grid,
        projection=projection.detach().cpu().numpy().astype(np.float32, copy=False),
        feature_metadata=metadata,
    )
    shuffled = None
    if include_shuffled:
        shuffled = FlowSignatureEncoding(
            order1_prefix=np.concatenate(shuffled_o1, axis=0),
            order2_prefix=np.concatenate(shuffled_o2, axis=0),
            total_variation=np.concatenate(shuffled_tv, axis=0),
            phase_grid=phase_grid,
            projection=chronological.projection,
            feature_metadata={**metadata, "increment_order": "deterministically_shuffled"},
        )
    return chronological, shuffled
