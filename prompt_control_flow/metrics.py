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
    question_token_indices: np.ndarray | None = None,
    response_token_start: int,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
    subspace_k: int,
    prefix_k: int,
    rng: np.random.Generator,
    center_subspaces: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute per-step prompt/prefix/random residual-flow metrics.

    ``layers`` contains post-block hidden-state depths.  Depth ``d`` means
    ``hidden_states[d]`` and its incoming block update is
    ``hidden_states[d] - hidden_states[d - 1]``.  For a token at position
    ``p``, ``logits[p - 1]`` predicts ``input_ids[p]``, so all online flow
    measurements use prediction positions ``p - 1``.
    """

    if _uses_torch_backend(hidden_states):
        return _compute_step_prompt_flow_metrics_torch(
            hidden_states,
            prompt_token_indices=prompt_token_indices,
            question_token_indices=question_token_indices,
            response_token_start=response_token_start,
            step_ranges=step_ranges,
            layers=layers,
            subspace_k=subspace_k,
            prefix_k=prefix_k,
            rng=rng,
            center_subspaces=center_subspaces,
        )

    if len(step_ranges) == 0:
        raise ValueError("empty step_ranges")
    n_steps = len(step_ranges)
    metric_names = [
        MetricNames.PROMPT_FRAC,
        MetricNames.PREFIX_FRAC,
        MetricNames.RANDOM_FRAC,
        MetricNames.OFF_PROMPT,
        MetricNames.QUESTION_FRAC,
        MetricNames.OFF_QUESTION,
        MetricNames.PROMPT_CONTROL_RATIO,
        MetricNames.PREFIX_LOCK_RATIO,
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
        question_fracs: List[np.ndarray] = []

        for layer in layers:
            depth = int(layer)
            if depth <= 0 or depth >= len(hidden_states):
                continue
            h_before = np.asarray(hidden_states[depth - 1], dtype=np.float64)
            h_after = np.asarray(hidden_states[depth], dtype=np.float64)
            valid_pred = pred_positions[
                (pred_positions >= 0)
                & (pred_positions < h_before.shape[0])
                & (pred_positions < h_after.shape[0])
            ]
            if valid_pred.size == 0:
                continue
            delta = h_after[valid_pred] - h_before[valid_pred]
            if delta.ndim != 2 or delta.shape[0] == 0:
                continue

            prompt_idx = prompt_token_indices[prompt_token_indices < h_before.shape[0]]
            prompt_idx = prompt_idx[prompt_idx >= 0]
            if prompt_idx.size >= 1:
                try:
                    q_basis = orthonormal_basis(
                        h_before[prompt_idx],
                        subspace_k,
                        center=center_subspaces,
                    ).basis
                    prompt_fracs.append(projection_energy_fraction(delta, q_basis))
                    random_fracs.append(projection_energy_fraction(delta, random_basis(delta.shape[1], q_basis.shape[1], rng)))
                except ValueError:
                    pass

            if question_token_indices is not None:
                question_idx = np.asarray(question_token_indices, dtype=np.int64)
                question_idx = question_idx[
                    (question_idx >= 0) & (question_idx < h_before.shape[0])
                ]
                if question_idx.size >= 1:
                    try:
                        question_basis = orthonormal_basis(
                            h_before[question_idx],
                            subspace_k,
                            center=center_subspaces,
                        ).basis
                        question_fracs.append(
                            projection_energy_fraction(delta, question_basis)
                        )
                    except ValueError:
                        pass

            prefix_idx = np.arange(
                int(response_token_start), max(int(a), int(response_token_start)), dtype=np.int64
            )
            prefix_idx = prefix_idx[
                (prefix_idx >= 0) & (prefix_idx < h_before.shape[0])
            ]
            if prefix_idx.size >= 2:
                try:
                    p_basis = orthonormal_basis(
                        h_before[prefix_idx],
                        prefix_k,
                        center=center_subspaces,
                    ).basis
                    prefix_fracs.append(projection_energy_fraction(delta, p_basis))
                except ValueError:
                    pass

        pf = np.concatenate(prompt_fracs) if prompt_fracs else np.asarray([], dtype=np.float64)
        prf = np.concatenate(prefix_fracs) if prefix_fracs else np.asarray([], dtype=np.float64)
        rf = np.concatenate(random_fracs) if random_fracs else np.asarray([], dtype=np.float64)
        qf = np.concatenate(question_fracs) if question_fracs else np.asarray([], dtype=np.float64)

        out[MetricNames.PROMPT_FRAC][j] = _safe_nanmean(pf)
        out[MetricNames.PREFIX_FRAC][j] = _safe_nanmean(prf)
        out[MetricNames.RANDOM_FRAC][j] = _safe_nanmean(rf)
        out[MetricNames.OFF_PROMPT][j] = 1.0 - out[MetricNames.PROMPT_FRAC][j] if np.isfinite(out[MetricNames.PROMPT_FRAC][j]) else float("nan")
        out[MetricNames.QUESTION_FRAC][j] = _safe_nanmean(qf)
        out[MetricNames.OFF_QUESTION][j] = 1.0 - out[MetricNames.QUESTION_FRAC][j] if np.isfinite(out[MetricNames.QUESTION_FRAC][j]) else float("nan")
        den = out[MetricNames.PREFIX_FRAC][j] + out[MetricNames.OFF_PROMPT][j] + EPS
        out[MetricNames.PROMPT_CONTROL_RATIO][j] = out[MetricNames.PROMPT_FRAC][j] / den if np.isfinite(den) else float("nan")
        out[MetricNames.PREFIX_LOCK_RATIO][j] = (out[MetricNames.PREFIX_FRAC][j] + out[MetricNames.OFF_PROMPT][j]) / (out[MetricNames.PROMPT_FRAC][j] + EPS)

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

    if _uses_torch_backend(hidden_states):
        return _compute_step_residual_vectors_torch(
            hidden_states, step_ranges=step_ranges, layers=layers
        )

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
            depth = int(layer)
            if depth <= 0 or depth >= len(hidden_states):
                chunks.append(np.zeros(hidden_dim, dtype=np.float32))
                continue
            h_before = np.asarray(hidden_states[depth - 1], dtype=np.float32)
            h_after = np.asarray(hidden_states[depth], dtype=np.float32)
            valid_pred = pred_positions[
                (pred_positions >= 0)
                & (pred_positions < h_before.shape[0])
                & (pred_positions < h_after.shape[0])
            ]
            if valid_pred.size == 0:
                chunks.append(np.zeros(hidden_dim, dtype=np.float32))
                continue
            delta = h_after[valid_pred] - h_before[valid_pred]
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

    if _uses_torch_backend(hidden_states):
        return _compute_step_layer_state_vectors_torch(
            hidden_states, step_ranges=step_ranges, layers=layers
        )

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


def compute_step_boundary_state_vectors(
    hidden_states: Sequence[np.ndarray],
    *,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal pre-step and descriptive step-end states.

    For inclusive step range ``[a, b]``, the pre-step state is at token
    ``a - 1`` and therefore cannot contain any token from that step.  The end
    state is at ``b`` and is useful only for retrospective geometry.  Both
    tensors have shape ``[step, layer, hidden]``.
    """

    if _uses_torch_backend(hidden_states):
        return _compute_step_boundary_state_vectors_torch(
            hidden_states, step_ranges=step_ranges, layers=layers
        )

    if len(step_ranges) == 0:
        return (
            np.zeros((0, len(layers), 0), dtype=np.float32),
            np.zeros((0, len(layers), 0), dtype=np.float32),
        )
    hidden_dim = next(
        (int(np.asarray(h).shape[1]) for h in hidden_states if np.asarray(h).ndim == 2),
        None,
    )
    if hidden_dim is None:
        raise ValueError("hidden_states must contain rank-2 arrays")
    pre = np.full(
        (len(step_ranges), len(layers), hidden_dim), np.nan, dtype=np.float32
    )
    end = np.full_like(pre, np.nan)
    for step_pos, (a, b) in enumerate(step_ranges):
        pre_pos = int(a) - 1
        end_pos = int(b)
        for layer_pos, layer in enumerate(layers):
            depth = int(layer)
            if depth < 0 or depth >= len(hidden_states):
                continue
            state = np.asarray(hidden_states[depth], dtype=np.float32)
            if 0 <= pre_pos < state.shape[0]:
                pre[step_pos, layer_pos] = state[pre_pos]
            if 0 <= end_pos < state.shape[0]:
                end[step_pos, layer_pos] = state[end_pos]
    return pre, end


def compute_response_token_layer_states(
    hidden_states: Sequence[np.ndarray],
    *,
    response_token_range: Tuple[int, int],
    layers: Sequence[int],
) -> np.ndarray:
    """Return selected-depth states on the exact response-token axis.

    The output shape is ``[response_token, layer, hidden]``.  These are
    post-token states and must not be used as online predictors for the same
    token; online analyses should shift to the preceding prediction position.
    """

    if _uses_torch_backend(hidden_states):
        return _compute_response_token_layer_states_torch(
            hidden_states,
            response_token_range=response_token_range,
            layers=layers,
        )

    start, stop = (int(response_token_range[0]), int(response_token_range[1]))
    if start < 0 or stop < start:
        raise ValueError(f"invalid response token range {(start, stop)}")
    hidden_dim = next(
        (int(np.asarray(h).shape[1]) for h in hidden_states if np.asarray(h).ndim == 2),
        None,
    )
    if hidden_dim is None:
        raise ValueError("hidden_states must contain rank-2 arrays")
    out = np.full(
        (stop - start, len(layers), hidden_dim), np.nan, dtype=np.float32
    )
    for layer_pos, layer in enumerate(layers):
        depth = int(layer)
        if depth < 0 or depth >= len(hidden_states):
            raise ValueError(
                f"requested hidden-state depth {depth} is outside [0, {len(hidden_states) - 1}]"
            )
        state = np.asarray(hidden_states[depth], dtype=np.float32)
        if stop > state.shape[0]:
            raise ValueError(
                f"response token range {(start, stop)} exceeds depth {depth} length {state.shape[0]}"
            )
        out[:, layer_pos] = state[start:stop]
    return out


def compute_prompt_token_layer_states(
    hidden_states: Sequence[np.ndarray],
    *,
    prompt_token_range: Tuple[int, int],
    layers: Sequence[int],
) -> np.ndarray:
    """Return selected-depth states on the exact rendered-prompt token axis."""

    return compute_response_token_layer_states(
        hidden_states,
        response_token_range=prompt_token_range,
        layers=layers,
    )


def _uses_torch_backend(hidden_states: Sequence[object]) -> bool:
    return any(
        hasattr(value, "detach") and hasattr(value, "device")
        for value in hidden_states
    )


def _torch_state_metadata(hidden_states: Sequence[object]) -> tuple[object, int]:
    for value in hidden_states:
        if hasattr(value, "detach") and len(value.shape) == 2:
            return value.device, int(value.shape[1])
    raise ValueError("hidden_states must contain rank-2 torch tensors")


def _torch_to_numpy(value) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _torch_basis(x, k: int, *, center: bool):
    """Top right-singular directions using a small token-token Gram matrix."""

    import torch

    value = x.float()
    value = value[torch.isfinite(value).all(dim=1)]
    if value.shape[0] == 0:
        raise ValueError("cannot build a basis from zero finite rows")
    mean = value.mean(dim=0, keepdim=True) if center else torch.zeros_like(value[:1])
    centered = value - mean
    if centered.shape[0] == 1:
        norm = torch.linalg.vector_norm(centered[0])
        if float(norm.item()) <= EPS:
            return centered.new_zeros((centered.shape[1], 0))
        return (centered[0] / norm).reshape(-1, 1)

    n_rows, n_dim = (int(centered.shape[0]), int(centered.shape[1]))
    requested = min(max(int(k), 0), n_rows, n_dim)
    if requested == 0:
        return centered.new_zeros((n_dim, 0))
    if n_rows <= n_dim:
        gram = centered @ centered.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        tolerance = (
            torch.finfo(centered.dtype).eps
            * max(n_rows, n_dim)
            * eigenvalues[0].clamp_min(1.0)
        )
        rank = min(requested, int(torch.sum(eigenvalues > tolerance).item()))
        if rank == 0:
            return centered.new_zeros((n_dim, 0))
        values = eigenvalues[:rank].clamp_min(float(tolerance.item()))
        basis = centered.T @ eigenvectors[:, :rank]
        basis = basis / torch.sqrt(values)[None, :]
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        return basis[:, :rank]
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    tolerance = (
        torch.finfo(centered.dtype).eps
        * max(n_rows, n_dim)
        * singular[0].clamp_min(1.0)
    )
    rank = min(requested, int(torch.sum(singular > tolerance).item()))
    return vh[:rank].T.contiguous()


def _torch_random_basis(dim: int, rank: int, rng: np.random.Generator, device):
    import torch

    rank = min(max(int(rank), 0), int(dim))
    if rank == 0:
        return torch.zeros((int(dim), 0), dtype=torch.float32, device=device)
    matrix = torch.as_tensor(
        rng.normal(size=(int(dim), rank)), dtype=torch.float32, device=device
    )
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis[:, :rank]


def _torch_projection_fraction(x, basis):
    import torch

    value = x.float()
    denominator = torch.sum(value * value, dim=1).clamp_min(EPS)
    if basis.numel() == 0 or basis.shape[1] == 0:
        numerator = torch.zeros_like(denominator)
    else:
        projected = value @ basis.float()
        numerator = torch.sum(projected * projected, dim=1)
    result = numerator / denominator
    return torch.clamp(result, 0.0, 1.0)


def _compute_step_prompt_flow_metrics_torch(
    hidden_states: Sequence[object],
    *,
    prompt_token_indices: np.ndarray,
    question_token_indices: np.ndarray | None,
    response_token_start: int,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
    subspace_k: int,
    prefix_k: int,
    rng: np.random.Generator,
    center_subspaces: bool,
) -> Dict[str, np.ndarray]:
    import torch

    if len(step_ranges) == 0:
        raise ValueError("empty step_ranges")
    metric_names = [
        MetricNames.PROMPT_FRAC,
        MetricNames.PREFIX_FRAC,
        MetricNames.RANDOM_FRAC,
        MetricNames.OFF_PROMPT,
        MetricNames.QUESTION_FRAC,
        MetricNames.OFF_QUESTION,
        MetricNames.PROMPT_CONTROL_RATIO,
        MetricNames.PREFIX_LOCK_RATIO,
        MetricNames.STEP_LEN,
        MetricNames.REL_POS,
    ]
    n_steps = len(step_ranges)
    out = {
        name: np.full(n_steps, np.nan, dtype=np.float64) for name in metric_names
    }
    prepared: dict[int, tuple[object, object, object, object, object]] = {}
    for layer in layers:
        depth = int(layer)
        if depth <= 0 or depth >= len(hidden_states):
            continue
        before = hidden_states[depth - 1]
        after = hidden_states[depth]
        prompt_idx = torch.as_tensor(
            np.asarray(prompt_token_indices, dtype=np.int64),
            dtype=torch.long,
            device=before.device,
        )
        prompt_idx = prompt_idx[
            (prompt_idx >= 0) & (prompt_idx < before.shape[0])
        ]
        prompt_basis = _torch_basis(
            before.index_select(0, prompt_idx).float(),
            subspace_k,
            center=center_subspaces,
        )
        random = _torch_random_basis(
            int(before.shape[1]), int(prompt_basis.shape[1]), rng, before.device
        )
        question_basis = before.new_zeros((before.shape[1], 0))
        if question_token_indices is not None:
            question_idx = torch.as_tensor(
                np.asarray(question_token_indices, dtype=np.int64),
                dtype=torch.long,
                device=before.device,
            )
            question_idx = question_idx[
                (question_idx >= 0) & (question_idx < before.shape[0])
            ]
            if question_idx.numel() > 0:
                question_basis = _torch_basis(
                    before.index_select(0, question_idx).float(),
                    subspace_k,
                    center=center_subspaces,
                )
        prepared[depth] = (before, after, prompt_basis, question_basis, random)

    with torch.inference_mode():
        for step_pos, (a, b) in enumerate(step_ranges):
            targets = torch.arange(
                max(int(a), 1),
                int(b) + 1,
                dtype=torch.long,
                device=next(iter(prepared.values()))[0].device,
            ) if prepared else torch.empty(0, dtype=torch.long)
            if targets.numel() == 0:
                continue
            out[MetricNames.STEP_LEN][step_pos] = float(targets.numel())
            out[MetricNames.REL_POS][step_pos] = float(
                step_pos / max(n_steps - 1, 1)
            )
            prompt_values = []
            prefix_values = []
            random_values = []
            question_values = []
            for before, after, prompt_basis, question_basis, random in prepared.values():
                prediction = targets.to(before.device) - 1
                prediction = prediction[
                    (prediction >= 0)
                    & (prediction < before.shape[0])
                    & (prediction < after.shape[0])
                ]
                if prediction.numel() == 0:
                    continue
                delta = after.index_select(0, prediction).float() - before.index_select(
                    0, prediction
                ).float()
                prompt_values.append(_torch_projection_fraction(delta, prompt_basis))
                random_values.append(_torch_projection_fraction(delta, random))
                if question_basis.shape[1] > 0:
                    question_values.append(
                        _torch_projection_fraction(delta, question_basis)
                    )
                prefix = torch.arange(
                    int(response_token_start),
                    max(int(a), int(response_token_start)),
                    dtype=torch.long,
                    device=before.device,
                )
                prefix = prefix[(prefix >= 0) & (prefix < before.shape[0])]
                if prefix.numel() >= 2:
                    prefix_basis = _torch_basis(
                        before.index_select(0, prefix).float(),
                        prefix_k,
                        center=center_subspaces,
                    )
                    prefix_values.append(
                        _torch_projection_fraction(delta, prefix_basis)
                    )

            def mean_or_nan(values) -> float:
                if not values:
                    return float("nan")
                return float(torch.cat(values).mean().item())

            prompt_fraction = mean_or_nan(prompt_values)
            prefix_fraction = mean_or_nan(prefix_values)
            random_fraction = mean_or_nan(random_values)
            question_fraction = mean_or_nan(question_values)
            out[MetricNames.PROMPT_FRAC][step_pos] = prompt_fraction
            out[MetricNames.PREFIX_FRAC][step_pos] = prefix_fraction
            out[MetricNames.RANDOM_FRAC][step_pos] = random_fraction
            out[MetricNames.QUESTION_FRAC][step_pos] = question_fraction
            if np.isfinite(prompt_fraction):
                out[MetricNames.OFF_PROMPT][step_pos] = 1.0 - prompt_fraction
            if np.isfinite(question_fraction):
                out[MetricNames.OFF_QUESTION][step_pos] = 1.0 - question_fraction
            off_prompt = out[MetricNames.OFF_PROMPT][step_pos]
            denominator = prefix_fraction + off_prompt + EPS
            if np.isfinite(denominator):
                out[MetricNames.PROMPT_CONTROL_RATIO][step_pos] = (
                    prompt_fraction / denominator
                )
            if np.isfinite(prefix_fraction) and np.isfinite(off_prompt):
                out[MetricNames.PREFIX_LOCK_RATIO][step_pos] = (
                    prefix_fraction + off_prompt
                ) / (prompt_fraction + EPS)
    return out


def _compute_step_residual_vectors_torch(
    hidden_states: Sequence[object],
    *,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
) -> np.ndarray:
    import torch

    device, hidden_dim = _torch_state_metadata(hidden_states)
    if len(step_ranges) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    result = torch.zeros(
        (len(step_ranges), len(layers), hidden_dim),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for step_pos, (a, b) in enumerate(step_ranges):
            prediction = torch.arange(
                max(int(a), 1) - 1,
                int(b),
                dtype=torch.long,
                device=device,
            )
            for layer_pos, layer in enumerate(layers):
                depth = int(layer)
                if depth <= 0 or depth >= len(hidden_states):
                    continue
                before = hidden_states[depth - 1]
                after = hidden_states[depth]
                valid = prediction[
                    (prediction >= 0)
                    & (prediction < before.shape[0])
                    & (prediction < after.shape[0])
                ]
                if valid.numel() > 0:
                    result[step_pos, layer_pos] = (
                        after.index_select(0, valid).float()
                        - before.index_select(0, valid).float()
                    ).mean(dim=0)
    return _torch_to_numpy(result.reshape(len(step_ranges), -1))


def _compute_step_layer_state_vectors_torch(
    hidden_states: Sequence[object],
    *,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
) -> np.ndarray:
    import torch

    device, hidden_dim = _torch_state_metadata(hidden_states)
    if len(step_ranges) == 0:
        return np.zeros((0, len(layers), 0), dtype=np.float32)
    result = torch.zeros(
        (len(step_ranges), len(layers), hidden_dim),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for step_pos, (a, b) in enumerate(step_ranges):
            positions = torch.arange(
                int(a), int(b) + 1, dtype=torch.long, device=device
            )
            for layer_pos, layer in enumerate(layers):
                depth = int(layer)
                state = hidden_states[depth]
                valid = positions[(positions >= 0) & (positions < state.shape[0])]
                if valid.numel() > 0:
                    result[step_pos, layer_pos] = state.index_select(
                        0, valid
                    ).float().mean(dim=0)
    return _torch_to_numpy(result)


def _compute_step_boundary_state_vectors_torch(
    hidden_states: Sequence[object],
    *,
    step_ranges: Sequence[Tuple[int, int]],
    layers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device, hidden_dim = _torch_state_metadata(hidden_states)
    if len(step_ranges) == 0:
        empty = np.zeros((0, len(layers), 0), dtype=np.float32)
        return empty, empty.copy()
    shape = (len(step_ranges), len(layers), hidden_dim)
    pre = torch.full(shape, float("nan"), dtype=torch.float32, device=device)
    end = torch.full_like(pre, float("nan"))
    with torch.inference_mode():
        for step_pos, (a, b) in enumerate(step_ranges):
            for layer_pos, layer in enumerate(layers):
                state = hidden_states[int(layer)]
                if 0 <= int(a) - 1 < state.shape[0]:
                    pre[step_pos, layer_pos] = state[int(a) - 1].float()
                if 0 <= int(b) < state.shape[0]:
                    end[step_pos, layer_pos] = state[int(b)].float()
    return _torch_to_numpy(pre), _torch_to_numpy(end)


def _compute_response_token_layer_states_torch(
    hidden_states: Sequence[object],
    *,
    response_token_range: Tuple[int, int],
    layers: Sequence[int],
) -> np.ndarray:
    import torch

    start, stop = (int(response_token_range[0]), int(response_token_range[1]))
    if start < 0 or stop < start:
        raise ValueError(f"invalid response token range {(start, stop)}")
    device, hidden_dim = _torch_state_metadata(hidden_states)
    result = torch.full(
        (stop - start, len(layers), hidden_dim),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for layer_pos, layer in enumerate(layers):
            depth = int(layer)
            state = hidden_states[depth]
            if stop > state.shape[0]:
                raise ValueError(
                    f"response token range {(start, stop)} exceeds depth {depth} "
                    f"length {state.shape[0]}"
                )
            result[:, layer_pos] = state[start:stop].float()
    return _torch_to_numpy(result)


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
