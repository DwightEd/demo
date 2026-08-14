from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .replay import _decision_contexts

FFN_DIRECTION_ID = -32768


def source_binned_residual_messages(
    attention,
    expanded_values,
    output_weight,
    *,
    source_step_ids: np.ndarray,
    query_position: int,
):
    """Aggregate exact decision-token attention writes by source step.

    Token contexts are grouped before the output projection, avoiding the
    prohibitively large dense ``[head, source_token, hidden]`` tensor.
    """
    import torch

    contexts, _mass, head_projection = _decision_contexts(
        attention,
        expanded_values,
        output_weight,
        query_position=query_position,
    )
    source_ids = np.asarray(source_step_ids)
    if source_ids.ndim != 1 or source_ids.shape[0] != contexts.shape[1]:
        raise ValueError("source_step_ids must align with the attention key axis")
    if not np.issubdtype(source_ids.dtype, np.integer):
        raise ValueError("source_step_ids must be integers")

    unique_ids = np.unique(source_ids.astype(np.int64, copy=False))
    messages = []
    for source_id in unique_ids:
        mask = torch.as_tensor(
            source_ids == source_id, device=contexts.device, dtype=torch.bool
        )
        grouped_context = contexts[:, mask].sum(dim=1)
        messages.append(
            torch.einsum("hd,ohd->o", grouped_context, head_projection)
        )
    stacked = torch.stack(messages, dim=0)
    return unique_ids, stacked, stacked.sum(dim=0)


def categorical_fisher_gram(
    probabilities: np.ndarray, logit_jacobian: np.ndarray
) -> np.ndarray:
    """Project the categorical Fisher metric onto intervention directions."""
    probability = np.asarray(probabilities, dtype=np.float64)
    jacobian = np.asarray(logit_jacobian, dtype=np.float64)
    if probability.ndim != 1 or probability.size < 2:
        raise ValueError("probabilities must be a vocabulary vector")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(probability.sum(), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("probabilities must sum to one")
    if jacobian.ndim != 2 or jacobian.shape[1] != probability.size:
        raise ValueError("logit_jacobian must have shape [direction,vocabulary]")
    if jacobian.shape[0] < 1 or not np.isfinite(jacobian).all():
        raise ValueError("logit_jacobian must contain finite directions")

    centered = jacobian - (jacobian @ probability)[:, None]
    gram = (centered * probability[None, :]) @ centered.T
    return 0.5 * (gram + gram.T)


def fisher_diagnostics(
    gram: np.ndarray, *, direction_source_ids: np.ndarray
) -> Mapping[str, float | int]:
    """Return coordinate-free spectrum and typed top-direction loadings."""
    matrix = np.asarray(gram, dtype=np.float64)
    source_ids = np.asarray(direction_source_ids)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be a square matrix")
    if source_ids.shape != (matrix.shape[0],):
        raise ValueError("direction_source_ids must align with gram")
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=1e-7, atol=1e-9
    ):
        raise ValueError("gram must be finite and symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-9
    if float(eigenvalues.min(initial=0.0)) < -tolerance:
        raise ValueError("gram must be positive semidefinite")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    positive = eigenvalues[eigenvalues > tolerance]
    if total > 0.0:
        normalized = eigenvalues[eigenvalues > 0.0] / total
        effective_rank = float(np.exp(-np.sum(normalized * np.log(normalized))))
        anisotropy = largest / total
        top_loading = np.square(eigenvectors[:, -1])
    else:
        effective_rank = 0.0
        anisotropy = 0.0
        top_loading = np.zeros(matrix.shape[0], dtype=np.float64)
    condition_number = (
        float(largest / positive[0]) if positive.size > 0 else 0.0
    )
    return {
        "largest_eigenvalue": largest,
        "fisher_trace": total,
        "effective_rank": effective_rank,
        "anisotropy": float(anisotropy),
        "condition_number": condition_number,
        "numerical_rank": int(positive.size),
        "top_prompt_loading": float(top_loading[source_ids == -1].sum()),
        "top_ffn_loading": float(
            top_loading[source_ids == FFN_DIRECTION_ID].sum()
        ),
    }


__all__ = [
    "FFN_DIRECTION_ID",
    "categorical_fisher_gram",
    "fisher_diagnostics",
    "source_binned_residual_messages",
]
