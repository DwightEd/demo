from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


EPS = 1e-12
RESIDUAL_RANKS = (1, 2, 4, 8, 16)


def _unit_rows(hidden: np.ndarray) -> np.ndarray:
    H = np.asarray(hidden, float)
    if H.ndim != 2:
        raise ValueError("token cloud must have shape [tokens, hidden_dim]")
    good = np.isfinite(H).all(axis=1)
    H = H[good]
    if H.size == 0:
        return np.empty((0, hidden.shape[1] if np.asarray(hidden).ndim == 2 else 0))
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    return H[norms[:, 0] > EPS] / norms[norms[:, 0] > EPS]


def gram_spectrum(token_hidden: np.ndarray, *, center: bool = False) -> np.ndarray:
    """Normalized non-zero spectrum of ``G = U U^T / n``.

    Building the token-token Gram matrix avoids a hidden-width by hidden-width
    covariance while preserving its non-zero eigenvalues.
    """
    U = _unit_rows(token_hidden)
    if len(U) == 0:
        return np.empty(0, dtype=float)
    if center:
        U = U - np.mean(U, axis=0, keepdims=True)
    G = (U @ U.T) / float(len(U))
    eig = np.linalg.eigvalsh(0.5 * (G + G.T))
    eig = np.clip(eig[::-1], 0.0, None)
    eig = eig[eig > EPS]
    total = float(eig.sum())
    return eig / total if total > EPS else np.empty(0, dtype=float)


def anchor_subspace(
    anchor_vectors: np.ndarray,
    *,
    rank: Optional[int] = None,
    energy: float = 0.99,
) -> np.ndarray:
    """Orthonormal row basis spanning the selected prompt-anchor directions."""
    A = np.asarray(anchor_vectors, float)
    if A.ndim != 2:
        raise ValueError("anchor_vectors must have shape [anchors, hidden_dim]")
    good = np.isfinite(A).all(axis=1) & (np.linalg.norm(np.nan_to_num(A), axis=1) > EPS)
    A = A[good]
    if len(A) == 0:
        return np.empty((0, A.shape[1]), dtype=float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    max_rank = min(vt.shape)
    if rank is not None:
        r = max(1, min(int(rank), max_rank))
    else:
        e = s**2
        r = int(np.searchsorted(np.cumsum(e) / max(float(e.sum()), EPS), float(energy)) + 1)
        r = max(1, min(r, max_rank))
    return vt[:r]


def anchor_residual_cloud(
    token_hidden: np.ndarray,
    anchor_vectors: np.ndarray,
    *,
    anchor_rank: Optional[int] = None,
    anchor_energy: float = 0.99,
) -> tuple[np.ndarray, float, int]:
    """Remove the prompt-anchor subspace from a row-normalized token cloud.

    Returns ``(residual_cloud, residual_energy_ratio, fitted_anchor_rank)``.
    The residual energy preserves how much of the response state is not
    explainable by prompt anchors; the residual Gram spectrum describes the
    geometry of that unexplained component.
    """
    U = _unit_rows(token_hidden)
    A = np.asarray(anchor_vectors, float)
    if A.ndim != 2:
        raise ValueError("anchor_vectors must have shape [anchors, hidden_dim]")
    if U.ndim != 2 or A.shape[1] != U.shape[1]:
        raise ValueError("token cloud and anchor hidden dimensions differ")
    basis = anchor_subspace(A, rank=anchor_rank, energy=anchor_energy)
    if len(U) == 0 or len(basis) == 0:
        return U.copy(), float("nan"), int(len(basis))
    projection = (U @ basis.T) @ basis
    residual = U - projection
    total = float(np.sum(U**2))
    ratio = float(np.sum(residual**2) / max(total, EPS))
    return residual, ratio, int(len(basis))


def _energy_rank(p: np.ndarray, q: float) -> float:
    if p.size == 0:
        return float("nan")
    return float(np.searchsorted(np.cumsum(p), float(q)) + 1)


def gram_features(token_hidden: np.ndarray, *, center: bool = False) -> Dict[str, float]:
    p = gram_spectrum(token_hidden, center=center)
    out: Dict[str, float] = {
        "eff_rank": float("nan"),
        "tail_auc": float("nan"),
        "k50": float("nan"),
        "k75": float("nan"),
        "k90": float("nan"),
        "gap12": float("nan"),
        "n_eigs": float(len(p)),
    }
    for k in RESIDUAL_RANKS:
        out[f"resid_k{k}"] = float("nan")
    if p.size == 0:
        return out

    entropy = -float(np.sum(p * np.log(np.maximum(p, EPS))))
    cumulative = np.cumsum(p)
    out.update(
        eff_rank=float(np.exp(entropy)),
        tail_auc=float(np.mean(1.0 - cumulative)),
        k50=_energy_rank(p, 0.50),
        k75=_energy_rank(p, 0.75),
        k90=_energy_rank(p, 0.90),
        gap12=float(p[0] - (p[1] if len(p) > 1 else 0.0)),
    )
    for k in RESIDUAL_RANKS:
        out[f"resid_k{k}"] = float(max(0.0, 1.0 - p[:k].sum()))
    return out


def spectrum_distance(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """L2 and Jensen-Shannon distance between variable-length spectra."""
    x = np.asarray(a, float).reshape(-1)
    y = np.asarray(b, float).reshape(-1)
    n = max(len(x), len(y))
    if n == 0:
        return {"spectral_l2": float("nan"), "spectral_js": float("nan")}
    xp = np.zeros(n, float)
    yp = np.zeros(n, float)
    xp[: len(x)] = np.clip(x, 0.0, None)
    yp[: len(y)] = np.clip(y, 0.0, None)
    if xp.sum() <= EPS or yp.sum() <= EPS:
        return {"spectral_l2": float("nan"), "spectral_js": float("nan")}
    xp /= xp.sum()
    yp /= yp.sum()
    m = 0.5 * (xp + yp)
    klx = np.sum(np.where(xp > 0, xp * np.log(np.maximum(xp, EPS) / np.maximum(m, EPS)), 0.0))
    kly = np.sum(np.where(yp > 0, yp * np.log(np.maximum(yp, EPS) / np.maximum(m, EPS)), 0.0))
    return {
        "spectral_l2": float(np.linalg.norm(xp - yp)),
        "spectral_js": float(np.sqrt(max(0.0, 0.5 * (klx + kly)))),
    }


def ambiguity_mask(
    n: int,
    *,
    condition: Optional[Sequence[bool]] = None,
    spread: Optional[Sequence[float]] = None,
    uncertainty: Optional[Sequence[float]] = None,
    spread_threshold: Optional[float] = None,
    uncertainty_threshold: Optional[float] = None,
    combine: str = "or",
) -> np.ndarray:
    """Select the regime where second moments are intended to add information."""
    if condition is not None:
        mask = np.asarray(condition, bool)
        if mask.shape != (n,):
            raise ValueError("condition length differs from cloud sequence")
        return mask
    tests = []
    if spread is not None and spread_threshold is not None:
        s = np.asarray(spread, float)
        if s.shape != (n,):
            raise ValueError("spread length differs from cloud sequence")
        tests.append(np.isfinite(s) & (s >= float(spread_threshold)))
    if uncertainty is not None and uncertainty_threshold is not None:
        u = np.asarray(uncertainty, float)
        if u.shape != (n,):
            raise ValueError("uncertainty length differs from cloud sequence")
        tests.append(np.isfinite(u) & (u <= float(uncertainty_threshold)))
    if not tests:
        return np.ones(n, dtype=bool)
    if combine not in {"or", "and"}:
        raise ValueError("combine must be 'or' or 'and'")
    return np.logical_or.reduce(tests) if combine == "or" else np.logical_and.reduce(tests)


def _anchor_sets(anchor_vectors, n: int) -> list[Optional[np.ndarray]]:
    if anchor_vectors is None:
        return [None] * n
    if isinstance(anchor_vectors, np.ndarray) and anchor_vectors.dtype != object:
        A = np.asarray(anchor_vectors, float)
        if A.ndim == 2:
            return [A] * n
        if A.ndim == 3 and A.shape[0] == n:
            return [A[i] for i in range(n)]
        raise ValueError("anchor_vectors must be [anchors, dim] or [windows, anchors, dim]")
    values = list(anchor_vectors)
    if len(values) != n:
        raise ValueError("per-window anchor_vectors must align with clouds")
    return [None if x is None else np.asarray(x, float) for x in values]


def conditional_gram_geometry(
    clouds: Sequence[np.ndarray],
    *,
    anchor_vectors=None,
    anchor_rank: Optional[int] = None,
    anchor_energy: float = 0.99,
    condition: Optional[Sequence[bool]] = None,
    spread: Optional[Sequence[float]] = None,
    uncertainty: Optional[Sequence[float]] = None,
    spread_threshold: Optional[float] = None,
    uncertainty_threshold: Optional[float] = None,
    combine: str = "or",
    center: bool = False,
) -> Dict[str, np.ndarray]:
    """Raw and anchor-residual Gram geometry with an explicit ambiguity gate.

    ``anchor_vectors`` can be one global ``[anchors, dim]`` bank or a list/
    tensor of one bank per window.  Anchor-residual features are unavailable
    (NaN) when no real bank is provided; the function never silently substitutes
    a random or q-partition anchor.
    """
    clouds = list(clouds)
    n = len(clouds)
    active = ambiguity_mask(
        n,
        condition=condition,
        spread=spread,
        uncertainty=uncertainty,
        spread_threshold=spread_threshold,
        uncertainty_threshold=uncertainty_threshold,
        combine=combine,
    )
    spectra = [gram_spectrum(H, center=center) for H in clouds]
    rows = [gram_features(H, center=center) for H in clouds]
    names = list(rows[0]) if rows else [
        "eff_rank", "tail_auc", "k50", "k75", "k90", "gap12", "n_eigs"
    ] + [f"resid_k{k}" for k in RESIDUAL_RANKS]
    out: Dict[str, np.ndarray] = {
        name: np.asarray([row.get(name, np.nan) for row in rows], float) for name in names
    }
    out["condition_active"] = active.astype(float)
    for name in names:
        out[f"conditional_{name}"] = np.where(active, out[name], np.nan)

    l2 = np.full(n, np.nan)
    js = np.full(n, np.nan)
    for t in range(1, n):
        d = spectrum_distance(spectra[t - 1], spectra[t])
        l2[t] = d["spectral_l2"]
        js[t] = d["spectral_js"]
    out["spectral_l2_change"] = l2
    out["spectral_js_change"] = js
    out["conditional_spectral_l2_change"] = np.where(active, l2, np.nan)
    out["conditional_spectral_js_change"] = np.where(active, js, np.nan)

    anchors = _anchor_sets(anchor_vectors, n)
    residual_rows: list[Optional[Dict[str, float]]] = []
    residual_spectra: list[np.ndarray] = []
    residual_energy = np.full(n, np.nan)
    fitted_rank = np.full(n, np.nan)
    for i, (cloud, bank) in enumerate(zip(clouds, anchors)):
        if bank is None:
            residual_rows.append(None)
            residual_spectra.append(np.empty(0, dtype=float))
            continue
        residual, ratio, r = anchor_residual_cloud(
            cloud,
            bank,
            anchor_rank=anchor_rank,
            anchor_energy=anchor_energy,
        )
        residual_energy[i] = ratio
        fitted_rank[i] = float(r)
        residual_rows.append(gram_features(residual, center=center))
        residual_spectra.append(gram_spectrum(residual, center=center))

    out["residual_energy_ratio"] = residual_energy
    out["anchor_subspace_rank"] = fitted_rank
    out["conditional_residual_energy_ratio"] = np.where(active, residual_energy, np.nan)
    for name in names:
        values = np.asarray(
            [np.nan if row is None else row.get(name, np.nan) for row in residual_rows],
            float,
        )
        out[f"residual_{name}"] = values
        out[f"conditional_residual_{name}"] = np.where(active, values, np.nan)

    residual_js = np.full(n, np.nan)
    for t in range(1, n):
        if anchors[t - 1] is None or anchors[t] is None:
            continue
        residual_js[t] = spectrum_distance(
            residual_spectra[t - 1], residual_spectra[t]
        )["spectral_js"]
    out["residual_spectral_js_change"] = residual_js
    out["conditional_residual_spectral_js_change"] = np.where(active, residual_js, np.nan)
    return out
