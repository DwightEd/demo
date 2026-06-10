"""Activation-degree / dimension geometry of a single (raw) vector.

Motivation
----------
The anchor hypothesis: error reasoning = activation DIFFUSE over more dimensions;
correct reasoning = concentrated in a low-dim but non-degenerate participating
subset. We want to measure "how many dimensions are meaningfully active" for the
model state, RAW (no per-dim healthy standardization), at the magnitude scale the
hidden states actually live at.

Why not just PR
---------------
participation_ratio  PR(h) = (sum h^2)^2 / sum h^4  is exactly the Renyi alpha=2
effective dimension of the energy distribution p_d = h_d^2 / ||h||^2:
    PR = (sum_d p_d^2)^{-1}.
alpha=2 is the MOST peak-dominated member of the family, so on transformer hidden
states a handful of "massive activation" dims (Sun et al. 2402.17762) dominate
sum h^4 and collapse PR toward "#massive dims", going nearly blind to the bulk
diffuseness we actually care about. We therefore compute a small Renyi family so
the data can pick the right alpha:
    alpha=2  -> PR        (peak-dominated; kept for comparability with prior work)
    alpha=1  -> AE        (Shannon; exp of activation entropy)
    alpha=.5 -> ed_half   (bulk/tail-sensitive)
plus an energy-width E_q (min #dims to reach q of the energy) and a
massive-removed AE_robust (AE after dropping the top-m magnitude dims).

Everything here is scale-invariant EXCEPT `norm`, which is the magnitude (模长)
signal and is deliberately measured on the raw, un-normalized vector.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Renyi effective dimension of the energy distribution p_d = h_d^2 / ||h||^2
# ---------------------------------------------------------------------------

def _energy_dist(h: np.ndarray, eps: float = 1e-12):
    h = np.asarray(h, dtype=np.float64).ravel()
    s2 = float(np.sum(h * h))
    if s2 <= eps:
        return None
    return (h * h) / s2


def renyi_eff_dim(h: np.ndarray, alpha: float, eps: float = 1e-12) -> float:
    """exp(H_alpha) of p_d = h_d^2/||h||^2  =  effective number of active dims.

    alpha=1 is the Shannon limit exp(-sum p log p); alpha=2 gives PR; alpha<1
    weights the bulk/tail more (more sensitive to diffuse activation).
    Range [1, d]. One coordinate -> 1; uniform over k coords -> k.
    """
    p = _energy_dist(h, eps)
    if p is None:
        return float("nan")
    p = p[p > eps]
    if p.size == 0:
        return float("nan")
    if abs(alpha - 1.0) < 1e-9:
        return float(np.exp(-np.sum(p * np.log(p))))
    s = float(np.sum(p ** alpha))
    if s <= eps:
        return float("nan")
    return float(s ** (1.0 / (1.0 - alpha)))


def participation_ratio(h: np.ndarray, eps: float = 1e-12) -> float:
    """PR(h) = (sum h^2)^2 / sum h^4  (Renyi alpha=2). Kept explicit for clarity."""
    h = np.asarray(h, dtype=np.float64).ravel()
    s2 = float(np.sum(h ** 2))
    s4 = float(np.sum(h ** 4))
    if s4 <= eps:
        return float("nan")
    return (s2 ** 2) / s4


def energy_width(h: np.ndarray, q: float = 0.9, eps: float = 1e-12) -> float:
    """Minimum #dimensions (sorted by energy, descending) to reach fraction q of
    the total energy ||h||^2. Robust, interpretable "effective dimension"; far
    less peak-dominated than PR. Returns a count in [1, d]."""
    p = _energy_dist(h, eps)
    if p is None:
        return float("nan")
    ps = np.sort(p)[::-1]
    c = np.cumsum(ps)
    k = int(np.searchsorted(c, q) + 1)
    return float(min(k, p.size))


def ae_robust(h: np.ndarray, massive_m: int = 4, eps: float = 1e-12) -> float:
    """Shannon effective dim (AE) AFTER removing the top-`massive_m` magnitude
    dims, i.e. the activation entropy of the BULK once the massive-activation
    dimensions (which dominate PR) are taken out. Un-standardized."""
    h = np.asarray(h, dtype=np.float64).ravel()
    if massive_m > 0 and h.size > massive_m:
        idx = np.argpartition(np.abs(h), -massive_m)[-massive_m:]
        h = h.copy()
        h[idx] = 0.0
    return renyi_eff_dim(h, alpha=1.0, eps=eps)


def anom_count(h: np.ndarray, k: float = 10.0, eps: float = 1e-12) -> float:
    """Number of anomalously-activated dims via an INTRINSIC threshold (no
    external/healthy baseline): #{ d : |h_d| > k * median_d(|h|) }. Captures the
    "异常激活维数" without standardization."""
    h = np.abs(np.asarray(h, dtype=np.float64).ravel())
    med = float(np.median(h))
    if med <= eps:
        return float("nan")
    return float(int(np.sum(h > k * med)))


# ---------------------------------------------------------------------------
# One call -> all per-vector geometry features (scalars)
# ---------------------------------------------------------------------------

GEOM_FEATURE_NAMES = (
    "norm", "pr", "ae", "ed_half", "e50", "e90", "ae_robust",
    "anom_k5", "anom_k10",
)


def twonn_dim(H, frac=0.9, eps=1e-12):
    """TwoNN intrinsic dimension (Facco et al. 2017) of a point cloud H (n, d).

    Length-ROBUST: estimates the manifold dimension from the ratio of 2nd to 1st
    nearest-neighbour distances, asymptotically independent of the number of
    points -- unlike the linear effective rank, which is bounded by n and so
    tracks token count. Meant for the WHOLE-chain pooled cloud (n ~ hundreds).
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]
    if n < 10:
        return float("nan")
    sq = (H * H).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (H @ H.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)
    np.fill_diagonal(D, np.inf)
    two = np.partition(D, 1, axis=1)[:, :2]          # 1st, 2nd NN distances
    mu = two[:, 1] / np.maximum(two[:, 0], eps)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1.0)])
    if mu.size < 10:
        return float("nan")
    Femp = np.arange(1, mu.size + 1) / (mu.size + 1.0)
    x, y = np.log(mu), -np.log(1.0 - Femp)
    keep = Femp < frac                                # drop the unstable tail
    if keep.sum() < 5:
        return float("nan")
    return float(np.sum(x[keep] * y[keep]) / np.sum(x[keep] ** 2))


def vector_features(h: np.ndarray, massive_m: int = 4) -> dict:
    """All RAW activation-degree features of one vector h.

    `norm` is the magnitude on the un-normalized vector; the participation
    family is scale-invariant. Keys == GEOM_FEATURE_NAMES.
    """
    h = np.asarray(h, dtype=np.float64).ravel()
    return {
        "norm":      float(np.linalg.norm(h)),
        "pr":        participation_ratio(h),
        "ae":        renyi_eff_dim(h, alpha=1.0),
        "ed_half":   renyi_eff_dim(h, alpha=0.5),
        "e50":       energy_width(h, q=0.5),
        "e90":       energy_width(h, q=0.9),
        "ae_robust": ae_robust(h, massive_m=massive_m),
        "anom_k5":   anom_count(h, k=5.0),
        "anom_k10":  anom_count(h, k=10.0),
    }
