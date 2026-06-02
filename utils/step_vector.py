"""Step-vector aggregation + activation-participation metrics.

Two ideas combined here:

1. STEP REPRESENTATION (how to turn a step's token cloud into ONE vector).
   Follows Streaming Hallucination Detection in Long CoT Reasoning
   (Lu et al., arXiv 2601.02170), whose ablation (their Fig. 4) finds a
   *within-step, time-aware EXPONENTIAL* weighting to be optimal -- it beats
   uniform mean / linear / global-accumulation aggregation by +3-6 AUC. The
   key reasons (their Property I & II):
     - Property I  (cross-step saturation): do NOT additively accumulate across
       steps; aggregate only WITHIN the current step.
     - Property II (within-step imbalance): later tokens summarize the step more
       completely, so weight them higher.
   Their Eq. 6:  z_t = sum_j softmax(w_j) h_{t,j},  w_n = (n-1)/(L_t-1),
   then L2-normalize.  First token weight ~ e^0 = 1, last ~ e^1 = e, and the
   position is normalized to [0,1] so it is invariant to step length.

   We implement four modes for comparison (their baselines + their method):
     "last"     -> last token only (the HARP/SAPLMA default)
     "mean"     -> uniform average (Step Time Average baseline)
     "linear"   -> linear-increasing weights (Global Linear-style baseline)
     "step_exp" -> Eq. 6, the paper's optimal choice

2. ACTIVATION PARTICIPATION (how many dimensions a single vector "uses").
   The hypothesis: when the model is uncertain, activation spreads over more
   dimensions (high participation); when confident, it concentrates (low
   participation, sparse). This needs NO point cloud -- it is computed on the
   single step vector, so it is immune to the "too few points per step" problem
   that breaks kNN intrinsic-dimension estimation.
     participation_ratio(h) = (sum h_i^2)^2 / sum h_i^4
     activation_entropy(h)  = exp(-sum p_i log p_i),  p_i = h_i^2 / sum h_j^2
   IMPORTANT: these are sensitive to activation magnitude / LayerNorm scale.
   Use whitened vectors (per-dimension standardized against a healthy baseline)
   if you want "which dims are unusually active" rather than raw magnitude.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Step representation: token cloud -> one vector
# ---------------------------------------------------------------------------

def step_vector(H: np.ndarray, mode: str = "step_exp",
                l2_normalize: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Aggregate a step's token cloud H (n_tokens, d) into one (d,) vector.

    Args:
        H: (n, d) token hidden states for ONE step at ONE layer, in token order
           (row 0 = first token of the step, row n-1 = last token).
        mode: "last" | "mean" | "linear" | "step_exp" (paper-optimal).
        l2_normalize: L2-normalize the result (the paper does; recommended for
            "step_exp" so magnitude does not dominate).
        eps: numerical floor.

    Returns:
        (d,) step vector, or None if H is empty.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < 1:
        return None
    n = H.shape[0]

    if mode == "last":
        z = H[-1]
    elif mode == "mean":
        z = H.mean(axis=0)
    elif mode == "linear":
        w = np.arange(1, n + 1, dtype=np.float64)      # 1,2,...,n
        z = (w[:, None] * H).sum(axis=0) / w.sum()
    elif mode == "step_exp":
        # paper Eq.6: w_n = (n-1)/(L-1) in [0,1]; weight = softmax(w)
        if n == 1:
            z = H[0]
        else:
            pos = np.arange(n, dtype=np.float64) / (n - 1)   # 0 .. 1
            wexp = np.exp(pos)
            wexp = wexp / wexp.sum()
            z = (wexp[:, None] * H).sum(axis=0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if l2_normalize:
        nrm = float(np.linalg.norm(z))
        if nrm > eps:
            z = z / nrm
    return z


# ---------------------------------------------------------------------------
# 2. Activation participation of a single vector
# ---------------------------------------------------------------------------

def participation_ratio(h: np.ndarray, eps: float = 1e-12) -> float:
    """PR(h) = (sum h_i^2)^2 / sum h_i^4.

    Energy in one coordinate -> 1; uniform over k coords -> k. A continuous
    "how many dimensions are active" for a single vector. Range [1, d].
    """
    h = np.asarray(h, dtype=np.float64)
    s2 = float(np.sum(h ** 2))
    s4 = float(np.sum(h ** 4))
    if s4 <= eps:
        return float("nan")
    return (s2 ** 2) / s4


def activation_entropy(h: np.ndarray, eps: float = 1e-12) -> float:
    """exp(-sum p_i log p_i) over p_i = h_i^2 / sum h_j^2.

    Same "effective number of active dimensions" idea as PR, via spectral-style
    entropy of the squared-coordinate distribution. Range [1, d].
    """
    h = np.asarray(h, dtype=np.float64)
    s2 = float(np.sum(h ** 2))
    if s2 <= eps:
        return float("nan")
    p = (h ** 2) / s2
    p = p[p > eps]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def activation_metrics(h: np.ndarray) -> tuple[float, float]:
    """Convenience: (participation_ratio, activation_entropy) for one vector."""
    return participation_ratio(h), activation_entropy(h)