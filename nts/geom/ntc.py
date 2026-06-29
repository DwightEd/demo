# nts/geom/ntc.py — Gao & Ganguli (2017) neural-task-complexity zero model, instantiated per step.
# participation ratio PR, autocorrelation length tau, and the length-normalized saturation ratio
# phi = PR * tau / T  (= PR / (T/tau), the single-parameter NTC bound). Dimensionless; on a smooth
# random manifold phi ~ O(1). The point: phi normalizes out the T-dependence (n-bias) that confounds
# raw effective rank / participation ratio.
import numpy as np


def participation_ratio(H):
    H = np.asarray(H, np.float64)
    if H.shape[0] < 2:
        return float("nan")
    H = H - H.mean(0)
    s = np.linalg.svd(H, compute_uv=False); lam = s ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum()) if lam.sum() > 0 else float("nan")


def autocorr_tau(H, max_lag=None):
    """Integrated autocorrelation length of the (centered, unit) token sequence, Sokal-truncated:
    tau = 1 + 2*sum_{d>=1} r(d) up to the first non-positive r(d), r(d) = mean_t cos(h_t, h_{t+d}).
    Truncating at the first non-positive lag avoids accumulating positive autocorrelation NOISE
    (which biases iid sequences upward); iid -> tau ~ 1, smooth trajectory -> tau large."""
    H = np.asarray(H, np.float64); T = len(H)
    if T < 3:
        return float("nan")
    H = H - H.mean(0)
    U = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    max_lag = max_lag or max(1, T // 2)
    tau = 1.0
    for d in range(1, min(max_lag, T - 1) + 1):
        r = float(np.mean(np.sum(U[:-d] * U[d:], axis=1)))
        if r <= 0:
            break
        tau += 2.0 * r
    return tau


def phi_saturation(H):
    T = len(H)
    pr = participation_ratio(H); tau = autocorr_tau(H)
    if T < 1 or not (np.isfinite(pr) and np.isfinite(tau)):
        return float("nan")
    return pr * tau / T
