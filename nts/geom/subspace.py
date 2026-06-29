# nts/geom/subspace.py — correct-reasoning subspace + off-subspace (off-manifold) energy.
# Step-free: operates on a trajectory's per-token cloud H (R x d), not on steps.
import numpy as np


class CorrectSubspace:
    """Top-k PCA subspace of CORRECT-trajectory token clouds.

    off_energy(H) = fraction of H's centered energy in the orthogonal complement of that
    subspace, in [0,1] (higher = more off the correct manifold => more error-like).
    Fitted by streaming an iterable of (R, d) clouds (single pass via raw moments), so the
    full per-token hidden never has to live in RAM at once.
    """

    def __init__(self, k=32, token_cap=256, seed=0):
        self.k = k; self.token_cap = token_cap; self.seed = seed

    def fit(self, clouds):
        rng = np.random.default_rng(self.seed)
        d = None; S1 = S2 = None; n = 0
        for H in clouds:
            H = np.asarray(H, np.float64)
            if H.ndim != 2 or H.shape[0] == 0:
                continue
            if self.token_cap and H.shape[0] > self.token_cap:
                H = H[rng.choice(H.shape[0], self.token_cap, replace=False)]
            if d is None:
                d = H.shape[1]; S1 = np.zeros(d); S2 = np.zeros((d, d))
            S1 += H.sum(0); S2 += H.T @ H; n += H.shape[0]
        if not n:
            raise ValueError("CorrectSubspace.fit: no tokens")
        self.mu = S1 / n
        cov = S2 / n - np.outer(self.mu, self.mu)
        w, V = np.linalg.eigh(cov)
        self.U = V[:, ::-1][:, :self.k]            # (d, k) top-k principal directions
        return self

    def off_energy(self, H):
        Hc = np.asarray(H, np.float64) - self.mu
        den = float((Hc * Hc).sum())
        if den <= 0:
            return float("nan")
        P = Hc @ self.U
        return 1.0 - float((P * P).sum()) / den

    def off_energy_windows(self, H, w=64, stride=32):
        H = np.asarray(H, np.float64)
        starts = range(0, max(1, len(H) - w + 1), stride)
        return np.array([self.off_energy(H[a:a + w]) for a in starts])
