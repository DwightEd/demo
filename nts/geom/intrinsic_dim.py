# nts/geom/intrinsic_dim.py — TwoNN intrinsic dimension + principal angle between subspaces
import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn(X, frac=0.9):
    X = np.asarray(X, float)
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]; ok = (r1 > 0) & (r2 > 0)
    mu = np.sort(r2[ok] / r1[ok]); N = len(mu)
    F = np.arange(1, N + 1) / N                 # empirical CDF over the FULL set
    keep = max(2, int(frac * N))                # discard noisy upper tail, keep CDF intact
    x = np.log(mu[:keep]); y = -np.log(np.clip(1 - F[:keep], 1e-12, None))
    return float(np.sum(x * y) / np.sum(x * x))


def principal_angle(U, V):
    Qu = np.linalg.qr(U)[0]; Qv = np.linalg.qr(V)[0]
    sv = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    return float(np.arccos(np.clip(sv.min(), -1, 1)))
