# nts/geom/tangent.py — local PCA tangent, displacement decomposition, per-chain energies
import numpy as np
from sklearn.covariance import LedoitWolf


def local_tangent(neighbors, dloc):
    X = np.asarray(neighbors, float); X = X - X.mean(0)
    cov = LedoitWolf().fit(X).covariance_
    _, V = np.linalg.eigh(cov)
    return V[:, ::-1][:, :dloc]   # (m, dloc) top eigenvectors


def decompose(delta, U):
    z = U.T @ delta
    return z, delta - U @ z      # (tangent coords, normal vector)


def chain_energies(reduced_chain, bank, k, dloc):
    """Per step t>=1: (tang_norm, normal_norm, speed). Anchor = previous step."""
    T = len(reduced_chain)
    Tn = np.full(T, np.nan); Nn = np.full(T, np.nan); Sp = np.full(T, np.nan)
    for t in range(1, T):
        nb, _ = bank.neighbors(reduced_chain[t - 1], k)
        U = local_tangent(nb, dloc)
        delta = reduced_chain[t] - reduced_chain[t - 1]
        z, normal = decompose(delta, U)
        Tn[t] = np.linalg.norm(z); Nn[t] = np.linalg.norm(normal); Sp[t] = np.linalg.norm(delta)
    return Tn, Nn, Sp
