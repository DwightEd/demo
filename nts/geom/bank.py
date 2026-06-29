# nts/geom/bank.py — kNN bank over reduced correct-chain step vectors
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Bank:
    def __init__(self, reduced_corr, cap=30000, seed=0):
        B = np.asarray(reduced_corr, float)
        if len(B) > cap:
            B = B[np.random.default_rng(seed).choice(len(B), cap, replace=False)]
        self.B = B
        self.nn = NearestNeighbors(n_neighbors=1).fit(B)

    def neighbors(self, query, k):
        kk = min(k, len(self.B))
        d, idx = self.nn.kneighbors(query[None, :], n_neighbors=kk)
        return self.B[idx[0]], d[0]

    def mean_dist(self, query, k):
        _, d = self.neighbors(query, k)
        return float(d.mean())
