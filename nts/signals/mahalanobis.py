# nts/signals/mahalanobis.py — in-subspace diagonal Mahalanobis to correct manifold (mahal_step.py logic)
import numpy as np
from .base import BaseSignal
from ..core.registry import SIGNALS


@SIGNALS.register("mahalanobis")
class MahalanobisSignal(BaseSignal):
    name = "mahalanobis"

    def fit(self, train):
        X = np.concatenate([c.vecs for c in train.correct_chains()], 0).astype(float)
        self.mu = X.mean(0); Xc = X - self.mu
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        kpc = int(self.params.get("k_pc", 50)); self.comp = Vt[:kpc]
        self.sd = (Xc @ self.comp.T).std(0) + 1e-6
        return self

    def score(self, test):
        out = []
        for c in test.chains:
            p = (c.vecs.astype(float) - self.mu) @ self.comp.T
            out.append(np.sqrt(((p / self.sd) ** 2).sum(1)))
        return np.concatenate(out) if out else np.array([])
