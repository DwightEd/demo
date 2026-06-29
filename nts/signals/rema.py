# nts/signals/rema.py — REMA isotropic kNN distance to correct manifold (NTS's differential control)
import numpy as np
from .base import BaseSignal
from ..core.registry import SIGNALS
from ..geom.reducer import fit_reducer
from ..geom.bank import Bank


@SIGNALS.register("rema")
class REMASignal(BaseSignal):
    name = "rema"

    def fit(self, train):
        X = np.concatenate([c.vecs for c in train.correct_chains()], 0)
        self.transform = fit_reducer(X, self.cfg.m, self.cfg.massive_drop)
        self.bank = Bank(self.transform(X), cap=self.cfg.bank_cap)
        return self

    def score(self, test):
        out = []
        for c in test.chains:
            red = self.transform(c.vecs)
            out.append(np.array([self.bank.mean_dist(red[t], self.cfg.k) for t in range(len(red))]))
        return np.concatenate(out) if out else np.array([])
