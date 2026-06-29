# nts/signals/nts.py — curvature-debiased residual normal-escape energy
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .base import BaseSignal
from ..core.registry import SIGNALS
from ..geom.reducer import fit_reducer
from ..geom.bank import Bank
from ..geom.tangent import chain_energies


@SIGNALS.register("nts")
class NTSSignal(BaseSignal):
    name = "nts"

    def fit(self, train):
        cc = train.correct_chains()
        X = np.concatenate([c.vecs for c in cc], 0)
        self.transform = fit_reducer(X, self.cfg.m, self.cfg.massive_drop)
        self.bank = Bank(self.transform(X), cap=self.cfg.bank_cap)
        # curvature regressor: tangent_norm -> normal_norm on correct chains
        tn, nn = [], []
        for c in cc:
            Tn, Nn, _ = chain_energies(self.transform(c.vecs), self.bank, self.cfg.k, self.cfg.dloc)
            m = np.isfinite(Tn); tn.append(Tn[m]); nn.append(Nn[m])
        tn = np.concatenate(tn); nn = np.concatenate(nn)
        self.curv = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
        self.curv.fit(tn[:, None], nn)
        return self

    def score(self, test):
        out = []
        for c in test.chains:
            Tn, Nn, _ = chain_energies(self.transform(c.vecs), self.bank, self.cfg.k, self.cfg.dloc)
            resid = np.full(len(Tn), np.nan); m = np.isfinite(Tn)
            resid[m] = Nn[m] - self.curv.predict(Tn[m][:, None])
            out.append(resid)
        return np.concatenate(out) if out else np.array([])
