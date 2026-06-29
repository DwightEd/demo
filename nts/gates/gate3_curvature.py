# nts/gates/gate3_curvature.py — does curvature debiasing add over raw normal on coherent-but-wrong?
import numpy as np
from sklearn.model_selection import GroupKFold
from .base import BaseGate, GateResult, crossfit_signal
from ..core.registry import GATES
from ..eval.metrics import bucket
from ..eval.confound import oof_logit, cluster_boot_increment
from ..geom.reducer import fit_reducer
from ..geom.bank import Bank
from ..geom.tangent import chain_energies


@GATES.register("gate3_curvature")
class Gate3(BaseGate):
    name = "gate3_curvature"

    def run(self, table):
        cfg = self.cfg; chains = table.chains; pid = np.array([c.problem_id for c in chains])
        RAW = [None] * len(chains)
        for tr, te in GroupKFold(cfg.folds).split(np.zeros(len(chains)), np.zeros(len(chains)), pid):
            ccx = [chains[i] for i in tr if chains[i].correct]
            if not ccx:
                continue
            Xt = np.concatenate([c.vecs for c in ccx], 0); tf = fit_reducer(Xt, cfg.m, cfg.massive_drop)
            bank = Bank(tf(Xt), cap=cfg.bank_cap)
            for i in te:
                _, Nn, _ = chain_energies(tf(chains[i].vecs), bank, cfg.k, cfg.dloc); RAW[i] = Nn
        raw = np.concatenate([RAW[i] for i in range(len(chains))])
        resid = crossfit_signal("nts", table, cfg, folds=cfg.folds)
        f = table.flat(); kmed = np.median(f.kappa[f.y == 0]); cbw = f.kappa >= kmed
        y, g = f.y[cbw], f.groups[cbw]
        r = GateResult(self.name)
        r.lines.append(f"gate3 curvature | cbw steps {int(cbw.sum())} err {int(y.sum())}")
        r.lines.append(f"  raw normal bucket {bucket(raw[cbw], y, f.speed[cbw]):.3f} | resid bucket {bucket(resid[cbw], y, f.speed[cbw]):.3f}")
        base = oof_logit(raw[cbw][:, None], y, g); full = oof_logit(np.column_stack([raw[cbw], resid[cbw]]), y, g)
        mean, lo, hi, sig = cluster_boot_increment(full, base, y, g)
        r.lines.append(f"  resid over raw normal: +{mean:.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'ns'}")
        r.kill = not sig
        return r
