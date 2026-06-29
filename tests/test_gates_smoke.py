# tests/test_gates_smoke.py — gate plumbing runs end-to-end on synthetic data (no real npz)
import numpy as np
import nts.signals  # register signals
import nts.gates     # register gates
from nts.core.types import ChainData, StepTable
from nts.core.config import GeomCfg
from nts.core.registry import GATES
from nts.gates.base import GateResult


def _synth(seed=0, m=30, dman=3, n_chains=60, T=8):
    rng = np.random.default_rng(seed); basis = np.linalg.qr(rng.normal(size=(m, dman)))[0]
    chains = []
    for i in range(n_chains):
        coords = np.cumsum(rng.normal(size=(T, dman)) * 0.3, 0)
        vecs = coords @ basis.T + 0.01 * rng.normal(size=(T, m))
        y = np.zeros(T, int); correct = (i % 2 == 0)
        if not correct:
            off = np.linalg.qr(rng.normal(size=(m, 1)))[0][:, 0]
            vecs[4:] += 1.5 * off; y[4] = 1
        sp = np.r_[np.nan, np.linalg.norm(np.diff(vecs, axis=0), axis=1)]
        kap = 0.5 + 0.05 * rng.normal(size=T)
        chains.append(ChainData(vecs=vecs.astype(np.float32), y=y,
                                length=np.full(T, 20.0) + rng.normal(size=T),
                                speed=sp, repetition=np.abs(rng.normal(scale=.05, size=T)),
                                kappa=kap, problem_id=i, correct=correct))
    return StepTable(chains)


def test_all_gates_run():
    tab = _synth(); cfg = GeomCfg(m=20, k=15, dloc=3, massive_drop=2, folds=5)
    for name in ["gate0_mahal", "gate1_estimability", "gate2_nts_vs_rema", "gate3_curvature"]:
        res = GATES.create(name, cfg=cfg, params={}).run(tab)  # gate1: npz=None -> skips ID curve
        assert isinstance(res, GateResult) and isinstance(res.kill, bool) and res.lines


def test_gates_run_without_kappa():
    # mirrors the real npz where 'resultant'/kappa is absent (kappa all-NaN)
    tab = _synth()
    for c in tab.chains:
        c.kappa = np.full(len(c.y), np.nan)
    cfg = GeomCfg(m=20, k=15, dloc=3, massive_drop=2, folds=5)
    for name in ["gate2_nts_vs_rema", "gate3_curvature"]:
        res = GATES.create(name, cfg=cfg, params={}).run(tab)
        assert isinstance(res, GateResult) and isinstance(res.kill, bool)
