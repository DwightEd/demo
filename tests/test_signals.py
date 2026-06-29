# tests/test_signals.py
import numpy as np
from nts.core.types import ChainData, StepTable
from nts.core.config import GeomCfg


def _synth(seed=0, m=30, dman=3, n_chains=60, T=8):
    rng = np.random.default_rng(seed); basis = np.linalg.qr(rng.normal(size=(m, dman)))[0]
    chains = []
    for i in range(n_chains):
        coords = np.cumsum(rng.normal(size=(T, dman)) * 0.3, 0)
        vecs = coords @ basis.T + 0.01 * rng.normal(size=(T, m))
        y = np.zeros(T, int); correct = (i % 2 == 0)
        if not correct:                       # inject off-manifold jump at step 4
            off = np.linalg.qr(rng.normal(size=(m, 1)))[0][:, 0]
            vecs[4:] += 1.5 * off; y[4] = 1
        sp = np.r_[np.nan, np.linalg.norm(np.diff(vecs, axis=0), axis=1)]
        chains.append(ChainData(vecs=vecs.astype(np.float32), y=y,
                                length=np.full(T, 20.0), speed=sp, repetition=np.zeros(T),
                                kappa=np.full(T, 0.5), problem_id=i, correct=correct))
    return StepTable(chains)


def test_nts_scores_offmanifold_error_high():
    from nts.signals.nts import NTSSignal
    from nts.eval.metrics import auroc, bdir
    tab = _synth(); cfg = GeomCfg(m=20, k=15, dloc=3, massive_drop=2)
    sig = NTSSignal(cfg=cfg); sig.fit(StepTable(tab.correct_chains()))
    s = sig.score(tab); y = tab.flat().y; m = np.isfinite(s)
    assert bdir(auroc(s[m], y[m])) > 0.75


def test_registry_has_all():
    import nts.signals  # triggers registration
    from nts.core.registry import SIGNALS
    assert {"nts", "rema", "kappa", "mahalanobis"} <= set(SIGNALS.list())
