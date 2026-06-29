# tests/test_nts_cloud.py — cloud signal + gate_cloud run end-to-end against on-disk shards
import os
import numpy as np
import nts.signals  # register
import nts.gates     # register
from nts.core.types import ChainData, StepTable
from nts.core.config import GeomCfg
from nts.core.registry import SIGNALS, GATES
from nts.gates.base import GateResult


def _make(tmp, d=48, k=4, n_chains=40, T=3, R=60):
    rng = np.random.default_rng(0); B = np.linalg.qr(rng.normal(size=(d, k)))[0]; chains = []
    for i in range(n_chains):
        H = (rng.normal(size=(R, k)) @ B.T) + 0.01 * rng.normal(size=(R, d))
        correct = (i % 2 == 0)
        if not correct:
            H = H + 0.6 * np.linalg.qr(rng.normal(size=(d, 1)))[0][:, 0]   # off-subspace shove
        p = os.path.join(tmp, f"c{i}.npy"); np.save(p, H[:, None, :].astype(np.float16))  # (R,1,d)
        vecs = H[:T].astype(np.float32)
        sp = np.r_[np.nan, np.linalg.norm(np.diff(vecs, axis=0), axis=1)]
        chains.append(ChainData(vecs=vecs, y=np.zeros(T, int), length=np.full(T, 20.0), speed=sp,
                                repetition=np.zeros(T), kappa=np.full(T, np.nan), problem_id=i,
                                correct=correct, hidden_path=p, hidden_col=0))
    return StepTable(chains)


def test_cloud_signal_scores(tmp_path):
    tab = _make(str(tmp_path)); cfg = GeomCfg()
    sig = SIGNALS.create("nts_cloud", cfg=cfg, params={"k": 4}).fit(StepTable(tab.correct_chains()))
    s = sig.score(tab)
    assert np.isfinite(s).all() and len(s) == sum(len(c.y) for c in tab.chains)


def test_gate_cloud_runs(tmp_path):
    tab = _make(str(tmp_path)); cfg = GeomCfg(m=20, k=10, dloc=3, massive_drop=2, folds=5)
    res = GATES.create("gate_cloud", cfg=cfg, params={}).run(tab)
    assert isinstance(res, GateResult) and isinstance(res.kill, bool) and res.lines
