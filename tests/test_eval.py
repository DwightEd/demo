# tests/test_eval.py
import numpy as np
from nts.eval.metrics import auroc, bdir, bucket
from nts.eval.confound import residualize, oof_logit, cluster_boot_increment


def test_auroc():
    y = np.array([0, 0, 1, 1]); s = np.array([.1, .2, .8, .9])
    assert abs(auroc(s, y) - 1) < 1e-9 and abs(bdir(auroc(-s, y)) - 1) < 1e-9


def test_bucket_runs():
    rng = np.random.default_rng(0); n = 600
    y = rng.integers(0, 2, n); s = y + rng.normal(scale=.5, size=n); nt = rng.normal(size=n)
    a = bucket(s, y, nt)
    assert 0.5 <= a <= 1.0


def test_residualize_removes_linear_confound():
    rng = np.random.default_rng(0); n = 2000
    g = rng.integers(0, 200, n); conf = rng.normal(size=n)
    correct = rng.integers(0, 2, n).astype(bool)
    sig = 3 * conf + rng.normal(scale=.1, size=n)
    r = residualize(sig, conf[:, None], correct, g, folds=5); m = np.isfinite(r)
    assert abs(np.corrcoef(r[m], conf[m])[0, 1]) < 0.2


def test_increment_detects_useful_signal():
    rng = np.random.default_rng(1); n = 3000
    g = rng.integers(0, 300, n); y = rng.integers(0, 2, n)
    base = rng.normal(size=n); useful = y + rng.normal(scale=.5, size=n)
    sb = oof_logit(base[:, None], y, g); sf = oof_logit(np.c_[base, useful], y, g)
    mean, lo, hi, sig = cluster_boot_increment(sf, sb, y, g, nboot=200)
    assert sig and mean > 0
