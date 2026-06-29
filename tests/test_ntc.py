# tests/test_ntc.py
import numpy as np
from nts.geom.ntc import participation_ratio, autocorr_tau, phi_saturation


def test_participation_ratio():
    rng = np.random.default_rng(0)
    assert participation_ratio(rng.normal(size=(500, 10))) > 7.0          # isotropic ~ d
    rank1 = np.outer(rng.normal(size=300), rng.normal(size=10))
    assert participation_ratio(rank1 + 1e-3 * rng.normal(size=(300, 10))) < 2.0


def test_autocorr_tau():
    rng = np.random.default_rng(1)
    assert autocorr_tau(rng.normal(size=(300, 8))) < 2.0                   # iid -> ~1
    walk = np.cumsum(0.1 * rng.normal(size=(300, 8)), axis=0)              # smooth random walk
    assert autocorr_tau(walk) > 3.0


def test_phi_finite():
    rng = np.random.default_rng(2)
    v = phi_saturation(rng.normal(size=(60, 16)))
    assert np.isfinite(v) and v > 0
