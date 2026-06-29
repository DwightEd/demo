# tests/test_geom.py
import numpy as np
from nts.geom.tangent import local_tangent, decompose
from nts.geom.intrinsic_dim import twonn, principal_angle


def test_tangent_recovers_subspace():
    rng = np.random.default_rng(0); m, d = 50, 3
    basis = np.linalg.qr(rng.normal(size=(m, d)))[0]
    pts = (rng.normal(size=(400, d)) @ basis.T) + 0.01 * rng.normal(size=(400, m))
    assert principal_angle(local_tangent(pts, d), basis) < 0.1


def test_inplane_vs_offplane_normal():
    rng = np.random.default_rng(1); m, d = 40, 2
    basis = np.linalg.qr(rng.normal(size=(m, d)))[0]
    pts = (rng.normal(size=(300, d)) @ basis.T) + 0.005 * rng.normal(size=(300, m))
    U = local_tangent(pts, d)
    din = basis @ rng.normal(size=d)
    dout = din + 0.5 * np.linalg.qr(rng.normal(size=(m, 1)))[0][:, 0]
    _, nin = decompose(din, U); _, nout = decompose(dout, U)
    assert np.linalg.norm(nin) < 0.1 * np.linalg.norm(din) + 1e-6
    assert np.linalg.norm(nout) > 3 * np.linalg.norm(nin)


def test_twonn():
    rng = np.random.default_rng(2)
    # TwoNN assumes locally-uniform density; calibrate on uniform-in-cube data
    assert 4.0 < twonn(rng.uniform(size=(3000, 5))) < 6.0
    # and the estimator must order dimensions correctly (its job in the ID curve)
    assert twonn(rng.uniform(size=(3000, 3))) < twonn(rng.uniform(size=(3000, 8)))
