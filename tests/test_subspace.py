# tests/test_subspace.py
import numpy as np
from nts.geom.subspace import CorrectSubspace


def test_off_energy_discriminates_offmanifold():
    rng = np.random.default_rng(0); d, k = 64, 5
    B = np.linalg.qr(rng.normal(size=(d, k)))[0]              # true correct subspace

    def correct():
        return (rng.normal(size=(200, k)) @ B.T) + 0.01 * rng.normal(size=(200, d))

    cs = CorrectSubspace(k=k, token_cap=256).fit([correct() for _ in range(20)])
    off_dir = np.linalg.qr(rng.normal(size=(d, 1)))[0][:, 0]  # a direction mostly off B
    e_correct = cs.off_energy(correct())
    e_error = cs.off_energy(correct() + 0.6 * off_dir)        # cloud shoved off-manifold
    assert e_correct < 0.2
    assert e_error > e_correct + 0.05


def test_windows_shape():
    rng = np.random.default_rng(1); d, k = 32, 4
    B = np.linalg.qr(rng.normal(size=(d, k)))[0]
    cs = CorrectSubspace(k=k).fit([(rng.normal(size=(150, k)) @ B.T) for _ in range(10)])
    w = cs.off_energy_windows(rng.normal(size=(300, d)) @ B.T @ B.T.T if False else (rng.normal(size=(300, k)) @ B.T), w=64, stride=32)
    assert w.ndim == 1 and len(w) >= 1 and np.all(np.isfinite(w))
