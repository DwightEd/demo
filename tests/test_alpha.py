# tests/test_alpha.py
import numpy as np
from nts.signals.alpha import spectral_alpha


def test_alpha_higher_for_concentrated():
    rng = np.random.default_rng(0); d, n = 64, 200
    concentrated = rng.normal(size=(n, 3)) @ rng.normal(size=(3, d)) + 0.01 * rng.normal(size=(n, d))
    flat = rng.normal(size=(n, d))
    assert spectral_alpha(concentrated) > spectral_alpha(flat)
