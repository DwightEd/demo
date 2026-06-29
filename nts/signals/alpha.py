# nts/signals/alpha.py — spectral power-law exponent alpha (Liu 2604.15350 "Spectral Geometry of
# Thought"): slope of log singular values vs log rank of the response cloud. Chain-level (their metric).
import numpy as np
from .base import BaseSignal
from ..core.registry import SIGNALS


def spectral_alpha(H):
    H = np.asarray(H, np.float64); H = H - H.mean(0)
    s = np.linalg.svd(H, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) < 5:
        return float("nan")
    x = np.log(np.arange(1, len(s) + 1))
    return float(-np.polyfit(x, np.log(s), 1)[0])   # sigma_k ~ k^{-alpha}; higher alpha = more concentrated


@SIGNALS.register("alpha")
class AlphaSignal(BaseSignal):
    """Their global scalar, reproduced for head-to-head. Chain-level, broadcast to steps. No fit."""
    name = "alpha"

    def fit(self, train):
        return self

    def score(self, test):
        out = []
        for c in test.chains:
            a = spectral_alpha(np.load(c.hidden_path, mmap_mode="r")[:, c.hidden_col, :]) if c.hidden_path else float("nan")
            out.append(np.full(len(c.y), a))
        return np.concatenate(out) if out else np.array([])
