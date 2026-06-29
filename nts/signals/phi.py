# nts/signals/phi.py — per-step NTC signals: raw participation ratio (pr) and the length-normalized
# saturation ratio phi = PR*tau/T (Gao&Ganguli zero-model). Both unsupervised, computed from the
# per-token cloud sliced by step. Score higher = more error-like is handled by best-direction AUROC.
import numpy as np
from .base import BaseSignal
from ..core.registry import SIGNALS
from ..geom.ntc import participation_ratio, autocorr_tau, phi_saturation


def _step_clouds(c):
    H = np.load(c.hidden_path, mmap_mode="r")[:, c.hidden_col, :]
    for (a, b) in c.step_ranges:
        yield np.asarray(H[int(a):int(b)])


@SIGNALS.register("pr")
class PrSignal(BaseSignal):
    """Raw per-step participation ratio (the confounded baseline to beat)."""
    name = "pr"

    def fit(self, train):
        return self

    def score(self, test):
        out = []
        for c in test.chains:
            if c.hidden_path is None or c.step_ranges is None:
                out.append(np.full(len(c.y), np.nan)); continue
            out.append(np.array([participation_ratio(seg) for seg in _step_clouds(c)]))
        return np.concatenate(out) if out else np.array([])


@SIGNALS.register("phi")
class PhiSignal(BaseSignal):
    """Length-normalized NTC saturation ratio phi = PR*tau/T per step."""
    name = "phi"

    def fit(self, train):
        return self

    def score(self, test):
        out = []
        for c in test.chains:
            if c.hidden_path is None or c.step_ranges is None:
                out.append(np.full(len(c.y), np.nan)); continue
            out.append(np.array([phi_saturation(seg) for seg in _step_clouds(c)]))
        return np.concatenate(out) if out else np.array([])
