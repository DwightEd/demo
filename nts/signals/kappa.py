# nts/signals/kappa.py — raw-kappa baseline (-resultant); higher concentration => lower error score
import numpy as np
from .base import BaseSignal
from ..core.registry import SIGNALS


@SIGNALS.register("kappa")
class KappaSignal(BaseSignal):
    name = "kappa"

    def fit(self, train):
        return self

    def score(self, test):
        return np.concatenate([-c.kappa for c in test.chains]) if test.chains else np.array([])
