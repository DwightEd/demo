# nts/signals/nts_cloud.py — step-free off-correct-subspace energy on the per-token cloud
import numpy as np
from .base import BaseSignal
from ..core.registry import SIGNALS
from ..geom.subspace import CorrectSubspace


def _cloud(c):
    return np.load(c.hidden_path, mmap_mode="r")[:, c.hidden_col, :]


@SIGNALS.register("nts_cloud")
class NTSCloudSignal(BaseSignal):
    """Chain-level (step-free): fraction of a trajectory's token-cloud energy off the
    correct-reasoning subspace learned from correct train chains. One value per chain,
    broadcast to its steps to fit the canonical step order."""
    name = "nts_cloud"

    def fit(self, train):
        k = int(self.params.get("k", 32)); cap = int(self.params.get("token_cap", 256))
        self.cs = CorrectSubspace(k=k, token_cap=cap).fit(
            _cloud(c) for c in train.chains if c.correct and c.hidden_path)
        return self

    def score(self, test):
        out = []
        for c in test.chains:
            e = self.cs.off_energy(_cloud(c)) if c.hidden_path else float("nan")
            out.append(np.full(len(c.y), e))
        return np.concatenate(out) if out else np.array([])
