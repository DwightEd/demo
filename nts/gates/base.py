# nts/gates/base.py — gate interface + result container + shared cross-fit scorer
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import numpy as np
from sklearn.model_selection import GroupKFold
from ..core.types import StepTable
from ..core.registry import SIGNALS


@dataclass
class GateResult:
    name: str
    lines: List[str] = field(default_factory=list)
    kill: bool = False

    @property
    def summary(self):
        return "\n".join(self.lines + [f"  KILL? {'YES' if self.kill else 'no'}"])


class BaseGate(ABC):
    name = "base"

    def __init__(self, cfg, params=None):
        self.cfg = cfg
        self.params = params or {}

    @abstractmethod
    def run(self, table: StepTable) -> GateResult:
        ...


def crossfit_signal(signal_name, table, cfg, params=None, folds=5):
    """GroupKFold by problem over chains; fit signal on train, score test.
    Returns a score array aligned to table.flat() (canonical chain/step order)."""
    chains = table.chains
    pid = np.array([c.problem_id for c in chains])
    score_by_chain = [None] * len(chains)
    for tr, te in GroupKFold(folds).split(np.zeros(len(chains)), np.zeros(len(chains)), pid):
        train = StepTable([chains[i] for i in tr])
        test = StepTable([chains[i] for i in te])
        sig = SIGNALS.create(signal_name, cfg=cfg, params=params).fit(train)
        s = sig.score(test); off = 0
        for i in te:
            T = len(chains[i].y); score_by_chain[i] = s[off:off + T]; off += T
    return np.concatenate([score_by_chain[i] for i in range(len(chains))])
