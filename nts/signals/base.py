# nts/signals/base.py — pluggable per-step signal interface (mirrors hallu-det BaseMethod)
from abc import ABC, abstractmethod
import numpy as np
from ..core.config import GeomCfg
from ..core.types import StepTable


class BaseSignal(ABC):
    name = "base"

    def __init__(self, cfg: GeomCfg = None, params: dict = None):
        self.cfg = cfg or GeomCfg()
        self.params = params or {}

    @abstractmethod
    def fit(self, train: StepTable) -> "BaseSignal":
        ...

    @abstractmethod
    def score(self, test: StepTable) -> np.ndarray:
        """Per-step score in canonical order (iterate test.chains, steps 0..T-1)."""
