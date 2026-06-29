# nts/core/config.py — geometry/eval hyperparameters
from dataclasses import dataclass


@dataclass
class GeomCfg:
    layer: int = 14
    m: int = 128          # PCA-whiten target dim
    k: int = 64           # kNN neighbors
    dloc: int = 8         # local tangent dim
    massive_drop: int = 5  # massive activation dims removed before reduce
    folds: int = 5
    bank_cap: int = 30000  # subsample bank above this many steps (speed)
