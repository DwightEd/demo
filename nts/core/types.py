# nts/core/types.py — per-chain + flattened step containers
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ChainData:
    vecs: np.ndarray        # (T, d) float32 raw step vectors at analysis layer
    y: np.ndarray           # (T,) int  step error label
    length: np.ndarray      # (T,) float step token length
    speed: np.ndarray       # (T,) float ||h_t - h_{t-1}|| raw (nan at t=0)
    repetition: np.ndarray  # (T,) float trigram repetition rate
    kappa: np.ndarray       # (T,) float raw-kappa (resultant)
    problem_id: int
    correct: bool           # whole chain has no error


@dataclass
class Flat:
    y: np.ndarray
    groups: np.ndarray
    chain_correct: np.ndarray
    length: np.ndarray
    speed: np.ndarray
    repetition: np.ndarray
    kappa: np.ndarray


@dataclass
class StepTable:
    chains: List[ChainData]

    def flat(self) -> Flat:
        cat = lambda f: np.concatenate([f(c) for c in self.chains]) if self.chains else np.array([])
        return Flat(
            y=cat(lambda c: c.y),
            groups=cat(lambda c: np.full(len(c.y), c.problem_id)),
            chain_correct=cat(lambda c: np.full(len(c.y), c.correct)),
            length=cat(lambda c: c.length),
            speed=cat(lambda c: c.speed),
            repetition=cat(lambda c: c.repetition),
            kappa=cat(lambda c: c.kappa),
        )

    def correct_chains(self):
        return [c for c in self.chains if c.correct]
