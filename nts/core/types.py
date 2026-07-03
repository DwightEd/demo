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
    hidden_path: str = None  # path to per-token hidden shard (R,L,d) for step-free cloud signals
    hidden_col: int = 0      # layer column within that shard
    step_ranges: np.ndarray = None  # (T,2) shard-relative [lo,hi) token slices per step (for per-step NTC)


def step_eval_mask(c) -> np.ndarray:
    """Step-level evaluation mask (True = step may be used in step-level metrics).

    ProcessBench only certifies steps up to and including the FIRST error
    (gold_error_step); later steps of an error chain are unjudged (and often
    also wrong), so using them as negatives contaminates the negative class.
    Error chains whose error step is not stored (y all zero, e.g. truncation)
    are excluded entirely. Chain-level scoring is unaffected by this mask."""
    ok = np.ones(len(c.y), bool)
    if not c.correct:
        if (c.y == 1).any():
            ok[int(np.argmax(c.y == 1)) + 1:] = False
        else:
            ok[:] = False
    return ok


@dataclass
class Flat:
    y: np.ndarray
    groups: np.ndarray
    chain_correct: np.ndarray
    length: np.ndarray
    speed: np.ndarray
    repetition: np.ndarray
    kappa: np.ndarray
    eval_ok: np.ndarray


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
            eval_ok=cat(step_eval_mask),
        )

    def correct_chains(self):
        return [c for c in self.chains if c.correct]
