from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .anchors import Anchor, fallback_anchors
from .data import Trace, unit


@dataclass
class AnchorBank:
    vectors: np.ndarray
    anchors: List[Anchor]
    mode: str


def _partition_vector(q: np.ndarray, idx: int, total: int) -> np.ndarray:
    q = np.asarray(q, float)
    d = len(q)
    mask = np.zeros(d, float)
    total = max(1, int(total))
    width = max(1, d // total)
    start = (idx % total) * width
    end = d if idx == total - 1 else min(d, start + width)
    mask[start:end] = 1.0
    v = q * mask
    if np.linalg.norm(v) <= 1e-9:
        v = np.roll(q, idx * max(1, width // 2))
    return unit(v)


def _text_seed_vector(text: str, d: int) -> np.ndarray:
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return unit(rng.normal(size=d))


def build_anchor_bank(
    trace: Trace,
    anchors: Optional[List[Anchor]] = None,
    *,
    max_anchors: int = 24,
    fallback_partitions: int = 8,
    random: bool = False,
    shuffle_kinds: bool = False,
    seed: int = 0,
) -> AnchorBank:
    """Build a first-pass anchor bank.

    If prompt-span hidden vectors are unavailable, this uses qvec partitions as a
    deterministic fallback. That is not a semantic anchor implementation; the
    audit reports this mode explicitly and uses random/shuffled controls.
    """
    if anchors is None or len(anchors) == 0:
        anchors = fallback_anchors()
    anchors = list(anchors)[:max_anchors]

    if trace.qvec is not None:
        q = unit(trace.qvec)
        d = len(q)
    elif trace.stepvec is not None and len(trace.stepvec):
        q = unit(np.nanmean(trace.stepvec, axis=0))
        d = trace.stepvec.shape[1]
    else:
        d = 64
        q = unit(_text_seed_vector(trace.chain_id, d))

    rng = np.random.default_rng(seed + int(trace.idx) * 1009)
    vecs = []
    if random:
        for _ in anchors:
            vecs.append(unit(rng.normal(size=d)))
        mode = "random"
    else:
        for i, a in enumerate(anchors):
            base = _partition_vector(q, i, min(fallback_partitions, max(1, len(anchors))))
            # Text jitter breaks exact duplicate partitions without dominating q.
            jitter = _text_seed_vector(a.kind + ":" + a.text, d)
            vecs.append(unit(0.92 * base + 0.08 * jitter))
        mode = "q_partition_fallback"

    if shuffle_kinds and len(anchors) > 1:
        kinds = [a.kind for a in anchors]
        rng.shuffle(kinds)
        anchors = [
            Anchor(a.anchor_id, kinds[i], a.text, a.char_span, a.token_span, a.value)
            for i, a in enumerate(anchors)
        ]
        mode += "_shuffled_kinds"

    return AnchorBank(np.asarray(vecs, float), anchors, mode)
