from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anchors import Anchor, fallback_anchors, parse_anchors
from .data import Trace, unit


EPS = 1e-9


@dataclass
class AnchorBank:
    vectors: np.ndarray
    anchors: List[Anchor]
    mode: str
    fallback_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def semantic(self) -> bool:
        """Whether every vector came from a real prompt-token span."""
        return bool(len(self.vectors)) and not bool(np.asarray(self.fallback_mask, bool).any())


def _stable_seed(text: str, seed: int = 0) -> int:
    """Process-independent seed; unlike Python ``hash``, this is reproducible."""
    payload = f"{int(seed)}\0{text}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


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
    if np.linalg.norm(v) <= EPS:
        v = np.roll(q, idx * max(1, width // 2))
    return unit(v)


def _text_seed_vector(text: str, d: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(text, seed))
    return unit(rng.normal(size=int(d)))


def char_span_to_token_span(
    offsets: Sequence[Sequence[int]],
    char_span: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Map a character span to an overlapping half-open token span ``[lo, hi)``.

    Special tokens with offset ``(0, 0)`` are ignored.  Offsets and character
    spans must refer to the exact same rendered prompt string.
    """
    if char_span is None:
        return None
    off = np.asarray(offsets, int)
    if off.ndim != 2 or off.shape[1] != 2:
        raise ValueError("prompt_offsets must have shape [n_tokens, 2]")
    cs, ce = int(char_span[0]), int(char_span[1])
    if ce <= cs:
        return None
    valid = (off[:, 1] > off[:, 0]) & (off[:, 1] > cs) & (off[:, 0] < ce)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1] + 1)


def prompt_span_vectors(
    anchors: Sequence[Anchor],
    prompt_offsets: Sequence[Sequence[int]],
    prompt_hidden: np.ndarray,
) -> Tuple[List[Anchor], List[Optional[np.ndarray]]]:
    """Pool real prompt hidden states over every parsed anchor span.

    ``prompt_hidden`` is a single selected layer with shape ``[tokens, dim]``.
    The returned anchors carry the resolved half-open token span.  Unmappable
    anchors retain ``None`` and can be filled by an explicitly marked fallback.
    """
    H = np.asarray(prompt_hidden, float)
    off = np.asarray(prompt_offsets, int)
    if H.ndim != 2:
        raise ValueError("prompt_hidden must have shape [n_tokens, hidden_dim]")
    if off.ndim != 2 or off.shape[1] != 2 or len(off) != len(H):
        raise ValueError("prompt_offsets must align one-to-one with prompt_hidden")

    resolved: List[Anchor] = []
    vectors: List[Optional[np.ndarray]] = []
    for a in anchors:
        span = char_span_to_token_span(off, a.char_span)
        vec: Optional[np.ndarray] = None
        if span is not None:
            lo, hi = span
            block = H[lo:hi]
            good = np.isfinite(block).all(axis=1)
            if good.any():
                pooled = np.mean(block[good], axis=0)
                if np.linalg.norm(pooled) > EPS:
                    vec = unit(pooled)
        resolved.append(
            Anchor(a.anchor_id, a.kind, a.text, a.char_span, span, a.value)
        )
        vectors.append(vec)
    return resolved, vectors


def select_prompt_hidden_layer(
    prompt_hidden: np.ndarray,
    *,
    prompt_hidden_layers: Optional[Sequence[int]] = None,
    layer: Optional[int] = None,
) -> np.ndarray:
    """Select ``[prompt_tokens, hidden_dim]`` from the trace-schema payload."""
    H = np.asarray(prompt_hidden, float)
    if H.ndim == 2:
        return H
    if H.ndim != 3:
        raise ValueError("prompt_hidden must be [P,D] or trace-schema [P,L,D]")
    if prompt_hidden_layers is None:
        if H.shape[1] != 1:
            raise ValueError("prompt_hidden_layers is required for multi-layer prompt hidden")
        return H[:, 0, :]
    layers = [int(x) for x in prompt_hidden_layers]
    if len(layers) != H.shape[1]:
        raise ValueError("prompt_hidden_layers does not match prompt_hidden layer axis")
    chosen = int(layer) if layer is not None else (layers[0] if len(layers) == 1 else None)
    if chosen is None or chosen not in layers:
        raise ValueError("requested trace layer is absent from prompt_hidden_layers")
    return H[:, layers.index(chosen), :]


def _feature_payload(trace: Trace, key: str):
    value = trace.features.get(key)
    if value is not None:
        return value
    aliases = {
        "prompt_offsets": "prompt_offsets",
        "token_offsets": "prompt_offsets",
        "prompt_hidden": "prompt_hidden",
        "prompt_hidden_layers": "prompt_hidden_layers",
        "question_char_span": "question_char_span",
        "target_question_char_span": "question_char_span",
    }
    attr = aliases.get(key)
    return None if attr is None else getattr(trace, attr, None)


def build_anchor_bank(
    trace: Trace,
    anchors: Optional[List[Anchor]] = None,
    *,
    max_anchors: int = 24,
    fallback_partitions: int = 8,
    prompt_offsets: Optional[np.ndarray] = None,
    prompt_hidden: Optional[np.ndarray] = None,
    prompt_hidden_layers: Optional[Sequence[int]] = None,
    prompt_layer: Optional[int] = None,
    random: bool = False,
    shuffle_kinds: bool = False,
    seed: int = 0,
) -> AnchorBank:
    """Build semantic prompt-span anchors, with a loudly labelled fallback.

    Real anchors require offsets and hidden states from the *same exact rendered
    prompt used for generation*.  For backward compatibility, those arrays may
    be passed explicitly or stored in ``trace.features`` under
    ``prompt_offsets`` and ``prompt_hidden``.  Missing spans fall back to q-vector
    partitions and are recorded in ``fallback_mask`` and ``metadata``.
    """
    if prompt_offsets is None:
        prompt_offsets = _feature_payload(trace, "prompt_offsets")
    if prompt_offsets is None:
        prompt_offsets = _feature_payload(trace, "token_offsets")
    if prompt_hidden is None:
        prompt_hidden = _feature_payload(trace, "prompt_hidden")
    if prompt_hidden_layers is None:
        prompt_hidden_layers = _feature_payload(trace, "prompt_hidden_layers")

    if anchors is None or len(anchors) == 0:
        target_span = _feature_payload(trace, "question_char_span")
        if target_span is None:
            target_span = _feature_payload(trace, "target_question_char_span")
        if trace.prompt_text:
            anchors = parse_anchors(
                trace.prompt_text,
                max_anchors=max_anchors,
                char_span=None if target_span is None else tuple(int(x) for x in target_span),
            )
        else:
            anchors = fallback_anchors()
    anchors = list(anchors)[: int(max_anchors)]

    span_vectors: List[Optional[np.ndarray]] = [None] * len(anchors)
    if prompt_offsets is not None and prompt_hidden is not None and anchors:
        selected_hidden = select_prompt_hidden_layer(
            prompt_hidden,
            prompt_hidden_layers=prompt_hidden_layers,
            layer=trace.layer if prompt_layer is None else prompt_layer,
        )
        selected_offsets = np.asarray(prompt_offsets, int)
        if len(selected_offsets) > len(selected_hidden):
            selected_offsets = selected_offsets[: len(selected_hidden)]
        anchors, span_vectors = prompt_span_vectors(
            anchors,
            selected_offsets,
            selected_hidden,
        )

    real_dims = [len(v) for v in span_vectors if v is not None]
    if real_dims:
        d = real_dims[0]
    elif trace.qvec is not None:
        d = int(np.asarray(trace.qvec).size)
    elif trace.stepvec is not None and len(trace.stepvec):
        d = int(np.asarray(trace.stepvec).shape[-1])
    else:
        d = 64

    if trace.qvec is not None and np.asarray(trace.qvec).size == d:
        q = unit(np.asarray(trace.qvec, float).reshape(-1))
    elif real_dims:
        q = unit(np.mean([v for v in span_vectors if v is not None], axis=0))
    elif trace.stepvec is not None and np.asarray(trace.stepvec).ndim == 2:
        q = unit(np.nanmean(np.asarray(trace.stepvec, float), axis=0))
    else:
        q = _text_seed_vector(trace.chain_id, d, seed=seed)

    rng = np.random.default_rng(_stable_seed(trace.chain_id, seed))
    vecs: List[np.ndarray] = []
    fallback: List[bool] = []
    if random:
        for _ in anchors:
            vecs.append(unit(rng.normal(size=d)))
            fallback.append(True)
        mode = "random_control"
    else:
        for i, (a, span_vec) in enumerate(zip(anchors, span_vectors)):
            if span_vec is not None:
                vecs.append(span_vec)
                fallback.append(False)
                continue
            base = _partition_vector(
                q, i, min(int(fallback_partitions), max(1, len(anchors)))
            )
            jitter = _text_seed_vector(a.kind + ":" + a.text, d, seed=seed)
            vecs.append(unit(0.92 * base + 0.08 * jitter))
            fallback.append(True)
        n_real = int(sum(not x for x in fallback))
        if n_real == len(anchors) and anchors:
            mode = "prompt_span_hidden"
        elif n_real:
            mode = "prompt_span_hidden_mixed_fallback"
        else:
            mode = "q_partition_fallback"

    if shuffle_kinds and len(anchors) > 1:
        perm = rng.permutation(len(anchors))
        kinds = [anchors[int(j)].kind for j in perm]
        anchors = [
            Anchor(a.anchor_id, kinds[i], a.text, a.char_span, a.token_span, a.value)
            for i, a in enumerate(anchors)
        ]
        mode += "_shuffled_kinds_control"

    fmask = np.asarray(fallback, dtype=bool)
    prompt_sha = hashlib.sha256(str(trace.prompt_text or "").encode("utf-8")).hexdigest()
    meta: Dict[str, object] = {
        "semantic_anchor_count": int((~fmask).sum()) if fmask.size else 0,
        "fallback_anchor_count": int(fmask.sum()) if fmask.size else 0,
        "vector_dim": int(d),
        "prompt_sha256": prompt_sha,
        "token_span_convention": "half_open",
    }
    return AnchorBank(np.asarray(vecs, float), anchors, mode, fmask, meta)
