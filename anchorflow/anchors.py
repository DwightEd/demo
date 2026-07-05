from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple


NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+)?%?")
CONSTRAINT_RE = re.compile(
    r"\b(total|each|every|more than|less than|left|remaining|remain|together|"
    r"altogether|difference|ratio|cost|price|twice|half|equal|greater|fewer|"
    r"after|before|if|when|per)\b",
    re.IGNORECASE,
)


@dataclass
class Anchor:
    anchor_id: int
    kind: str
    text: str
    char_span: Optional[Tuple[int, int]] = None
    token_span: Optional[Tuple[int, int]] = None
    value: Optional[float] = None


def _parse_value(text: str) -> Optional[float]:
    s = text.strip().rstrip("%")
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


def _context_span(text: str, start: int, end: int, *, radius: int = 28) -> Tuple[int, int]:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    while lo > 0 and text[lo - 1].isalnum():
        lo -= 1
    while hi < len(text) and text[hi].isalnum():
        hi += 1
    return lo, hi


def _goal_span(text: str) -> Tuple[int, int]:
    stripped = text.strip()
    if not stripped:
        return (0, 0)
    q = max(stripped.rfind("?"), stripped.rfind("."))
    if q <= 0:
        return (0, len(stripped))
    prev = max(stripped.rfind(".", 0, q), stripped.rfind("\n", 0, q))
    start = 0 if prev < 0 else prev + 1
    return (start, min(len(stripped), q + 1))


def parse_anchors(prompt_text: str, *, max_anchors: int = 24) -> List[Anchor]:
    text = str(prompt_text or "")
    anchors: List[Anchor] = []

    gid = 0
    gs, ge = _goal_span(text)
    goal = text[gs:ge].strip() or "question goal"
    anchors.append(Anchor(gid, "goal", goal, (gs, ge), None, None))
    gid += 1

    seen = {("goal", goal.lower())}
    for m in NUMBER_RE.finditer(text):
        if len(anchors) >= max_anchors:
            break
        raw = m.group(0)
        key = ("number", raw.lower())
        if key in seen:
            continue
        anchors.append(Anchor(gid, "number", raw, (m.start(), m.end()), None, _parse_value(raw)))
        gid += 1
        seen.add(key)
        if len(anchors) >= max_anchors:
            break
        cs, ce = _context_span(text, m.start(), m.end())
        ctx = text[cs:ce].strip()
        ckey = ("entity", ctx.lower())
        if ctx and ckey not in seen:
            anchors.append(Anchor(gid, "entity", ctx, (cs, ce), None, None))
            gid += 1
            seen.add(ckey)

    for m in CONSTRAINT_RE.finditer(text):
        if len(anchors) >= max_anchors:
            break
        cs, ce = _context_span(text, m.start(), m.end(), radius=36)
        ctx = text[cs:ce].strip()
        key = ("constraint", ctx.lower())
        if ctx and key not in seen:
            anchors.append(Anchor(gid, "constraint", ctx, (cs, ce), None, None))
            gid += 1
            seen.add(key)

    return anchors[:max_anchors]


def fallback_anchors(kinds: Iterable[str] = ("goal", "number", "constraint", "entity")) -> List[Anchor]:
    anchors = []
    for i, kind in enumerate(kinds):
        anchors.append(Anchor(i, kind, f"{kind} fallback"))
    return anchors


def anchors_to_jsonable(anchors: Iterable[Anchor]) -> List[Dict[str, object]]:
    return [asdict(a) for a in anchors]


def anchors_to_jsonl(rows: Iterable[Tuple[str, List[Anchor]]]) -> str:
    out = []
    for key, anchors in rows:
        out.append(json.dumps({"id": key, "anchors": anchors_to_jsonable(anchors)}, ensure_ascii=False))
    return "\n".join(out)


def anchor_coverage_stats(anchor_lists: Iterable[List[Anchor]]) -> Dict[str, object]:
    counts = []
    by_kind: Dict[str, int] = {}
    for anchors in anchor_lists:
        counts.append(len(anchors))
        for a in anchors:
            by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
    if not counts:
        return {"n": 0, "mean": 0.0, "min": 0, "max": 0, "by_kind": by_kind}
    return {
        "n": len(counts),
        "mean": float(sum(counts) / len(counts)),
        "min": int(min(counts)),
        "max": int(max(counts)),
        "by_kind": by_kind,
    }
