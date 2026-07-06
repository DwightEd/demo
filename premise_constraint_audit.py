#!/usr/bin/env python3
"""Premise/constraint audit for same-problem reasoning samples.

This audit is intentionally not another hidden-geometry variant.  It tests a
different failure channel: whether a generated reasoning step preserves local
numeric premises and arithmetic constraints.  The target use case is the
coherent-but-wrong slice where geometry/entropy can look healthy while a chain
quietly commits to a false intermediate value.

The implementation is lightweight and local:

* parse per-step text from the multisample npz;
* build an online bank of question/previous-step numbers;
* flag invalid explicit equations, unsupported newly introduced numbers, and
  numbers derived from already tainted premises;
* evaluate the resulting risk scores with same-problem paired AUROC;
* report how often the constraint score rescues error/correct pairs missed by
  the strongest geometry/entropy baseline.

It is a diagnostic bridge toward a richer premise graph verifier, not a final
symbolic theorem prover.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multisample_temporal_rupture_audit import (
    auroc_signed,
    build_base_sequences,
    descriptive,
    finite_json,
    label_policy,
    paired_delta,
    problem_groups,
    safe_mean,
    signal_sign,
    within_pair_auroc,
)


EPS = 1e-12
NUMBER_RE = re.compile(
    r"(?<![\w/])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?"
)
ALLOWED_EXPR_CHARS = set("0123456789.,+-*/() %$")


@dataclass
class StepConstraintMetrics:
    n_numbers: int = 0
    n_new_numbers: int = 0
    valid_equations: int = 0
    invalid_equations: int = 0
    unsupported_numbers: int = 0
    tainted_dependencies: int = 0
    risk: float = 0.0


def parse_number(raw: str) -> Optional[float]:
    s = raw.strip().replace(",", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            val = float(a) / float(b)
        else:
            val = float(s)
    except Exception:
        return None
    if pct:
        val /= 100.0
    if not math.isfinite(val):
        return None
    return float(val)


def close_num(a: float, b: float, *, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b), 1.0)


def normalize_math_text(text: str) -> str:
    return (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("＝", "=")
        .replace("≈", "=")
    )


def extract_numbers(text: str) -> List[float]:
    """Extract numeric literals while filtering common step indices."""
    vals: List[float] = []
    text = normalize_math_text(text)
    for m in NUMBER_RE.finditer(text):
        prefix = text[max(0, m.start() - 14) : m.start()].lower()
        token = m.group(0)
        if re.search(r"(step|part|case|line|#)\s*$", prefix):
            continue
        if token in {"-", "+", ""}:
            continue
        val = parse_number(token)
        if val is not None:
            vals.append(val)
    return vals


def strip_to_expr(text: str, *, side: str) -> str:
    text = normalize_math_text(text)
    if side == "left":
        i = len(text) - 1
        while i >= 0 and text[i] in ALLOWED_EXPR_CHARS:
            i -= 1
        return text[i + 1 :].strip()
    i = 0
    while i < len(text) and text[i] in ALLOWED_EXPR_CHARS:
        i += 1
    return text[:i].strip()


def normalize_expr(expr: str) -> str:
    expr = normalize_math_text(expr)
    expr = expr.replace("$", "").replace(",", "")
    expr = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expr)
    # Keep only a conservative arithmetic subset.
    expr = "".join(ch for ch in expr if ch in set("0123456789.+-*/() "))
    return expr.strip()


def safe_eval_expr(expr: str) -> Optional[float]:
    expr = normalize_expr(expr)
    if not expr or not re.search(r"\d", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = ev(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a = ev(node.left)
            b = ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if abs(b) < EPS:
                raise ZeroDivisionError
            return a / b
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    try:
        val = ev(tree)
    except Exception:
        return None
    return float(val) if math.isfinite(val) else None


def equation_pairs(text: str) -> List[Tuple[float, float, str, str]]:
    """Return evaluable adjacent equation pairs from a reasoning step."""
    text = normalize_math_text(text)
    if "=" not in text:
        return []
    parts = re.split(r"=+", text)
    pairs: List[Tuple[float, float, str, str]] = []
    for left_raw, right_raw in zip(parts[:-1], parts[1:]):
        left = strip_to_expr(left_raw, side="left")
        right = strip_to_expr(right_raw, side="right")
        if not left or not right:
            continue
        lv = safe_eval_expr(left)
        rv = safe_eval_expr(right)
        if lv is None or rv is None:
            continue
        pairs.append((lv, rv, normalize_expr(left), normalize_expr(right)))
    return pairs


class KnownNumberBank:
    """Small online bank of known numbers with taint propagation."""

    def __init__(self, *, max_items: int = 80) -> None:
        self.max_items = max_items
        self.values: List[Tuple[float, bool]] = []

    def add(self, value: float, *, tainted: bool) -> None:
        for old, old_tainted in self.values:
            if close_num(value, old):
                if tainted and not old_tainted:
                    # Preserve the untainted support if it already exists.
                    return
                return
        self.values.append((float(value), bool(tainted)))
        if len(self.values) > self.max_items:
            self.values = self.values[-self.max_items :]

    def seed(self, values: Iterable[float]) -> None:
        for v in values:
            self.add(float(v), tainted=False)

    def direct_support(self, value: float) -> Optional[bool]:
        for old, tainted in self.values:
            if close_num(value, old):
                return bool(tainted)
        return None

    def derived_support(self, value: float) -> Optional[bool]:
        vals = self.values[-min(len(self.values), 50) :]
        for i, (a, ta) in enumerate(vals):
            for b, tb in vals[i:]:
                candidates = [a + b, a - b, b - a, a * b]
                if abs(b) > EPS:
                    candidates.append(a / b)
                if abs(a) > EPS:
                    candidates.append(b / a)
                if any(close_num(value, c) for c in candidates):
                    return bool(ta or tb)
        return None

    def support(self, value: float) -> Tuple[bool, bool]:
        direct = self.direct_support(value)
        if direct is not None:
            return True, direct
        derived = self.derived_support(value)
        if derived is not None:
            return True, derived
        return False, False


def step_constraint_metrics(text: str, bank: KnownNumberBank) -> StepConstraintMetrics:
    nums = extract_numbers(text)
    pairs = equation_pairs(text)
    valid_eq = 0
    invalid_eq = 0
    invalid_values: List[float] = []
    for lv, rv, _left, _right in pairs:
        if close_num(lv, rv):
            valid_eq += 1
        else:
            invalid_eq += 1
            invalid_values.extend([lv, rv])

    unsupported = 0
    tainted_deps = 0
    new_numbers = 0
    for v in nums:
        direct = bank.direct_support(v)
        if direct is not None:
            if direct:
                tainted_deps += 1
            continue
        new_numbers += 1
        supported, dep_tainted = bank.support(v)
        is_invalid_value = any(close_num(v, bad) for bad in invalid_values)
        if not supported:
            unsupported += 1
        elif dep_tainted:
            tainted_deps += 1
        bank.add(v, tainted=(not supported) or dep_tainted or is_invalid_value)

    denom = max(1, len(nums))
    eq_penalty = float(invalid_eq)
    unsupported_rate = unsupported / denom
    taint_rate = tainted_deps / denom
    risk = eq_penalty + unsupported_rate + 0.5 * taint_rate
    return StepConstraintMetrics(
        n_numbers=len(nums),
        n_new_numbers=new_numbers,
        valid_equations=valid_eq,
        invalid_equations=invalid_eq,
        unsupported_numbers=unsupported,
        tainted_dependencies=tainted_deps,
        risk=float(risk),
    )


def object_to_str_list(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    try:
        return [str(x) for x in list(obj)]
    except Exception:
        return []


def optional_text_column(data: np.lib.npyio.NpzFile, names: Sequence[str], i: int) -> str:
    for name in names:
        if name not in data.files:
            continue
        col = data[name]
        try:
            if len(col) == len(data["problem_ids"]):
                return str(col[i])
            if len(col) > int(data["problem_ids"][i]):
                return str(col[int(data["problem_ids"][i])])
        except Exception:
            continue
    return ""


def step_texts_for_sample(data: np.lib.npyio.NpzFile, i: int) -> List[str]:
    if "steps_text" in data.files:
        steps = object_to_str_list(data["steps_text"][i])
        if steps:
            return steps
    if "responses" in data.files:
        txt = str(data["responses"][i])
        chunks = [x.strip() for x in re.split(r"\n+|(?<=[.;])\s+(?=(?:Then|Next|So|Thus|Therefore|We)\b)", txt) if x.strip()]
        return chunks if chunks else [txt]
    return []


def constraint_sequences(data: np.lib.npyio.NpzFile) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    n = len(data["problem_ids"])
    seqs: List[Dict[str, np.ndarray]] = []
    coverage = {
        "samples_with_steps": 0,
        "samples_with_equations": 0,
        "total_steps": 0,
        "steps_with_numbers": 0,
        "steps_with_equations": 0,
    }
    question_names = ("questions", "question", "problem_text", "problem_texts", "prompts", "prompt")
    for i in range(n):
        question = optional_text_column(data, question_names, i)
        bank = KnownNumberBank()
        bank.seed(extract_numbers(question))
        steps = step_texts_for_sample(data, i)
        if steps:
            coverage["samples_with_steps"] += 1
        rows: List[StepConstraintMetrics] = []
        for step in steps:
            m = step_constraint_metrics(step, bank)
            rows.append(m)
            coverage["total_steps"] += 1
            if m.n_numbers:
                coverage["steps_with_numbers"] += 1
            if m.valid_equations or m.invalid_equations:
                coverage["steps_with_equations"] += 1
        if any(r.valid_equations or r.invalid_equations for r in rows):
            coverage["samples_with_equations"] += 1
        seqs.append(
            {
                "risk": np.asarray([r.risk for r in rows], dtype=np.float64),
                "invalid_eq": np.asarray([r.invalid_equations for r in rows], dtype=np.float64),
                "valid_eq": np.asarray([r.valid_equations for r in rows], dtype=np.float64),
                "unsupported": np.asarray([r.unsupported_numbers for r in rows], dtype=np.float64),
                "tainted_dep": np.asarray([r.tainted_dependencies for r in rows], dtype=np.float64),
                "n_numbers": np.asarray([r.n_numbers for r in rows], dtype=np.float64),
                "n_new_numbers": np.asarray([r.n_new_numbers for r in rows], dtype=np.float64),
            }
        )
    return seqs, coverage


def seq_score(v: np.ndarray, mode: str) -> Tuple[float, float]:
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).any():
        return float("nan"), float("nan")
    pos = np.arange(x.size, dtype=np.float64) / max(1, x.size - 1)
    if mode == "mean":
        return safe_mean(x), float("nan")
    if mode == "late_mean":
        m = pos >= 0.60
        return safe_mean(x[m]), float("nan")
    if mode == "sum":
        return float(np.nansum(x)), float("nan")
    k = int(np.nanargmax(x))
    return float(x[k]), float(pos[k])


def evaluate_score(
    name: str,
    vals: np.ndarray,
    pos: np.ndarray,
    *,
    y_err: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[np.ndarray],
) -> Dict[str, Any]:
    m = mask & np.isfinite(vals)
    err = vals[m & (y_err == 1)]
    cor = vals[m & (y_err == 0)]
    w, pairs = within_pair_auroc(groups, vals, y_err)
    return {
        "score": name,
        "n": int(m.sum()),
        "n_error": int((m & (y_err == 1)).sum()),
        "n_correct": int((m & (y_err == 0)).sum()),
        "cross_auroc_error_high": auroc_signed(err, cor),
        "within_pair_auroc_error_high": w,
        "within_pairs": int(pairs),
        "paired_delta_error_minus_correct": paired_delta(groups, vals, y_err),
        "error": descriptive(err),
        "correct": descriptive(cor),
        "argpos_error": descriptive(pos[m & (y_err == 1)]),
        "argpos_correct": descriptive(pos[m & (y_err == 0)]),
    }


def score_constraint_sequences(
    seqs: Sequence[Mapping[str, np.ndarray]],
    *,
    y_err: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[np.ndarray],
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    score_arrays: Dict[str, np.ndarray] = {}
    modes = ("max", "mean", "late_mean", "sum")
    for channel in ("risk", "invalid_eq", "unsupported", "tainted_dep"):
        for mode in modes:
            vals = np.full(len(seqs), np.nan, dtype=np.float64)
            pos = np.full(len(seqs), np.nan, dtype=np.float64)
            for i, s in enumerate(seqs):
                vals[i], pos[i] = seq_score(np.asarray(s[channel], dtype=np.float64), mode)
            name = f"constraint_{channel}_{mode}"
            score_arrays[name] = vals
            rows.append(evaluate_score(name, vals, pos, y_err=y_err, mask=mask, groups=groups))
    rows.sort(key=lambda r: (np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0)), reverse=True)
    return rows, score_arrays


def baseline_scores(
    data: np.lib.npyio.NpzFile,
    *,
    bands: Sequence[str],
    y_err: np.ndarray,
    mask: np.ndarray,
    groups: Sequence[np.ndarray],
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    seqs = build_base_sequences(data, bands=bands)
    names = sorted({k for s in seqs for k in s.keys() if k != "step_pos"})
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    for ch in names:
        for mode in ("mean", "late_mean", "max"):
            vals = np.full(len(seqs), np.nan, dtype=np.float64)
            pos = np.full(len(seqs), np.nan, dtype=np.float64)
            sign = signal_sign(ch)
            for i, s in enumerate(seqs):
                if ch not in s:
                    continue
                vals[i], pos[i] = seq_score(sign * np.asarray(s[ch], dtype=np.float64), mode)
            name = f"baseline_{ch}_{mode}"
            arrays[name] = vals
            rows.append(evaluate_score(name, vals, pos, y_err=y_err, mask=mask, groups=groups))
    if "n_steps" in data.files:
        vals = data["n_steps"].astype(float)
        pos = np.full(len(vals), np.nan)
        arrays["baseline_n_steps"] = vals
        rows.append(evaluate_score("baseline_n_steps", vals, pos, y_err=y_err, mask=mask, groups=groups))
    rows = [r for r in rows if r["n"] > 0]
    rows.sort(key=lambda r: (np.nan_to_num(r["within_pair_auroc_error_high"], nan=-1.0)), reverse=True)
    return rows, arrays


def bootstrap_within_increment(
    new_score: np.ndarray,
    base_score: np.ndarray,
    *,
    groups: Sequence[np.ndarray],
    y_err: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    point_new, _ = within_pair_auroc(groups, new_score, y_err)
    point_base, _ = within_pair_auroc(groups, base_score, y_err)
    point = point_new - point_base
    if not np.isfinite(point) or not groups:
        return {"point": None, "lo": None, "hi": None, "sig": False}
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        picked = [groups[int(j)] for j in rng.integers(0, len(groups), size=len(groups))]
        a, _ = within_pair_auroc(picked, new_score, y_err)
        b, _ = within_pair_auroc(picked, base_score, y_err)
        if np.isfinite(a) and np.isfinite(b):
            vals.append(a - b)
    if not vals:
        return {"point": float(point), "lo": None, "hi": None, "sig": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


def pair_rescue_report(
    new_score: np.ndarray,
    base_score: np.ndarray,
    *,
    groups: Sequence[np.ndarray],
    y_err: np.ndarray,
) -> Dict[str, Any]:
    missed = 0
    rescued = 0
    both_ok = 0
    total = 0
    for idx in groups:
        err = [i for i in idx if y_err[i] == 1 and np.isfinite(new_score[i]) and np.isfinite(base_score[i])]
        cor = [i for i in idx if y_err[i] == 0 and np.isfinite(new_score[i]) and np.isfinite(base_score[i])]
        for e in err:
            for c in cor:
                total += 1
                base_ok = base_score[e] > base_score[c]
                new_ok = new_score[e] > new_score[c]
                both_ok += int(base_ok and new_ok)
                if not base_ok:
                    missed += 1
                    rescued += int(new_ok)
    return {
        "finite_pairs": int(total),
        "baseline_missed_pairs": int(missed),
        "rescued_pairs": int(rescued),
        "rescue_rate_among_baseline_misses": float(rescued / missed) if missed else float("nan"),
        "both_correct_pairs": int(both_ok),
    }


def run(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "problem_ids" not in data.files:
        raise SystemExit("npz missing problem_ids")
    y_err, mask, desc = label_policy(data, args.policy)
    problem_ids = data["problem_ids"].astype(int)
    groups = problem_groups(problem_ids, y_err, mask, args.min_per_class)
    if not groups:
        raise SystemExit(f"policy {args.policy!r} has no contrastive same-problem groups")

    constraint_seqs, coverage = constraint_sequences(data)
    constraint_rows, constraint_arrays = score_constraint_sequences(
        constraint_seqs, y_err=y_err, mask=mask, groups=groups
    )
    bands = [x.strip() for x in args.bands.split(",") if x.strip()]
    baseline_rows, baseline_arrays = baseline_scores(
        data, bands=bands, y_err=y_err, mask=mask, groups=groups
    )
    best_constraint = constraint_rows[0] if constraint_rows else None
    best_baseline = baseline_rows[0] if baseline_rows else None
    increment: Dict[str, Any] = {}
    rescue: Dict[str, Any] = {}
    if best_constraint and best_baseline:
        cscore = constraint_arrays[best_constraint["score"]]
        bscore = baseline_arrays[best_baseline["score"]]
        increment = bootstrap_within_increment(
            cscore,
            bscore,
            groups=groups,
            y_err=y_err,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
        rescue = pair_rescue_report(cscore, bscore, groups=groups, y_err=y_err)

    return {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "policy": args.policy,
            "policy_description": desc,
            "n_samples_policy": int(mask.sum()),
            "n_error_policy": int(y_err[mask].sum()),
            "n_correct_policy": int(mask.sum() - y_err[mask].sum()),
            "n_contrastive_problems": int(len(groups)),
            "bands": bands,
            "coverage": coverage,
            "method": {
                "core_hypothesis": "Coherent wrong steps can preserve geometry/entropy while violating local premise or arithmetic constraints.",
                "non_probe_property": "Scores are computed from explicit step text constraints and same-problem pair tests, not from a supervised hidden-state classifier.",
            },
        },
        "headline": {
            "best_constraint": best_constraint,
            "best_baseline": best_baseline,
            "increment_over_best_baseline": increment,
            "baseline_miss_rescue": rescue,
        },
        "constraint_scores": constraint_rows,
        "baseline_scores": baseline_rows[:20],
    }


def write_markdown(path: str, res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]
    lines: List[str] = []
    lines.append("# Premise Constraint Audit\n")
    lines.append(f"- input: `{meta['input']}`")
    lines.append(f"- policy: `{meta['policy']}`")
    lines.append(f"- contrastive problems: {meta['n_contrastive_problems']}")
    cov = meta["coverage"]
    lines.append(
        f"- text coverage: samples with steps {cov['samples_with_steps']}, "
        f"samples with equations {cov['samples_with_equations']}, "
        f"steps with equations {cov['steps_with_equations']} / {cov['total_steps']}"
    )
    lines.append("")
    bc = head.get("best_constraint") or {}
    bb = head.get("best_baseline") or {}
    lines.append("## Headline\n")
    if bc:
        lines.append(
            f"- Best constraint: `{bc['score']}` within AUROC {bc['within_pair_auroc_error_high']:.3f} "
            f"(cross {bc['cross_auroc_error_high']:.3f})."
        )
    if bb:
        lines.append(
            f"- Best baseline: `{bb['score']}` within AUROC {bb['within_pair_auroc_error_high']:.3f} "
            f"(cross {bb['cross_auroc_error_high']:.3f})."
        )
    inc = head.get("increment_over_best_baseline") or {}
    if inc:
        lines.append(
            f"- Increment over best baseline: {inc.get('point')} "
            f"CI=[{inc.get('lo')}, {inc.get('hi')}] sig={inc.get('sig')}."
        )
    rescue = head.get("baseline_miss_rescue") or {}
    if rescue:
        lines.append(
            f"- Baseline missed {rescue.get('baseline_missed_pairs')} finite same-problem pairs; "
            f"constraint rescued {rescue.get('rescued_pairs')} "
            f"({rescue.get('rescue_rate_among_baseline_misses')})."
        )
    lines.append("")
    lines.append("## Top Constraint Scores\n")
    lines.append("| score | within AUROC | cross AUROC | pairs |")
    lines.append("|---|---:|---:|---:|")
    for r in res["constraint_scores"][:12]:
        lines.append(
            f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
            f"{r['cross_auroc_error_high']:.3f} | {r['within_pairs']} |"
        )
    lines.append("")
    lines.append("## Top Baselines\n")
    lines.append("| score | within AUROC | cross AUROC | pairs |")
    lines.append("|---|---:|---:|---:|")
    for r in res["baseline_scores"][:12]:
        lines.append(
            f"| {r['score']} | {r['within_pair_auroc_error_high']:.3f} | "
            f"{r['cross_auroc_error_high']:.3f} | {r['within_pairs']} |"
        )
    lines.append("")
    lines.append("## Anti-Degradation Checks\n")
    lines.append("- Same-problem paired AUROC is the headline metric.")
    lines.append("- The baseline is selected from saved geometry/entropy/length channels before computing the rescue report.")
    lines.append("- `baseline_miss_rescue` only counts pairs where the strongest baseline fails to rank the error sample above the correct sample.")
    lines.append("- This audit is local and parser-bounded; low equation coverage should be read as insufficient evidence, not as absence of premise errors.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_outputs(res: Mapping[str, Any], output_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jpath = os.path.join(output_dir, stem + ".json")
    mpath = os.path.join(output_dir, stem + ".md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, ensure_ascii=False, indent=2)
    write_markdown(mpath, finite_json(res))
    return jpath, mpath


def _object_array(xs: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest(path: str, *, n_problems: int = 24, samples_per_problem: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    problem_ids: List[int] = []
    sample_idx: List[int] = []
    is_correct: List[int] = []
    steps_text: List[np.ndarray] = []
    questions: List[str] = []
    n_steps: List[int] = []
    ent: List[np.ndarray] = []
    clouds: List[np.ndarray] = []
    sizes: List[np.ndarray] = []
    pr: List[np.ndarray] = []
    ae: List[np.ndarray] = []
    for p in range(n_problems):
        a = int(rng.integers(2, 9))
        b = int(rng.integers(2, 9))
        c = int(rng.integers(2, 7))
        gold = (a + b) * c
        wrong_mid = a + b + int(rng.choice([-2, -1, 1, 2]))
        wrong = wrong_mid * c
        for s in range(samples_per_problem):
            err = int(s >= samples_per_problem // 2)
            problem_ids.append(p)
            sample_idx.append(s)
            is_correct.append(1 - err)
            questions.append(f"Alice has {a} red marbles and {b} blue marbles. She makes {c} identical boxes. How many marbles are used?")
            if err:
                steps = np.array(
                    [
                        f"Step 1: Combine the marbles: {a} + {b} = {wrong_mid}.",
                        f"Step 2: Put that amount in each box: {wrong_mid} * {c} = {wrong}.",
                        f"#### {wrong}",
                    ],
                    dtype=object,
                )
            else:
                mid = a + b
                steps = np.array(
                    [
                        f"Step 1: Combine the marbles: {a} + {b} = {mid}.",
                        f"Step 2: Put that amount in each box: {mid} * {c} = {gold}.",
                        f"#### {gold}",
                    ],
                    dtype=object,
                )
            steps_text.append(steps)
            n_steps.append(len(steps))
            # Keep entropy/spread deliberately uninformative so the selftest
            # exercises complementarity rather than hidden-geometry leakage.
            entropy = 0.25 + 0.02 * rng.normal(size=len(steps))
            ent.append(entropy.astype(np.float32))
            step_sizes = np.array([5, 5, 3], dtype=np.int32)
            sizes.append(step_sizes)
            H = []
            for t, sz in enumerate(step_sizes):
                center = rng.normal(size=12)
                center /= np.linalg.norm(center)
                H.append(center[None, :] + 0.04 * rng.normal(size=(int(sz), 12)))
            clouds.append(np.concatenate(H, axis=0)[:, None, :].astype(np.float32))
            pr.append((2.0 + 0.03 * rng.normal(size=(len(steps), 33))).astype(np.float32))
            ae.append((0.8 + 0.03 * rng.normal(size=(len(steps), 33))).astype(np.float32))
    np.savez(
        path,
        problem_ids=np.asarray(problem_ids, dtype=np.int32),
        sample_idx=np.asarray(sample_idx, dtype=np.int32),
        is_correct=np.asarray(is_correct, dtype=np.int32),
        is_correct_strict=np.asarray(is_correct, dtype=np.int32),
        format_ok=np.ones(len(problem_ids), dtype=bool),
        questions=np.asarray(questions, dtype=object),
        steps_text=_object_array(steps_text),
        n_steps=np.asarray(n_steps, dtype=np.int32),
        sv_out_entropy=_object_array(ent),
        sv_clouds=_object_array(clouds),
        cloud_sizes=_object_array(sizes),
        sv_pr_step_exp=_object_array(pr),
        sv_ae_step_exp=_object_array(ae),
        model_name=np.asarray("selftest"),
        prompt_style=np.asarray("premise-constraint-selftest"),
        step_split=np.asarray("selftest"),
    )


def assert_selftest(res: Mapping[str, Any]) -> None:
    bc = res["headline"]["best_constraint"]
    if bc is None or bc["within_pair_auroc_error_high"] < 0.95:
        raise SystemExit("selftest failed: constraint score did not recover arithmetic premise errors")
    rescue = res["headline"]["baseline_miss_rescue"]
    if rescue.get("baseline_missed_pairs", 0) <= 0:
        raise SystemExit("selftest failed: baseline had no misses, complementarity was not tested")
    if rescue.get("rescue_rate_among_baseline_misses", 0.0) < 0.90:
        raise SystemExit("selftest failed: constraint score did not rescue baseline misses")


def print_summary(res: Mapping[str, Any]) -> None:
    meta = res["meta"]
    head = res["headline"]
    print(f"\n===== premise constraint audit | {meta['basename']} | {meta['policy']} =====")
    print(
        f"samples {meta['n_samples_policy']} | err {meta['n_error_policy']} | "
        f"contrastive problems {meta['n_contrastive_problems']}"
    )
    cov = meta["coverage"]
    print(
        f"text coverage: steps {cov['samples_with_steps']} samples | "
        f"equation samples {cov['samples_with_equations']} | "
        f"equation steps {cov['steps_with_equations']}/{cov['total_steps']}"
    )
    bc = head.get("best_constraint")
    bb = head.get("best_baseline")
    if bc:
        print(
            f"best constraint {bc['score']} within={bc['within_pair_auroc_error_high']:.3f} "
            f"cross={bc['cross_auroc_error_high']:.3f}"
        )
    if bb:
        print(
            f"best baseline   {bb['score']} within={bb['within_pair_auroc_error_high']:.3f} "
            f"cross={bb['cross_auroc_error_high']:.3f}"
        )
    inc = head.get("increment_over_best_baseline") or {}
    if inc:
        print(f"increment over baseline: {inc.get('point')} CI=[{inc.get('lo')}, {inc.get('hi')}] sig={inc.get('sig')}")
    rescue = head.get("baseline_miss_rescue") or {}
    if rescue:
        print(
            f"baseline misses rescued: {rescue.get('rescued_pairs')}/"
            f"{rescue.get('baseline_missed_pairs')} "
            f"({rescue.get('rescue_rate_among_baseline_misses')})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="same-problem multisample npz")
    ap.add_argument("--policy", default="answer_format_ok", choices=["answer", "strict", "answer_format_ok"])
    ap.add_argument("--min_per_class", type=int, default=1)
    ap.add_argument("--bands", default="mid,deep")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="outputs/premise_constraint_audit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            npz = os.path.join(td, "premise_constraint_selftest.npz")
            make_selftest(npz, seed=args.seed)
            res = run(npz, args)
            assert_selftest(res)
            jpath, mpath = write_outputs(res, args.output_dir, "premise_constraint_selftest")
            print_summary(res)
            print(f"\nselftest passed; saved: {jpath}\nselftest passed; saved: {mpath}")
        return

    if not args.input:
        raise SystemExit("pass --input or --selftest")
    res = run(args.input, args)
    stem = os.path.splitext(os.path.basename(args.input))[0] + f"_{args.policy}"
    jpath, mpath = write_outputs(res, args.output_dir, stem)
    print_summary(res)
    print(f"\nsaved: {jpath}\nsaved: {mpath}")


if __name__ == "__main__":
    main()
