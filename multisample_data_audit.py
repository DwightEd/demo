#!/usr/bin/env python3
"""Audit same-problem multi-sampling npz files before trajectory modeling.

The within-problem experiments only mean what their data support.  This script
records the sample/problem layout, label/format failure mass, saved arrays, and
basic text/step statistics so later detectors can separate reasoning failures
from prompt-format artifacts and difficulty confounds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


EPS = 1e-12


def finite_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def qstats(v: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "q10": float(np.quantile(a, 0.10)),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "q90": float(np.quantile(a, 0.90)),
        "max": float(np.max(a)),
    }


def scalar_str(data: np.lib.npyio.NpzFile, key: str, default: str = "unknown") -> str:
    if key not in data.files:
        return default
    x = data[key]
    try:
        if np.ndim(x) == 0:
            return str(x.item())
        if len(x) == 1:
            return str(x[0])
    except Exception:
        pass
    return str(x)


def array_shape_summary(data: np.lib.npyio.NpzFile, key: str, *, max_items: int = 8) -> Dict[str, Any]:
    arr = data[key]
    info: Dict[str, Any] = {
        "dtype": str(arr.dtype),
        "shape": tuple(int(x) for x in arr.shape),
    }
    if arr.dtype == object and arr.size:
        shapes = []
        for obj in arr[: min(max_items, arr.size)]:
            try:
                shapes.append(tuple(int(x) for x in np.asarray(obj).shape))
            except Exception:
                shapes.append("unavailable")
        info["example_object_shapes"] = shapes
    return info


def value_counts(xs: Iterable[Any], *, top: int = 20) -> Dict[str, int]:
    c = Counter(str(x) for x in xs)
    return dict(c.most_common(top))


def policy_labels(data: np.lib.npyio.NpzFile, policy: str) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(data["problem_ids"])
    if policy == "answer":
        return (data["is_correct"].astype(int) == 0).astype(int), np.ones(n, bool), "lenient answer incorrect"
    if policy == "strict":
        return (data["is_correct_strict"].astype(int) == 0).astype(int), np.ones(n, bool), "strict marker+answer incorrect"
    if policy == "answer_format_ok":
        if "format_ok" not in data.files:
            return (data["is_correct"].astype(int) == 0).astype(int), np.ones(n, bool), "answer incorrect; format_ok missing"
        return (
            (data["is_correct"].astype(int) == 0).astype(int),
            data["format_ok"].astype(bool),
            "answer incorrect among samples with requested final-answer marker",
        )
    raise ValueError(policy)


def problem_layout(problem_ids: np.ndarray, y_err: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    by_problem: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(problem_ids):
        if mask[i]:
            by_problem[int(p)].append(i)

    n_samples = []
    n_err = []
    n_cor = []
    n_pairs = 0
    mixed = all_err = all_cor = 0
    for idxs in by_problem.values():
        e = int(np.sum(y_err[idxs] == 1))
        c = int(np.sum(y_err[idxs] == 0))
        n_samples.append(len(idxs))
        n_err.append(e)
        n_cor.append(c)
        n_pairs += e * c
        if e > 0 and c > 0:
            mixed += 1
        elif e > 0:
            all_err += 1
        else:
            all_cor += 1

    return {
        "n_problems": int(len(by_problem)),
        "n_samples": int(sum(n_samples)),
        "contrastive_problems": int(mixed),
        "all_correct_problems": int(all_cor),
        "all_error_problems": int(all_err),
        "same_problem_error_correct_pairs": int(n_pairs),
        "samples_per_problem": qstats(n_samples),
        "errors_per_problem": qstats(n_err),
        "correct_per_problem": qstats(n_cor),
    }


def safe_steps_text(data: np.lib.npyio.NpzFile) -> List[List[str]]:
    if "steps_text" not in data.files:
        return []
    out: List[List[str]] = []
    for obj in data["steps_text"]:
        try:
            out.append([str(x) for x in list(obj)])
        except Exception:
            out.append([])
    return out


def text_audit(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "responses" in data.files:
        responses = [str(x) for x in data["responses"]]
        out["response_chars"] = qstats([len(x) for x in responses])
        out["has_final_marker_rate"] = float(np.mean([bool(re.search(r"####\s*[-+]?\d", x)) for x in responses]))
        out["looks_truncated_rate"] = float(
            np.mean(
                [
                    (not re.search(r"####\s*[-+]?\d", x))
                    and bool(re.search(r"(therefore|so|we get|equals|=)\s*$", x.strip(), re.I))
                    for x in responses
                ]
            )
        )
    steps = safe_steps_text(data)
    if steps:
        step_lens = [len(s) for chain in steps for s in chain]
        out["parsed_steps_per_sample"] = qstats([len(chain) for chain in steps])
        out["step_chars"] = qstats(step_lens)
        out["empty_step_chains"] = int(sum(1 for chain in steps if not chain))
    return out


def metric_payload_audit(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in data.files:
        if key.startswith("sv_") or key in {"layers_used", "cloud_layers", "clouds_stored"}:
            payload[key] = array_shape_summary(data, key)
    payload["has_raw_step_vectors"] = bool(data.get("sv_vectors_stored", np.array(False)))
    payload["has_token_clouds"] = bool(data.get("clouds_stored", np.array(False)))
    payload["has_token_uncertainty"] = "sv_tok_entropy" in data.files or "sv_out_entropy" in data.files
    payload["has_raw_responses"] = "responses" in data.files
    payload["has_steps_text"] = "steps_text" in data.files
    return payload


def label_audit(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    n = len(data["problem_ids"])
    out: Dict[str, Any] = {"n_samples": int(n)}
    is_correct = data["is_correct"].astype(int) if "is_correct" in data.files else np.zeros(n, int)
    strict = data["is_correct_strict"].astype(int) if "is_correct_strict" in data.files else None
    fmt = data["format_ok"].astype(int) if "format_ok" in data.files else None

    out["answer_correct"] = int(is_correct.sum())
    out["answer_error"] = int(n - is_correct.sum())
    if strict is not None:
        out["strict_correct"] = int(strict.sum())
        out["strict_error"] = int(n - strict.sum())
        out["lenient_minus_strict_correct"] = int(is_correct.sum() - strict.sum())
    if fmt is not None:
        out["format_ok"] = int(fmt.sum())
        out["format_fail"] = int(n - fmt.sum())
        ok = fmt.astype(bool)
        out["answer_format_ok_correct"] = int(is_correct[ok].sum())
        out["answer_format_ok_error"] = int(ok.sum() - is_correct[ok].sum())
        fail = ~ok
        out["format_fail_lenient_correct"] = int(is_correct[fail].sum())
        out["format_fail_lenient_error"] = int(fail.sum() - is_correct[fail].sum())
    if "pred_source" in data.files:
        pred_source = [str(x) for x in data["pred_source"]]
        out["pred_source_counts"] = value_counts(pred_source)
        by_src: Dict[str, Dict[str, int]] = {}
        for src in sorted(set(pred_source)):
            m = np.array([x == src for x in pred_source], bool)
            by_src[src] = {
                "n": int(m.sum()),
                "answer_correct": int(is_correct[m].sum()),
                "answer_error": int(m.sum() - is_correct[m].sum()),
            }
        out["pred_source_by_answer"] = by_src
    return out


def run(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "problem_ids" not in data.files:
        raise SystemExit("npz missing problem_ids; not a same-problem multisample feature file")

    problem_ids = data["problem_ids"].astype(int)
    result: Dict[str, Any] = {
        "meta": {
            "input": os.path.abspath(path),
            "basename": os.path.basename(path),
            "model_name": scalar_str(data, "model_name"),
            "dataset": scalar_str(data, "dataset"),
            "prompt_style": scalar_str(data, "prompt_style"),
            "step_split": scalar_str(data, "step_split"),
            "whitened": scalar_str(data, "whitened", "unknown"),
            "n_keys": int(len(data.files)),
        },
        "saved_fields": {key: array_shape_summary(data, key) for key in data.files},
        "payload": metric_payload_audit(data),
        "labels": label_audit(data),
        "text": text_audit(data),
        "policies": {},
    }

    n_steps = data["n_steps"].astype(int) if "n_steps" in data.files else None
    for pol in ("answer", "strict", "answer_format_ok"):
        try:
            y_err, mask, desc = policy_labels(data, pol)
        except Exception:
            continue
        sec = problem_layout(problem_ids, y_err, mask)
        sec["description"] = desc
        sec["n_error_samples"] = int(y_err[mask].sum())
        sec["n_correct_samples"] = int(mask.sum() - y_err[mask].sum())
        if n_steps is not None:
            sec["n_steps_error"] = qstats(n_steps[mask & (y_err == 1)])
            sec["n_steps_correct"] = qstats(n_steps[mask & (y_err == 0)])
            sec["n_steps_all"] = qstats(n_steps[mask])
        result["policies"][pol] = sec

    return result


def write_outputs(res: Mapping[str, Any], output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = f"multisample_data_audit_{os.path.splitext(str(res['meta']['basename']))[0]}"
    json_path = os.path.join(output_dir, stem + ".json")
    md_path = os.path.join(output_dir, stem + ".md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(finite_json(res), f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        meta = res["meta"]
        labels = res["labels"]
        f.write(f"# Multisample Data Audit: {meta['basename']}\n\n")
        f.write("## Result Analysis\n\n")
        f.write(
            f"- Samples are grouped by `problem_ids`; prompt style is `{meta['prompt_style']}`, "
            f"step split is `{meta['step_split']}`, model is `{meta['model_name']}`.\n"
        )
        f.write(
            f"- Label mass: answer-correct {labels.get('answer_correct', 'NA')}, "
            f"answer-error {labels.get('answer_error', 'NA')}, format-ok {labels.get('format_ok', 'NA')}, "
            f"format-fail {labels.get('format_fail', 'NA')}.\n"
        )
        if "format_fail_lenient_correct" in labels:
            f.write(
                f"- Format failures include {labels['format_fail_lenient_correct']} samples that lenient scoring "
                "would count as correct via fallback; treat these separately from reasoning errors.\n"
            )
        for name, sec in res["policies"].items():
            f.write(
                f"- `{name}`: {sec['n_correct_samples']} correct / {sec['n_error_samples']} error over "
                f"{sec['n_problems']} problems; {sec['contrastive_problems']} contrastive problems and "
                f"{sec['same_problem_error_correct_pairs']} same-problem error-correct pairs.\n"
            )
        f.write("\n## Follow-Up Research Direction\n\n")
        f.write("- Prioritize `answer_format_ok` for reasoning-failure analysis; analyze format failures as a separate failure mode.\n")
        f.write("- Use same-problem contrastive problems as the first gate for any detector that claims to learn trajectory failure rather than difficulty.\n")
        f.write("- If token clouds or raw step vectors are present, test boundary-state and within-step reconvergence signals before adding extra verifier forwards.\n")
        f.write("\n## Optimization Suggestions\n\n")
        f.write("- Report same-problem paired AUROC, contrastive-problem count, and length/format baselines with every new detector.\n")
        f.write("- Stratify by error type: wrong final number with marker, format failure, no-number/truncation, and lenient-only correctness.\n")
        f.write("- Avoid cumulative trajectory scores unless the statistic is explicitly time-debiased and evaluated on correct-chain false alarms.\n")
    return json_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="same-problem multisample .npz")
    ap.add_argument("--output_dir", default="outputs/multisample_data_audit")
    args = ap.parse_args()

    res = run(args.input)
    json_path, md_path = write_outputs(res, args.output_dir)
    meta = res["meta"]
    print(f"\n===== multisample data audit | {meta['basename']} =====")
    print(f"prompt={meta['prompt_style']} step_split={meta['step_split']} model={meta['model_name']}")
    lab = res["labels"]
    print(
        f"samples {lab.get('n_samples')} | answer err {lab.get('answer_error')} | "
        f"format fail {lab.get('format_fail', 'NA')}"
    )
    for name, sec in res["policies"].items():
        print(
            f"  {name:16s} problems {sec['n_problems']} contrastive {sec['contrastive_problems']} "
            f"samples {sec['n_samples']} err {sec['n_error_samples']} pairs {sec['same_problem_error_correct_pairs']}"
        )
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
