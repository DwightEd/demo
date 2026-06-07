"""Convert a multisample npz (from 10_sample_and_extract.py) into a
per-chain JSON record that is easy to read / grep / hand-audit.

The general-purpose npz_to_json.py dumps each array as one key, which is
correct for analysis tables (probe_*.npz, ens_*.npz, ...) but unhelpful
for a multisample file: to inspect chain p17_s3 you would have to cross-
reference 8 separate arrays at index 73 by hand. This script flips the
layout: one record per chain, with all the chain-level fields glued
together.

Output schema (per chain):
    {
      "id"               : "p17_s3",
      "problem_id"       : 17,
      "sample_idx"       : 3,
      "gold"             : 18.0,
      "pred"             : 18.0,
      "is_correct"       : 1,                 # lenient (v1 compat)
      "is_correct_strict": 1,                 # if present in npz
      "format_ok"        : 1,                 # if present
      "pred_source"      : "marker",          # if present
      "n_steps"          : 5,
      "response"         : "...",             # full generation
      "steps"            : ["step 1 text", ...]
    }

Hidden-state arrays (sv_pr_*, sv_ae_*, sv_vec_*, sv_clouds) are NOT
included by default -- they belong to analysis npz, not to a text audit
file. Pass --include_signal_summary to add per-chain summary stats
(mean PR per band etc.) without the raw arrays.

Filters:
    --only_errors       only chains with is_correct == 0
    --only_format_fail  only chains with format_ok == 0  (needs strict labels)
    --only_strict_disagree    only chains where lenient and strict labels
                              differ (= isolates the last-number fallback mass)
    --problems P0,P1    keep only these problem_ids
    --limit N           take first N matching records

Usage:
    # full dump
    python multisample_to_json.py \
        --input data/gsm8k_v2_custom.npz \
        --output data/gsm8k_v2_custom.records.json

    # only errors, first 50, with per-chain signal means
    python multisample_to_json.py \
        --input data/gsm8k_v2_custom.npz \
        --only_errors --limit 50 --include_signal_summary \
        --output data/gsm8k_v2_custom.errors50.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _jsonify_scalar(x):
    """Coerce numpy scalar / NaN / Inf to JSON-safe."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return v
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return repr(x)
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return _jsonify_scalar(x.item())
    return str(x)


def _get(data, key, default=None):
    """Return the npz field as a python object, or default if missing."""
    return data[key] if key in data.files else default


def _safe_str_list(obj):
    """obj is a numpy object-array element that should be a list of strings."""
    if obj is None:
        return None
    arr = np.asarray(obj, dtype=object).ravel().tolist()
    return [str(x) for x in arr]


# ----------------------------------------------------------------------------
# Band-level signal summary (optional)
# ----------------------------------------------------------------------------

def _band_cols(L_sub, band):
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    return np.arange(L_sub)


def _signal_summary(data, i, n_steps_i):
    """For chain index i, compute compact summary stats for PR/AE per band.

    Returns dict like:
        {"pr_mid_late_mean": 0.7124, "ae_mid_late_mean": 0.6987, ...}
    Skipped silently if the npz has no PR/AE arrays.
    """
    out = {}
    PR = _get(data, "sv_pr_step_exp")
    AE = _get(data, "sv_ae_step_exp")
    OE = _get(data, "sv_out_entropy")
    if PR is None or AE is None:
        return out
    Pm = np.asarray(PR[i], dtype=np.float64)        # (T, L_sub)
    Am = np.asarray(AE[i], dtype=np.float64)
    if Pm.ndim != 2 or Pm.shape[0] == 0:
        return out
    L_sub = Pm.shape[1]
    T = Pm.shape[0]
    fr = (np.arange(T) / max(1, T - 1))
    late = fr >= 0.6
    if not late.any():
        late = fr >= fr.max()
    for band in ("mid", "deep", "all"):
        cols = _band_cols(L_sub, band)
        with np.errstate(invalid="ignore"):
            out[f"pr_{band}_late_mean"] = float(np.nanmean(np.nanmean(Pm[late][:, cols], axis=1)))
            out[f"ae_{band}_late_mean"] = float(np.nanmean(np.nanmean(Am[late][:, cols], axis=1)))
            out[f"pr_{band}_full_mean"] = float(np.nanmean(np.nanmean(Pm[:, cols], axis=1)))
            out[f"ae_{band}_full_mean"] = float(np.nanmean(np.nanmean(Am[:, cols], axis=1)))
    if OE is not None:
        oe = np.asarray(OE[i], dtype=np.float64)
        oe = oe[np.isfinite(oe)]
        if oe.size:
            out["out_entropy_mean"] = float(oe.mean())
            out["out_entropy_max"] = float(oe.max())
    # JSON-clean NaNs
    return {k: _jsonify_scalar(v) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="multisample npz path")
    ap.add_argument("--output", default=None,
                    help="output JSON (default: alongside input with .records.json)")
    ap.add_argument("--only_errors", action="store_true",
                    help="keep only chains with is_correct == 0")
    ap.add_argument("--only_format_fail", action="store_true",
                    help="keep only chains with format_ok == 0 (needs new 10)")
    ap.add_argument("--only_strict_disagree", action="store_true",
                    help="keep only chains where is_correct != is_correct_strict "
                         "(isolates the last-number fallback mass)")
    ap.add_argument("--problems", default=None,
                    help="comma-separated problem_ids to keep, e.g. '3,17,42'")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap at first N matching records")
    ap.add_argument("--include_signal_summary", action="store_true",
                    help="add per-chain PR/AE mean per band (no raw vectors)")
    ap.add_argument("--max_steps_to_print", type=int, default=0,
                    help="0 = include all step strings; >0 = truncate to first N "
                         "step strings (response itself is always kept full)")
    args = ap.parse_args()

    out_path = args.output or (os.path.splitext(args.input)[0] + ".records.json")
    data = np.load(args.input, allow_pickle=True)

    # core arrays (always present in any 10-output npz)
    problem_ids = data["problem_ids"].astype(int)
    sample_idx = data["sample_idx"].astype(int)
    is_correct = data["is_correct"].astype(int)
    n_steps = data["n_steps"].astype(int)
    pred_answers = data["pred_answers"].astype(float) if "pred_answers" in data.files else None
    gold_answers = data["gold_answers"].astype(float) if "gold_answers" in data.files else None
    ids_field = _get(data, "ids")

    N = problem_ids.size

    # v2 fields (may be missing on v1 data)
    has_strict = "is_correct_strict" in data.files
    has_format = "format_ok" in data.files
    has_pred_src = "pred_source" in data.files
    has_resp = "responses" in data.files
    has_steps_text = "steps_text" in data.files

    is_correct_strict = data["is_correct_strict"].astype(int) if has_strict else None
    format_ok = data["format_ok"].astype(int) if has_format else None
    pred_source = data["pred_source"] if has_pred_src else None
    responses = data["responses"] if has_resp else None
    steps_text = data["steps_text"] if has_steps_text else None

    # filter mask
    keep = np.ones(N, dtype=bool)
    if args.only_errors:
        keep &= (is_correct == 0)
    if args.only_format_fail:
        if not has_format:
            raise SystemExit("--only_format_fail needs format_ok field (v2 npz)")
        keep &= (format_ok == 0)
    if args.only_strict_disagree:
        if not has_strict:
            raise SystemExit("--only_strict_disagree needs is_correct_strict (v2 npz)")
        keep &= (is_correct != is_correct_strict)
    if args.problems:
        want = set(int(x) for x in args.problems.split(",") if x.strip())
        keep &= np.isin(problem_ids, list(want))

    kept_idx = np.where(keep)[0]
    if args.limit is not None and args.limit > 0:
        kept_idx = kept_idx[:args.limit]

    print(f"Loaded {N} chains from {args.input}")
    print(f"  v2 fields present: strict={has_strict} format_ok={has_format} "
          f"pred_source={has_pred_src} responses={has_resp} steps_text={has_steps_text}")
    print(f"  After filters: {len(kept_idx)} / {N} chains kept "
          f"(filters: errors={args.only_errors} format_fail={args.only_format_fail} "
          f"strict_disagree={args.only_strict_disagree}"
          f"{' problems=' + args.problems if args.problems else ''}"
          f"{' limit=' + str(args.limit) if args.limit else ''})")

    # build records
    chains = []
    for i in kept_idx:
        rec = {
            "id": str(ids_field[i]) if ids_field is not None else f"p{int(problem_ids[i])}_s{int(sample_idx[i])}",
            "problem_id": int(problem_ids[i]),
            "sample_idx": int(sample_idx[i]),
            "n_steps": int(n_steps[i]),
            "is_correct": int(is_correct[i]),
        }
        if pred_answers is not None:
            rec["pred"] = _jsonify_scalar(pred_answers[i])
        if gold_answers is not None:
            rec["gold"] = _jsonify_scalar(gold_answers[i])
        if has_strict:
            rec["is_correct_strict"] = int(is_correct_strict[i])
        if has_format:
            rec["format_ok"] = int(format_ok[i])
        if has_pred_src:
            rec["pred_source"] = str(pred_source[i])
        if has_resp:
            rec["response"] = str(responses[i])
        if has_steps_text:
            steps_list = _safe_str_list(steps_text[i])
            if steps_list is not None:
                if args.max_steps_to_print > 0 and len(steps_list) > args.max_steps_to_print:
                    rec["steps"] = steps_list[:args.max_steps_to_print]
                    rec["steps_truncated"] = True
                    rec["steps_total"] = len(steps_list)
                else:
                    rec["steps"] = steps_list
        if args.include_signal_summary:
            sigsum = _signal_summary(data, int(i), int(n_steps[i]))
            if sigsum:
                rec["signal_summary"] = sigsum
        chains.append(rec)

    # contrastive bookkeeping for the meta block
    by_prob = {}
    for r in chains:
        by_prob.setdefault(r["problem_id"], []).append(r["is_correct"])
    n_contrastive = sum(1 for v in by_prob.values()
                        if any(v) and not all(v))

    out = {
        "_meta": {
            "input_file": os.path.basename(args.input),
            "n_chains_in_input": int(N),
            "n_chains_kept": len(chains),
            "n_problems_kept": len(by_prob),
            "n_contrastive_problems_kept": int(n_contrastive),
            "model": str(_get(data, "model_name", "?")),
            "dataset": str(_get(data, "dataset", "?")),
            "prompt_style": str(_get(data, "prompt_style", "?")),
            "step_split": str(_get(data, "step_split", "?")),
            "whitened": bool(_get(data, "whitened", np.array(False))),
            "v2_fields_present": {
                "is_correct_strict": has_strict,
                "format_ok": has_format,
                "pred_source": has_pred_src,
                "responses": has_resp,
                "steps_text": has_steps_text,
            },
            "filters_applied": {
                "only_errors": bool(args.only_errors),
                "only_format_fail": bool(args.only_format_fail),
                "only_strict_disagree": bool(args.only_strict_disagree),
                "problems": args.problems,
                "limit": args.limit,
            },
            "include_signal_summary": bool(args.include_signal_summary),
        },
        "chains": chains,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"Saved -> {out_path}  ({size_kb} KB)")
    if not has_resp:
        print("  NOTE: this npz does NOT contain 'responses' (v1 data).  "
              "Re-extract with the new 10_sample_and_extract.py to get the "
              "raw generated text per chain.")


if __name__ == "__main__":
    main()
