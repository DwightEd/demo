"""Convert .npz analysis outputs to human-readable JSON.

Why: the data/*.npz files saved by 11/12/19/20/22-25 use numpy's binary
format -- efficient for large hidden-state tensors but un-greppable. For the
small npz that hold AUROC tables / Wilcoxon stats / correlation matrices,
JSON is the right representation: plain text, diff-friendly, no Python
import required.

Usage
-----
  # one file
  python npz_to_json.py --input data/probe_all.npz --output data/probe_all.json

  # one file, full mode (dump every array as a list -- can be large)
  python npz_to_json.py --input data/probe_all.npz --mode full

  # batch (all data/*.npz next to it)
  python npz_to_json.py --batch 'data/*.npz' --out_dir json_summaries/

Modes
-----
  summary (default) : scalars and small arrays (<= --max_inline elements) are
                      dumped verbatim; large arrays get
                      {"_summary": {"shape": [...], "dtype": "...", "n": ...,
                                    "mean": ..., "std": ..., "min": ..., "max": ...,
                                    "head": [first 10], "tail": [last 10]}}
                      Object arrays of strings (e.g. responses, steps_text) are
                      dumped verbatim regardless of size, with a per-element
                      length cap to avoid 100 MB JSON files.
  full              : every array is dumped as a nested list. Object arrays of
                      large strings are NOT truncated. Use only for small npz.

The resulting JSON is sorted by key and pretty-printed (indent=2). Numpy
scalar types (np.int64, np.float32, ...) are coerced to Python builtins so
the JSON loads cleanly anywhere.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np


# Cap for inlining numeric arrays in summary mode (#elements, NOT bytes).
DEFAULT_MAX_INLINE = 64
# Cap for individual string elements in summary mode (chars). responses can be
# multi-KB; we keep the full text but warn so people can re-export with --mode full.
DEFAULT_STR_CAP = 4096


def _coerce(x: Any) -> Any:
    """Convert numpy scalar -> Python scalar; passthrough Python builtins."""
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        # NaN/Inf -> string so json.dumps does not crash (JSON has no NaN literal)
        v = float(x)
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return v
    return x


def _arr_summary(a: np.ndarray, max_inline: int) -> Any:
    """Either inline-list (small array) or a stats summary dict (large array)."""
    # try to summarise floats / ints
    if a.size == 0:
        return {"_summary": {"shape": list(a.shape), "dtype": str(a.dtype), "n": 0}}
    if a.size <= max_inline:
        return _to_jsonable(a.tolist())
    # numeric path
    if np.issubdtype(a.dtype, np.number):
        flat = a.astype(np.float64).ravel()
        finite = flat[np.isfinite(flat)]
        n_finite = int(finite.size)
        head = flat[:10].tolist()
        tail = flat[-10:].tolist()
        summary = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "n": int(a.size),
            "n_finite": n_finite,
            "n_nan": int(np.isnan(flat).sum()),
            "head": _to_jsonable(head),
            "tail": _to_jsonable(tail),
        }
        if n_finite > 0:
            summary.update({
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if n_finite > 1 else 0.0,
                "min": float(finite.min()),
                "max": float(finite.max()),
                "median": float(np.median(finite)),
                "q25": float(np.quantile(finite, 0.25)),
                "q75": float(np.quantile(finite, 0.75)),
            })
        return {"_summary": _to_jsonable(summary)}
    # non-numeric large array: dump the first chunk
    return {
        "_summary": {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "n": int(a.size),
            "head": _to_jsonable(a.ravel()[:10].tolist()),
        }
    }


def _truncate_str(s: str, cap: int) -> Any:
    if cap <= 0 or len(s) <= cap:
        return s
    return {"_truncated_str": True, "len": len(s),
            "head": s[:cap // 2], "tail": s[-cap // 2:]}


def _to_jsonable(obj: Any, *, str_cap: int = DEFAULT_STR_CAP) -> Any:
    """Recursively convert numpy / nested structures to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        if np.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return obj
    if isinstance(obj, str):
        return _truncate_str(obj, str_cap)
    if isinstance(obj, bytes):
        try:
            return _truncate_str(obj.decode("utf-8", errors="replace"), str_cap)
        except Exception:
            return repr(obj)
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return _coerce(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, str_cap=str_cap) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x, str_cap=str_cap) for x in obj]
    if isinstance(obj, np.ndarray):
        # 0-d arrays -> coerce to scalar
        if obj.ndim == 0:
            return _to_jsonable(obj.item(), str_cap=str_cap)
        # object arrays of strings (responses, steps_text, pred_source, ...)
        # -> nested lists (with per-string truncation)
        if obj.dtype == object:
            return [_to_jsonable(x, str_cap=str_cap) for x in obj.ravel().tolist()
                    ] if obj.ndim == 1 else _to_jsonable(obj.tolist(),
                                                        str_cap=str_cap)
        return _to_jsonable(obj.tolist(), str_cap=str_cap)
    # numpy generic scalars caught above; fall back to str
    return str(obj)


def npz_to_dict(path: str, mode: str = "summary",
                max_inline: int = DEFAULT_MAX_INLINE,
                str_cap: int = DEFAULT_STR_CAP) -> dict:
    """Load one .npz and return a JSON-ready dict."""
    d = np.load(path, allow_pickle=True)
    out: dict = {"_meta": {"file": os.path.basename(path),
                            "keys": sorted(d.files),
                            "mode": mode}}
    for k in sorted(d.files):
        arr = d[k]
        try:
            arr = np.asarray(arr)
        except Exception:
            out[k] = repr(d[k])
            continue
        if mode == "full":
            out[k] = _to_jsonable(arr, str_cap=0)        # 0 = no truncation
        else:
            if arr.dtype == object:
                # object arrays: always inline (with per-string cap)
                out[k] = _to_jsonable(arr, str_cap=str_cap)
            elif arr.ndim == 0 or arr.size <= max_inline:
                out[k] = _to_jsonable(arr, str_cap=str_cap)
            else:
                out[k] = _arr_summary(arr, max_inline)
    return out


def _default_output(path: str, out_dir: str | None = None) -> str:
    base = os.path.splitext(os.path.basename(path))[0] + ".json"
    return os.path.join(out_dir, base) if out_dir else \
        os.path.splitext(path)[0] + ".json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="one .npz file")
    ap.add_argument("--output", help="JSON output (default: alongside .npz)")
    ap.add_argument("--batch", help="glob over .npz, e.g. 'data/*.npz'")
    ap.add_argument("--out_dir", default=None,
                    help="output directory for batch mode")
    ap.add_argument("--mode", default="summary",
                    choices=["summary", "full"],
                    help="summary=stats for large arrays (default); "
                         "full=dump every array verbatim (large files)")
    ap.add_argument("--max_inline", type=int, default=DEFAULT_MAX_INLINE,
                    help="inline arrays with <= this many elements in summary "
                         f"mode (default {DEFAULT_MAX_INLINE})")
    ap.add_argument("--str_cap", type=int, default=DEFAULT_STR_CAP,
                    help="per-string char cap in summary mode "
                         f"(default {DEFAULT_STR_CAP}; 0=no cap)")
    ap.add_argument("--stdout", action="store_true",
                    help="print to stdout instead of writing a file")
    args = ap.parse_args()

    if not args.input and not args.batch:
        ap.error("provide --input or --batch")

    targets = []
    if args.input:
        targets.append((args.input,
                        args.output or _default_output(args.input,
                                                      args.out_dir)))
    if args.batch:
        files = sorted(glob.glob(args.batch))
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
        for f in files:
            targets.append((f, _default_output(f, args.out_dir)))

    for in_path, out_path in targets:
        try:
            d = npz_to_dict(in_path, mode=args.mode,
                            max_inline=args.max_inline,
                            str_cap=args.str_cap)
        except Exception as e:
            print(f"[skip] {in_path}: {e}")
            continue
        text = json.dumps(d, indent=2, ensure_ascii=False, sort_keys=True)
        if args.stdout:
            print(f"=== {in_path} ===")
            print(text)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  {in_path}  ->  {out_path}  ({len(text) // 1024} KB)")


if __name__ == "__main__":
    main()
