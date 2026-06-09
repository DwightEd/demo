"""Inspect a saved multisample npz: overview of all keys + one full data record.

  python inspect_npz.py --input data/gsm8k_v2_5shot_full.npz            # overview only
  python inspect_npz.py --input data/gsm8k_v2_5shot_full.npz --row 0    # + one chain
  python inspect_npz.py --input ... --row 0 --resp_chars 600            # longer response
"""
from __future__ import annotations
import argparse
import numpy as np


def short(a):
    """One-line description of an array value."""
    a = np.asarray(a)
    if a.dtype == object:
        if a.ndim == 0:
            return f"object scalar = {a.item()!r}"
        el = np.asarray(a[0]) if len(a) else None
        es = f"elem0 {el.shape} {el.dtype}" if el is not None and el.ndim else \
             (f"elem0={a[0]!r}" if len(a) else "empty")
        return f"object[{len(a)}]  ({es})"
    if a.ndim == 0:
        return f"scalar {a.dtype} = {a.item()!r}"
    if a.size <= 8:
        return f"{a.dtype}{list(a.shape)} = {a.tolist()}"
    return f"{a.dtype}{list(a.shape)}"


def fmt_vec(v, k=6):
    v = np.asarray(v, dtype=float).ravel()
    head = ", ".join(f"{x:.3f}" for x in v[:k])
    nan = int(np.sum(~np.isfinite(v)))
    return f"len={len(v)} nan={nan} [{head}{' ...' if len(v) > k else ''}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--resp_chars", type=int, default=400)
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=True)
    keys = sorted(d.files)

    print(f"=== {args.input} : {len(keys)} keys ===\n")
    meta, perchain = [], []
    for k in keys:
        a = np.asarray(d[k])
        (perchain if (a.dtype == object and a.ndim == 1) or
         (a.ndim >= 1 and a.shape[0] > 50) else meta).append(k)

    print("-- meta / scalars --")
    for k in meta:
        print(f"  {k:24s} {short(d[k])}")
    print("\n-- per-chain arrays (len = #chains) --")
    for k in perchain:
        print(f"  {k:24s} {short(d[k])}")

    if args.row is None:
        print("\n(pass --row N to dump one chain)")
        return

    i = args.row
    N = len(d["problem_ids"]) if "problem_ids" in d.files else len(d[perchain[0]])
    if not (0 <= i < N):
        raise SystemExit(f"row {i} out of range [0,{N})")

    print(f"\n========== ROW {i} (of {N}) ==========")
    lab = ["ids", "problem_id" if "problem_id" in d.files else "problem_ids", "sample_idx",
           "is_correct", "is_correct_strict", "format_ok", "pred_source",
           "pred_answers", "gold_answers", "n_steps"]
    print("\n[labels]")
    for k in lab:
        if k in d.files:
            print(f"  {k:18s} = {np.asarray(d[k])[i]!r}")

    if "responses" in d.files:
        r = str(d["responses"][i])
        print(f"\n[response]  ({len(r)} chars)\n  " + r[:args.resp_chars].replace("\n", "\n  ")
              + (" ..." if len(r) > args.resp_chars else ""))
    if "steps_text" in d.files:
        st = list(d["steps_text"][i])
        print(f"\n[steps_text]  {len(st)} steps")
        for j, s in enumerate(st[:6]):
            print(f"  step{j}: {str(s)[:90]}")
        if len(st) > 6:
            print(f"  ... (+{len(st)-6} more)")

    print("\n[representation per (step,layer)]")
    for k in [k for k in keys if k.startswith(("sv_pr_", "sv_ae_")) or k == "sv_D"]:
        el = np.asarray(d[k][i])
        print(f"  {k:18s} shape={list(el.shape)}  "
              f"mean={np.nanmean(el):.3f}  (T_steps={el.shape[0]})")
    if "sv_vec_step_exp" in d.files and d["sv_vec_step_exp"][i] is not None:
        el = np.asarray(d["sv_vec_step_exp"][i])
        print(f"  {'sv_vec_step_exp':18s} shape={list(el.shape)}  (T,L,dim)")

    print("\n[uncertainty]")
    for k in ["sv_out_entropy", "sv_out_committal"]:
        if k in d.files and d[k][i] is not None:
            print(f"  {k:18s} per-STEP  {fmt_vec(d[k][i])}")
    for k in ["sv_tok_entropy", "sv_tok_committal"]:
        if k in d.files and d[k][i] is not None:
            print(f"  {k:18s} per-TOKEN {fmt_vec(d[k][i])}")


if __name__ == "__main__":
    main()
