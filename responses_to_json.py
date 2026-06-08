"""Dump saved model responses from a multisample npz into readable JSON.

Each chain becomes one record with the raw response text, the parsed answer, the
gold answer, and a CATEGORY that makes the strict-vs-lenient / format confound visible:

  correct_clean          format_ok=1 and answer right   (formatted + correct)
  reasoning_error        format_ok=1 and answer wrong    (formatted + WRONG = genuine reasoning error)
  format_fail_but_correct format_ok=0 and answer right   (no '####' marker but answer matched)
  format_fail_wrong      format_ok=0 and answer wrong

Usage:
  python responses_to_json.py --input data/gsm8k_v2_5shot.npz
  python responses_to_json.py --input ... --filter format_fail_but_correct --n 20
  python responses_to_json.py --input ... --problem 7
"""
from __future__ import annotations
import argparse, json
import numpy as np


def categorize(fmt, ic, ics):
    if fmt and ics == 1:        return "correct_clean"
    if fmt and ics == 0:        return "reasoning_error"
    if (not fmt) and ic == 1:   return "format_fail_but_correct"
    return "format_fail_wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="default: <input>.responses.json")
    ap.add_argument("--filter", default="all",
                    help="all / correct_clean / reasoning_error / format_fail_but_correct / format_fail_wrong")
    ap.add_argument("--problem", type=int, default=None, help="only this problem_id")
    ap.add_argument("--n", type=int, default=0, help="cap number of records written (0 = all)")
    ap.add_argument("--max_chars", type=int, default=0, help="truncate response text (0 = full)")
    ap.add_argument("--print", type=int, default=3, help="print this many records to stdout")
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=True)
    if "responses" not in d.files:
        raise SystemExit("npz has no 'responses' field -- needs the v2 extraction (10 with text saved).")
    N = len(d["responses"])

    def col(name, default=None):
        return d[name] if name in d.files else ([default]*N)

    ids = col("ids"); pid = d["problem_ids"]; sidx = col("sample_idx")
    ic = d["is_correct"].astype(int)
    ics = d["is_correct_strict"].astype(int) if "is_correct_strict" in d.files else ic
    fmt = d["format_ok"].astype(int) if "format_ok" in d.files else np.ones(N, int)
    psrc = col("pred_source", ""); pred = col("pred_answers"); gold = col("gold_answers")
    nstep = col("n_steps"); resp = d["responses"]; steps = col("steps_text")

    # category counts (always printed)
    cats = [categorize(bool(fmt[i]), int(ic[i]), int(ics[i])) for i in range(N)]
    from collections import Counter
    cnt = Counter(cats)
    print(f"N={N}  category counts:")
    for k in ["correct_clean", "reasoning_error", "format_fail_but_correct", "format_fail_wrong"]:
        print(f"  {k:24s} {cnt.get(k,0)}")

    recs = []
    for i in range(N):
        if args.filter != "all" and cats[i] != args.filter:
            continue
        if args.problem is not None and int(pid[i]) != args.problem:
            continue
        txt = str(resp[i])
        if args.max_chars and len(txt) > args.max_chars:
            txt = txt[:args.max_chars] + " …[truncated]"
        recs.append({
            "id": str(ids[i]), "problem_id": int(pid[i]),
            "sample_idx": int(sidx[i]) if sidx[i] is not None else None,
            "category": cats[i],
            "format_ok": int(fmt[i]), "is_correct_lenient": int(ic[i]),
            "is_correct_strict": int(ics[i]), "pred_source": str(psrc[i]),
            "pred": (None if pred[i] is None or (isinstance(pred[i], float) and np.isnan(pred[i])) else float(pred[i])),
            "gold": (None if gold[i] is None else float(gold[i])),
            "n_steps": int(nstep[i]) if nstep[i] is not None else None,
            "response": txt,
            "steps_text": list(steps[i]) if steps[i] is not None else None,
        })
        if args.n and len(recs) >= args.n:
            break

    out = args.output or (args.input.rsplit(".", 1)[0] + ".responses.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(recs)} records (filter={args.filter}) -> {out}")

    for r in recs[:args.print]:
        print("\n" + "="*70)
        print(f"[{r['category']}] problem {r['problem_id']} sample {r['sample_idx']}  "
              f"pred={r['pred']} gold={r['gold']} format_ok={r['format_ok']} src={r['pred_source']}")
        print("-"*70)
        print(r["response"][:1200])


if __name__ == "__main__":
    main()
