"""Best-of-N selection with a GEOMETRIC veto -- use the validated DETECTION strength (not repair).

Confident hallucinations have HIGH confidence (low entropy) -> confidence-based selectors
(DeepConf / TokUR) TRUST and keep them. But geometry flags them (directional collapse). So as a
SELECTOR over N sampled solutions, the geometric signal should remove the confident-wrong chains
that confidence-based selection keeps -> higher pass@1. This plays entirely to our strong, validated
detection (no fragile repair/steer).

Per problem: sample N solutions; for each chain score the WORST step (same-problem samples -> same
difficulty, so RAW values are comparable, no within-chain residual needed):
    ent_bad  = max step mean-entropy        (highest-entropy step; DeepConf-style uncertainty)
    geom_bad = 1 - min step resultant       (most directionally-diffuse step; our signal)
Selection methods at a MATCHED filter fraction f (drop the worst-f by each badness, then majority):
    self-consistency (no filter) | confidence-filter (DeepConf) | geometry-filter (ours) | fused
Metric = pass@1 of the selected answer. Headline: geometry-filter > confidence-filter, because
geometry vetoes the confident-wrong chains confidence trusts.

Reuses Solver/helpers from intervene_prototype. Needs a model + a jsonl of {question/problem, answer}.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from collections import Counter
from intervene_prototype import Solver, extract_answer, correct


def pick(chains, badness_key, drop_frac):
    """drop the worst drop_frac of chains by badness_key, majority-vote the rest. returns answer."""
    valid = [c for c in chains if c["ans"] is not None]
    if not valid:
        return None
    if badness_key is not None and 0 < drop_frac < 1 and len(valid) >= 3:
        valid = sorted(valid, key=lambda c: c[badness_key])
        keep = max(1, int(round(len(valid) * (1 - drop_frac))))
        valid = valid[:keep]                       # keep the LEAST bad
    votes = Counter(_key(c["ans"]) for c in valid)
    return votes.most_common(1)[0][0]


def _key(a):
    from intervene_prototype import _norm
    return _norm(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=100, help="number of problems")
    ap.add_argument("--samples", type=int, default=8, help="N solutions per problem")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--drop", type=float, default=0.5, help="filter fraction (worst-f dropped)")
    ap.add_argument("--data_jsonl", required=True)
    args = ap.parse_args()

    probs = []
    for line in open(args.data_jsonl, encoding="utf-8"):
        d = json.loads(line); q = d.get("question") or d.get("problem")
        probs.append((q, str(d["answer"]).strip()))
    probs = probs[:args.n]

    S = Solver(args.model, args.layer)
    per_prob = []                                  # (gold_key, [chains])
    for qi, (q, gold) in enumerate(probs):
        prompt = S.prompt(q); chains = []
        for _ in range(args.samples):
            sol = S.generate(prompt, temp=args.temp)
            ans = extract_answer(sol)
            res, en, sp, vecs = S.signals(prompt, sol)
            ent_bad = float(np.nanmax(en)) if len(en) and np.isfinite(en).any() else 0.0
            geom_bad = (1.0 - float(np.nanmin(res))) if len(res) and np.isfinite(res).any() else 0.0
            zb = None                              # filled after standardization
            chains.append(dict(ans=ans, ok=correct(ans, gold), ent_bad=ent_bad, geom_bad=geom_bad))
        per_prob.append((_key(gold), chains))
        if (qi + 1) % 10 == 0:
            print(f"  [{qi+1}/{len(probs)}] sampled")

    # standardize ent_bad/geom_bad across ALL chains for the fused score
    allc = [c for _, ch in per_prob for c in ch]
    def z(arrkey):
        v = np.array([c[arrkey] for c in allc]); m, s = v.mean(), v.std() + 1e-9
        for c in allc:
            c["z_" + arrkey] = (c[arrkey] - m) / s
    z("ent_bad"); z("geom_bad")
    for c in allc:
        c["fused_bad"] = max(c["z_ent_bad"], c["z_geom_bad"])

    methods = [("self-consistency", None), ("confidence (DeepConf)", "ent_bad"),
               ("geometry (ours)", "geom_bad"), ("fused", "fused_bad")]
    acc = {nm: 0 for nm, _ in methods}; oracle = 0
    for gold, chains in per_prob:
        if any(c["ok"] for c in chains):
            oracle += 1                            # pass@N ceiling: correct answer present among samples
        for nm, bk in methods:
            sel = pick(chains, bk, args.drop)
            acc[nm] += int(sel is not None and sel == gold)
    npb = max(len(per_prob), 1)

    print(f"\nmodel {args.model} | layer {args.layer} | problems {len(per_prob)} | "
          f"N={args.samples} | temp {args.temp} | drop {args.drop}")
    print(f"oracle pass@{args.samples} (correct present): {oracle/npb:.3f}")
    print(f"\n{'selection method':24s} {'pass@1':>8s}")
    for nm, _ in methods:
        print(f"  {nm:22s} {acc[nm]/npb:>8.3f}")

    # blind-spot diagnostic: of WRONG chains that confidence TRUSTS (low ent_bad), does geometry flag?
    wrong = [c for c in allc if not c["ok"]]
    if wrong:
        et = np.median([c["ent_bad"] for c in allc]); gt = np.median([c["geom_bad"] for c in allc])
        conf_wrong = [c for c in wrong if c["ent_bad"] <= et]          # wrong but low-entropy (confident)
        geo_flag = [c for c in conf_wrong if c["geom_bad"] > gt]       # geometry flags them
        print(f"\nblind-spot: confident-wrong chains (wrong & low entropy) = {len(conf_wrong)}/{len(wrong)}; "
              f"geometry flags {len(geo_flag)}/{max(len(conf_wrong),1)} = {len(geo_flag)/max(len(conf_wrong),1):.2f}")
    print("\nread: HEADLINE = geometry (ours) pass@1 > confidence (DeepConf) pass@1, because geometry vetoes "
          "the confident-wrong chains confidence keeps. fused should be >= both. self-consistency is the no-"
          "selection baseline; oracle is the ceiling. The blind-spot line shows geometry catching confident-"
          "wrong chains the entropy/confidence selector trusts -- the mechanism of the win. Sweep --drop / "
          "--samples; scale --n.")


if __name__ == "__main__":
    main()
