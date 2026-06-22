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


def edis(H, w=8, tb=1.36, tr=1.33):
    """Entropy Dynamics Instability Score (Zhu et al. 2026): burst spikes (cumulative entropy growth
    over window w) + peak-valley rebounds (rise above running min), times (1+variance). Higher =
    less stable = worse. Computed on the per-TOKEN entropy trajectory. THE strong entropy baseline."""
    H = np.asarray(H, float)
    if len(H) < 3:
        return 0.0
    burst = sum(1 for t in range(len(H) - w) if H[t + w] - H[t] > tb) if len(H) > w else 0
    rebound = 0; rmin = H[0]
    for t in range(1, len(H)):
        if H[t] - rmin > tr:
            rebound += 1
        rmin = min(rmin, H[t])
    return 0.5 * (burst + rebound) * (1.0 + float(H.var()))


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
            res, en, sp, vecs, ent_tok, Rtraj = S.signals(prompt, sol)
            ent_bad = float(np.nanmean(en)) if len(en) and np.isfinite(en).any() else 0.0  # sequence entropy (static)
            edis_bad = edis(ent_tok)                                                        # EDIS = entropy DYNAMICS
            geom_bad = (1.0 - float(np.nanmin(res))) if len(res) and np.isfinite(res).any() else 0.0  # resultant (static)
            gdis_bad = edis(1.0 - Rtraj[np.isfinite(Rtraj)])           # GDIS = geometric DYNAMICS (EDIS formula on 1-R)
            chains.append(dict(ans=ans, ok=correct(ans, gold), ent_bad=ent_bad, edis_bad=edis_bad,
                               geom_bad=geom_bad, gdis_bad=gdis_bad))
        per_prob.append((_key(gold), chains))
        if (qi + 1) % 10 == 0:
            print(f"  [{qi+1}/{len(probs)}] sampled")

    # standardize each badness across ALL chains; fused = geometry on top of the STRONGEST entropy (EDIS)
    allc = [c for _, ch in per_prob for c in ch]
    def z(arrkey):
        v = np.array([c[arrkey] for c in allc]); m, s = v.mean(), v.std() + 1e-9
        for c in allc:
            c["z_" + arrkey] = (c[arrkey] - m) / s
    z("ent_bad"); z("edis_bad"); z("geom_bad"); z("gdis_bad")
    for c in allc:
        c["fused_eg"] = max(c["z_edis_bad"], c["z_gdis_bad"])   # EDIS + GDIS (both dynamics)

    methods = [("self-consistency", None), ("sequence-entropy (static)", "ent_bad"),
               ("resultant (static)", "geom_bad"), ("EDIS = entropy-dyn", "edis_bad"),
               ("GDIS = geom-dyn (ours)", "gdis_bad"), ("GDIS+EDIS", "fused_eg")]
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

    # diagnostic: of WRONG chains EDIS ranks stable (its blind spot), does GDIS (geometric dynamics) flag them?
    wrong = [c for c in allc if not c["ok"]]
    if wrong:
        et = np.median([c["edis_bad"] for c in allc]); gt = np.median([c["gdis_bad"] for c in allc])
        edis_blind = [c for c in wrong if c["edis_bad"] <= et]
        gdis_flag = [c for c in edis_blind if c["gdis_bad"] > gt]
        print(f"\nEDIS-blind wrong chains (no entropy instability) = {len(edis_blind)}/{len(wrong)}; "
              f"GDIS flags {len(gdis_flag)}/{max(len(edis_blind),1)} = {len(gdis_flag)/max(len(edis_blind),1):.2f}")
    print("\nread: HEADLINE = GDIS (geometric dynamics, ours) pass@1 > EDIS (entropy dynamics, SOTA) -- the "
          "SAME instability-counting lens applied to the resultant trajectory, which is CLEANER than the "
          "entropy trajectory. Sanity: EDIS > sequence-entropy (our EDIS impl is faithful) and resultant-static "
          "should already be decent. If GDIS > EDIS, geometric dynamics BEATS entropy dynamics (the strong "
          "top-venue claim, not a complement). If GDIS < EDIS, fall back to fusion / the static-vs-dynamic "
          "asymmetry. Sweep --drop/--samples; scale --n.")


if __name__ == "__main__":
    main()
