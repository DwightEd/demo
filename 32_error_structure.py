"""Step 32: do errors have TYPES? (structure among error chains, not endpoint distance)

For each problem, take its ERROR chains (wrong answer). Group them by the WRONG ANSWER they
produced. Ask: are error chains that reached the SAME wrong answer closer in activation space
than error chains that reached DIFFERENT wrong answers?
  same-answer distance  <  different-answer distance  =>  systematic error MODES
  (the model fails in a few repeatable ways, not randomly)
  same ~= different                                   =>  no error structure beyond the answer

This is NOT the trivial "error far from correct centroid". It is about whether the error
cloud has internal, answer-grounded structure.

Per qualifying problem (>=2 error chains sharing an answer AND >=2 distinct wrong answers):
  per-problem AUROC = P(distance of a different-answer pair > distance of a same-answer pair)
Aggregate: mean per-problem AUROC, fraction of problems with AUROC>0.5, sign test, mean
same/different distance ratio. Also a correct-chain control (shuffle answers) is reported.

Label: answer-based. Distances in space standardized by the correct-chain per-dim std.
"""
from __future__ import annotations
import argparse
import numpy as np
from math import comb
from itertools import combinations


def band_cols(L, band):
    if band == "deep": return np.arange(int(L*0.6), L)
    if band == "mid": return np.arange(int(L*0.3), int(L*0.7))
    return np.arange(L)


def late_vectors(VEC, cols, late_lo, d):
    N = len(VEC); X = np.full((N, d), np.nan)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            P = np.nanmean(V, axis=1)
        T = P.shape[0]; fr = (np.arange(T)/(T-1)) if T > 1 else np.array([0.0])
        m = fr >= late_lo
        if not m.any(): m = fr >= fr.max()
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(P[m], axis=0)
    return X


def signtest(k, n):
    if n == 0 or n > 1000: return float("nan")
    s = sum(comb(n, i) for i in range(k, n+1)) if k >= n/2 else sum(comb(n, i) for i in range(0, k+1))
    return min(1.0, 2*s/2**n)


def pair_auroc(same_d, diff_d):
    if not same_d or not diff_d: return None
    s = np.array(same_d); df = np.array(diff_d)
    allv = np.concatenate([s, df]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    # AUROC that a different-pair distance exceeds a same-pair distance
    return (ranks[len(s):].sum() - len(df)*(len(df)+1)/2) / (len(s)*len(df))


def problem_structure(idx, Xs, ans, rng):
    """idx: chain indices (same group, e.g. error chains of a problem). ans: their answers.
    Returns (per-problem AUROC same<diff, n_same, n_diff)."""
    same_d, diff_d = [], []
    for a, b in combinations(range(len(idx)), 2):
        dist = np.sqrt(((Xs[idx[a]] - Xs[idx[b]])**2).sum())
        (same_d if ans[a] == ans[b] else diff_d).append(dist)
    au = pair_auroc(same_d, diff_d)
    return au, len(same_d), len(diff_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--bands", default="mid,all")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--min_err", type=int, default=4, help="min error chains per problem")
    ap.add_argument("--format_ok", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    VEC = dd[f"sv_vec_{args.mode}"]; pid = dd["problem_ids"].astype(int)
    y = (dd["is_correct"].astype(int) == 0).astype(int)
    pred = dd["pred_answers"].astype(float) if "pred_answers" in dd.files else None
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(VEC), bool)
    if pred is None:
        raise SystemExit("npz lacks pred_answers -- need it to group errors by wrong answer.")
    N = len(VEC); L = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    keep = fmt if args.format_ok else np.ones(N, bool)
    ans_key = np.round(pred, 4)
    rng = np.random.default_rng(args.seed)

    for band in [b.strip() for b in args.bands.split(",")]:
        cols = band_cols(L, band)
        X = late_vectors(VEC, cols, args.late_lo, d)
        ok = keep & np.isfinite(X).all(1)
        sd = X[ok & (y == 0)].std(0) + 1e-6
        Xs = X / sd
        prob = {}
        for i in np.where(ok)[0]:
            prob.setdefault(int(pid[i]), []).append(i)

        aucs, ratios = [], []
        # control: shuffle answers within each problem's error set
        aucs_ctrl = []
        nq = 0
        for p, v in prob.items():
            ei = np.array([i for i in v if y[i] == 1])
            if len(ei) < args.min_err: continue
            ea = ans_key[ei]
            uniq, cnts = np.unique(ea, return_counts=True)
            if uniq.size < 2 or cnts.max() < 2:    # need >=2 distinct answers and a repeat
                continue
            nq += 1
            au, ns, nd = problem_structure(ei, Xs, ea, rng)
            if au is None: continue
            aucs.append(au)
            # control: permuted answers
            ea_sh = rng.permutation(ea)
            au_c, _, _ = problem_structure(ei, Xs, ea_sh, rng)
            if au_c is not None: aucs_ctrl.append(au_c)
        if not aucs:
            print(f"\n[{band}] 无合格题(需每题>= {args.min_err} 错误链且>=2种错误答案且有重复)"); continue
        aucs = np.array(aucs); k = int((aucs > 0.5).sum()); n = len(aucs)
        print(f"\n##### band={band}  子集={'仅格式合规' if args.format_ok else '全链'} #####")
        print(f"  合格题={nq}")
        print(f"  同答案错误链更近(AUROC>0.5)的题占比 = {k}/{n} = {k/n:.2f}  (符号检验 p={signtest(k,n):.1e})")
        print(f"  每题AUROC均值 = {aucs.mean():.3f}   对照(打乱答案)均值 = {np.mean(aucs_ctrl):.3f}")
        print(f"  解读: 显著>0.5 且 高于打乱对照 => 同一错误答案的链在激活上成簇 = 系统性错误模式")


if __name__ == "__main__":
    main()
