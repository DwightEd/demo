"""Step 28b: STRICTLY within-problem geometry of correct vs error chains.

28 measured effective dimension on a CROSS-problem pooled (per-problem-centered) cloud.
This script stays INSIDE each problem: for every problem that has both correct and error
chains, it compares the two groups' representations directly, then aggregates. No pooling
across problems, so no residual cross-problem confound.

Per qualifying problem p (>= min_c correct AND >= min_e error chains):
  spread_C = sqrt(mean ||x - centroid_C||^2) over correct chains   (within-problem scatter)
  spread_E = same over error chains
  centroid_sep = ||centroid_E - centroid_C||
  + leave-one-out: each correct chain's distance to the correct centroid (excluding itself)
    vs each error chain's distance to the correct centroid -> within-pair AUROC
    ("is an error chain farther from THIS problem's correct chains than a correct chain is")

Aggregated across problems:
  mean spread_C / spread_E, ratio, fraction(spread_E > spread_C) + sign test
  mean centroid_sep (relative to spreads)
  pooled within-pair AUROC of distance-to-correct-centroid

Label: answer-based (answer correct = correct). Also reported for format_ok subset.
Per-dim standardized by the correct chains' std so massive-activation dims don't dominate.
"""
from __future__ import annotations
import argparse
import numpy as np
from math import comb


def band_cols(L, band):
    if band == "all": return np.arange(L)
    if band == "deep": return np.arange(int(L*0.6), L)
    if band == "mid": return np.arange(int(L*0.3), int(L*0.7))
    return np.array([int(x) for x in band.split(",") if x.strip()])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--bands", default="mid,deep,all")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--min_c", type=int, default=2)
    ap.add_argument("--min_e", type=int, default=2)
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    VEC = dd[f"sv_vec_{args.mode}"]
    pid = dd["problem_ids"].astype(int)
    if "is_correct" not in dd.files:
        raise SystemExit("npz lacks is_correct.")
    y = (dd["is_correct"].astype(int) == 0).astype(int)             # 1 = wrong answer
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(VEC), bool)
    N = len(VEC); L = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]

    for sub_name, sub in [("全链(只看答案)", np.ones(N, bool)), ("仅格式合规", fmt)]:
        print(f"\n##### 子集 = {sub_name} #####")
        for band in [b.strip() for b in args.bands.split(",")]:
            cols = band_cols(L, band)
            X = late_vectors(VEC, cols, args.late_lo, d)
            ok = sub & np.isfinite(X).all(1)
            # standardize per-dim by correct-chain std (so distances are sensible)
            cmask = ok & (y == 0)
            sd = X[cmask].std(0) + 1e-6
            Xs = X / sd

            prob = {}
            for i in np.where(ok)[0]:
                prob.setdefault(int(pid[i]), []).append(i)

            sC, sE, sep, fracE = [], [], [], 0
            d_cor, d_err = [], []           # for within-pair: distance to correct centroid
            nq = 0
            for p, v in prob.items():
                v = np.array(v)
                ci = v[y[v] == 0]; ei = v[y[v] == 1]
                if len(ci) < args.min_c or len(ei) < args.min_e:
                    continue
                nq += 1
                cenC = Xs[ci].mean(0); cenE = Xs[ei].mean(0)
                spreadC = np.sqrt(np.mean(((Xs[ci]-cenC)**2).sum(1)))
                spreadE = np.sqrt(np.mean(((Xs[ei]-cenE)**2).sum(1)))
                sC.append(spreadC); sE.append(spreadE); fracE += int(spreadE > spreadC)
                sep.append(np.sqrt(((cenE-cenC)**2).sum()))
                # leave-one-out distance to correct centroid
                for a in ci:
                    cen_loo = Xs[ci[ci != a]].mean(0)
                    d_cor.append(np.sqrt(((Xs[a]-cen_loo)**2).sum()))
                for b in ei:
                    d_err.append(np.sqrt(((Xs[b]-cenC)**2).sum()))
            if nq == 0:
                print(f"  [{band:4s}] 无合格题(min_c={args.min_c},min_e={args.min_e})"); continue
            sC = np.array(sC); sE = np.array(sE); sep = np.array(sep)
            # within-pair AUROC: but d_cor/d_err are pooled, not paired-by-problem;
            # report a simple pooled AUROC of error>correct on distance-to-correct-centroid
            dc = np.array(d_cor); de = np.array(d_err)
            # Mann-Whitney style AUROC = P(de > dc)
            allv = np.concatenate([dc, de]); order = allv.argsort()
            ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(allv)+1)
            r_de = ranks[len(dc):].sum()
            auroc = (r_de - len(de)*(len(de)+1)/2) / (len(dc)*len(de))
            # sign test p for fracE
            k = fracE; n = len(sC)
            p_two = min(1.0, 2*sum(comb(n, i) for i in range(k, n+1))/2**n) if n <= 1000 else float("nan")
            print(f"  [{band:4s}] 合格题={nq}  题内散布: 正确均值={sC.mean():.2f} 错误均值={sE.mean():.2f} "
                  f"错误/正确={sE.mean()/sC.mean():.2f}")
            print(f"         错误更散的题占比={fracE}/{n}={fracE/n:.2f} (符号检验 p={p_two:.1e})  "
                  f"质心间距均值={sep.mean():.2f}")
            print(f"         到正确质心距离 错误>正确 的 AUROC(留一,池化)={auroc:.3f}")

    print("\n解读:")
    print("  错误/正确散布比 >1 且 占比显著 => 题内错误组确实更散(锚点的题内版)。")
    print("  质心间距 vs 散布 => 对错是否还有位移分离。")
    print("  AUROC ~ 单链在题内被'离正确质心远'检出的力(预计~0.6-0.66, 集合更散的单链投影上限)。")


if __name__ == "__main__":
    main()
