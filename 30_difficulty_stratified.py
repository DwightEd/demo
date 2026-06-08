"""Step 30: does error-detectability rise with problem DIFFICULTY?

Hypothesis (user): on easy problems the model is confident and errors are careless slips
that look just like correct chains (low detectability); on hard problems the model 'knows
it lacks the knowledge' so errors carry a clearer geometric signature. Prediction:
within-problem error detection AUROC should INCREASE with problem difficulty.

Test: stratify problems by difficulty, then within each stratum pool the within-problem
(error vs correct) comparisons of a clean geometric signal and report AUROC + effective-dim.

Signal per chain = distance to its problem's correct centroid (leave-one-out for correct
chains), standardized by the global correct-chain std. Difficulty proxies:
  failrate : fraction of the problem's chains that are wrong (answer-based)
  nsteps   : mean reasoning length over the problem's chains (a fail-rate-independent proxy)
Label: answer-based.  Subsets: all chains and format_ok only.
"""
from __future__ import annotations
import argparse
import numpy as np


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


def eff_dim(M):
    if len(M) < 4: return float("nan")
    Mc = M - M.mean(0); ev = np.linalg.eigvalsh(Mc @ Mc.T); ev = ev[ev > 1e-9]
    return float(ev.sum()**2 / (ev**2).sum()) if ev.size else float("nan")


def auroc_pairs(d_cor, d_err):
    if not d_cor or not d_err: return float("nan")
    dc = np.array(d_cor); de = np.array(d_err)
    allv = np.concatenate([dc, de]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    a = (ranks[len(dc):].sum() - len(de)*(len(de)+1)/2) / (len(dc)*len(de))
    return max(a, 1-a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--band", default="mid")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--proxy", default="failrate", choices=["failrate", "nsteps"])
    ap.add_argument("--nbins", type=int, default=4)
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    VEC = dd[f"sv_vec_{args.mode}"]; pid = dd["problem_ids"].astype(int)
    y = (dd["is_correct"].astype(int) == 0).astype(int)
    nsteps = dd["n_steps"].astype(float) if "n_steps" in dd.files else np.zeros(len(VEC))
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(VEC), bool)
    N = len(VEC); L = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_cols(L, args.band)
    X = late_vectors(VEC, cols, args.late_lo, d)

    for sub_name, sub in [("全链", np.ones(N, bool)), ("仅格式合规", fmt)]:
        ok = sub & np.isfinite(X).all(1)
        sd = X[ok & (y == 0)].std(0) + 1e-6
        Xs = X / sd
        prob = {}
        for i in np.where(ok)[0]:
            prob.setdefault(int(pid[i]), []).append(i)
        # per-problem difficulty + chain indices, keep contrastive problems
        items = []
        for p, v in prob.items():
            v = np.array(v); ci = v[y[v] == 0]; ei = v[y[v] == 1]
            if len(ci) < 2 or len(ei) < 1: continue
            diff = (len(ei)/len(v)) if args.proxy == "failrate" else float(np.mean(nsteps[v]))
            items.append((p, ci, ei, diff))
        if len(items) < args.nbins * 2:
            print(f"\n##### {sub_name} #####  合格题太少({len(items)}),跳过"); continue
        diffs = np.array([it[3] for it in items])
        edges = np.quantile(diffs, np.linspace(0, 1, args.nbins+1))
        print(f"\n##### 子集={sub_name}  band={args.band}  难度代理={args.proxy} #####")
        print(f"  {'难度区间':>16s} {'题数':>5s} {'题内AUROC':>9s} {'有效维 正/错':>14s}")
        for b in range(args.nbins):
            lo, hi = edges[b], edges[b+1]
            sel = [it for it in items if (lo <= it[3] <= hi if b == args.nbins-1 else lo <= it[3] < hi)]
            if not sel: continue
            d_cor, d_err = [], []
            cor_all, err_all = [], []
            for p, ci, ei, _ in sel:
                cenC = Xs[ci].mean(0)
                for a in ci:
                    cl = Xs[ci[ci != a]].mean(0); d_cor.append(np.sqrt(((Xs[a]-cl)**2).sum()))
                for e in ei:
                    d_err.append(np.sqrt(((Xs[e]-cenC)**2).sum()))
                # per-problem-centered pooled for eff dim
                cor_all.append(Xs[ci]-Xs[ci].mean(0)); err_all.append(Xs[ei]-Xs[ei].mean(0))
            au = auroc_pairs(d_cor, d_err)
            Dc = eff_dim(np.vstack(cor_all)); De = eff_dim(np.vstack(err_all))
            print(f"  [{lo:6.2f},{hi:6.2f}] {len(sel):5d} {au:9.3f}   {Dc:5.1f}/{De:5.1f}")

    print("\n解读:若题内AUROC随难度上升 => 用户假设成立(难题错误更可检测);"
          " 若平平 => GSM8K 内难度范围太窄, 需 MATH 验证。")


if __name__ == "__main__":
    main()
