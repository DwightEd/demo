"""Step 31: per-PROBLEM record of correct vs error geometry (no averaging across problems).

For every problem that has both correct and error chains, record SEPARATELY the correct
group and the error group on:
  setdim   : effective dimension of that group's chain cloud (>=min_grp chains needed)
  pr       : mean participation ratio of the chains (per-vector "active dims")
  norm     : mean activation magnitude (||late-window band-mean vector||)
for mid and deep bands. Writes one CSV row per problem so you can eyeball whether error is
UNIVERSALLY higher (not just higher on average), plus a sign-test summary.

Label: answer-based (answer correct = correct). Optional --format_ok.
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
from math import comb


def band_cols(L, band):
    if band == "deep": return np.arange(int(L*0.6), L)
    if band == "mid": return np.arange(int(L*0.3), int(L*0.7))
    return np.arange(L)


def chain_pr(PR_i, cols, n, late_lo=0.6):
    M = np.asarray(PR_i)[:, cols]            # (T, |cols|)
    T = M.shape[0]; fr = (np.arange(T)/(T-1)) if T > 1 else np.array([0.0])
    m = fr >= late_lo
    if not m.any(): m = fr >= fr.max()
    return float(np.nanmean(M[m]))


def chain_vec(VEC_i, cols, late_lo=0.6):
    V = np.asarray(VEC_i, dtype=np.float64)[:, cols, :]
    with np.errstate(invalid="ignore"):
        P = np.nanmean(V, axis=1)
    T = P.shape[0]; fr = (np.arange(T)/(T-1)) if T > 1 else np.array([0.0])
    m = fr >= late_lo
    if not m.any(): m = fr >= fr.max()
    with np.errstate(invalid="ignore"):
        return np.nanmean(P[m], axis=0)


def eff_dim(M):
    if len(M) < 4: return None
    Mc = M - M.mean(0); ev = np.linalg.eigvalsh(Mc @ Mc.T); ev = ev[ev > 1e-9]
    return float(ev.sum()**2/(ev**2).sum()) if ev.size else None


def signtest(k, n):
    if n == 0 or n > 1000: return float("nan")
    return min(1.0, 2*sum(comb(n, i) for i in range(k, n+1))/2**n) if k >= n/2 else \
           min(1.0, 2*sum(comb(n, i) for i in range(0, k+1))/2**n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--min_grp", type=int, default=4, help="min chains per group for effective dim")
    ap.add_argument("--format_ok", action="store_true")
    ap.add_argument("--out", default="results_answer/per_problem.csv")
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    VEC = dd[f"sv_vec_{args.mode}"]; PR = dd["sv_pr_step_exp"]
    pid = dd["problem_ids"].astype(int)
    y = (dd["is_correct"].astype(int) == 0).astype(int)            # 1 = wrong
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(VEC), bool)
    nstep = dd["n_steps"].astype(int)
    N = len(VEC); L = np.asarray(VEC[0]).shape[1]
    keep = fmt if args.format_ok else np.ones(N, bool)
    cm = {b: band_cols(L, b) for b in ("mid", "deep")}

    # per-chain scalars
    pr = {b: np.full(N, np.nan) for b in cm}
    vec = {b: [None]*N for b in cm}
    for i in range(N):
        if not keep[i]: continue
        for b, cols in cm.items():
            pr[b][i] = chain_pr(PR[i], cols, nstep[i], args.late_lo)
            vec[b][i] = chain_vec(VEC[i], cols, args.late_lo)

    prob = {}
    for i in np.where(keep)[0]:
        prob.setdefault(int(pid[i]), []).append(i)

    rows = []
    for p, v in prob.items():
        v = np.array(v); ci = v[y[v] == 0]; ei = v[y[v] == 1]
        if len(ci) < 1 or len(ei) < 1: continue
        r = {"problem_id": p, "n_cor": len(ci), "n_err": len(ei),
             "failrate": round(len(ei)/len(v), 3)}
        for b in cm:
            r[f"pr_{b}_cor"] = round(float(np.nanmean(pr[b][ci])), 3)
            r[f"pr_{b}_err"] = round(float(np.nanmean(pr[b][ei])), 3)
            r[f"norm_{b}_cor"] = round(float(np.mean([np.linalg.norm(vec[b][i]) for i in ci])), 2)
            r[f"norm_{b}_err"] = round(float(np.mean([np.linalg.norm(vec[b][i]) for i in ei])), 2)
            Dc = eff_dim(np.array([vec[b][i] for i in ci])) if len(ci) >= args.min_grp else None
            De = eff_dim(np.array([vec[b][i] for i in ei])) if len(ei) >= args.min_grp else None
            r[f"setdim_{b}_cor"] = round(Dc, 2) if Dc else ""
            r[f"setdim_{b}_err"] = round(De, 2) if De else ""
        rows.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} per-problem rows -> {args.out}  (subset={'format_ok' if args.format_ok else 'all'})")

    # universality summary: fraction of problems where error > correct
    print(f"\n{'指标':16s} {'题数':>5s} {'错>对题占比':>11s} {'符号p':>9s} {'对均值':>8s} {'错均值':>8s}")
    def summ(name, ckey, ekey, need_both=True):
        pairs = [(r[ckey], r[ekey]) for r in rows if r[ckey] != "" and r[ekey] != ""]
        if not pairs: print(f"{name:16s}  无足够数据"); return
        c = np.array([a for a, _ in pairs]); e = np.array([b for _, b in pairs])
        k = int((e > c).sum()); n = len(pairs)
        print(f"{name:16s} {n:5d} {k/n:11.2f} {signtest(k,n):9.1e} {c.mean():8.2f} {e.mean():8.2f}")
    for b in cm:
        summ(f"参与度 {b}", f"pr_{b}_cor", f"pr_{b}_err")
        summ(f"激活幅度 {b}", f"norm_{b}_cor", f"norm_{b}_err")
        summ(f"集合维数 {b}", f"setdim_{b}_cor", f"setdim_{b}_err")

    print("\n前 8 题示例:")
    cols_show = ["problem_id", "n_cor", "n_err", "failrate",
                 "pr_mid_cor", "pr_mid_err", "norm_mid_cor", "norm_mid_err",
                 "setdim_mid_cor", "setdim_mid_err"]
    print("  " + " ".join(f"{c}" for c in cols_show))
    for r in rows[:8]:
        print("  " + " ".join(str(r[c]) for c in cols_show))


if __name__ == "__main__":
    main()
