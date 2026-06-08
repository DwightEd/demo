"""Step 38: representation channel (D/PR/AE) vs uncertainty channel (UD/UC), unified.

On the SAME chains, build per-step series for both channels and ask, rigorously:
  - STEP level: within a chain, when uncertainty spikes at step t, does the
    representation (spectral eff-rank D / participation PR / activation-entropy AE)
    also move at step t?  (pooled over all (chain,step), AND mean of per-chain
    step-series correlations = the within-chain co-movement, problem/chain partialled out)
  - CHAIN level: do the chain summaries correlate?  Summaries include the DYNAMIC
    curve-fit features (slope, r2) -- the same shape features the uncertainty paper
    (2605.07776) uses for its late-stage signal -- not just the static mean.
  - DETECTION: representation-only / uncertainty-only / fusion within-problem AUROC
    -> are the two channels redundant or complementary for predicting error?

Representation per step (deep/mid layer band, mean over layers):
  D  = sv_D            spectral effective rank of the step token cloud
  PR = sv_pr_<mode>    participation ratio of the step vector
  AE = sv_ae_<mode>    activation entropy of the step vector
Uncertainty per step:
  UD = sv_out_entropy  distributional aleatoric (next-token entropy at step boundary)
  UC = sv_out_committal committal aleatoric p(1-p) of the realized boundary token
(UE / epistemic = gradient norm is NOT computed; this is a 5-signal, 2-channel test.)

Label: answer-based. Within-problem = difficulty-controlled. Writes a JSON.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def band_cols(L, band):
    if band == "deep": return np.arange(int(L*0.6), L)
    if band == "mid":  return np.arange(int(L*0.3), int(L*0.7))
    return np.arange(L)


def steptrace(M, cols):
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2: return None
    cc = cols[cols < M.shape[1]]
    with np.errstate(invalid="ignore"):
        return np.nanmean(M[:, cc], axis=1)


def profile3(e):
    """[mean, slope, r2] of a series over normalized position (curve-fit dynamics)."""
    e = np.asarray(e, dtype=np.float64); e = e[np.isfinite(e)]
    T = len(e)
    if T < 4: return None
    pos = np.linspace(0, 1, T); A = np.vstack([pos, np.ones(T)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    pred = A @ coef; ssr = ((e-pred)**2).sum(); sst = ((e-e.mean())**2).sum()+1e-12
    return np.array([e.mean(), coef[0], 1 - ssr/sst])


def sp(a, b):
    a = np.asarray(a); b = np.asarray(b); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12: return np.nan
    return float(spearmanr(a[m], b[m]).correlation)


def within_pair_auroc(groups, f, y):
    conc = 0.0; npair = 0
    for idx in groups:
        inc = [f[i] for i in idx if y[i] == 1 and np.isfinite(f[i])]
        cor = [f[i] for i in idx if y[i] == 0 and np.isfinite(f[i])]
        if not inc or not cor: continue
        for u in inc:
            for v in cor:
                conc += 1.0 if u > v else (0.5 if u == v else 0.0)
        npair += len(inc)*len(cor)
    a = conc/npair if npair else np.nan
    return max(a, 1-a) if np.isfinite(a) else np.nan


def gfolds(g, k, s):
    u = np.unique(g); r = np.random.default_rng(s); r.shuffle(u)
    fo = {int(x): i % k for i, x in enumerate(u)}; f = np.array([fo[int(x)] for x in g])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def channel_auroc(F, y, pp, groups, k, seeds):
    vals = []
    for s in range(seeds):
        oof = np.full(len(y), np.nan)
        for tr, te in gfolds(pp, k, s):
            if len(np.unique(y[tr])) < 2: continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced"))
            clf.fit(F[tr], y[tr]); oof[te] = clf.predict_proba(F[te])[:, 1]
        vals.append(within_pair_auroc(groups, oof, y))
    return round(float(np.nanmean(vals)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--band", default="deep", choices=["deep", "mid", "all"])
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--format_ok", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=True)
    REP = {}
    if "sv_D" in d.files: REP["D"] = d["sv_D"]
    if f"sv_pr_{args.mode}" in d.files: REP["PR"] = d[f"sv_pr_{args.mode}"]
    if f"sv_ae_{args.mode}" in d.files: REP["AE"] = d[f"sv_ae_{args.mode}"]
    UNC = {}
    if "sv_out_entropy" in d.files: UNC["UD"] = d["sv_out_entropy"]
    if "sv_out_committal" in d.files: UNC["UC"] = d["sv_out_committal"]
    if not REP or not UNC:
        raise SystemExit(f"need representation + uncertainty per-step arrays. files={sorted(d.files)}")
    miss = [k for k in ("D",) if k not in REP] + [k for k in ("UC",) if k not in UNC]
    if miss: print(f"  note: missing {miss} (re-run 10b after the 01 update to add them)")

    pid = d["problem_ids"].astype(int)
    y = (d["is_correct"].astype(int) == 0).astype(int)
    fmt = d["format_ok"].astype(bool) if "format_ok" in d.files else np.ones(len(pid), bool)
    keep = fmt if args.format_ok else np.ones(len(pid), bool)
    N = len(pid)
    L = None
    for a in list(REP.values()):
        s = np.asarray(a[0])
        if s.ndim == 2: L = s.shape[1]; break
    cols = band_cols(L, args.band) if L else None

    rep_keys = list(REP); unc_keys = list(UNC)
    # per-step series per chain
    ser = {k: [None]*N for k in rep_keys + unc_keys}
    for i in range(N):
        if not keep[i]: continue
        for k in rep_keys:
            ser[k][i] = steptrace(REP[k][i], cols)
        for k in unc_keys:
            v = np.asarray(UNC[k][i], dtype=np.float64)
            ser[k][i] = v if v.ndim == 1 else None

    # ---------- STEP-level correlations ----------
    step_pooled = {}; step_within = {}
    for rk in rep_keys:
        for uk in unc_keys:
            A = []; B = []; wc = []
            for i in range(N):
                if not keep[i]: continue
                a = ser[rk][i]; b = ser[uk][i]
                if a is None or b is None: continue
                T = min(len(a), len(b))
                if T < 4: continue
                a = a[:T]; b = b[:T]
                A.append(a); B.append(b)
                c = sp(a, b)
                if np.isfinite(c): wc.append(c)
            step_pooled[f"{rk}~{uk}"] = round(sp(np.concatenate(A), np.concatenate(B)), 3) if A else None
            step_within[f"{rk}~{uk}"] = round(float(np.mean(wc)), 3) if wc else None

    # ---------- CHAIN-level features (mean/slope/r2) ----------
    feat = {k: [] for k in rep_keys + unc_keys}
    yy = []; pp = []
    for i in range(N):
        if not keep[i]: continue
        ps = {k: (profile3(ser[k][i]) if ser[k][i] is not None else None) for k in rep_keys + unc_keys}
        if any(v is None for v in ps.values()): continue
        for k in rep_keys + unc_keys: feat[k].append(ps[k])
        yy.append(y[i]); pp.append(pid[i])
    yy = np.array(yy); pp = np.array(pp)
    feat = {k: np.array(v) for k, v in feat.items()}     # each (M,3): mean,slope,r2

    prob = {}
    for j, p in enumerate(pp): prob.setdefault(int(p), []).append(j)
    groups = [np.array(v) for v in prob.values() if any(yy[v] == 1) and any(yy[v] == 0)]

    # chain-level correlation matrix (on the mean summary; pooled + within-problem)
    def wcenter(v):
        o = v.astype(float).copy()
        for p in np.unique(pp):
            m = pp == p; o[m] -= np.nanmean(o[m])
        return o
    chain_pooled = {}; chain_within = {}
    for rk in rep_keys:
        for uk in unc_keys:
            a = feat[rk][:, 0]; b = feat[uk][:, 0]      # mean summary
            chain_pooled[f"{rk}~{uk}"] = round(sp(a, b), 3)
            chain_within[f"{rk}~{uk}"] = round(sp(wcenter(a), wcenter(b)), 3)

    # ---------- channel detection AUROC ----------
    def stackch(keys): return np.column_stack([feat[k] for k in keys])  # (M, 3*len)
    kw = dict(k=args.kfold, seeds=args.n_seeds)
    rep_F = stackch(rep_keys); unc_F = stackch(unc_keys)
    auroc = {
        "representation_only": channel_auroc(rep_F, yy, pp, groups, **kw),
        "uncertainty_only":    channel_auroc(unc_F, yy, pp, groups, **kw),
        "fusion":              channel_auroc(np.column_stack([rep_F, unc_F]), yy, pp, groups, **kw),
    }
    # static (mean only) vs dynamic (slope+r2 only), each channel
    rep_static = np.column_stack([feat[k][:, 0] for k in rep_keys])
    rep_dyn = np.column_stack([feat[k][:, 1:] for k in rep_keys])
    unc_static = np.column_stack([feat[k][:, 0] for k in unc_keys])
    unc_dyn = np.column_stack([feat[k][:, 1:] for k in unc_keys])
    auroc_split = {
        "rep_static": channel_auroc(rep_static, yy, pp, groups, **kw),
        "rep_dynamic": channel_auroc(rep_dyn, yy, pp, groups, **kw),
        "unc_static": channel_auroc(unc_static, yy, pp, groups, **kw),
        "unc_dynamic": channel_auroc(unc_dyn, yy, pp, groups, **kw),
    }

    out = {"meta": {"input": os.path.basename(args.input), "band": args.band, "mode": args.mode,
                    "N": int(len(yy)), "n_incorrect": int(yy.sum()), "contrastive_problems": len(groups),
                    "rep_channel": rep_keys, "unc_channel": unc_keys,
                    "subset": "format_ok" if args.format_ok else "all", "label": "answer-based"},
           "step_corr_pooled": step_pooled, "step_corr_within_chain": step_within,
           "chain_corr_pooled": chain_pooled, "chain_corr_within_problem": chain_within,
           "channel_within_auroc": auroc, "static_vs_dynamic_within_auroc": auroc_split}

    o = args.out or f"results_uncertainty/channels_{args.mode}_{args.band}.json"
    os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
    json.dump(out, open(o, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"N={len(yy)} incorrect={int(yy.sum())} contrastive={len(groups)} "
          f"rep={rep_keys} unc={unc_keys} band={args.band}")
    print("step within-chain corr:", json.dumps(step_within, ensure_ascii=False))
    print("chain within-problem corr:", json.dumps(chain_within, ensure_ascii=False))
    print("channel AUROC:", json.dumps(auroc, ensure_ascii=False))
    print("static vs dynamic:", json.dumps(auroc_split, ensure_ascii=False))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
