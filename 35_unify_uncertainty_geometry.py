"""Step 35: unify uncertainty-trace features with activation geometry (same npz).

Question (from the 2605.07776 read + our geometry results): are the uncertainty-trace
features and the activation-geometry features (participation ratio = effective dim,
activation entropy) DIFFERENT READINGS OF ONE MECHANISM, or independent signals?

We already store, per chain:
  uncertainty : sv_tok_entropy / sv_tok_committal (per token), sv_out_entropy (per step)
  geometry    : sv_pr_<mode> (T,L) participation ratio, sv_ae_<mode> (T,L) activation entropy

For each chain build two PARALLEL trajectory-profile feature sets (mu_early, mu_mid,
mu_late, slope, r2) -- one from the uncertainty trace, one from the geometry trace
(layer-band-averaged per step). Then:

  (1) Spearman correlation, uncertainty-feature x geometry-feature, computed BOTH
      pooled (cross-problem) and within-problem (subtract per-problem mean first).
      Headline: corr(uncertainty LEVEL, PR LEVEL)  -> is "uncertain <=> more diffuse"?
                corr(slope, slope), corr(r2, r2)   -> do the DYNAMICS align?
  (2) within-problem paired AUROC for each single feature -> which features predict
      error, and do the predictive uncertainty features overlap with the predictive
      geometry features?
  (3) combined within-problem AUROC: uncertainty-only / geometry-only / both -> are
      they redundant (no gain) or complementary (both > either)?

Label: answer-based. Writes results_uncertainty/unify_<mode>_<band>.json.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ---------- trajectory profile (same as 34) ----------
def profile(e, n=0):
    e = np.asarray(e, dtype=np.float64); e = e[np.isfinite(e)]
    if n: e = e[:n]
    T = len(e)
    if T < 4: return None
    pos = np.linspace(0, 1, T); A = np.vstack([pos, np.ones(T)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    pred = A @ coef; ssr = ((e-pred)**2).sum(); sst = ((e-e.mean())**2).sum()+1e-12
    q = max(1, T//4)
    mid = e[q:T-q].mean() if T-2*q > 0 else e.mean()
    return np.array([e[:q].mean(), mid, e[-q:].mean(), coef[0], 1-ssr/sst, e.mean()])


def band_cols(L, band):
    if band == "deep": return np.arange(int(L*0.6), L)
    if band == "mid":  return np.arange(int(L*0.3), int(L*0.7))
    return np.arange(L)


def mat_to_steptrace(M, cols):
    """(T,L) per-(step,layer) -> (T,) per-step scalar averaged over the layer band."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2: return None
    cols = cols[cols < M.shape[1]]
    with np.errstate(invalid="ignore"):
        return np.nanmean(M[:, cols], axis=1)


# ---------- within-problem paired AUROC for a single scalar ----------
def within_pair_auroc(groups, f, y):
    conc = 0.0; npair = 0
    for idx in groups:
        inc = [f[i] for i in idx if y[i] == 1 and np.isfinite(f[i])]
        cor = [f[i] for i in idx if y[i] == 0 and np.isfinite(f[i])]
        if not inc or not cor: continue
        for a in inc:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        npair += len(inc)*len(cor)
    a = conc/npair if npair else float("nan")
    return max(a, 1-a) if np.isfinite(a) else float("nan")


def gfolds(g, k, s):
    u = np.unique(g); r = np.random.default_rng(s); r.shuffle(u)
    fo = {int(x): i % k for i, x in enumerate(u)}; f = np.array([fo[int(x)] for x in g])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def cv(F, y, folds):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2: continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(F[tr], y[tr]); oof[te] = clf.predict_proba(F[te])[:, 1]
    return oof


def combined_within_auroc(F, y, pp, groups, kfold, n_seeds, seed):
    vals = []
    for s in range(n_seeds):
        og = cv(F, y, gfolds(pp, kfold, seed+s))
        vals.append(within_pair_auroc(groups, og, y))
    return round(float(np.nanmean(vals)), 4)


def within_center(v, pp):
    """subtract per-problem mean (partial out problem) -> the chain-to-chain signal."""
    out = np.array(v, dtype=np.float64).copy()
    for p in np.unique(pp):
        m = pp == p
        out[m] = out[m] - np.nanmean(out[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp", help="geometry sv_ mode key")
    ap.add_argument("--band", default="deep", choices=["deep", "mid", "all"])
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format_ok", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    need = [f"sv_pr_{args.mode}", f"sv_ae_{args.mode}", "sv_tok_entropy", "sv_tok_committal"]
    miss = [k for k in need if k not in dd.files]
    if miss:
        raise SystemExit(f"npz missing {miss}; files={sorted(dd.files)}")
    ENT = dd["sv_tok_entropy"]; COM = dd["sv_tok_committal"]
    PR = dd[f"sv_pr_{args.mode}"]; AE = dd[f"sv_ae_{args.mode}"]
    pid = dd["problem_ids"].astype(int)
    y = (dd["is_correct"].astype(int) == 0).astype(int)
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(ENT), bool)
    keep = fmt if args.format_ok else np.ones(len(ENT), bool)

    L = np.asarray(PR[0]).shape[1]; cols = band_cols(L, args.band)
    NAMES = ["early", "mid", "late", "slope", "r2", "mean"]
    rows = []                     # per-chain feature dict
    for i in range(len(ENT)):
        if not keep[i]: continue
        pe = profile(ENT[i]); pc = profile(COM[i])
        prt = mat_to_steptrace(PR[i], cols); aet = mat_to_steptrace(AE[i], cols)
        pp_ = profile(prt) if prt is not None else None
        pa = profile(aet) if aet is not None else None
        if any(x is None for x in (pe, pc, pp_, pa)): continue
        d = {"y": int(y[i]), "pid": int(pid[i])}
        for nm, vec, tag in [("ent", pe, "u"), ("com", pc, "u"), ("pr", pp_, "g"), ("ae", pa, "g")]:
            for j, s in enumerate(NAMES):
                d[f"{nm}_{s}"] = float(vec[j])
        rows.append(d)

    if len(rows) < 20:
        raise SystemExit(f"too few usable chains ({len(rows)})")
    pp = np.array([r["pid"] for r in rows]); yy = np.array([r["y"] for r in rows])
    prob = {}
    for j, p in enumerate(pp): prob.setdefault(int(p), []).append(j)
    groups = [np.array(v) for v in prob.values() if any(yy[v] == 1) and any(yy[v] == 0)]
    col = lambda k: np.array([r[k] for r in rows], dtype=np.float64)

    out = {"meta": {"input": os.path.basename(args.input), "mode": args.mode, "band": args.band,
                    "N": len(rows), "n_incorrect": int(yy.sum()), "contrastive_problems": len(groups),
                    "label": "answer-based", "subset": "format_ok" if args.format_ok else "all"}}

    # ---- (1) correlations: uncertainty x geometry, pooled & within-problem ----
    unc_keys = [f"{m}_{s}" for m in ("ent", "com") for s in ("mean", "slope", "r2")]
    geo_keys = [f"{m}_{s}" for m in ("pr", "ae") for s in ("mean", "slope", "r2")]
    corr = {"pooled": {}, "within": {}}
    for uk in unc_keys:
        u = col(uk); uw = within_center(u, pp)
        for gk in geo_keys:
            g = col(gk); gw = within_center(g, pp)
            mp = np.isfinite(u) & np.isfinite(g)
            corr["pooled"][f"{uk}~{gk}"] = round(float(spearmanr(u[mp], g[mp]).correlation), 3)
            mw = np.isfinite(uw) & np.isfinite(gw)
            corr["within"][f"{uk}~{gk}"] = round(float(spearmanr(uw[mw], gw[mw]).correlation), 3)
    out["correlations"] = corr
    out["headline"] = {
        "level_pooled (ent_mean~pr_mean)": corr["pooled"]["ent_mean~pr_mean"],
        "level_within (ent_mean~pr_mean)": corr["within"]["ent_mean~pr_mean"],
        "slope_within (ent_slope~pr_slope)": corr["within"]["ent_slope~pr_slope"],
        "r2_within (ent_r2~pr_r2)": corr["within"]["ent_r2~pr_r2"],
    }

    # ---- (2) per-feature within-problem paired AUROC (which features detect error) ----
    feat_auroc = {}
    for k in unc_keys + geo_keys:
        feat_auroc[k] = round(within_pair_auroc(groups, col(k), yy), 4)
    out["per_feature_within_auroc"] = feat_auroc

    # ---- (3) combined within-problem AUROC: unc / geom / both ----
    def stack(keys): return np.column_stack([col(k) for k in keys])
    kw = dict(kfold=args.kfold, n_seeds=args.n_seeds, seed=args.seed)
    out["combined_within_auroc"] = {
        "uncertainty_only": combined_within_auroc(stack(unc_keys), yy, pp, groups, **kw),
        "geometry_only":    combined_within_auroc(stack(geo_keys), yy, pp, groups, **kw),
        "both":             combined_within_auroc(stack(unc_keys+geo_keys), yy, pp, groups, **kw),
    }

    o = args.out or f"results_uncertainty/unify_{args.mode}_{args.band}.json"
    os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
    json.dump(out, open(o, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"N={len(rows)} incorrect={int(yy.sum())} contrastive={len(groups)} band={args.band}")
    print("headline:", json.dumps(out["headline"], ensure_ascii=False))
    print("combined within AUROC:", json.dumps(out["combined_within_auroc"], ensure_ascii=False))
    print(f"wrote {o}")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
