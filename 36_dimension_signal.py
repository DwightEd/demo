"""Step 36: per-dimension SIGNED deviation features (which dims, how much, outliers).

Motivation: PR / AE collapse each activation vector to ONE scale-free scalar -> they
cannot express magnitude, dimension identity, or which dims are abnormally high. We
instead keep the per-dimension signed deviation of each chain's late-stage activation
from the HEALTHY (correct-chain) reference, difficulty-controlled (per-problem correct
mean removed), and build features that preserve identity + sign + magnitude:

  v_i   = late-step, deep-layer mean of sv_vec_<mode>[i]      (signed, magnitude kept)
  z_i,j = (v_i,j - mu_c,j[problem]) / sigma_c,j[global]        (per-dim signed deviation)

Features / detectors (all within-problem paired AUROC):
  magnitude (label-free) : ||z||2, max|z|, #{|z|>2}, #{|z|>3}
  supervised projection  : w = mean(z|err) - mean(z|cor), held-out GroupKFold, score=z.w
  sparse L1 probe        : which / how-many dims carry it (n_active, top dims)
  per-dim |z| err-vs-cor : ranks the abnormal dimensions (identity)

Then vs uncertainty: Spearman(best dim score, ent_mean) pooled+within, and combined
within AUROC (uncertainty-only / dim-magnitude-only / both) -> redundant or complementary.

Caveat: dims are in the STORED vector basis (if V_R projection was used at extraction,
they are reasoning-subspace coords, not raw neurons) -> detection is valid; neuron-level
identity needs a raw (no-V_R) re-extract. Label: answer-based.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def band_cols(L, band):
    if band == "deep": return np.arange(int(L*0.6), L)
    if band == "mid":  return np.arange(int(L*0.3), int(L*0.7))
    return np.arange(L)


def late_deep_vec(sv, cols, late_lo):
    M = np.asarray(sv, dtype=np.float32)
    if M.ndim != 3: return None
    T, L, d = M.shape
    cc = cols[cols < L]
    fr = (np.arange(T)/(T-1)) if T > 1 else np.array([0.0])
    m = fr >= late_lo
    if not m.any(): m = fr >= fr.max()
    sub = M[m][:, cc, :].reshape(-1, d)
    with np.errstate(invalid="ignore"):
        v = np.nanmean(sub, axis=0)
    return v.astype(np.float64)


def profile_mean(e):
    e = np.asarray(e, dtype=np.float64); e = e[np.isfinite(e)]
    return float(e.mean()) if len(e) else np.nan


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


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    return round(float(spearmanr(a[m], b[m]).correlation), 3)


def within_center(v, pp):
    out = np.array(v, dtype=np.float64).copy()
    for p in np.unique(pp):
        m = pp == p; out[m] = out[m] - np.nanmean(out[m])
    return out


def oof_proba(F, y, g, k, seeds, clf_factory):
    """out-of-fold P(err) averaged over seeds; GroupKFold by problem."""
    acc = np.zeros(len(y))
    for s in range(seeds):
        oof = np.full(len(y), np.nan)
        for tr, te in gfolds(g, k, s):
            if len(np.unique(y[tr])) < 2: continue
            clf = clf_factory(); clf.fit(F[tr], y[tr])
            oof[te] = clf.predict_proba(F[te])[:, 1]
        acc += oof
    return acc/seeds


def supervised_proj_oof(Z, y, g, k, seeds):
    """held-out signed-deviation direction w = mean(z|err)-mean(z|cor); score = z.w."""
    acc = np.zeros(len(y))
    for s in range(seeds):
        oof = np.full(len(y), np.nan)
        for tr, te in gfolds(g, k, s):
            if len(np.unique(y[tr])) < 2: continue
            w = Z[tr][y[tr] == 1].mean(0) - Z[tr][y[tr] == 0].mean(0)
            oof[te] = Z[te] @ w
        acc += oof
    return acc/seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--band", default="deep", choices=["deep", "mid", "all"])
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--format_ok", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    key = f"sv_vec_{args.mode}"
    if key not in dd.files:
        raise SystemExit(f"npz lacks {key}; re-extract with --store_vectors. files={sorted(dd.files)}")
    SV = dd[key]; ENT = dd["sv_tok_entropy"]; COM = dd["sv_tok_committal"]
    pid = dd["problem_ids"].astype(int)
    y = (dd["is_correct"].astype(int) == 0).astype(int)
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(SV), bool)
    keep0 = fmt if args.format_ok else np.ones(len(SV), bool)

    L = np.asarray(SV[0]).shape[1]; cols = band_cols(L, args.band)
    V, yy, pp, ent_m, com_m, idxs = [], [], [], [], [], []
    for i in range(len(SV)):
        if not keep0[i]: continue
        v = late_deep_vec(SV[i], cols, args.late_lo)
        if v is None or not np.isfinite(v).all(): continue
        V.append(v); yy.append(int(y[i])); pp.append(int(pid[i]))
        ent_m.append(profile_mean(ENT[i])); com_m.append(profile_mean(COM[i]))
    V = np.array(V); yy = np.array(yy); pp = np.array(pp)
    ent_m = np.array(ent_m); com_m = np.array(com_m)
    d = V.shape[1]

    # healthy reference: per-problem correct mean (difficulty control) + global correct std
    glob_cor = V[yy == 0]
    mu_glob = glob_cor.mean(0); sg = glob_cor.std(0) + 1e-6
    muc = {}
    for p in np.unique(pp):
        cm = (pp == p) & (yy == 0)
        muc[p] = V[cm].mean(0) if cm.any() else mu_glob
    Z = np.vstack([(V[i] - muc[pp[i]]) / sg for i in range(len(V))])

    prob = {}
    for j, p in enumerate(pp): prob.setdefault(int(p), []).append(j)
    groups = [np.array(v) for v in prob.values() if any(yy[v] == 1) and any(yy[v] == 0)]

    out = {"meta": {"input": os.path.basename(args.input), "mode": args.mode, "band": args.band,
                    "N": len(V), "d": int(d), "n_incorrect": int(yy.sum()),
                    "contrastive_problems": len(groups),
                    "subset": "format_ok" if args.format_ok else "all", "label": "answer-based"}}

    # --- magnitude features (label-free) ---
    l2 = np.sqrt((Z**2).sum(1)); linf = np.abs(Z).max(1)
    n2 = (np.abs(Z) > 2).sum(1).astype(float); n3 = (np.abs(Z) > 3).sum(1).astype(float)
    out["magnitude_within_auroc"] = {
        "l2_norm": round(within_pair_auroc(groups, l2, yy), 4),
        "max_abs": round(within_pair_auroc(groups, linf, yy), 4),
        "n_gt2":   round(within_pair_auroc(groups, n2, yy), 4),
        "n_gt3":   round(within_pair_auroc(groups, n3, yy), 4),
    }

    # --- supervised held-out projection ---
    sp = supervised_proj_oof(Z, yy, pp, args.kfold, args.n_seeds)
    out["supervised_proj_within_auroc"] = round(within_pair_auroc(groups, sp, yy), 4)

    # --- sparse L1 probe (which / how many dims) ---
    l1 = oof_proba(Z, yy, pp, args.kfold, args.n_seeds,
                   lambda: make_pipeline(StandardScaler(),
                       LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                                          max_iter=2000, class_weight="balanced")))
    out["l1_probe_within_auroc"] = round(within_pair_auroc(groups, l1, yy), 4)
    full = make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                                            max_iter=2000, class_weight="balanced"))
    full.fit(Z, yy)
    coef = full[-1].coef_.ravel()
    nz = np.where(np.abs(coef) > 1e-8)[0]
    order = nz[np.argsort(-np.abs(coef[nz]))][:15]
    out["l1_active_dims"] = int(len(nz))
    out["l1_top_dims"] = [{"dim": int(j), "coef": round(float(coef[j]), 3)} for j in order]

    # --- which dims abnormal: per-dim mean|z| error vs correct ---
    az_e = np.abs(Z[yy == 1]).mean(0); az_c = np.abs(Z[yy == 0]).mean(0)
    diff = az_e - az_c; top = np.argsort(-diff)[:15]
    out["top_abnormal_dims"] = [{"dim": int(j), "err_absz": round(float(az_e[j]), 3),
                                 "cor_absz": round(float(az_c[j]), 3),
                                 "diff": round(float(diff[j]), 3)} for j in top]

    # --- baseline PR for reference (same chains) ---
    if f"sv_pr_{args.mode}" in dd.files:
        PR = dd[f"sv_pr_{args.mode}"]
        prm = []
        kept = 0
        for i in range(len(SV)):
            if not keep0[i]: continue
            v = late_deep_vec(SV[i], cols, args.late_lo)
            if v is None or not np.isfinite(v).all(): continue
            M = np.asarray(PR[i], float)
            cc = cols[cols < M.shape[1]]
            prm.append(float(np.nanmean(M[:, cc])) if M.ndim == 2 else np.nan); kept += 1
        prm = np.array(prm)
        out["pr_mean_within_auroc_ref"] = round(within_pair_auroc(groups, prm, yy), 4)

    # --- vs uncertainty: correlation + complementarity ---
    best_name = max(out["magnitude_within_auroc"], key=out["magnitude_within_auroc"].get)
    best_feat = {"l2_norm": l2, "max_abs": linf, "n_gt2": n2, "n_gt3": n3}[best_name]
    out["vs_uncertainty"] = {
        "best_magnitude_feature": best_name,
        "corr_pooled(best~ent_mean)": spearman(best_feat, ent_m),
        "corr_within(best~ent_mean)": spearman(within_center(best_feat, pp), within_center(ent_m, pp)),
        "corr_pooled(supervised~ent_mean)": spearman(sp, ent_m),
    }

    def stack(*cols_): return np.column_stack(cols_)
    def comb(F):
        oof = oof_proba(F, yy, pp, args.kfold, args.n_seeds,
                        lambda: make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced")))
        return round(within_pair_auroc(groups, oof, yy), 4)
    out["combined_within_auroc"] = {
        "uncertainty_only": comb(stack(ent_m, com_m)),
        "dim_magnitude_only": comb(stack(l2, linf, n2, n3)),
        "both": comb(stack(ent_m, com_m, l2, linf, n2, n3)),
    }

    o = args.out or f"results_uncertainty/dimsignal_{args.mode}_{args.band}.json"
    os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
    json.dump(out, open(o, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"N={len(V)} d={d} incorrect={int(yy.sum())} contrastive={len(groups)} band={args.band}")
    print("magnitude:", json.dumps(out["magnitude_within_auroc"], ensure_ascii=False))
    print("supervised proj:", out["supervised_proj_within_auroc"],
          " L1 probe:", out["l1_probe_within_auroc"], " (active dims", out["l1_active_dims"], ")")
    print("PR ref:", out.get("pr_mean_within_auroc_ref"))
    print("vs uncertainty:", json.dumps(out["vs_uncertainty"], ensure_ascii=False))
    print("combined:", json.dumps(out["combined_within_auroc"], ensure_ascii=False))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
