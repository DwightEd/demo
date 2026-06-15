"""The OTHER geometric family we never tested at step level: DISPLACEMENT to the
healthy manifold (Mahalanobis / SPE), not within-step concentration.

Concentration (norm/resultant) = how spread the tokens are INSIDE a step.
Displacement (this script)      = WHERE the step's pooled vector sits relative to
the distribution of CORRECT steps. A step can be internally coherent yet far from
the healthy region, or vice versa -> potentially an orthogonal, possibly stronger
axis (chain-level Mahalanobis was historically the strongest signal, 0.66-0.83).

Per step, pooled vector z_j (d=4096). Cross-fit by chain (GroupKFold): on
correct-chain steps in TRAIN, fit PCA(k) + mean mu; then for every step
  Mahalanobis = || (z-mu) projected onto PCs, z-scored ||   (in-subspace distance)
  SPE         = || residual of (z-mu) outside the k PCs || / ||z-mu||  (leakage, Q-stat)
Report AUROC (raw + within-length-bucket) and the KEY question: increment of
displacement OVER [confound + U_D/U_C + concentration(norm)] -- is displacement an
INDEPENDENT geometric axis beyond the concentration family?

Needs step vectors: extract with --store_step_vectors --sv_layers L (+ --cloud_eff_rank
for the concentration columns). Uses the pooled step vector `stepvec`.
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("needs scikit-learn")


def auroc(score, y):
    m = np.isfinite(score); s, yy = score[m], y[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def bucket_auroc(score, y, nt, nb=5):
    edges = np.quantile(nt, np.linspace(0, 1, nb + 1)); edges[-1] += 1
    b = np.clip(np.digitize(nt, edges[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb
        ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        a = bdir(auroc(score[m], y[m]))
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(np.argsort(np.argsort(a[m])).astype(float),
                            np.argsort(np.argsort(b[m])).astype(float))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=50, help="PCA dim of healthy subspace")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("step_vectors_stored", np.array(False))):
        raise SystemExit("no step vectors; re-extract with --store_step_vectors --sv_layers ...")
    svl = [int(x) for x in (z["sv_layers"] if "sv_layers" in z.files else z["layers_used"])]
    if args.layer not in svl:
        raise SystemExit(f"layer {args.layer} not in sv_layers {svl}")
    svk = svl.index(args.layer)
    SV = z["stepvec"]; SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG = z["stepgeom"]
    SC = z["stepcloud"] if ("stepcloud" in z.files and bool(z.get("cloud_stored", np.array(False)))) else None
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    Z, NT, Y, G, NORM, RES, UDv, UCv = [], [], [], [], [], [], [], []
    for i in range(len(SV)):
        sv = np.asarray(SV[i], np.float32)
        if sv.ndim != 3 or sv.shape[1] <= svk:
            continue
        sg = np.asarray(SG[i], float)
        sc = np.asarray(SC[i], float) if SC is not None else None
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = sv.shape[0]
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            zj = sv[j, svk, :]
            if not np.isfinite(zj).all():
                continue
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            Z.append(zj); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); Y.append(y); G.append(i)
            NORM.append(sg[j, li, gnames.index("norm")])
            RES.append(sc[j, li, cnames.index("resultant")] if (sc is not None and "resultant" in cnames) else np.nan)
            lo = max(0, int(rng[j, 0]) - a0); hi = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            UDv.append(np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan)
            UCv.append(np.nanmean(uc[lo:hi]) if (uc is not None and hi > lo) else np.nan)
    Z = np.asarray(Z, np.float64); NT = np.asarray(NT, float); Y = np.asarray(Y, int); G = np.asarray(G, int)
    NORM = np.asarray(NORM, float); RES = np.asarray(RES, float)
    UDv = np.asarray(UDv, float); UCv = np.asarray(UCv, float)
    for a in (NORM, RES, UDv, UCv):
        a[~np.isfinite(a)] = np.nanmean(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0
    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())} "
          f"| d={Z.shape[1]} | PCA k={args.k}")

    # cross-fit: healthy subspace (PCA + mean) on CORRECT-chain steps in train
    MAH = np.full(len(Y), np.nan); SPE = np.full(len(Y), np.nan)
    gkf = GroupKFold(args.folds)
    for tr, te in gkf.split(Z, Y, G):
        htr = tr[Y[tr] == 0]
        if len(htr) < args.k + 10:
            continue
        mu = Z[htr].mean(0)
        pca = PCA(n_components=args.k, random_state=0).fit(Z[htr] - mu)
        comp = pca.components_                       # (k, d)
        proj_h = (Z[htr] - mu) @ comp.T              # healthy projections
        sd = proj_h.std(0) + 1e-6
        dz = Z[te] - mu
        p = dz @ comp.T                              # (n_te, k)
        MAH[te] = np.sqrt(((p / sd) ** 2).sum(1))    # in-subspace Mahalanobis (diag)
        recon = p @ comp
        SPE[te] = np.linalg.norm(dz - recon, axis=1) / (np.linalg.norm(dz, axis=1) + 1e-9)

    print(f"\n=== displacement metrics (raw / within-length-bucket AUROC) ===")
    for nm, v in [("Mahalanobis", MAH), ("SPE(leakage)", SPE)]:
        print(f"  {nm:14s} raw {bdir(auroc(v, Y)):.3f}   bucket {bucket_auroc(v, Y, NT):.3f}")
    print(f"  {'norm(concentr)':14s} raw {bdir(auroc(NORM, Y)):.3f}   bucket {bucket_auroc(NORM, Y, NT):.3f}")

    print(f"\n=== is displacement INDEPENDENT of concentration / uncertainty? ===")
    print(f"  corr(Mahalanobis, norm)      {spearman(MAH, NORM):+.2f}")
    print(f"  corr(Mahalanobis, resultant) {spearman(MAH, RES):+.2f}")
    print(f"  corr(Mahalanobis, U_D)       {spearman(MAH, UDv):+.2f}")
    print(f"  corr(Mahalanobis, log n_tok) {spearman(MAH, np.log(np.maximum(NT,1))):+.2f}")
    print(f"  corr(SPE, norm)              {spearman(SPE, NORM):+.2f}")

    # increment of displacement over [confound + U + concentration]
    logn = np.log(np.maximum(NT, 1))
    base = np.c_[logn, UDv, UCv, NORM]               # confound + uncertainty + concentration
    def oof(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p.fit(X[tr], Y[tr]); s[te] = p.predict_proba(X[te])[:, 1]
        return s
    s_b = oof(base); s_bm = oof(np.c_[base, MAH]); s_bs = oof(np.c_[base, MAH, SPE])
    a_b, a_bm, a_bs = auroc(s_b, Y), auroc(s_bm, Y), auroc(s_bs, Y)
    rng = np.random.default_rng(0); chains = np.unique(G); d = []
    for _ in range(500):
        cb = rng.choice(chains, size=len(chains), replace=True)
        m = np.concatenate([np.where(G == c)[0] for c in cb])
        d.append(auroc(s_bs[m], Y[m]) - auroc(s_b[m], Y[m]))
    d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
    print(f"\n=== increment over [length+U_D+U_C+norm] ===")
    print(f"  baseline (len+U+norm):            {a_b:.3f}")
    print(f"  + Mahalanobis:                    {a_bm:.3f}")
    print(f"  + Mahalanobis + SPE:              {a_bs:.3f}")
    print(f"  DISPLACEMENT INCREMENT: +{a_bs-a_b:.3f}  [{lo:+.3f},{hi:+.3f}]  "
          f"{'SIGNIFICANT' if lo > 0 else 'ns'}")
    print("\nread: if displacement increment > 0 significant AND corr with norm is low, "
          "it is a SECOND geometric axis beyond concentration -> geometry NOT exhausted.")


if __name__ == "__main__":
    main()
