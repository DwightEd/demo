"""Within-chain first-error LOCALIZER fusing geometry (concentration) + uncertainty.

Motivated by FINDINGS sec.13: geometry (coherence/resultant) and uncertainty (U_D)
are complementary -- geometry localizes better on easy data, U_D on hard. So fuse
them and ask: does the fused score localize the first-error step WITHIN a chain
better than either alone -- and BEYOND length?

Protocol: GroupKFold by chain. Train a logistic on the detection-labeled steps
(first-error vs correct+pre-error) of the train chains; predict a per-step
error-score for ALL steps of the test chains (incl post-error). Then evaluate
WITHIN-CHAIN localization of that score:
  wc_loc      : per error chain, frac of OTHER steps (pre+post) the first-error beats
  MRR         : mean 1/rank of first-error among its chain's steps
  wc_loc(⊥nt) : same after residualizing the score on n_tok WITHIN each chain
                (localization BEYOND 'error step is longer') -- THE headline number
Compares feature sets: geom | unc | geom+unc | all(+len,pos), plus single-signal refs.

Needs _coh.npz with resultant/coherence/norm/cloud_D + U_D/U_C.
"""

from __future__ import annotations

import argparse
import numpy as np

try:
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


def wc_metrics(score, chains, min_steps, resid_nt=False):
    """score: full per-step array; chains: list of (idx_array, k_local, n_tok_array)."""
    locs, w, mrrs = [], [], []
    for idx, kloc, nt in chains:
        v = score[idx]
        fin = np.isfinite(v)
        if resid_nt:
            f2 = fin & np.isfinite(nt)
            if f2.sum() >= 3:
                b = np.polyfit(nt[f2], v[f2], 1); v = v - (b[0] * nt + b[1])
        T = len(idx)
        if T < min_steps or not np.isfinite(v[kloc]):
            continue
        others = np.array([t for t in range(T) if t != kloc and np.isfinite(v[t])])
        if len(others) < 2:
            continue
        locs.append(np.mean(v[others] < v[kloc])); w.append(len(others))
        mrrs.append(1.0 / (1 + int(np.sum(v[others] >= v[kloc]))))
    if not locs:
        return float("nan"), float("nan")
    return np.average(locs, weights=np.asarray(w, float)), float(np.mean(mrrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min_steps", type=int, default=4)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG, SC = z["stepgeom"], z["stepcloud"]; SR = z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None
    FNAMES = ["resultant", "coherence", "norm", "cloud_D", "U_D", "U_C", "n_tok", "position"]

    # flat per-step arrays + chain bookkeeping
    F, G, JIS, KIS, TIS, DET = [], [], [], [], [], []
    for i in range(len(SG)):
        sg = np.asarray(SG[i], float); sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            def cf(nm): return sc[j, li, cnames.index(nm)] if nm in cnames else np.nan
            ntok = int(rng[j, 1] - rng[j, 0] + 1)
            lo = max(0, int(rng[j, 0]) - a0); hi = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            row = [cf("resultant"), cf("coherence"), sg[j, li, gnames.index("norm")], cf("cloud_D"),
                   np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan,
                   np.nanmean(uc[lo:hi]) if (uc is not None and hi > lo) else np.nan,
                   ntok, j / max(1, T - 1)]
            F.append(row); G.append(i); JIS.append(j); KIS.append(k); TIS.append(T)
            DET.append(1 if (not correct and j == k) else (0 if (correct or j < k) else np.nan))
    F = np.asarray(F, float); G = np.asarray(G, int); JIS = np.asarray(JIS, int)
    KIS = np.asarray(KIS, int); DET = np.asarray(DET, float)
    for c in range(F.shape[1]):                       # impute feature NaNs (column mean)
        col = F[:, c]; col[~np.isfinite(col)] = np.nanmean(col[np.isfinite(col)])

    # per-error-chain bookkeeping for within-chain eval
    chains = []
    nt_col = FNAMES.index("n_tok")
    for c in np.unique(G):
        m = np.where(G == c)[0]
        k = KIS[m[0]]
        if k < 0:
            continue
        order = np.argsort(JIS[m]); m = m[order]
        kloc = int(np.where(JIS[m] == k)[0][0]) if (JIS[m] == k).any() else -1
        if kloc < 0:
            continue
        chains.append((m, kloc, F[m, nt_col]))

    sets = {
        "geom (res,coh,norm,clD)": [0, 1, 2, 3],
        "unc (U_D,U_C)":           [4, 5],
        "geom+unc":                [0, 1, 2, 3, 4, 5],
        "all (+len,pos)":          [0, 1, 2, 3, 4, 5, 6, 7],
        "resultant only":          [0],
        "U_D only":                [4],
    }
    gkf = GroupKFold(args.folds)

    def oof_full(cols):
        s = np.full(len(F), np.nan)
        X = F[:, cols]
        for tr, te in gkf.split(X, np.nan_to_num(DET), G):
            lab = np.isfinite(DET[tr])
            ytr = DET[tr][lab]
            if len(np.unique(ytr)) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr][lab], ytr.astype(int))
            s[te] = clf.predict_proba(X[te])[:, 1]      # predict ALL test steps
        return s

    print(f"file: {args.npz} | layer {args.layer} | error-chains {len(chains)} "
          f"| det-labeled steps {int(np.isfinite(DET).sum())}")
    print(f"\n{'feature set':24s} {'pooled_det':>11s} {'wc_loc':>8s} {'MRR':>7s} {'wc_loc(⊥nt)':>12s}")
    detmask = np.isfinite(DET)
    for nm, cols in sets.items():
        s = oof_full(cols)
        a_det = auroc(s[detmask], DET[detmask].astype(int))
        wl, mrr = wc_metrics(s, chains, args.min_steps, resid_nt=False)
        wlr, _ = wc_metrics(s, chains, args.min_steps, resid_nt=True)
        print(f"{nm:24s} {a_det:11.3f} {wl:8.3f} {mrr:7.3f} {wlr:12.3f}")

    print("\nwc_loc(⊥nt) = within-chain localization BEYOND length (residualize score on "
          "n_tok per chain). Key comparison: does geom+unc beat 'resultant only' AND "
          "'U_D only'? If yes, fusion of complementary axes is a better within-chain "
          "first-error localizer.")


if __name__ == "__main__":
    main()
