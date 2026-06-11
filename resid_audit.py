"""Covariate residualization audit -- how much of the naive geometric signal is
confound (step length n_tok, chain position j/T, content density)?

For a metric M, per layer, fit E[M | n_tok, j/T, density] on PROCESS-CORRECT
chains' steps (cross-fit by chain, gradient boosting), and replace M by its
residual. Then compare step-level AUROC (first-error vs correct/pre-error,
ProcessBench labels) RAW vs RESIDUALIZED, per layer + a combined logistic.

A large drop raw->residual = the naive signal was mostly confound. This is the
defensive-as-contribution section ("how much discriminative power survives a
rigorous confound control").

Needs a ProcessBench npz with stepgeom/stepcloud + step_token_ranges + steps_text.
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
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


def density(text):
    t = str(text)
    if not t:
        return float("nan")
    letters = sum(ch.isalpha() for ch in t)
    return 1.0 - letters / len(t)              # non-letter fraction (formula/number proxy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--metric", default="cloud_D")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    layers = [int(x) for x in z["layers_used"]]
    L = len(layers)
    geom_names = [str(x) for x in z["geom_feature_names"]]
    cloud_names = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    if args.metric in geom_names:
        SRC, fi = z["stepgeom"], geom_names.index(args.metric)
    elif args.metric in cloud_names and bool(z.get("cloud_stored", np.array(False))):
        SRC, fi = z["stepcloud"], cloud_names.index(args.metric)
    else:
        raise SystemExit(f"metric {args.metric} not found")
    ges = z["gold_error_step"].astype(int)
    SR = z["step_token_ranges"]; ST = z["steps_text"]

    M, NUIS, Y, G, H = [], [], [], [], []
    for i in range(len(SRC)):
        g = np.asarray(SRC[i], float)
        if g.ndim == 3:
            g = g[:, :, fi]
        T = g.shape[0]
        rng = np.asarray(SR[i], int)
        txt = list(ST[i]) if i < len(ST) else []
        k = int(ges[i]); correct = (k < 0)
        for j in range(T):
            if not np.isfinite(g[j]).all():
                continue
            if correct:
                y, keepit = 0, True
            elif j < k:
                y, keepit = 0, True
            elif j == k:
                y, keepit = 1, True
            else:
                keepit = False
            if not keepit:
                continue
            ntok = int(rng[j, 1] - rng[j, 0] + 1) if j < rng.shape[0] else np.nan
            dens = density(txt[j]) if j < len(txt) else np.nan
            M.append(g[j]); NUIS.append([ntok, j / max(1, T - 1), dens])
            Y.append(y); G.append(i); H.append(correct)
    M = np.asarray(M, float); NUIS = np.asarray(NUIS, float)
    Y = np.asarray(Y, int); G = np.asarray(G, int); H = np.asarray(H, bool)
    # impute nuisance NaNs with column means
    for c in range(NUIS.shape[1]):
        col = NUIS[:, c]; col[~np.isfinite(col)] = np.nanmean(col)

    print(f"file: {args.npz} | metric {args.metric} | layers {layers}")
    print(f"steps: {len(Y)} | first-error: {int(Y.sum())} | correct-chain: {int(H.sum())}")

    # cross-fit residuals per layer (regressor trained on correct-chain steps)
    resid = np.full_like(M, np.nan)
    gkf = GroupKFold(args.folds)
    for tr, te in gkf.split(M, Y, G):
        Htr = H[tr]
        if Htr.sum() < 20:
            continue
        Xtr = NUIS[tr][Htr]
        for l in range(L):
            reg = GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                            random_state=0)
            reg.fit(Xtr, M[tr][Htr][:, l])
            resid[te, l] = M[te][:, l] - reg.predict(NUIS[te])

    # per-layer raw vs residual AUROC
    print(f"\n{'layer':>6s} {'raw':>7s} {'resid':>7s} {'drop':>7s}")
    print("-" * 30)
    for l in range(L):
        ar = bdir(auroc(M[:, l], Y)); rr = bdir(auroc(resid[:, l], Y))
        print(f"{layers[l]:6d} {ar:7.3f} {rr:7.3f} {ar - rr:+7.3f}")
    braw = max(bdir(auroc(M[:, l], Y)) for l in range(L))
    bres = max(bdir(auroc(resid[:, l], Y)) for l in range(L))

    # nuisance-only baseline + combined logistic raw vs residual (OOF)
    def oof_logit(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            p = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000))
            p.fit(X[tr], Y[tr]); s[te] = p.predict_proba(X[te])[:, 1]
        return auroc(s, Y)
    a_nuis = oof_logit(NUIS)
    a_raw = oof_logit(M)
    a_res = oof_logit(np.nan_to_num(resid, nan=0.0))

    print(f"\nbest single layer   raw {braw:.3f} -> resid {bres:.3f}  ({braw-bres:+.3f})")
    print(f"nuisance-only AUROC (n_tok,j/T,density): {a_nuis:.3f}  <- how much is confound alone")
    print(f"all-layer logistic  raw {a_raw:.3f} -> resid {a_res:.3f}  ({a_raw-a_res:+.3f})")
    print("\nlarge drop => naive geometric signal was mostly confound; residual = honest geometry.")


if __name__ == "__main__":
    main()
