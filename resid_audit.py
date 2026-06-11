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
    ap.add_argument("--nuis", default="n_tok,pos,density",
                    help="which nuisances to residualize on (subset of "
                         "n_tok,pos,density). e.g. 'n_tok,density' keeps position.")
    args = ap.parse_args()
    NUIS_NAMES = ["n_tok", "pos", "density"]
    use_nuis = [NUIS_NAMES.index(x) for x in args.nuis.split(",") if x in NUIS_NAMES]

    z = np.load(args.npz, allow_pickle=True)
    geom_names = [str(x) for x in z["geom_feature_names"]]
    cloud_names = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    if args.metric == "grad":
        if not bool(z.get("grad_stored", np.array(False))):
            raise SystemExit("no gradprof; re-extract with --grad_profile")
        SRC, fi = z["gradprof"], None
        layers = [int(x) for x in z["grad_layers"]]
    elif args.metric in geom_names:
        SRC, fi = z["stepgeom"], geom_names.index(args.metric)
        layers = [int(x) for x in z["layers_used"]]
    elif args.metric in cloud_names and bool(z.get("cloud_stored", np.array(False))):
        SRC, fi = z["stepcloud"], cloud_names.index(args.metric)
        layers = [int(x) for x in z["layers_used"]]
    else:
        raise SystemExit(f"metric {args.metric} not found")
    L = len(layers)
    ges = z["gold_error_step"].astype(int)
    SR = z["step_token_ranges"]; ST = z["steps_text"]

    M, NUIS, Y, G, H = [], [], [], [], []
    for i in range(len(SRC)):
        if SRC[i] is None:
            continue
        g = np.asarray(SRC[i], float)
        if fi is not None and g.ndim == 3:
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

    # --- per-nuisance breakdown: are error steps really longer/later/denser? ---
    print(f"\n=== nuisance breakdown (error-step vs good-step) ===")
    print(f"{'nuisance':9s} {'err mean':>9s} {'good mean':>9s} {'standalone AUROC':>17s}")
    for c, nm in enumerate(NUIS_NAMES):
        col = NUIS[:, c]
        me, mg = col[Y == 1].mean(), col[Y == 0].mean()
        print(f"{nm:9s} {me:9.3f} {mg:9.3f} {bdir(auroc(col, Y)):17.3f}")
    print(f"residualizing on: {[NUIS_NAMES[i] for i in use_nuis]}")

    # cross-fit residuals per layer (regressor trained on correct-chain steps)
    resid = np.full_like(M, np.nan)
    gkf = GroupKFold(args.folds)
    for tr, te in gkf.split(M, Y, G):
        Htr = H[tr]
        if Htr.sum() < 20:
            continue
        Xtr = NUIS[tr][Htr][:, use_nuis]
        Xte = NUIS[te][:, use_nuis]
        for l in range(L):
            reg = GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                            random_state=0)
            reg.fit(Xtr, M[tr][Htr][:, l])
            resid[te, l] = M[te][:, l] - reg.predict(Xte)

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
    a_nuis = oof_logit(NUIS[:, use_nuis])
    a_nuis_M = oof_logit(np.c_[NUIS[:, use_nuis], M])      # non-discarding increment test
    a_raw = oof_logit(M)
    a_res = oof_logit(np.nan_to_num(resid, nan=0.0))

    print(f"\nbest single layer   raw {braw:.3f} -> resid {bres:.3f}  ({braw-bres:+.3f})")
    print(f"nuisance-only AUROC ({args.nuis}): {a_nuis:.3f}  <- confound alone")
    print(f"all-layer logistic  raw {a_raw:.3f} -> resid {a_res:.3f}  ({a_raw-a_res:+.3f})")
    print(f"INCREMENT (non-discarding): nuisance {a_nuis:.3f} -> nuisance+metric "
          f"{a_nuis_M:.3f}  (+{a_nuis_M-a_nuis:.3f})  <- does the metric add OVER the "
          f"confounds, without throwing anything away?")
    print("\nincrement ~0 => metric carries no info beyond the confounds. residual drop "
          "alone can over-correct; the increment is the fair test.")


if __name__ == "__main__":
    main()
