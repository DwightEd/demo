"""Full detector: LDA (within-class whitening = remove the shared direction) on multi-layer kappa + own entropy, vs EDIS; cross-fit AUROC + bucket + bootstrap CI + strict-baseline geometry increment."""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def auroc(s, y):
    m = np.isfinite(s); s, y = s[m], y[m]
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if not p or not n:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def bucket(s, y, nt, nb=6):
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne = int(y[mm].sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def edis(H, w=8, tb=1.36, tr=1.33):
    H = np.asarray(H, float); H = H[np.isfinite(H)]
    if len(H) < 3:
        return 0.0
    ww = min(w, max(2, len(H) // 2))
    burst = sum(1 for t in range(len(H) - ww) if H[t + ww] - H[t] > tb) if len(H) > ww else 0
    reb = 0; rmin = H[0]
    for t in range(1, len(H)):
        if H[t] - rmin > tr:
            reb += 1
        rmin = min(rmin, H[t])
    return 0.5 * (burst + reb) * (1.0 + float(H.var()))


def unc_feats(e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]
    if len(e) < 2:
        return [float(e.mean()) if len(e) else 0.0, 0, 0, 0, 0, 0]
    t = np.arange(len(e)); sl = float(np.polyfit(t, e, 1)[0]) if len(e) >= 3 else 0.0
    return [float(e.mean()), float(e.var()), float(e.max()), float(e.max() - e.min()), sl, float(e[-max(1, len(e) // 3):].mean())]


def oof(X, y, g, lda):
    X = np.column_stack(X) if isinstance(X, list) else X; s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = (make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) if lda
               else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]; l14 = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int); UD = z["tok_U_D"]
    kfam = [cn.index(c) for c in ("resultant", "resultant_bulk", "resultant_unif", "coherence") if c in cn]; ri = cn.index("resultant")
    GM, G14, UN, ED, NT, Y, GG = [], [], [], [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            if hi - lo < 2:
                continue
            GM.append(sc[j][:, kfam].ravel()); G14.append(sc[j, l14, ri]); UN.append(unc_feats(ud[lo:hi]))
            ED.append(edis(ud[lo:hi])); NT.append(float(hi - lo)); Y.append(lab); GG.append(i)
    GM = np.asarray(GM, float); G14 = np.asarray(G14); UN = np.asarray(UN, float); ED = np.asarray(ED)
    NT = np.asarray(NT, float); Y = np.asarray(Y, int); GG = np.asarray(GG, int)
    cm = np.nanmean(np.where(np.isfinite(GM), GM, np.nan), 0); GM = np.where(np.isfinite(GM), GM, cm)
    uc = [UN[:, c] for c in range(UN.shape[1])]

    s_ours = oof(np.column_stack([GM] + uc), Y, GG, lda=True)       # LDA: multilayer geom + own entropy, zero EDIS
    s_log = oof(np.column_stack([GM] + uc), Y, GG, lda=False)
    print(f"{args.npz} | layers {lyu} | steps {len(Y)} err {int(Y.sum())}")
    for nm, v in [("EDIS", ED), ("geom L14", -G14), ("OURS-LDA", s_ours), ("OURS-logit", s_log)]:
        print(f"  {nm:12s} AUROC {bdir(auroc(v, Y)):.3f}  bkt {bucket(v, Y, NT):.3f}")
    aE = bdir(auroc(ED, Y)); aO = bdir(auroc(s_ours, Y))
    ch = np.unique(GG); rng = np.random.default_rng(0); ci = {c: np.where(GG == c)[0] for c in ch}; ds = []
    for _ in range(200):
        idx = np.concatenate([ci[c] for c in rng.choice(ch, len(ch), replace=True)])
        ds.append(bdir(auroc(s_ours[idx], Y[idx])) - bdir(auroc(ED[idx], Y[idx])))
    cl, cu = np.percentile(ds, [2.5, 97.5])
    print(f"  OURS-LDA - EDIS = {aO - aE:+.3f}  95% CI [{cl:+.3f}, {cu:+.3f}]")
    base = oof(np.column_stack(uc + [ED, NT]), Y, GG, lda=False); bg = oof(np.column_stack([GM] + uc + [ED, NT]), Y, GG, lda=False)
    print(f"  STRICT geom over [entropy+EDIS+len]: {bdir(auroc(base, Y)):.3f} -> {bdir(auroc(bg, Y)):.3f} ({bdir(auroc(bg,Y))-bdir(auroc(base,Y)):+.3f})")


if __name__ == "__main__":
    main()
