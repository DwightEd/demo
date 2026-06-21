"""Nested logistic fusion: the honest decomposition of the step-level detector.

A single scalar (resultant ~0.77) understates what a linear model can do, but a flat fusion
(~0.85) hides how much is just length. So we fit NESTED GroupKFold logistic models and read off
each block's marginal contribution:

  M_len   = [log n_tok, position]                         confound-only baseline
  M_unc   = M_len + [U_D, U_C]                            + token uncertainty
  M_geo   = M_len + [resultant, coherence, norm, cloud_D] + geometry (no uncertainty)
  M_full  = M_len + uncertainty + geometry               the full fused detector

Key increments (chain-paired bootstrap CI):
  geometry over [len+unc]   = AUROC(M_full) - AUROC(M_unc)   <- the real geometric contribution
  uncertainty over [len+geo]= AUROC(M_full) - AUROC(M_geo)
Reported pooled AND within-length-bucket, per config. The full number is the headline; the
geometry-over-[len+unc] increment with its CI is the honest scientific claim.

Needs coh.npz: stepcloud(resultant/coherence/cloud_D) + geom(norm) + tok_U_D/tok_U_C +
step_token_ranges + gold_error_step.
"""

from __future__ import annotations
import argparse
import numpy as np

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("needs scikit-learn")


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


def bucket(s, y, nt, nb=5):
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb; a = bdir(auroc(s[m], y[m])); ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def oof(X, y, grp, folds):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def boot_delta(sa, sb, y, grp, n=1000, seed=0):
    rng = np.random.default_rng(seed); ch = np.unique(grp)
    idx = {c: np.where(grp == c)[0] for c in ch}
    d0 = bdir(auroc(sb, y)) - bdir(auroc(sa, y)); ds = []
    for _ in range(n):
        pk = rng.choice(ch, len(ch), replace=True); ii = np.concatenate([idx[c] for c in pk])
        a, b = auroc(sa[ii], y[ii]), auroc(sb[ii], y[ii])
        if np.isfinite(a) and np.isfinite(b):
            ds.append(max(b, 1 - b) - max(a, 1 - a))
    lo, hi = np.percentile(ds, [2.5, 97.5]) if ds else (np.nan, np.nan)
    return d0, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    gn = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC = z["stepcloud"]; SG = z["stepgeom"] if "stepgeom" in z.files else None
    SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    def cidx(nm): return cn.index(nm) if nm in cn else None
    GEO_C = [m for m in ["resultant", "coherence", "cloud_D"] if m in cn]

    F, Y, NT, POS, G = [], [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); sg = np.asarray(SG[i], float) if SG is not None else None
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0); T = rng.shape[0]
        a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            geo = [sc[j, li, cidx(m)] for m in GEO_C]
            geo.append(sg[j, li, gn.index("norm")] if (sg is not None and "norm" in gn) else np.nan)
            lo = max(0, int(rng[j, 0]) - a0)
            hiu = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            udv = np.nanmean(ud[lo:hiu]) if (ud is not None and hiu > lo) else np.nan
            ucv = np.nanmean(uc[lo:hiu]) if (uc is not None and hiu > lo) else np.nan
            ntok = int(rng[j, 1] - rng[j, 0] + 1)
            F.append(geo + [udv, ucv, np.log(max(ntok, 1)), j / max(1, T - 1)])
            Y.append(lab); NT.append(ntok); POS.append(j / max(1, T - 1)); G.append(i)
    F = np.asarray(F, float); Y = np.asarray(Y, int); NT = np.asarray(NT, float); G = np.asarray(G, int)
    for c in range(F.shape[1]):
        col = F[:, c]; col[~np.isfinite(col)] = np.nanmean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0

    ng = len(GEO_C) + 1                       # geometry columns: GEO_C + norm
    gi = list(range(ng)); ui = [ng, ng + 1]; ci = [ng + 2, ng + 3]      # geom / unc / [logn,pos]
    blocks = {
        "M_len  [len+pos]": ci,
        "M_unc  [+U_D,U_C]": ci + ui,
        "M_geo  [+geometry]": ci + gi,
        "M_full [geo+unc+len]": ci + ui + gi,
    }
    sc_ = {}
    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} first-error {int(Y.sum())} | geom cols {GEO_C}+norm")
    print(f"\n{'model':24s} {'pooled':>7s} {'bucket':>7s}")
    for nm, cols in blocks.items():
        s = oof(F[:, cols], Y, G, args.folds); sc_[nm] = s
        print(f"  {nm:24s} {bdir(auroc(s, Y)):7.3f} {bucket(s, Y, NT):7.3f}")
    # single resultant for reference
    ridx = GEO_C.index("resultant") if "resultant" in GEO_C else 0
    print(f"  {'(single resultant)':24s} {bdir(auroc(F[:, ridx], Y)):7.3f} {bucket(F[:, ridx], Y, NT):7.3f}")

    print(f"\nincrements (chain-paired bootstrap 95% CI):")
    dg, lg, hg = boot_delta(sc_["M_unc  [+U_D,U_C]"], sc_["M_full [geo+unc+len]"], Y, G, args.boot)
    du, lu, hu = boot_delta(sc_["M_geo  [+geometry]"], sc_["M_full [geo+unc+len]"], Y, G, args.boot)
    print(f"  geometry over [len+unc]    {dg:+.3f}  [{lg:+.3f},{hg:+.3f}]  {'*' if lg>0 else 'ns'}  <- real geometric contribution")
    print(f"  uncertainty over [len+geo] {du:+.3f}  [{lu:+.3f},{hu:+.3f}]  {'*' if lu>0 else 'ns'}")

    print("\nread: M_full is the headline fused detector. The DECISIVE scientific number is geometry-"
          "over-[len+unc]: how much the directional-collapse signal adds beyond length AND uncertainty, "
          "with chain-paired bootstrap CI. '*' = CI clears 0 = a real, confound-controlled contribution. "
          "Compare pooled (length-inflated) vs bucket (length-controlled).")


if __name__ == "__main__":
    main()
