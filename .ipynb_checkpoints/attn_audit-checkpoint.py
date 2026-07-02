"""Attention-sink channel audit: does the DIRECT attention-end read of anchoring
add signal ORTHOGONAL to the (one) geometric hidden-state signal?

Per (step, layer) features (extract_features --attn_sink):
  sink_frac    attention mass on position 0 (BOS attention sink)
  q_frac       attention mass on the question/prompt span (problem-relevant region)
  attn_entropy entropy of each token's attention (low = focused/anchored, high = diffuse)

Reports: each feature's AUROC; corr with the geometric signal (resultant) -> orthogonal?;
and the KEY increment -- attention features added on top of [confound + U + geometry],
GroupKFold by chain, chain-paired bootstrap. If SIG > 0 and corr low, attention sink is a
new orthogonal axis -> fusing it levels up the detector (the user's strengthening bet).
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


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m])))[0, 1])


def density(t):
    t = str(t); return 1.0 - sum(c.isalpha() for c in t) / len(t) if t else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("attn_stored", np.array(False))):
        raise SystemExit("no stepattn; re-extract with --attn_sink")
    an = [str(x) for x in z["attn_names"]]
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SA, SG, SC, SR = z["stepattn"], z["stepgeom"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int); ST = z["steps_text"]
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None
    rkey = "resultant" if "resultant" in cnames else None

    cols = {k: [] for k in an + ["resultant", "U_D", "U_C", "n_tok", "pos", "dens"]}
    Y, G = [], []
    for i in range(len(SA)):
        sa = np.asarray(SA[i], float); sg = np.asarray(SG[i], float)
        sc = np.asarray(SC[i], float) if rkey else None
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0); T = rng.shape[0]
        a0 = int(rng[0, 0]); txt = list(ST[i]) if i < len(ST) else []
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            for ai, nm in enumerate(an):
                cols[nm].append(sa[j, li, ai])
            cols["resultant"].append(sc[j, li, cnames.index("resultant")] if rkey else np.nan)
            cols["n_tok"].append(int(rng[j, 1] - rng[j, 0] + 1))
            cols["pos"].append(j / max(1, T - 1))
            cols["dens"].append(density(txt[j]) if j < len(txt) else np.nan)
            lo = max(0, int(rng[j, 0]) - a0); hi = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            cols["U_D"].append(np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan)
            cols["U_C"].append(np.nanmean(uc[lo:hi]) if (uc is not None and hi > lo) else np.nan)
            Y.append(y); G.append(i)
    for k in cols:
        c = np.asarray(cols[k], float); c[~np.isfinite(c)] = np.nanmean(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
        cols[k] = c
    Y = np.asarray(Y, int); G = np.asarray(G, int)

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'attn feature':14s} {'AUROC':>7s} {'corr(.,resultant)':>18s}")
    for nm in an:
        print(f"{nm:14s} {bdir(auroc(cols[nm], Y)):7.3f} {spear(cols[nm], cols['resultant']):18.2f}")

    gkf = GroupKFold(args.folds)
    def oof(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p.fit(X[tr], Y[tr]); s[te] = p.predict_proba(X[te])[:, 1]
        return s
    base = np.c_[cols["n_tok"], cols["pos"], cols["dens"], cols["U_D"], cols["U_C"], cols["resultant"]]
    attnX = np.c_[[cols[nm] for nm in an]].T
    s_b = oof(base); s_ba = oof(np.c_[base, attnX])
    a_b, a_ba = auroc(s_b, Y), auroc(s_ba, Y)
    rng = np.random.default_rng(0); ch = np.unique(G); d = []
    for _ in range(2000):
        cb = rng.choice(ch, len(ch), replace=True)
        m = np.concatenate([np.where(G == c)[0] for c in cb])
        d.append(auroc(s_ba[m], Y[m]) - auroc(s_b[m], Y[m]))
    d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
    print(f"\n=== does attention add over [confound+U+geometry]? ===")
    print(f"  baseline (n_tok,pos,dens,U_D,U_C,resultant): {a_b:.3f}")
    print(f"  + attention (sink,q,entropy):                {a_ba:.3f}")
    print(f"  ATTENTION INCREMENT: +{np.nanmean(d):.3f} [{lo:+.3f},{hi:+.3f}] "
          f"{'SIGNIFICANT' if lo > 0 else 'ns'}")
    print("\nread: SIG>0 + low corr with resultant => attention sink is a NEW orthogonal axis, "
          "fusing it levels up the detector. ns => redundant with the geometric/uncertainty signal.")


if __name__ == "__main__":
    main()
