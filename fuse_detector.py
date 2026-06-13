"""R1 fused step-level detector: does combining the geometric structure beat the
single hand-crafted scalar (resultant), UNDER the anti-difficulty-inflation
protocol?

Step-level, ProcessBench process labels (positive = gold first-error step;
negative = correct-chain steps + pre-error steps). GroupKFold BY CHAIN so no
problem leaks train->test (this is what kills difficulty inflation). We report
out-of-fold AUROC and a chain-paired bootstrap CI on the increments:

    baseline      = nuisance (n_tok, pos, density) [+ U_D/U_C if --with_ud]
    + best single = baseline + the single best (metric,layer) column
    + FUSED       = baseline + ALL (metric x layer) geometric columns

The two questions the paper needs answered:
  (1) FUSED - baseline      : does geometry add signal beyond confound+uncertainty?
  (2) FUSED - best single   : does the FULL STRUCTURE beat ONE scalar (resultant)?
      -> if (2) ~ 0, the single feature already captures it (pooling/scalar is
         enough, ML over these features is overkill -> go richer: token clouds);
      -> if (2) > 0 significant, cross-layer / multi-metric structure is real
         signal the scalar throws away.

Concatenating each metric across ALL stored layers gives the cross-layer profile
for free (the structure a single-layer scalar discards).
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
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
    return 1.0 - letters / len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--metrics", default="resultant,coherence,cloud_D,cloud_C,norm,pr,ae",
                    help="comma list; each resolved in stepgeom or stepcloud, "
                         "concatenated across --layers")
    ap.add_argument("--layers", default="all", help="'all' or comma list of layers")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--with_ud", action="store_true",
                    help="put per-step U_D/U_C in the baseline (increment = beyond "
                         "uncertainty too)")
    ap.add_argument("--model", default="logit", choices=["logit", "gbm"])
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    geom_names = [str(x) for x in z["geom_feature_names"]]
    cloud_names = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers_used = [int(x) for x in z["layers_used"]]
    if args.layers == "all":
        sel_layers = layers_used
    else:
        sel_layers = [int(x) for x in args.layers.split(",")]
    li_sel = [layers_used.index(l) for l in sel_layers]

    # resolve each metric -> (source object-array, feature index)
    cloud_ok = bool(z.get("cloud_stored", np.array(False)))
    srcs = {}
    for m in args.metrics.split(","):
        m = m.strip()
        if m in geom_names:
            srcs[m] = (z["stepgeom"], geom_names.index(m))
        elif m in cloud_names and cloud_ok:
            srcs[m] = (z["stepcloud"], cloud_names.index(m))
        else:
            raise SystemExit(f"metric {m!r} not found (geom={geom_names}, cloud={cloud_names})")
    metric_list = list(srcs.keys())

    ges = z["gold_error_step"].astype(int)
    SR = z["step_token_ranges"]; ST = z["steps_text"]
    UD = z["tok_U_D"] if ("tok_U_D" in z.files and args.with_ud) else None
    UC = z["tok_U_C"] if ("tok_U_C" in z.files and args.with_ud) else None

    # column layout of the geometric block: [metric0 x layers..., metric1 x layers..., ...]
    col_names = [f"{m}@L{layers_used[li]}" for m in metric_list for li in li_sel]

    NUIS, GEO, UEX, Y, G = [], [], [], [], []
    nchain = len(z["stepgeom"])
    for i in range(nchain):
        # need every requested source present for this chain
        rng = np.asarray(SR[i], int)
        txt = list(ST[i]) if i < len(ST) else []
        k = int(ges[i]); correct = (k < 0)
        # pre-pull metric arrays for the chain
        marr = {}
        ok = True
        for m in metric_list:
            SRC, fi = srcs[m]
            if SRC[i] is None:
                ok = False; break
            g = np.asarray(SRC[i], float)
            marr[m] = g[:, :, fi] if g.ndim == 3 else g
        if not ok:
            continue
        T = next(iter(marr.values())).shape[0]
        a0 = int(rng[0, 0])
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            geo_row = []
            bad = False
            for m in metric_list:
                row = marr[m][j, li_sel]
                if not np.isfinite(row).all():
                    bad = True; break
                geo_row.extend(row.tolist())
            if bad:
                continue
            ntok = int(rng[j, 1] - rng[j, 0] + 1) if j < rng.shape[0] else np.nan
            dens = density(txt[j]) if j < len(txt) else np.nan
            NUIS.append([ntok, j / max(1, T - 1), dens]); GEO.append(geo_row)
            Y.append(y); G.append(i)
            if ud is not None:
                lo = int(rng[j, 0]) - a0; hi = int(rng[j, 1]) - a0 + 1
                lo, hi = max(0, lo), min(len(ud), hi)
                UEX.append([np.nanmean(ud[lo:hi]) if hi > lo else np.nan,
                            np.nanmean(uc[lo:hi]) if hi > lo else np.nan])

    NUIS = np.asarray(NUIS, float); GEO = np.asarray(GEO, float)
    Y = np.asarray(Y, int); G = np.asarray(G, int)
    for c in range(NUIS.shape[1]):
        col = NUIS[:, c]; col[~np.isfinite(col)] = np.nanmean(col)
    base_X = NUIS.copy(); base_lbl = "n_tok,pos,density"
    if UEX:
        UEX = np.asarray(UEX, float)
        for c in range(UEX.shape[1]):
            col = UEX[:, c]; col[~np.isfinite(col)] = np.nanmean(col)
        base_X = np.c_[base_X, UEX]; base_lbl += "+U_D,U_C"

    print(f"file: {args.npz} | model {args.model}")
    print(f"metrics: {metric_list}  x  layers {sel_layers}  -> {GEO.shape[1]} geo cols")
    print(f"steps: {len(Y)} | first-error: {int(Y.sum())} | chains: {len(np.unique(G))}")

    gkf = GroupKFold(args.folds)

    def oof(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            if args.model == "gbm":
                clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                 random_state=0)
                clf.fit(X[tr], Y[tr])
            else:
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=3000, C=1.0))
                clf.fit(X[tr], Y[tr])
            s[te] = clf.predict_proba(X[te])[:, 1]
        return s

    # single best geo column (the "one scalar" champion, e.g. resultant@L14)
    col_auc = [bdir(auroc(GEO[:, c], Y)) for c in range(GEO.shape[1])]
    cbest = int(np.nanargmax(col_auc))
    print(f"\nbest single geo column: {col_names[cbest]}  (raw AUROC {col_auc[cbest]:.3f})")

    s_base = oof(base_X)
    s_one = oof(np.c_[base_X, GEO[:, [cbest]]])
    s_fused = oof(np.c_[base_X, GEO])
    a_base, a_one, a_fused = auroc(s_base, Y), auroc(s_one, Y), auroc(s_fused, Y)

    print(f"\n{'detector':32s} {'AUROC':>7s}")
    print(f"{'baseline ('+base_lbl+')':32s} {a_base:7.3f}")
    print(f"{'+ best single ('+col_names[cbest]+')':32s} {a_one:7.3f}")
    print(f"{'+ FUSED (all geo)':32s} {a_fused:7.3f}")

    # chain-paired bootstrap CIs on the two increments
    rng = np.random.default_rng(0); chains = np.unique(G)
    d_fb, d_fo = [], []
    for _ in range(args.boot):
        cb = rng.choice(chains, size=len(chains), replace=True)
        mask = np.concatenate([np.where(G == c)[0] for c in cb])
        d_fb.append(auroc(s_fused[mask], Y[mask]) - auroc(s_base[mask], Y[mask]))
        d_fo.append(auroc(s_fused[mask], Y[mask]) - auroc(s_one[mask], Y[mask]))

    def ci(d, name):
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        sig = "SIGNIFICANT" if lo > 0 else "ns"
        print(f"  {name:42s} +{np.nanmean(d):.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}")

    print("\n=== increments (chain-paired bootstrap) ===")
    ci(d_fb, "FUSED - baseline (geo beyond confound+unc)")
    ci(d_fo, "FUSED - best single (structure beyond 1 scalar)")
    print("\nread: (1)>0 => geometry adds beyond confound+uncertainty. "
          "(2)>0 => the FULL cross-layer/multi-metric STRUCTURE beats the single "
          "scalar -> ML over these features is worth it; (2)~0 => the scalar "
          "already captures it -> go richer (token clouds, R2), not a bigger classifier.")


if __name__ == "__main__":
    main()
