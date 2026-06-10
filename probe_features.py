"""Tracing-style regression over the EXISTING signals (no SPE).

For every per-step / per-token signal -- U_D, U_C (paper) and the geometry family
(norm, pr, ae, ed_half, e50, e90, ae_robust, anom_k5, anom_k10 per layer) --
summarize each chain by the 5 trace-profile features (mu_early, mu_mid, mu_late,
slope m, linear-fit r2), then train a within-problem logistic regression and ask:

  do the DYNAMIC features (slope, r2) strengthen discrimination over the static
  means alone? and does the geometry add anything over the paper U_D/U_C?

Within-problem AUROC = paired AUROC of out-of-fold probe scores under GroupKFold
by problem_id (difficulty controlled + generalization). Ablations compare
feature subsets. --format_ok_only isolates the reasoning signal from format.

Needs an extract_features.py npz (uses profile_paper + stepgeom; no step vectors).
"""

from __future__ import annotations

import argparse
import numpy as np

from features.trace_profile import profile, PROFILE_STATS

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("probe_features needs scikit-learn (pip install scikit-learn)")


def within_auroc(score, y, pid):
    conc = tie = npair = 0.0
    for p in np.unique(pid):
        m = (pid == p) & np.isfinite(score)
        se, sc = score[m & (y == 1)], score[m & (y == 0)]
        if se.size and sc.size:
            d = se[:, None] - sc[None, :]
            conc += (d > 0).sum(); tie += (d == 0).sum(); npair += d.size
    return (conc + 0.5 * tie) / npair if npair else float("nan")


def auroc(score, y):
    m = np.isfinite(score); s, yy = score[m], y[m]
    if (yy == 1).sum() == 0 or (yy == 0).sum() == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    npos = (yy == 1).sum(); nneg = (yy == 0).sum()
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def build_table(z):
    """Return (X (N,P), colnames) of trace-profile features for all signals."""
    cols = [str(c) for c in z["profile_cols"]]
    P = np.asarray(z["profile_paper"], float)                 # (N, len(cols))
    names = [str(x) for x in z["geom_feature_names"]]
    layers = [int(x) for x in z["layers_used"]]
    SG = z["stepgeom"]
    N = len(SG)

    geo_cols = [f"{fn}_L{ly}_{st}" for ly in layers for fn in names
                for st in PROFILE_STATS]
    geo = np.full((N, len(geo_cols)), np.nan)
    for i in range(N):
        g = np.asarray(SG[i], float)                          # (T, L, F)
        c = 0
        for li in range(len(layers)):
            for fi in range(len(names)):
                pr = profile(g[:, li, fi]) if g.ndim == 3 else {}
                for st in PROFILE_STATS:
                    geo[i, c] = pr.get(st, np.nan); c += 1
    blocks = [P, geo]
    allcols = cols + geo_cols
    # point-cloud D/V/C profiles (same per-step profiling as geometry)
    if "stepcloud" in z.files and bool(z.get("cloud_stored", np.array(False))):
        cnames = [str(x) for x in z["cloud_feature_names"]]
        SC = z["stepcloud"]
        cl_cols = [f"{fn}_L{ly}_{st}" for ly in layers for fn in cnames
                   for st in PROFILE_STATS]
        cl = np.full((N, len(cl_cols)), np.nan)
        for i in range(N):
            g = np.asarray(SC[i], float)
            c = 0
            for li in range(len(layers)):
                for fi in range(len(cnames)):
                    pr = profile(g[:, li, fi]) if g.ndim == 3 else {}
                    for st in PROFILE_STATS:
                        cl[i, c] = pr.get(st, np.nan); c += 1
        blocks.append(cl); allcols = allcols + cl_cols
    # whole-chain intrinsic dimension (already chain-level, one per layer)
    if "chain_intrinsic" in z.files and bool(z.get("intrinsic_stored", np.array(False))):
        inames = [str(x) for x in z["intrinsic_names"]]
        CI = np.asarray(z["chain_intrinsic"], float)             # (N, L, n_est)
        id_cols = [f"{en}_L{ly}" for li, ly in enumerate(layers) for en in inames]
        idm = np.column_stack([CI[:, li, ei] for li in range(len(layers))
                               for ei in range(len(inames))])
        blocks.append(idm); allcols = allcols + id_cols
    X = np.column_stack(blocks)
    # length baseline appended last
    if "n_steps" in z.files:
        X = np.column_stack([X, z["n_steps"].astype(float)])
        allcols = allcols + ["n_steps"]
    # drop all-NaN columns (e.g. UE_* when U_E was skipped) -> no imputer warnings
    good = np.isfinite(X).any(axis=0)
    X = X[:, good]
    allcols = [c for c, g in zip(allcols, good) if g]
    return X, allcols


def oof(X, y, groups, mask, C, folds=5):
    Xs = X[:, mask]
    if Xs.shape[1] == 0:
        return np.full(len(y), np.nan)
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(Xs, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000))
        pipe.fit(Xs[tr], y[tr])
        s[te] = pipe.predict_proba(Xs[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--C", type=float, default=0.5, help="L2 inverse-strength.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--format_ok_only", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    X, cols = build_table(z)
    pid = z["problem_ids"].astype(int)
    y = (z["is_correct"].astype(int) == 0).astype(int)        # error=1 (answer-only)
    keep = np.ones(len(pid), bool)
    if args.format_ok_only and "format_ok" in z.files:
        keep = z["format_ok"].astype(int) == 1
        print(f"[format_ok=1 subset: {int(keep.sum())}/{len(keep)}]")
    X, y, pid = X[keep], y[keep], pid[keep]

    cset = lambda f: np.array([f(c) for c in cols])
    is_dyn = cset(lambda c: c.endswith(("_slope", "_r2")))
    is_len = cset(lambda c: c in ("n_steps", "n_resp_tokens"))
    is_paper = cset(lambda c: c.startswith(("UD_", "UC_", "UE_")))
    is_id = cset(lambda c: c.startswith("id_"))
    is_static = ~is_dyn & ~is_len                 # means + chain-level scalars (id_*)
    is_geom = ~is_paper & ~is_len

    sets = {
        "length(n_steps)":   is_len,
        "paper static":      is_paper & is_static,
        "paper +dyn":        is_paper,
        "geom static":       is_geom & is_static,
        "geom +dyn":         is_geom,
        "id_dim only":       is_id,
        "paper + id_dim":    is_paper | is_id,
        "ALL static":        is_static,
        "ALL +dyn":          is_static | is_dyn,
        "dyn only":          is_dyn,
    }

    print(f"file: {args.npz} | chains {len(y)} | error(answer) {int(y.sum())} | "
          f"features total {len(cols)} (static {int(is_static.sum())}, "
          f"dyn {int(is_dyn.sum())}) | C={args.C}")
    print(f"\n{'feature set':18s} {'n_feat':>6s} {'within_ans':>10s} {'cross_ans':>9s}")
    print("-" * 48)
    results = []
    for name, mask in sets.items():
        nf = int(mask.sum())
        if nf == 0:
            continue
        s = oof(X, y, pid, mask, args.C, args.folds)
        wa, ca = within_auroc(s, y, pid), auroc(s, y)
        print(f"{name:18s} {nf:6d} {wa:10.3f} {ca:9.3f}")
        results.append({"set": name, "n_feat": nf,
                        "within_ans": round(float(wa), 3) if np.isfinite(wa) else None,
                        "cross_ans": round(float(ca), 3) if np.isfinite(ca) else None})

    print("\nKey: 'ALL +dyn' vs 'ALL static' = does slope/r2 add discrimination; "
          "'geom +dyn' vs 'paper +dyn' = does our geometry add over U_D/U_C.")
    if args.json:
        import json
        print("\n=== JSON ===")
        print(json.dumps({"meta": {"file": args.npz, "chains": int(len(y)),
                                   "format_ok_only": bool(args.format_ok_only),
                                   "C": args.C}, "results": results}, indent=2))


if __name__ == "__main__":
    main()
